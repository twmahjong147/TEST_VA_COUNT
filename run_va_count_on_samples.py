#!/usr/bin/env python3
"""
Run VA-Count model on all images in `samples/` and save outputs to `outputs/`.

This script loads the model from `weights/checkpoint_FSC.pth` by default and
processes each image in `samples/` producing a JSON with predicted count and
a density heatmap PNG in `outputs/`.
"""
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import math
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi

import sys
from pathlib import Path

# ensure VA-Count folder is on path so imports like `models_mae_cross` and `util` resolve
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'VA-Count'))

import models_mae_cross
import util.misc as misc


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def make_exemplar_boxes(img_w, img_h, box_size=64, shot_num=3):
    """Return a list of `shot_num` boxes (each as [x1,y1,x2,y2]) on the resized image."""
    boxes = []
    cx, cy = img_w // 2, img_h // 2
    # center
    boxes.append((cx - box_size // 2, cy - box_size // 2, cx + box_size // 2, cy + box_size // 2))
    # top-left-ish
    boxes.append((max(0, cx - img_w // 4 - box_size // 2), max(0, cy - img_h // 4 - box_size // 2),
                  max(0, cx - img_w // 4 - box_size // 2) + box_size, max(0, cy - img_h // 4 - box_size // 2) + box_size))
    # bottom-right-ish
    boxes.append((min(img_w - box_size, cx + img_w // 4 - box_size // 2), min(img_h - box_size, cy + img_h // 4 - box_size // 2),
                  min(img_w - box_size, cx + img_w // 4 - box_size // 2) + box_size, min(img_h - box_size, cy + img_h // 4 - box_size // 2) + box_size))
    return boxes[:shot_num]


def preprocess_image(pil_img, target_size=384):
    pil_img = pil_img.convert('RGB')
    resized = pil_img.resize((target_size, target_size), Image.BICUBIC)
    return resized


def crop_and_prepare(resized_img, boxes, crop_size=64):
    crops = []
    for (x1, y1, x2, y2) in boxes:
        x1c = int(max(0, x1))
        y1c = int(max(0, y1))
        x2c = int(min(resized_img.width, x2))
        y2c = int(min(resized_img.height, y2))
        crop = resized_img.crop((x1c, y1c, x2c, y2c)).resize((crop_size, crop_size), Image.BICUBIC)
        crop_t = TF.to_tensor(crop)
        crops.append(crop_t)
    if len(crops) == 0:
        crop = resized_img.crop((resized_img.width//2 - crop_size//2, resized_img.height//2 - crop_size//2,
                                 resized_img.width//2 + crop_size//2, resized_img.height//2 + crop_size//2)).resize((crop_size, crop_size), Image.BICUBIC)
        crops = [TF.to_tensor(crop) for _ in range(3)]
    return torch.stack(crops)


def normalize_tensor(tensor):
    # create mean/std on the same device and dtype as `tensor`
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
    dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else torch.float32
    mean = torch.tensor(IM_NORM_MEAN, device=device, dtype=dtype).view(3,1,1)
    std = torch.tensor(IM_NORM_STD, device=device, dtype=dtype).view(3,1,1)
    return (tensor - mean) / std


def visualize_and_save(density, out_png_path):
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(density, cmap='jet')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def _normalize_density(d):
    d = d.astype('float32')
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    return d


def detect_peaks(density, sigma=2, min_distance=10, rel_thresh=0.12):
    """Return list of (x,y,conf) peaks in density image (coords in density pixel space)."""
    den_s = ndi.gaussian_filter(density, sigma=sigma)
    footprint = np.ones((2 * min_distance + 1, 2 * min_distance + 1))
    local_max = (den_s == ndi.maximum_filter(den_s, footprint=footprint))
    thresh = max(rel_thresh * den_s.max(), 0.01)
    ys, xs = np.where(local_max & (den_s >= thresh))
    peaks = [(int(x), int(y), float(den_s[y, x])) for (y, x) in zip(ys, xs)]
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks


def peaks_to_fixed_boxes(peaks, box_size=40, img_shape=None):
    boxes = []
    h = img_shape[0] if img_shape is not None else None
    w = img_shape[1] if img_shape is not None else None
    half = box_size // 2
    for x, y, conf in peaks:
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = x1 + box_size
        y2 = y1 + box_size
        if w is not None:
            x2 = min(w, x2)
            x1 = max(0, x2 - box_size)
        if h is not None:
            y2 = min(h, y2)
            y1 = max(0, y2 - box_size)
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return boxes


def draw_boxes_cv(img, boxes, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out


def save_boxes_json(path, boxes):
    j = [{"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "conf": float(conf)} for (x1, y1, x2, y2, conf) in boxes]
    with open(path, 'w') as f:
        json.dump(j, f, indent=2)


def run(args):
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model.eval()

    args.resume = args.resume
    misc.load_model_FSC(args, model)

    samples_dir = Path(args.samples_dir)
    imgs = sorted([p for p in samples_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if len(imgs) == 0:
        print('No images found in', samples_dir)
        return

    results = {}
    for p in imgs:
        try:
            pil_img = Image.open(p)
            resized = preprocess_image(pil_img, target_size=384)

            boxes_xy = make_exemplar_boxes(resized.width, resized.height, box_size=64, shot_num=3)
            neg_boxes_xy = [(max(0,x-20), max(0,y-20), min(resized.width, x2-20), min(resized.height, y2-20)) for (x,y,x2,y2) in boxes_xy]

            img_t = TF.to_tensor(resized).unsqueeze(0).to(device)
            img_t = normalize_tensor(img_t.squeeze(0)).unsqueeze(0).to(device)

            boxes_t = crop_and_prepare(resized, boxes_xy, crop_size=64).to(device)
            neg_boxes_t = crop_and_prepare(resized, neg_boxes_xy, crop_size=64).to(device)

            boxes_batch = boxes_t.unsqueeze(0)
            neg_boxes_batch = neg_boxes_t.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_t, boxes_batch, 3)
            pred_np = pred.squeeze(0).cpu().numpy()
            pred_count = float(pred_np.sum() / 60.0)

            base_name = p.stem
            out_json = out_dir / (base_name + '.json')
            out_png = out_dir / (base_name + '_density.png')
            visualize_and_save(pred_np, out_png)

            # additional outputs: detect peaks -> boxes and draw overlay
            den_norm = _normalize_density(pred_np)
            peaks = detect_peaks(den_norm, sigma=args.det_sigma, min_distance=args.det_min_distance, rel_thresh=args.det_rel_thresh)
            boxes = peaks_to_fixed_boxes(peaks, box_size=args.det_box_size, img_shape=den_norm.shape)

            # draw overlay on the resized image
            resized_bgr = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
            overlay = draw_boxes_cv(resized_bgr, boxes, color=(0,255,0), thickness=2)
            out_overlay = out_dir / (base_name + '_overlay.png')
            cv2.imwrite(str(out_overlay), overlay)

            # save boxes JSON
            out_boxes_json = out_dir / (base_name + '_boxes.json')
            save_boxes_json(str(out_boxes_json), boxes)

            results[p.name] = {
                'pred_count': pred_count,
                'density_png': str(out_png),
                'overlay_png': str(out_overlay),
                'boxes_json': str(out_boxes_json),
                'n_boxes': len(boxes)
            }
            with open(out_json, 'w') as f:
                json.dump(results[p.name], f)

            print(f'Processed {p.name}: predicted count {pred_count:.2f}')
        except Exception as e:
            print('Error processing', p, e)

    summary_path = out_dir / 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mae_vit_base_patch16', help='model name in models_mae_cross')
    parser.add_argument('--resume', default='weights/checkpoint_FSC.pth', help='path to checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--samples_dir', default='samples', help='directory with test images')
    parser.add_argument('--output_dir', default='outputs', help='where to save outputs')
    parser.add_argument('--norm_pix_loss', action='store_true')
    # detection/tuning params for converting density -> boxes
    parser.add_argument('--det_box_size', type=int, default=40, help='fixed box size for peak->box conversion')
    parser.add_argument('--det_sigma', type=float, default=2.0, help='gaussian smoothing sigma for density before peak detection')
    parser.add_argument('--det_min_distance', type=int, default=10, help='minimum pixel distance between detected peaks')
    parser.add_argument('--det_rel_thresh', type=float, default=0.12, help='relative threshold for peak acceptance (fraction of max)')
    args = parser.parse_args()
    run(args)
