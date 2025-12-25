#!/usr/bin/env python3
"""
Convert VA-Count PyTorch model to Core ML .mlpackage

Usage:
  python convert_to_coreml.py --resume weights/checkpoint_FSC.pth --model mae_vit_base_patch16 --output weights/checkpoint_FSC.mlpackage
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'VA-Count'))

import models_mae_cross
try:
    import util.misc as misc
    _HAVE_MISC = True
except Exception:
    misc = None
    _HAVE_MISC = False


class ModelForCoreML(torch.nn.Module):
    def __init__(self, model, shot_num=3):
        super().__init__()
        self.model = model
        self.shot_num = shot_num

    def forward(self, imgs, boxes):
        # imgs: [B,3,384,384], boxes: [B,num_boxes,3,64,64]
        return self.model(imgs, boxes, self.shot_num)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='weights/checkpoint_FSC.pth')
    parser.add_argument('--model', default='mae_vit_base_patch16')
    parser.add_argument('--output', default='weights/checkpoint_FSC.mlpackage')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--shot_num', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    print('Instantiating model:', args.model)
    model = models_mae_cross.__dict__[args.model](norm_pix_loss=False)
    model.to(device)
    model.eval()

    # load checkpoint (use existing helper to be robust)
    class CArgs:
        resume = args.resume
    try:
        misc.load_model_FSC(CArgs, model)
    except Exception as e:
        print('Warning: misc.load_model_FSC failed, attempting direct load:', e)
        # Newer torch versions may default to weights_only=True; try allowing pickled objects first
        try:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location='cpu')
        if 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)

    # wrap model to fix shot_num (easier to trace)
    wrapper = ModelForCoreML(model, shot_num=args.shot_num)
    wrapper.to('cpu')
    wrapper.eval()

    # example inputs
    img_ex = torch.randn(1, 3, 384, 384)
    boxes_ex = torch.randn(1, 3, 3, 64, 64)

    print('Tracing the model (this may take a moment)...')
    try:
        traced = torch.jit.trace(wrapper, (img_ex, boxes_ex), strict=False)
    except Exception as e:
        print('Torchscript trace failed:', e)
        print('Trying scripting instead...')
        traced = torch.jit.script(wrapper)

    try:
        import coremltools as ct
    except Exception as e:
        print('coremltools is not installed. Install it with: pip install coremltools')
        raise

    print('Converting to Core ML (.mlpackage) ...')
    try:
        # Attempt to produce an .mlpackage directly (if supported by installed coremltools)
        mlpackage = ct.convert(
            traced,
            source='pytorch',
            inputs=[
                ct.TensorType(name='image', shape=img_ex.shape),
                ct.TensorType(name='boxes', shape=boxes_ex.shape),
            ],
            convert_to='mlpackage'
        )
    except Exception as e:
        print('Direct .mlpackage conversion not supported in this coremltools:', e)
        print('Falling back to create a .mlmodel file instead.')
        mlmodel = ct.convert(
            traced,
            source='pytorch',
            inputs=[
                ct.TensorType(name='image', shape=img_ex.shape),
                ct.TensorType(name='boxes', shape=boxes_ex.shape),
            ],
        )
        out_path = Path(args.output)
        # if user asked for .mlpackage, change to .mlmodel and save
        if out_path.suffix == '.mlpackage':
            tmp = out_path
        else:
            tmp = out_path.with_suffix('.mlmodel')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(tmp))
        print('Saved .mlmodel to', tmp)
        print('To produce a .mlpackage, upgrade coremltools or package the model with Xcode/Apple tools.')
        return

    # save mlpackage
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print('Saving to', out)
    try:
        mlpackage.save(str(out))
    except Exception as e:
        # for some versions mlpackage is an object with save_package
        try:
            mlpackage.save_package(str(out))
        except Exception:
            print('Failed to save mlpackage object:', e)
            raise

    print('Conversion complete. Output:', out)


if __name__ == '__main__':
    main()
