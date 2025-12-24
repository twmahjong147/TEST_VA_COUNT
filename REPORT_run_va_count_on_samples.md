# Technical Report: run_va_count_on_samples.py

## Purpose
Run the VA-Count model on all images in `samples/` and save per-image outputs to `outputs/` for quick verification and visualization.

## Location
`run_va_count_on_samples.py` (project root)

## Summary of behavior
- Loads model from `weights/checkpoint_FSC.pth` (default) using `VA-Count/models_mae_cross` and helper `util.misc` loader.
- For each image in `samples/` (jpg/png), resizes to 384×384 and prepares 3 exemplar crops (default shot=3, crop_size=64).
- Runs the model to produce a density map (numpy array) per image.
- Computes a scalar predicted count as `pred_np.sum() / 60.0`.
- Saves a density heatmap PNG, an overlay PNG with fixed-size detected boxes, a boxes JSON, a per-image JSON summary, and a global `results_summary.json`.

## Key functions (in-script)
- `make_exemplar_boxes(img_w, img_h, box_size=64, shot_num=3)`: compute 3 exemplar box coordinates on resized image.
- `preprocess_image(pil_img, target_size=384)`: convert to RGB and bicubic resize.
- `crop_and_prepare(resized_img, boxes, crop_size=64)`: extract crops, resize to `crop_size`, return stacked tensors.
- `normalize_tensor(tensor)`: channel-wise normalization using IM_NORM_MEAN and IM_NORM_STD.
- `visualize_and_save(density, out_png_path)`: save density map as a colorized PNG (jet colormap).
- `detect_peaks(density, sigma=2, min_distance=10, rel_thresh=0.12)`: gaussian-smooth the density and find local maxima above threshold.
- `peaks_to_fixed_boxes(peaks, box_size=40, img_shape=None)`: convert peaks to fixed-size boxes with confidences.
- `draw_boxes_cv(img, boxes, color=(0,255,0), thickness=2)`: draw boxes on image and save.

## Detailed explanation: peak detection and box conversion

This section expands the two post-processing functions used to turn model density maps into discrete detections and visualization boxes: `detect_peaks` and `peaks_to_fixed_boxes`.

### `detect_peaks(density, sigma=2, min_distance=10, rel_thresh=0.12)`

1) Parameters — meaning and guidance
- `density` (2D numpy array): raw model output density in the resized image pixel space (e.g., 384×384). Values can be arbitrary float magnitudes depending on model scaling.
- `sigma` (float, default 2): standard deviation for Gaussian smoothing (pixels). Larger `sigma` blurs more, merging nearby peaks and suppressing small noisy spikes; smaller `sigma` preserves finer peaks but is more sensitive to noise. Choose `sigma` roughly comparable to the expected half-width (spread) of single-object responses in the density map. Typical values: 1.0–4.0.
- `min_distance` (int, default 10): the minimum allowed pixel distance between two detected peaks. Implements a simple spatial suppression (via a maximum filter footprint). Larger `min_distance` enforces sparser detections (useful when objects are not densely packed); smaller values allow closer detections (needed for crowded scenes). Choose based on expected object-to-object center distance in the resized image.
- `rel_thresh` (float, default 0.12): relative acceptance threshold expressed as a fraction of the (smoothed) density maximum. Peaks with smoothed value below `rel_thresh * max` are discarded. Larger `rel_thresh` raises the bar (fewer false positives, but may miss weak true objects); smaller values accept weaker peaks (higher recall, more false positives).

Guidance for selecting values:
- If detections are noisy (many spurious peaks): increase `sigma` and/or `rel_thresh` and/or `min_distance`.
- If you miss closely spaced objects: decrease `min_distance` and/or `sigma` (but monitor noise increase).
- If densities are low-magnitude overall, lower `rel_thresh` (but consider normalizing densities first).

2) Algorithm (step-by-step)
- Apply a Gaussian filter to `density` with standard deviation `sigma` to produce a smoothed map `den_s`. Gaussian smoothing reduces high-frequency noise and consolidates local mass associated with each object.
- Build a binary mask of local maxima by comparing `den_s` to a running maximum filter computed using a square `footprint` of size `(2 * min_distance + 1)`. A pixel is a local maximum if it equals the maximum within the footprint.
- Compute an absolute threshold `thresh = max(rel_thresh * den_s.max(), 0.01)` to avoid tiny thresholds when the map is near-zero. Retain only local maxima whose smoothed value >= `thresh`.
- Collect the (x, y, conf) triples for retained peaks, and sort them by confidence (descending).

Pseudocode:

```
den_s = gaussian_filter(density, sigma)
footprint = ones((2*min_distance+1, 2*min_distance+1))
local_max = (den_s == maximum_filter(den_s, footprint))
thresh = max(rel_thresh * den_s.max(), 0.01)
peaks = [(x,y,den_s[y,x]) for (y,x) where local_max and den_s[y,x] >= thresh]
sort peaks by confidence descending
return peaks
```

3) Why this methodology
- Gaussian smoothing effectively integrates nearby activations belonging to the same object while suppressing sensor/model noise. It makes peak detection robust to small localization jitter and distributed density responses.
- The maximum-filter-based local-maximum test performs a simple, fast non-maximum suppression (NMS) in image space without requiring heavy iterative NMS code. Using a square footprint parameterized by `min_distance` approximates the idea of forbidding multiple detections within a given radius.
- Relative thresholding (`rel_thresh * max`) adapts to variations in map amplitude between images: it keeps detections that are significant relative to the strongest response in the image, which is useful if absolute scale varies between images.

Trade-offs and limitations:
- This pipeline is simple and computationally light, suitable for visualization and quick inspection. However, it is heuristic: fixed `min_distance` and `box_size` assume roughly constant object sizes and spacing in pixel space.
- In highly heterogeneous scenes (objects of many sizes), consider adaptive methods (multi-scale smoothing, watershed segmentation, or learned keypoint regressors) instead.

### `peaks_to_fixed_boxes(peaks, box_size=40, img_shape=None)`

1) Parameters — meaning and guidance
- `peaks`: list of `(x, y, conf)` tuples (pixel coords in density image space) from `detect_peaks`.
- `box_size` (int, default 40): side-length in pixels of the returned square box. Larger `box_size` produces bigger visual boxes (may cover multiple objects if set too large); smaller `box_size` gives tighter boxes but may clip object parts. Choose according to expected object size in the resized image.
- `img_shape` (tuple or None): `(height, width)` of the image/density map. Used to clip boxes to image boundaries and to ensure boxes remain `box_size` in size when clipped.

Guidance:
- Align `box_size` with the expected object bounding box size after resizing to the model's input (384 in this repo). If uncertain, inspect a few annotated examples or tune by visual inspection.

2) Algorithm (step-by-step)
- For each peak `(x, y, conf)`, compute centered square coordinates using `half = box_size // 2`:
  - `x1 = x - half`, `y1 = y - half`, `x2 = x1 + box_size`, `y2 = y1 + box_size`.
- If `img_shape` is provided, clip `x2` to `width` and `y2` to `height`. If clipping shrinks the box, shift `x1` (or `y1`) so the box still has `box_size` width (i.e., `x1 = max(0, x2 - box_size)`). Same for vertical dimension.
- Append `(x1, y1, x2, y2, conf)` to output list.

Pseudocode:

```
boxes = []
half = box_size // 2
for (x, y, conf) in peaks:
  x1 = max(0, x - half)
  y1 = max(0, y - half)
  x2 = x1 + box_size
  y2 = y1 + box_size
  if img_shape:
    x2 = min(width, x2); x1 = max(0, x2 - box_size)
    y2 = min(height, y2); y1 = max(0, y2 - box_size)
  boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
return boxes
```

3) Why this methodology
- Fixed-size boxes are a pragmatic, deterministic way to visualize per-peak detections produced by a density map. They are computationally cheap and easy to interpret when object sizes are approximately uniform in pixel space.
- Using the peak confidence as the box `conf` provides a simple ordering to filter or rank detections.

When to replace with more advanced options:
- If object sizes vary widely, use an adaptive box estimator: e.g., estimate local object size from density spread, use multi-scale peak detection, or train a small regressor to predict width/height from local features.
- For precise localization and overlapping objects, consider a learned detection head or performing clustering on the density map (mean-shift, watershed on inverted density, or local contour extraction).

Combined workflow note
- Together, `detect_peaks` + `peaks_to_fixed_boxes` convert a continuous density image into a ranked set of point detections with simple square visualizations. The key hyperparameters (`sigma`, `min_distance`, `rel_thresh`, `box_size`) control the balance of noise suppression, detection granularity, and visualization scale — tune them on a representative validation set using visual inspection and simple metrics (precision/recall vs. ground-truth points).

## Input / Output
- Inputs:
  - `--samples_dir` (default `samples`): folder with images (.jpg/.jpeg/.png).
  - `--resume` (default `weights/checkpoint_FSC.pth`): model checkpoint.
- Outputs (to `--output_dir`, default `outputs`):
  - `{image_stem}.json` — per-image summary with `pred_count`, paths to `density_png`, `overlay_png`, `boxes_json`, and `n_boxes`.
  - `{image_stem}_density.png` — heatmap visualization of raw density.
  - `{image_stem}_overlay.png` — resized image with detected boxes drawn.
  - `{image_stem}_boxes.json` — array of boxes with x1,y1,x2,y2,conf.
  - `results_summary.json` — mapping of image filenames to their per-image summary.

## CLI arguments (defaults)
- `--model maevit_vit_base_patch16` (actual default in script: `mae_vit_base_patch16`)
- `--resume weights/checkpoint_FSC.pth`
- `--device cuda` (falls back to CPU if CUDA unavailable)
- `--samples_dir samples`
- `--output_dir outputs`
- `--norm_pix_loss` (flag)
- Detection tuning params:
  - `--det_box_size 40`
  - `--det_sigma 2.0`
  - `--det_min_distance 10`
  - `--det_rel_thresh 0.12`

## Important implementation notes
- The script rescales every input to 384×384. All exemplars and overlays are based on this resized image.
- Model output `pred` is assumed to be a 2D density map; the script computes the predicted scalar count by summing density values and dividing by 60.0. (The divisor `60.0` is a dataset/model scaling heuristic — check training/data preprocessing if counts appear off.)
- Exemplar negatives are created by offsetting exemplar boxes by -20 px in both x/y (clipped to image bounds) and passed to the model as `neg_boxes_batch` (though the model call in this script uses only `boxes_batch` and a fixed `3` argument; review `models_mae_cross` signature for additional behavior).
- Normalize uses ImageNet mean/std constants; tensors are normalized before model inference.

## Dependencies / Environment
- Python packages used directly in the script: `torch`, `torchvision`, `Pillow`, `numpy`, `matplotlib`, `opencv-python`, `scipy`.
- Use `pip install -r VA-Count/requirements.txt` to match the repository environment.

## Example runs
- Quick run using GPU (if available):

```bash
python run_va_count_on_samples.py --device cuda --samples_dir samples --output_dir outputs
```

- Use CPU explicitly:

```bash
python run_va_count_on_samples.py --device cpu
```

- Tweak detection sensitivity (example: smaller sigma, higher relative threshold):

```bash
python run_va_count_on_samples.py --det_sigma 1.0 --det_rel_thresh 0.18
```

## Troubleshooting
- "No images found": ensure `--samples_dir` points to a folder with supported image extensions.
- CUDA not used: script falls back to CPU if CUDA unavailable; ensure `torch.cuda.is_available()` and correct CUDA-enabled PyTorch wheel are installed.
- Count scaling seems wrong: check the divisor `/ 60.0` in the script and confirm expected units from model/training code.
- If `models_mae_cross` import fails: confirm `VA-Count` is accessible on `sys.path` (script already inserts it) and that `VA-Count` dependencies are installed.

## Recommendations / Next steps
- Document the origin/meaning of the `60.0` scaling factor (add comment or configuration flag) for transparency.
- Optionally add a `--save_raw_density` flag to save raw .npy densities for offline analysis.
- Log exceptions with stacktraces for easier debugging (script currently prints a short error message).

---

Generated by developer tooling on the repository; place this file next to the project README for quick reference.
