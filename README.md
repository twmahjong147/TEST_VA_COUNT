# TEST_VA_COUNT

Lightweight workspace for running the VA-Count zero-shot object counting code (ECCV 2024).

This repository contains the `VA-Count` implementation and helper scripts to run inference on sample images.

**Contents**
- `VA-Count/` — model code, training and inference scripts, and original project README.
- `weights/` — pretrained checkpoint(s) (e.g., `checkpoint_FSC.pth`).
- `samples/` — sample images for quick runs.
- `outputs/` — inference outputs (JSON) created by sample runner.
- `run_va_count_on_samples.py` — convenience script to run inference on samples.

## Quickstart
1. Install Python dependencies (recommended in a virtualenv):

```bash
pip install -r VA-Count/requirements.txt
```

2. Run the sample runner from the project root (reads `samples/`, writes `outputs/`):

```bash
python run_va_count_on_samples.py
```

3. Or run inference directly with the model (example):

```bash
cd VA-Count
python FSC_test.py --output_dir ../outputs --resume ../weights/checkpoint_FSC.pth
```

Notes:
- The workspace already includes `weights/checkpoint_FSC.pth`. Place other checkpoints in `weights/` as needed.
- See [VA-Count/README.md](VA-Count/README.md) for full dataset preparation, training, and advanced usage.

## Citation
If you use this code or models, please cite the VA-Count paper:

```bibtex
@inproceedings{zhu2024zero,
  title={Zero-shot Object Counting with Good Exemplars},
  author={Zhu, Huilin and Yuan, Jingling and Yang, Zhengwei and Guo, Yu and Wang, Zheng and Zhong, Xian and He, Shengfeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

## Contact
Questions or issues: jsj_zhl@whut.edu.cn
