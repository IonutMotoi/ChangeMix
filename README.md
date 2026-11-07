# ChangeMix

Official implementation of

> **[ChangeMix: Change-Aware Cut-and-Paste Augmentation for Semi-Supervised Remote Sensing Building Change Detection](https://ieeexplore.ieee.org/document/11775324)**<br>
> Ionut M. Motoi, Wei Huang, Thomas A. Ciarfuglia, Daniele Nardi, Xiao Xiang Zhu<br>
> *IEEE Transactions on Geoscience and Remote Sensing*, 2026, doi: [10.1109/TGRS.2026.3750046](https://doi.org/10.1109/TGRS.2026.3750046)

ChangeMix is a change-aware cut-and-paste augmentation for semi-supervised change detection. It maintains an Instance Bank of change regions collected from the scarce ground truth and from high-confidence teacher pseudo-labels, and pastes multiple paired pre-/post-change instances, together with their change and confidence maps, into the strongly augmented unlabeled pairs of a teacher-student framework, randomly swapping their temporal order. It replaces the CutMix step of the framework and works with FixMatch, UniMatch V1 and UniMatch V2, and with both DPT/DINOv2 and DeepLabV3+ models.

## Using ChangeMix in your own pipeline

[`changemix.py`](changemix.py) is self-contained (PyTorch, NumPy, OpenCV):

```python
from changemix import InstanceBank, paste_instances

bank = InstanceBank(bank_size=500, min_confidence_threshold=0.95, max_confidence_threshold=0.99,
                    min_instance_size=8, max_instance_size=192, padding=32)

# at every iteration, after the teacher prediction on the weakly augmented unlabeled pair
bank.update(img_a_l, img_b_l, mask_l, torch.ones_like(mask_l).float(), is_labeled=True)  # ground truth
bank.update(img_a_u, img_b_u, pseudo_label, confidence, is_labeled=False)                # pseudo-labels

mask_s, conf_s = pseudo_label.clone(), confidence.clone()
if random.random() < 0.5:
    img_a_s, img_b_s, mask_s, conf_s = paste_instances(img_a_s, img_b_s, mask_s, conf_s, bank,
                                                       instances_per_image=10, paste_attempts=10)
# unsupervised loss: cross-entropy between student(img_a_s, img_b_s) and mask_s where conf_s >= 0.95
```

Images are `(B, 3, H, W)` tensors, change maps `(B, H, W)` with 1 = changed, 0 = unchanged and 255 = ignored, confidence maps `(B, H, W)`. To see exactly how ChangeMix replaces the paired CutMix of UniMatch V2, compare `unimatch_v2.py` with `unimatch_v2_changemix.py`.

## Repository

| File | Contents |
| --- | --- |
| `changemix.py` | ChangeMix: Instance Bank and instance pasting |
| `fixmatch.py`, `unimatch_v1.py`, `unimatch_v2.py` | SSL frameworks with the paired CutMix baseline (DPT/DINOv2) |
| `fixmatch_changemix.py`, `unimatch_v1_changemix.py`, `unimatch_v2_changemix.py` | the same frameworks with ChangeMix |
| `unimatch_v2_deeplab.py`, `unimatch_v2_changemix_deeplab.py` | UniMatch V2 with DeepLabV3+/ResNet-50, without and with ChangeMix |
| `supervised.py` | Labeled Only and Fully Supervised baselines |
| `eval_reversed_order.py` | test-set evaluation with the temporal order of the pairs reversed (Sec. IV-E) |
| `configs/levir.yaml`, `configs/whu.yaml` | training and ChangeMix hyperparameters (Sec. IV-C) |
| `configs/experiments/` | every training run of the paper's tables |
| `run_experiments.py` | lists or runs the training runs of an experiment file |
| `splits/` | labeled / unlabeled partitions, identical to those of UniMatch V1 and V2 |

## Installation

```bash
git clone https://github.com/IonutMotoi/ChangeMix.git
cd ChangeMix
pip install -r requirements.txt
```

The code was developed with Python 3.12, PyTorch 2.9.1 and torchvision 0.24.1.

**Pretrained encoders.** Download [DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth), [DINOv2-Base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and, for DeepLabV3+, [ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) (from UniMatch), and save them as:

```
pretrained/
├── dinov2_small.pth
├── dinov2_base.pth
└── resnet50.pth
```

**Datasets.** LEVIR-CD and WHU-CD cropped into 256 × 256 tiles, as distributed by UniMatch V2: [LEVIR-CD](https://huggingface.co/LiheYoung/UniMatch-V2/resolve/main/LEVIR-CD.zip), [WHU-CD](https://huggingface.co/LiheYoung/UniMatch-V2/resolve/main/WHU-CD.zip). By default the configs expect (set `data_root` in `configs/levir.yaml` and `configs/whu.yaml` otherwise):

```
data/
├── LEVIR-CD/{A,B,label}
└── WHU-CD/{A,B,label}
```

## Training

A single run, e.g. UniMatch V2 + ChangeMix on LEVIR-CD with 20% of the labels:

```bash
torchrun --nproc_per_node=1 unimatch_v2_changemix.py --config configs/levir.yaml \
    --labeled-id-path splits/levir/20%/labeled.txt --unlabeled-id-path splits/levir/20%/unlabeled.txt \
    --save-path exp/levir/20%/unimatch_v2_changemix --seed 42
```

Use `unimatch_v2.py` for the baseline with paired CutMix and `--backbone dinov2_small` for DPT/DINOv2-Small. The ChangeMix hyperparameters can be overridden from the command line (`--bank_size`, `--max_instance_size`, `--min_conf_thresh`, `--max_conf_thresh`, `--padding`, ...), and `--wandb` enables Weights & Biases logging. At the end of training, the model with the best changed-class IoU on the validation set (the EMA teacher for FixMatch and UniMatch V2) is evaluated on the test set and the test metrics are printed; checkpoints and TensorBoard curves are saved in `--save-path`.

**Experiments of the paper.** Each file in `configs/experiments/` lists the training runs behind a group of tables:

| Tables | Experiment file | Runs |
| --- | --- | --- |
| I–II: SSL frameworks, DPT/DINOv2-Base | `configs/experiments/frameworks.yaml` | 82 |
| III–IV: DPT/DINOv2-Small and DeepLabV3+/ResNet-50 | `configs/experiments/backbones.yaml` | 56 |
| VI–IX: Instance Bank hyperparameters | `configs/experiments/hyperparameters.yaml` | 56 |

```bash
python run_experiments.py configs/experiments/frameworks.yaml          # print the commands
python run_experiments.py configs/experiments/frameworks.yaml --run    # run them one after another
python run_experiments.py configs/experiments/frameworks.yaml --run --filter dataset=whu split=20%
```

Protocol details:

- All results use a single GPU, 60 epochs and a batch size of 8 (8 labeled and 8 unlabeled pairs per iteration in the semi-supervised scripts).
- The semi-supervised runs at the 20% budget of Tables I–II, and the DeepLabV3+ runs on WHU-CD at 20% in Table IV, use seeds 42, 43 and 44 (mean and standard deviation in the tables); every other entry is a single run with seed 42.
- The supervised cold start of Sec. IV-C3 (`--only_sup_ratio 0.1`) is used on WHU-CD at 5% and in every DeepLabV3+ semi-supervised run.
- The DeepLabV3+ scripts use SGD (lr 0.02, momentum 0.9, weight decay 1e-4); the DPT/DINOv2 scripts use AdamW with the learning rates of the config.
- `supervised.py` follows the supervised baseline of UniMatch V2: the batch size of the config is doubled and, with `--supervision semi`, the labeled set is resampled to 3,000 pairs per epoch.
- cuDNN autotuning is enabled, so results can differ slightly across GPUs and library versions.

**Reversed temporal order (Sec. IV-E).**

```bash
python eval_reversed_order.py --config configs/levir.yaml \
    --checkpoint exp/levir/10%/unimatch_v2_changemix/dinov2_base/seed42/best_ema_weights.pth
```

## Citation

```bibtex
@article{motoi2026changemix,
  title   = {ChangeMix: Change-Aware Cut-and-Paste Augmentation for Semi-Supervised Remote Sensing Building Change Detection},
  author  = {Motoi, Ionut M. and Huang, Wei and Ciarfuglia, Thomas A. and Nardi, Daniele and Zhu, Xiao Xiang},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  doi     = {10.1109/TGRS.2026.3750046}
}
```

## Acknowledgements

This code builds on [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2) (remote sensing change detection code, MIT License) and [UniMatch](https://github.com/LiheYoung/UniMatch) (DeepLabV3+, MIT License). The DINOv2 backbone code comes from [DINOv2](https://github.com/facebookresearch/dinov2) (Apache License 2.0, see the file headers). The processed LEVIR-CD and WHU-CD datasets and the partitions are those released with UniMatch V2, originally prepared by [SemiCD](https://github.com/wgcban/SemiCD).

## License

MIT License, see [LICENSE](LICENSE).
