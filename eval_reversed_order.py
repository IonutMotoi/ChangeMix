"""Evaluate a DPT/DINOv2 change detection model on the test set in both temporal orders.

The models fuse the bi-temporal features through the symmetric absolute difference
|f(I_A) - f(I_B)|, so the predictions on the reversed pairs (I_B, I_A) are identical to those
on the original pairs (I_A, I_B) (Sec. IV-E of the paper).

    python eval_reversed_order.py --config configs/levir.yaml \
        --checkpoint exp/levir/10%/unimatch_v2_changemix/dinov2_base/seed42/best_ema_weights.pth
"""

import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset.semicd import SemiCDDataset
from model.semseg.dpt import DPT


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}
MULTIPLIER = 14  # DPT/DINOv2 patch size, as in evaluate_v2


def build_model(backbone, nclass, checkpoint_path, device):
    model = DPT(**{**MODEL_CONFIGS[backbone], 'nclass': nclass})

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_ema' in ckpt:
        state_dict = ckpt['model_ema']  # training checkpoint (latest.pth) of an EMA framework: use the teacher
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']      # training checkpoint (latest.pth) without EMA
    else:
        state_dict = ckpt               # best_ema_weights.pth / best_weights.pth
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    return model.to(device).eval()


@torch.no_grad()
def evaluate_direction(model, loader, nclass, reverse, device):
    """Per-class IoU, precision, recall and F1 (%) of the changed class for one temporal order."""
    inter = torch.zeros(nclass, device=device)
    union = torch.zeros(nclass, device=device)
    tp = torch.zeros((), device=device)
    fp = torch.zeros((), device=device)
    fn = torch.zeros((), device=device)

    for imgA, imgB, mask, _ in loader:
        imgA, imgB, mask = imgA.to(device), imgB.to(device), mask.to(device)
        if reverse:
            imgA, imgB = imgB, imgA

        ori_h, ori_w = imgA.shape[-2:]
        new_h = int(ori_h / MULTIPLIER + 0.5) * MULTIPLIER
        new_w = int(ori_w / MULTIPLIER + 0.5) * MULTIPLIER
        imgA = F.interpolate(imgA, (new_h, new_w), mode='bilinear', align_corners=True)
        imgB = F.interpolate(imgB, (new_h, new_w), mode='bilinear', align_corners=True)

        pred = model(imgA, imgB)
        pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
        pred = pred.argmax(dim=1)

        valid = mask != 255
        for cls in range(nclass):
            p = (pred == cls) & valid
            g = (mask == cls) & valid
            inter[cls] += (p & g).sum()
            union[cls] += (p | g).sum()
        pred_bin = (pred == 1) & valid
        mask_bin = (mask == 1) & valid
        tp += (pred_bin & mask_bin).sum()
        fp += (pred_bin & ~mask_bin).sum()
        fn += (~pred_bin & mask_bin).sum()

    iou = (inter / (union + 1e-10)) * 100.0
    precision = (tp / (tp + fp + 1e-10)).item() * 100.0
    recall = (tp / (tp + fn + 1e-10)).item() * 100.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return {'iou_unchanged': iou[0].item(), 'iou_changed': iou[1].item(),
            'precision': precision, 'recall': recall, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description='Test-set evaluation in the original and in the reversed temporal order.')
    parser.add_argument('--config', type=str, required=True, help='configs/levir.yaml or configs/whu.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='best_ema_weights.pth or best_weights.pth (or a latest.pth training checkpoint)')
    parser.add_argument('--backbone', type=str, help='dinov2_base or dinov2_small (default: the backbone of the config)')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    backbone = (args.backbone or cfg['backbone']).split('_')[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    model = build_model(backbone, cfg['nclass'], args.checkpoint, device)
    testset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'test')
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    results = {'original (I_A, I_B)': evaluate_direction(model, loader, cfg['nclass'], reverse=False, device=device),
               'reversed (I_B, I_A)': evaluate_direction(model, loader, cfg['nclass'], reverse=True, device=device)}

    print(f'\n{cfg["dataset"]}  {args.checkpoint}')
    for name, r in results.items():
        print(f'  {name}: IoU_c {r["iou_changed"]:6.2f}  IoU_u {r["iou_unchanged"]:6.2f}  '
              f'P {r["precision"]:6.2f}  R {r["recall"]:6.2f}  F1 {r["f1"]:6.2f}')


if __name__ == '__main__':
    main()
