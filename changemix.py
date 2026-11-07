"""ChangeMix: change-aware cut-and-paste augmentation for semi-supervised change detection.

Self-contained implementation (PyTorch, NumPy, OpenCV) of the augmentation proposed in

    I. M. Motoi, W. Huang, T. A. Ciarfuglia, D. Nardi, X. X. Zhu,
    "ChangeMix: Change-Aware Cut-and-Paste Augmentation for Semi-Supervised
    Remote Sensing Building Change Detection",
    IEEE Transactions on Geoscience and Remote Sensing, 2026, doi: 10.1109/TGRS.2026.3750046.

ChangeMix replaces the CutMix step of a weak-to-strong (teacher-student) SSL framework.
At every training iteration:

  1. ``InstanceBank.update`` extracts change instances (connected components of the change
     map) from the labeled batch, using the ground truth, and from the unlabeled batch, using
     the teacher pseudo-labels, and stores them in a FIFO bank as tuples (x_a, x_b, m, c):
     the pre- and post-change crops, the local change map and the local confidence map.
  2. ``paste_instances`` pastes up to N non-overlapping instances from the bank into the
     strongly augmented unlabeled views, randomly swapping the pre- and post-change crops,
     and writes their change and confidence maps into the pseudo-label and confidence maps
     used by the unsupervised loss.

Minimal usage inside a training loop (unimatch_v2_changemix.py shows the complete integration):

    bank = InstanceBank(bank_size=500, min_confidence_threshold=0.95, max_confidence_threshold=0.99,
                        min_instance_size=8, max_instance_size=192, padding=32)

    # images: (B, 3, H, W); change maps: (B, H, W) with 1 = changed, 0 = unchanged, 255 = ignored
    bank.update(img_a_l, img_b_l, mask_l, torch.ones_like(mask_l).float(), is_labeled=True)  # ground truth
    bank.update(img_a_u, img_b_u, mask_u, conf_u, is_labeled=False)                           # teacher pseudo-labels

    mask_s, conf_s = mask_u.clone(), conf_u.clone()
    if random.random() < 0.5:  # p_mix
        img_a_s, img_b_s, mask_s, conf_s = paste_instances(img_a_s, img_b_s, mask_s, conf_s, bank,
                                                           instances_per_image=10, paste_attempts=10)
    # unsupervised loss on the student prediction for (img_a_s, img_b_s), target mask_s, weight conf_s >= 0.95
"""

import cv2
import numpy as np
import torch

__all__ = ['InstanceBank', 'paste_instances']


class InstanceBank:
    """Fixed-capacity FIFO bank of paired change instances.

    Each instance is a tuple (x_a, x_b, m, c): the pre-change crop x_a and the post-change
    crop x_b (C, h, w), the local change map m (h, w) and the local confidence map c (h, w).
    Since m is a window of the full change map, changed pixels of neighbouring instances that
    fall inside the crop remain labeled as changed.

    Args:
        bank_size: capacity B of the bank. When the bank is full, the oldest instance is removed.
        min_confidence_threshold: tau_min. A pseudo-labeled instance is kept only if the mean
            confidence over its changed pixels lies in [tau_min, tau_max].
        max_confidence_threshold: tau_max (see above). Ground-truth instances are always kept.
        min_instance_size: s_min, minimum width and height of the instance bounding box (pixels).
        max_instance_size: s_max, maximum width and height of the instance bounding box (pixels).
        padding: p, context pixels added on each side of the bounding box, clipped to the valid
            (non-padded) region of the image.
    """

    def __init__(self, bank_size: int, min_confidence_threshold: float, max_confidence_threshold: float,
                 min_instance_size: int, max_instance_size: int, padding: int):
        self.bank_size = bank_size
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_threshold = max_confidence_threshold
        self.min_instance_size = min_instance_size
        self.max_instance_size = max_instance_size
        self.padding = padding
        self.instances = []

    def __len__(self):
        return len(self.instances)

    def update(self, imgs_a: torch.Tensor, imgs_b: torch.Tensor, masks: torch.Tensor, confs: torch.Tensor,
               is_labeled: bool = False):
        """Add the change instances of a batch to the bank.

        Args:
            imgs_a: pre-change images (B, C, H, W).
            imgs_b: post-change images (B, C, H, W).
            masks: change maps (B, H, W): 1 = changed, 0 = unchanged, 255 = ignored (padding).
            confs: confidence maps (B, H, W); all ones for ground-truth change maps.
            is_labeled: True for ground-truth change maps, which skip the confidence filter.
        """
        for b in range(imgs_a.shape[0]):
            for slice_y, slice_x in self._extract_instances(masks[b], confs[b], is_labeled):
                self._add_instance(imgs_a[b], imgs_b[b], masks[b], confs[b], slice_y, slice_x)

    def get_instances(self, n: int) -> list:
        """Return n distinct instances sampled uniformly from the bank (all of them if the bank holds fewer)."""
        if len(self.instances) == 0 or n <= 0:
            return []
        indices = np.random.choice(len(self.instances), min(n, len(self.instances)), replace=False)
        return [self.instances[idx] for idx in indices]

    def _extract_instances(self, mask: torch.Tensor, conf: torch.Tensor, is_labeled: bool = False) -> list:
        """Return the (slice_y, slice_x) windows of the accepted change instances of one change map."""
        mask_np = mask.cpu().numpy()
        H, W = mask_np.shape
        valid_map = (mask_np != 255)
        foreground_mask = (mask_np == 1).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask)
        instances = []

        # Bounding box of the valid (non-padded) region of the image
        rows_with_valid_data = np.any(valid_map, axis=1)
        cols_with_valid_data = np.any(valid_map, axis=0)
        valid_top = np.argmax(rows_with_valid_data)
        valid_bottom = H - np.argmax(rows_with_valid_data[::-1])
        valid_left = np.argmax(cols_with_valid_data)
        valid_right = W - np.argmax(cols_with_valid_data[::-1])

        for comp_id in range(1, num_labels):
            x, y, w, h, _ = stats[comp_id]

            # Size filter on the tight bounding box: s_min <= w, h <= s_max
            if self.min_instance_size <= w <= self.max_instance_size and self.min_instance_size <= h <= self.max_instance_size:
                # Expand the bounding box by the padding, clipped to the valid region
                x1 = max(valid_left, x - self.padding)
                y1 = max(valid_top, y - self.padding)
                x2 = min(valid_right, x + w + self.padding)
                y2 = min(valid_bottom, y + h + self.padding)

                slice_y = slice(y1, y2)
                slice_x = slice(x1, x2)

                if is_labeled:
                    # Ground-truth instances are not filtered by confidence
                    instances.append((slice_y, slice_x))
                else:
                    # Pseudo-labeled instances: mean confidence over the changed pixels of the component
                    mask_component = torch.from_numpy(labels == comp_id)[slice_y, slice_x].to(conf.device)
                    mean_conf = conf[slice_y, slice_x][mask_component].mean().item()
                    if self.min_confidence_threshold <= mean_conf <= self.max_confidence_threshold:
                        instances.append((slice_y, slice_x))
        return instances

    def _add_instance(self, img_a: torch.Tensor, img_b: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor,
                      slice_y: slice, slice_x: slice):
        """Store the (x_a, x_b, m, c) tuple of one instance, dropping the oldest instance if the bank is full."""
        instance = (img_a[:, slice_y, slice_x].clone(),
                    img_b[:, slice_y, slice_x].clone(),
                    mask[slice_y, slice_x].clone(),
                    conf[slice_y, slice_x].clone())

        if len(self.instances) >= self.bank_size:
            self.instances.pop(0)
        self.instances.append(instance)


def paste_instances(imgs_a: torch.Tensor, imgs_b: torch.Tensor, masks: torch.Tensor, confs: torch.Tensor,
                    instance_bank: InstanceBank, instances_per_image: int = 10, paste_attempts: int = 10):
    """Paste instances from the bank into a batch of image pairs (in place).

    For every image, up to ``instances_per_image`` distinct instances are sampled from the bank.
    Each one is placed at a random location that does not overlap the instances already pasted
    into the same image; if no such location is found in ``paste_attempts`` tries, the instance
    is skipped. The crops replace the image pixels directly (no blending), and with probability
    0.5 the pre- and post-change crops are swapped (temporal swap). The change map m and the
    confidence map c of the instance replace the (pseudo-)label and confidence of the window.

    Args:
        imgs_a: pre-change images (B, C, H, W), typically the strongly augmented unlabeled view.
        imgs_b: post-change images (B, C, H, W).
        masks: (pseudo-)label maps (B, H, W).
        confs: confidence maps (B, H, W).
        instance_bank: the InstanceBank to sample from.
        instances_per_image: N, maximum number of instances pasted into each image.
        paste_attempts: L, placement attempts per instance.

    Returns:
        The updated (imgs_a, imgs_b, masks, confs); the input tensors are modified in place.
    """
    B, _, H, W = imgs_a.shape

    for b in range(B):
        instances = instance_bank.get_instances(instances_per_image)
        if not instances:
            continue

        pasted_regions = []  # (x, y, width, height) of the instances already pasted into this image

        for img_a_slice, img_b_slice, mask_slice, conf_slice in instances:
            ih, iw = mask_slice.shape
            if ih > H or iw > W:
                continue

            for _ in range(paste_attempts):
                x = np.random.randint(0, W - iw + 1)
                y = np.random.randint(0, H - ih + 1)

                is_overlapping = False
                for pr_x, pr_y, pr_w, pr_h in pasted_regions:
                    if not (x + iw <= pr_x or x >= pr_x + pr_w or y + ih <= pr_y or y >= pr_y + pr_h):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    dest_slice_y = slice(y, y + ih)
                    dest_slice_x = slice(x, x + iw)

                    if np.random.random() < 0.5:
                        # Original temporal order: pre-change crop into image A, post-change crop into image B
                        imgs_a[b, :, dest_slice_y, dest_slice_x] = img_a_slice
                        imgs_b[b, :, dest_slice_y, dest_slice_x] = img_b_slice
                    else:
                        # Temporal swap: post-change crop into image A, pre-change crop into image B
                        imgs_a[b, :, dest_slice_y, dest_slice_x] = img_b_slice
                        imgs_b[b, :, dest_slice_y, dest_slice_x] = img_a_slice

                    # Binary change labels do not depend on the temporal order: m and c are pasted as they are
                    masks[b, dest_slice_y, dest_slice_x] = mask_slice
                    confs[b, dest_slice_y, dest_slice_x] = conf_slice

                    pasted_regions.append((x, y, iw, ih))
                    break

    return imgs_a, imgs_b, masks, confs
