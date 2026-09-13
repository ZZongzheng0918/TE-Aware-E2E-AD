from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from navsim.agents.transfuser_te.transfuser_te_config import TransfuserTEConfig
from navsim.agents.transfuser_te.transfuser_te_features import BoundingBox2DIndex

import numpy as np


def transfuser_te_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserTEConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    te_hm_loss = _te_hm_loss(predictions["te_heat_map"], targets["te_heat_map"], config)
    loss = (
        config.trajectory_weight * trajectory_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        + config.te_hm_weight * te_hm_loss
    )
    loss_dict = {
        "loss": loss,
        "trajectory_loss": config.trajectory_weight * trajectory_loss,
        "agent_class_loss": config.agent_class_weight * agent_class_loss,
        "agent_box_loss": config.agent_box_weight * agent_box_loss,
        "bev_semantic_loss": config.bev_semantic_weight * bev_semantic_loss,
        "te_hm_loss": config.te_hm_weight * te_hm_loss,
    }
    return loss_dict


def _agent_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserTEConfig):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


def draw_umich_gaussian(heatmap, center, radius, k=1.0):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def _te_hm_loss(pred_hm: torch.Tensor, gt_hm: torch.Tensor, config: TransfuserTEConfig) -> torch.Tensor:
    """
    GaussianFocal Loss for TE heatmap.
    pred_hm: (C,H,W) or (B,C,H,W), logits
    gt_hm:   (C,H,W) or (B,C,H,W), sparse binary points
    """
    if pred_hm.dim() == 3:
        pred_hm = pred_hm.unsqueeze(0)
    if gt_hm.dim() == 3:
        gt_hm = gt_hm.unsqueeze(0)

    B, C, H, W = gt_hm.shape
    device = pred_hm.device

    # 1) sparse gt -> gaussian gt (numpy draw, then back to torch)
    radius = int(getattr(config, "te_gaussian_radius", 2))
    gt_np = gt_hm.detach().cpu().numpy()
    gaussian_gt_np = np.zeros_like(gt_np, dtype=np.float32)

    for b in range(B):
        for c in range(C):
            ys, xs = np.where(gt_np[b, c] > 0)
            hm = gaussian_gt_np[b, c]
            for y, x in zip(ys, xs):
                draw_umich_gaussian(hm, center=(x, y), radius=radius, k=1.0)

    gaussian_gt = torch.from_numpy(gaussian_gt_np).to(device=device, dtype=pred_hm.dtype)

    # 2) gaussian focal loss
    alpha = float(getattr(config, "te_focal_alpha", 2.0))
    gamma = float(getattr(config, "te_focal_gamma", 4.0))
    eps = 1e-6

    pred = torch.sigmoid(pred_hm).clamp(min=eps, max=1.0 - eps)

    pos_inds = (gaussian_gt.eq(1)).float()
    neg_inds = (gaussian_gt.lt(1)).float()
    neg_weights = torch.pow(1 - gaussian_gt, gamma)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    num_pos = pos_inds.sum()
    num_neg = neg_inds.sum().clamp(min=1.0)

    # The negative-sample weight can be reduced when needed.
    neg_weight = float(getattr(config, "te_neg_weight", 1.0))

    if num_pos < 1:
        # Normalize negatives when no positive samples are present.
        return -neg_weight * (neg_loss / num_neg)

    # Normalize positive and negative samples separately for stability.
    return -(pos_loss / num_pos + neg_weight * (neg_loss / num_neg))
