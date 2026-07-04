# Vendored/adapted from https://github.com/sczhou/ProPainter (NTU S-Lab License 1.0 -- non-commercial use only)
# Pipeline logic ported from `inference_propainter.py` in the upstream repo.
# See https://github.com/sczhou/ProPainter/blob/main/LICENSE for full terms.
"""
Thin inference wrapper around the vendored ProPainter sub-networks
(RAFT flow estimator, recurrent flow completion, recurrent inpainting
transformer). Mirrors the control flow of upstream `inference_propainter.py`
but takes/returns plain numpy RGB frame lists instead of files, since
`ScratchRemovalRestorer` already owns frame I/O and scratch-mask detection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from restorax.restorers.artifact_removal.propainter.modules.flow_comp_raft import RAFT_bi
from restorax.restorers.artifact_removal.propainter.propainter import InpaintGenerator
from restorax.restorers.artifact_removal.propainter.recurrent_flow_completion import (
    RecurrentFlowCompleteNet,
)

# Default inference hyperparameters (upstream `inference_propainter.py` CLI defaults).
_RAFT_ITERS = 20
_REF_STRIDE = 10
_NEIGHBOR_LENGTH = 10
_SUBVIDEO_LENGTH = 80


def _resolve_weight(weight_dir: Path, filename: str) -> str:
    """ProPainter's HF snapshot may place weights either flat or under weights/."""
    for candidate in (weight_dir / filename, weight_dir / "weights" / filename):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"ProPainter weight '{filename}' not found under {weight_dir} "
        f"(looked in '{weight_dir}' and '{weight_dir / 'weights'}')."
    )


def _get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """Verbatim port of upstream `get_ref_index`."""
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class ProPainterPipeline:
    """Loads the three ProPainter sub-networks and runs full video inpainting."""

    def __init__(self, weight_dir: str, device: torch.device) -> None:
        self.device = device
        wdir = Path(weight_dir)

        raft_ckpt = _resolve_weight(wdir, "raft-things.pth")
        flow_ckpt = _resolve_weight(wdir, "recurrent_flow_completion.pth")
        inpaint_ckpt = _resolve_weight(wdir, "ProPainter.pth")

        self.fix_raft = RAFT_bi(raft_ckpt, device=device)

        self.fix_flow_complete = RecurrentFlowCompleteNet(flow_ckpt)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(device).eval()

        self.model = InpaintGenerator(model_path=inpaint_ckpt).to(device)
        self.model.eval()

    @torch.no_grad()
    def inpaint(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        """
        frames: list of HxWx3 RGB uint8 arrays.
        masks: list of HxW bool/uint8 arrays (True/nonzero = region to inpaint).
        Returns a list of HxWx3 RGB uint8 arrays, same length/size as input.
        """
        device = self.device
        video_length = len(frames)
        orig_h, orig_w = frames[0].shape[:2]

        # ponytail: pad up to a multiple of 8 (network requirement) instead of
        # upstream's resize-down-then-resize-back — preserves exact pixel content.
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        h, w = orig_h + pad_h, orig_w + pad_w

        frames_t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        frames_t = frames_t.unsqueeze(0) * 2 - 1  # (1, T, 3, H, W), range [-1, 1]

        masks_t = torch.stack(
            [torch.from_numpy(np.asarray(m, dtype=np.float32) > 0) for m in masks]
        ).float()
        masks_t = masks_t.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)

        if pad_h or pad_w:
            frames_t = F.pad(frames_t, (0, pad_w, 0, pad_h), mode="replicate")
            masks_t = F.pad(masks_t, (0, pad_w, 0, pad_h), mode="replicate")

        frames_t, masks_t = frames_t.to(device), masks_t.to(device)
        flow_masks, masks_dilated = masks_t, masks_t

        # ---- compute bidirectional optical flow (RAFT), chunked for memory ----
        short_clip_len = 12 if w <= 640 else 8 if w <= 720 else 4 if w <= 1280 else 2
        if video_length > short_clip_len:
            flows_f_list, flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                chunk = frames_t[:, f:end_f] if f == 0 else frames_t[:, f - 1:end_f]
                ff, fb = self.fix_raft(chunk, iters=_RAFT_ITERS)
                flows_f_list.append(ff)
                flows_b_list.append(fb)
            gt_flows_bi = (torch.cat(flows_f_list, dim=1), torch.cat(flows_b_list, dim=1))
        else:
            gt_flows_bi = self.fix_raft(frames_t, iters=_RAFT_ITERS)

        # ---- complete flow in occluded/masked regions ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > _SUBVIDEO_LENGTH:
            pred_f, pred_b = [], []
            pad_len = 5
            for f in range(0, flow_length, _SUBVIDEO_LENGTH):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + _SUBVIDEO_LENGTH + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + _SUBVIDEO_LENGTH)
                sub_bi, _ = self.fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f:e_f + 1],
                )
                sub_bi = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    sub_bi,
                    flow_masks[:, s_f:e_f + 1],
                )
                pred_f.append(sub_bi[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_b.append(sub_bi[1][:, pad_len_s:e_f - s_f - pad_len_e])
            pred_flows_bi = (torch.cat(pred_f, dim=1), torch.cat(pred_b, dim=1))
        else:
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

        # ---- flow-guided image propagation ----
        masked_frames = frames_t * (1 - masks_dilated)
        subvideo_len_prop = min(100, _SUBVIDEO_LENGTH)
        if video_length > subvideo_len_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_len_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_len_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_len_prop)
                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                flows_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                    masked_frames[:, s_f:e_f], flows_sub, masks_dilated[:, s_f:e_f], "nearest"
                )
                updated_frames_sub = (
                    frames_t[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                    + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                )
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = self.model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest"
            )
            updated_frames = frames_t * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)

        # ---- recurrent feature propagation + sparse transformer inpainting ----
        ori_frames = frames_t.clone()
        comp_frames: list[np.ndarray | None] = [None] * video_length
        neighbor_stride = _NEIGHBOR_LENGTH // 2
        ref_num = _SUBVIDEO_LENGTH // _REF_STRIDE if video_length > _SUBVIDEO_LENGTH else -1

        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
            ref_ids = _get_ref_index(f, neighbor_ids, video_length, _REF_STRIDE, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
            selected_flows = (
                pred_flows_bi[0][:, neighbor_ids[:-1]],
                pred_flows_bi[1][:, neighbor_ids[:-1]],
            )
            l_t = len(neighbor_ids)
            pred_img = self.model(selected_imgs, selected_flows, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            ori_np = (ori_frames[0].cpu().permute(0, 2, 3, 1).numpy() + 1) / 2 * 255

            for i, idx in enumerate(neighbor_ids):
                img = pred_img[i].astype(np.uint8) * binary_masks[i] + ori_np[idx].astype(np.uint8) * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(np.uint8)

        # Crop back to the original (pre-padding) resolution.
        return [np.clip(frame[:orig_h, :orig_w], 0, 255).astype(np.uint8) for frame in comp_frames]
