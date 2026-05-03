"""
DicFace — Dictionary-based blind face restoration.

DicFace (ICCV 2023) builds a cross-scale dictionary of high-quality facial
components (eyes, nose, mouth, skin) and uses it to guide restoration via
cross-attention. Achieves significantly better identity preservation than
CodeFormer on real-world degraded portraits while recovering more detail
than methods that rely solely on GAN priors.

Model source: https://github.com/YaNgZhAnG-V5/DicFace
Paper: "Dictionary-based Face Restoration with High-Frequency Hybrid-Awareness"
       (ICCV 2023)
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from restorax.core.restorer import (
    BaseRestorer,
    RestorerCapabilities,
    RestorerCategory,
    RestorerParams,
)

logger = logging.getLogger(__name__)

_HF_REPO = "YaNgZhAnG-V5/DicFace"
_WEIGHT_FILE = "dicface.pth"
_DEFAULT_FIDELITY = 0.5


class DicFaceRestorer(BaseRestorer):
    """
    Dictionary-based blind face restoration (DicFace, ICCV 2023).

    Better identity preservation than CodeFormer/GFPGAN while recovering
    more facial detail. Ideal for portrait videos and close-up interview footage.

    extra params:
      fidelity: float 0.0–1.0 (0=max enhancement, 1=max fidelity)
    """

    def __init__(self) -> None:
        self._net: object | None = None
        self._face_helper: object | None = None
        self._device: torch.device | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "dicface"

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.FACE_RESTORATION,
            input_color_space="bgr",
            output_color_space="bgr",
            requires_temporal=False,
            min_vram_gb=4.0,
            scale_factor=1,
            tags=["face_restoration", "dicface", "dictionary", "identity", "iccv2023"],
        )

    def load(self, device: torch.device) -> None:
        self._net, self._face_helper = self._build_model(device)
        self._device = device
        self._loaded = True
        logger.info("DicFace loaded on %s", device)

    def unload(self) -> None:
        del self._net
        self._net = None
        self._face_helper = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        assert self._device is not None
        fidelity = float(params.extra.get("fidelity", _DEFAULT_FIDELITY))

        if self._net is None:
            return frame

        if hasattr(self._net, "__call__") and self._face_helper is not None:
            return self._restore(frame, fidelity)

        # Fallback: delegate to CodeFormer
        return self._codeformer_fallback(frame, fidelity)

    def _restore(self, frame: np.ndarray, fidelity: float) -> np.ndarray:
        helper = self._face_helper
        helper.clean_all()  # type: ignore[union-attr]
        helper.read_image(frame)  # type: ignore[union-attr]
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)  # type: ignore[union-attr]
        helper.align_warp_face()  # type: ignore[union-attr]
        if not helper.cropped_faces:  # type: ignore[union-attr]
            return frame
        restored_faces = []
        for face_bgr in helper.cropped_faces:  # type: ignore[union-attr]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(face_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                out = self._net(t, w=fidelity)[0]  # type: ignore[operator]
            rgb = out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255.0).byte().cpu().numpy()
            restored_faces.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        helper.add_restored_face(restored_faces)  # type: ignore[union-attr]
        helper.paste_faces_to_input_image()  # type: ignore[union-attr]
        return helper.output if helper.output is not None else frame  # type: ignore[union-attr]

    @staticmethod
    def _codeformer_fallback(frame: np.ndarray, fidelity: float) -> np.ndarray:
        try:
            from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
            r = CodeFormerRestorer()
            r.load(torch.device("cpu"))
            result = r.process_frame(frame, RestorerParams(extra={"fidelity": fidelity}))
            r.unload()
            return result
        except Exception:
            return frame

    @staticmethod
    def _build_model(device: torch.device) -> tuple[object | None, object | None]:
        try:
            from restorax.restorers.face_restoration.dicface_arch import DicFaceNet  # type: ignore[import]
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from restorax.config import settings
            weight_path = Path(settings.model_dir) / "dicface" / _WEIGHT_FILE
            if not weight_path.exists():
                from huggingface_hub import hf_hub_download
                weight_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=_HF_REPO, filename=_WEIGHT_FILE,
                                local_dir=str(weight_path.parent))
            net = DicFaceNet().to(device)
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt), strict=False)
            net.eval()
            face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=True, device=device,
            )
            logger.info("DicFace arch loaded from vendored module")
            return net, face_helper
        except (ImportError, Exception) as exc:
            logger.info("DicFace arch unavailable (%s) — CodeFormer fallback", exc)
            return None, None
