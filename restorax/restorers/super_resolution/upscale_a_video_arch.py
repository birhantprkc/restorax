# Vendored from https://github.com/sczhou/Upscale-A-Video (S-Lab License 1.0 — non-commercial use only)
# See https://github.com/sczhou/Upscale-A-Video/blob/master/LICENSE for full terms.
# Commercial use requires contacting Dr. Shangchen Zhou (shangchenzhou@gmail.com).
#
# ponytail: This is a minimal ADAPTER shim, not a faithful port. The real upstream
# pipeline (models_video/pipeline_upscale_a_video.py, class VideoUpscalePipeline)
# depends on ~6000 lines of custom upstream modules that are NOT vendored here:
#   models_video/unet_video.py, autoencoder_kl_cond_video.py, propagation_module.py,
#   attention.py, temporal_module.py, unet_blocks.py, resnet.py,
#   diffusers_attention.py, scheduling_ddim.py, color_correction.py, RAFT/*
# Porting those faithfully is out of scope for this change (see task notes).
# `from_pretrained` therefore raises ImportError — caught by
# `upscale_a_video.py::_build_pipeline` and converted to a clean RestorerLoadError
# — rather than silently faking a working diffusion pipeline.
from __future__ import annotations

import diffusers  # noqa: F401 — required upstream dependency; raises ImportError if missing
import torch


class UpscaleAVideoPipeline:
    """Adapter matching the call surface `UpscaleAVideoRestorer` expects.

    Real signature (upstream `VideoUpscalePipeline`): constructed from
    text_encoder/tokenizer/vae/unet/scheduler/propagator components, called as
    `pipe(image=frames, num_inference_steps=.., guidance_scale=..)` returning an
    object with a `.frames` attribute.
    """

    def __init__(
        self,
        vae: object | None = None,
        text_encoder: object | None = None,
        tokenizer: object | None = None,
        unet: object | None = None,
        scheduler: object | None = None,
        propagator: object | None = None,
    ) -> None:
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.propagator = propagator
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "UpscaleAVideoPipeline":
        self.device = device
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str) -> "UpscaleAVideoPipeline":
        raise ImportError(
            "Upscale-A-Video's custom UNetVideoModel/AutoencoderKLVideo/Propagation/"
            "RAFT modules are not vendored in restorax — only this adapter shim is. "
            "Real weight loading is unimplemented."
        )

    def __call__(self, image, num_inference_steps: int = 30, guidance_scale: float = 7.5):
        raise NotImplementedError(
            "UpscaleAVideoPipeline inference requires the un-vendored upstream "
            "UNetVideoModel/AutoencoderKLVideo/Propagation modules."
        )
