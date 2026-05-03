"""GET /models — list available restorers and their capabilities."""
from fastapi import APIRouter

from restorax.api.schemas.model import ModelListResponse, RestorerInfo
from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
from restorax.restorers.face_restoration.dicface import DicFaceRestorer
from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer
from restorax.restorers.colorization.ddcolor import DDColorRestorer
from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer
from restorax.restorers.frame_interpolation.rife import RIFERestorer
from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer
from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
from restorax.restorers.stabilization.gavs import GaVSRestorer
from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer
from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from restorax.restorers.super_resolution.tdm import TDMRestorer
from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
from restorax.restorers.super_resolution.vrt import VRTRestorer

router = APIRouter(prefix="/models", tags=["models"])

_RESTORER_CLASSES = [
    RealESRGANx4Restorer, BasicVSRPlusPlusRestorer, UpscaleAVideoRestorer,
    VRTRestorer, MambaIRRestorer, TDMRestorer, SeedVRRestorer,
    Waifu2xRestorer, FlashVSRRestorer, EvTextureRestorer,
    CodeFormerRestorer, CodeFormerPlusPlusRestorer, GFPGANRestorer, DicFaceRestorer,
    DDColorRestorer, RIFERestorer,
    ScratchRemovalRestorer, HDRTVDMRestorer, VideoStabilizationRestorer,
    GaVSRestorer, AIDeinterlaceRestorer,
]


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    restorers = []
    for cls in _RESTORER_CLASSES:
        instance = object.__new__(cls)
        caps = cls.capabilities.fget(instance)  # type: ignore[attr-defined]
        restorers.append(
            RestorerInfo(
                name=cls.name.fget(instance),  # type: ignore[attr-defined]
                category=caps.category.value,
                input_color_space=caps.input_color_space,
                output_color_space=caps.output_color_space,
                requires_temporal=caps.requires_temporal,
                min_vram_gb=caps.min_vram_gb,
                scale_factor=caps.scale_factor,
                tags=caps.tags,
                loaded=False,  # Phase 3: wire into live registry
            )
        )
    return ModelListResponse(restorers=restorers)
