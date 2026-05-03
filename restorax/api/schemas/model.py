from pydantic import BaseModel


class RestorerInfo(BaseModel):
    name: str
    category: str
    input_color_space: str
    output_color_space: str
    requires_temporal: bool
    min_vram_gb: float
    scale_factor: int
    tags: list[str]
    loaded: bool


class ModelListResponse(BaseModel):
    restorers: list[RestorerInfo]
