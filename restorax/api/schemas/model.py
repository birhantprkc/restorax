from pydantic import BaseModel


class RestorerInfo(BaseModel):
    name: str
    category: str
    input_color_space: str | None = None
    output_color_space: str | None = None
    requires_temporal: bool | None = None
    min_vram_gb: float | None = None
    scale_factor: int | None = None
    min_ram_gb: float | None = None
    supports_stereo: bool | None = None
    sample_rates: list[int] | None = None
    tags: list[str]
    loaded: bool


class ModelListResponse(BaseModel):
    restorers: list[RestorerInfo]
