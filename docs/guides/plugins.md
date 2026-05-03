# Writing a Plugin

RestoraX supports third-party restorer plugins via Python's `importlib.metadata`
entry points. A plugin is any installed Python package that:

1. Defines a class extending `BaseRestorer`
2. Registers it under the `restorax.restorers` entry-point group

Plugins are auto-discovered at worker startup — no code change to RestoraX required.

## Minimal plugin structure

```
my-restorer-plugin/
├── pyproject.toml
├── my_plugin/
│   ├── __init__.py
│   └── my_restorer.py
```

## Implement BaseRestorer

```python
# my_plugin/my_restorer.py
import numpy as np
import torch
from restorax.core.restorer import BaseRestorer, RestorerCapabilities, RestorerCategory, RestorerParams

class MySuperRestorer(BaseRestorer):

    @property
    def name(self) -> str:
        return "my_super_restorer"  # unique slug

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",
            output_color_space="rgb",
            scale_factor=4,
            min_vram_gb=4.0,
        )

    def load(self, device: torch.device) -> None:
        # Load your model weights here
        self._model = ...
        self._loaded = True

    def unload(self) -> None:
        del self._model
        self._loaded = False

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        # frame: HxWx3 uint8 RGB
        # return: HxWx3 uint8 RGB (upscaled by scale_factor)
        ...
```

## Register the entry point

```toml
# pyproject.toml
[project.entry-points."restorax.restorers"]
my_super_restorer = "my_plugin.my_restorer:MySuperRestorer"
```

## Install and use

```bash
pip install -e ./my-restorer-plugin

# Verify it appears in the catalog
restorax models
# my_super_restorer   super_resolution   4×   4 GB   ...

# Use in a preset
# configs/presets/my_pipeline.yaml
# stages:
#   - restorer: my_super_restorer
#     scale: 4
```

## Plugin contract

- `name` must be globally unique (no collision with built-in slugs)
- `process_frame` must return `np.ndarray` with `dtype=uint8`
- Output spatial dimensions must equal `input * scale_factor`
- `load()` must be idempotent; `unload()` must free all VRAM
- For temporal models: override `process_sequence` and set `requires_temporal=True`
