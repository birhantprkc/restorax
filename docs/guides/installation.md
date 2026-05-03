# Installation

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11 |
| CUDA | — (CPU works) | 12.1+ |
| GPU VRAM | — | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB (code + deps) | 20 GB (+ model weights) |

## Conda (recommended)

```bash
conda create -n restorax python=3.11
conda activate restorax

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core package + dependencies
pip install -e ".[dev]"
pip install basicsr av opencv-python-headless
```

## pip (virtual env)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
pip install basicsr av opencv-python-headless
```

## Optional extras

```bash
# Face restoration (CodeFormer + GFPGAN)
pip install -e ".[face]"

# ONNX Runtime for accelerated inference
pip install onnxruntime-gpu

# Documentation
pip install mkdocs-material mkdocstrings[python]

# Prometheus monitoring
pip install prometheus-fastapi-instrumentator
```

## Docker (no local Python setup)

```bash
# Production (GPU required)
docker-compose up --build

# Development (CPU, hot-reload)
docker-compose -f docker-compose.dev.yml up
```

See [Docker Deployment](docker.md) for full details.
