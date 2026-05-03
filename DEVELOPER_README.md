# RestoraX — Developer Guide

Everything you need to set up a local development environment, understand
the architecture, run tests, add a restorer, write a plugin, and ship a release.

---

## Table of contents

- [Local dev setup](#local-dev-setup)
- [Project layout](#project-layout)
- [Architecture overview](#architecture-overview)
- [Running the stack locally](#running-the-stack-locally)
- [Running tests](#running-tests)
- [Adding a new restorer](#adding-a-new-restorer)
- [Writing a plugin package](#writing-a-plugin-package)
- [Vendoring a model architecture](#vendoring-a-model-architecture)
- [ONNX export](#onnx-export)
- [Multi-GPU workers](#multi-gpu-workers)
- [API reference](#api-reference)
- [Frontend development](#frontend-development)
- [Code style](#code-style)
- [Release process](#release-process)

---

## Local dev setup

### 1. Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11 | `conda create -n restorax python=3.11` |
| CUDA Toolkit | 12.1 | [NVIDIA docs](https://developer.nvidia.com/cuda-12-1-0-download-archive) |
| FFmpeg | any recent | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| Node.js | 20 LTS | `nvm install 20` |
| honcho | latest | `pip install honcho` — reads the Procfile |
| Docker | 24+ | [docs.docker.com](https://docs.docker.com/get-docker/) — for `docker-compose.deps.yml` |

### 2. Python environment

```bash
conda activate restorax

# PyTorch with CUDA 12.1 (replace cu121 with cpu for CPU-only dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install RestoraX with all dev/test dependencies + process manager
pip install -e ".[dev]" && pip install honcho
```

### 3. Frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 4. Environment file

```bash
cp .env.example .env
# Key settings to review:
#   RESTORAX_DEVICE=cuda        (use cpu if no GPU)
#   RESTORAX_MODEL_DIR=./models
#   RESTORAX_DATABASE_URL       (SQLite by default — no Postgres needed in dev)
```

### 5. Start external services

```bash
# Start Redis + Postgres with a single command:
docker compose -f docker-compose.deps.yml up -d
```

This starts Redis on `localhost:6379` and Postgres on `localhost:5432`.
SQLite is the default database — Postgres is optional for local dev.

### 6. Start the app

```bash
# All processes: API + worker + frontend + Flower
honcho start -f Procfile.dev

# Headless — no UI, no flower
honcho start -f Procfile.dev api worker

# API only
honcho start -f Procfile.dev api
```

Processes defined in [Procfile.dev](Procfile.dev):

| Process | URL | Description |
|---|---|---|
| `api` | <http://localhost:8000> | FastAPI REST API + WebSocket |
| `worker` | — | Celery GPU worker |
| `frontend` | <http://localhost:3000> | Next.js web UI |
| `flower` | <http://localhost:5555> | Celery monitor (optional) |

### 7. Verify installation

```bash
# Python tests (unit + integration + system)
python -m pytest tests/ -q
# 309 passed

# Frontend tests
cd frontend && npm test
# 25 passed

restorax models    # lists all 26 video + 3 audio restorers
restorax presets   # lists all 7 built-in pipeline presets
```

---

## Project layout

```
restorax/
├── restorax/                  Python package
│   ├── api/                   FastAPI app + routers + schemas
│   │   ├── app.py             create_app() with lifespan
│   │   ├── deps.py            FastAPI dependency injection
│   │   ├── middleware.py      Request ID, timing headers
│   │   └── routers/
│   │       ├── jobs.py        POST /jobs, /jobs/batch, GET, DELETE, download
│   │       ├── models.py      GET /models
│   │       ├── pipelines.py   CRUD /pipelines
│   │       └── ws.py          WebSocket /ws/jobs/{id}/progress
│   ├── core/                  Domain logic — NO framework imports
│   │   ├── restorer.py        BaseRestorer ABC + RestorerParams + capabilities
│   │   ├── pipeline.py        Pipeline, Stage, PipelineRunner (sequential chunks)
│   │   ├── registry.py        ModelRegistry (LRU VRAM cache)
│   │   ├── job.py             Job, JobStatus value objects
│   │   ├── plugin.py          discover_plugins() / register_plugins()
│   │   └── exceptions.py      Domain exceptions
│   ├── restorers/             One sub-package per restoration category
│   │   ├── super_resolution/  real_esrgan, basicvsr_pp, vrt, upscale_a_video
│   │   ├── colorization/      ddcolor
│   │   ├── face_restoration/  codeformer, gfpgan
│   │   ├── frame_interpolation/ rife
│   │   ├── artifact_removal/  scratch_removal (ProPainter)
│   │   ├── hdr/               hdrtvdm
│   │   ├── stabilization/     deep_flow_stab
│   │   └── deinterlacing/     ai_deinterlace
│   ├── tasks/                 Celery tasks
│   │   ├── celery_app.py      Celery() instance + queue config
│   │   ├── job_tasks.py       run_job task + module-level ModelRegistry
│   │   ├── progress.py        ProgressReporter → Redis pub/sub
│   │   └── gpu_router.py      Round-robin GPU queue selection
│   ├── db/                    Database layer
│   │   ├── models.py          SQLAlchemy ORM (JobModel, PipelineTemplateModel)
│   │   ├── session.py         Async session factory + create_tables()
│   │   └── repositories/      job_repo.py, pipeline_repo.py
│   ├── storage/               File storage backends
│   │   ├── base.py            StorageBackend Protocol
│   │   ├── local.py           LocalStorageBackend
│   │   └── s3.py              S3StorageBackend (MinIO-compatible)
│   ├── video/                 Video I/O
│   │   ├── reader.py          VideoReader — PyAV frame iterator + metadata
│   │   ├── writer.py          VideoWriter — PyAV encoder + audio passthrough
│   │   └── utils.py           Color space, padding, tiling, Gaussian merge
│   ├── metrics/               Quality assessment
│   │   ├── full_reference.py  PSNR, SSIM, LPIPS, compute_all()
│   │   └── no_reference.py    NIQE, BRISQUE stubs
│   ├── utils/
│   │   └── onnx_export.py     Export restorer models to ONNX
│   ├── config.py              Settings(BaseSettings) — reads .env
│   └── cli.py                 Click CLI: run / models / presets
├── frontend/                  Next.js 14 web UI
│   ├── app/                   App Router pages
│   │   ├── page.tsx           Job dashboard
│   │   └── jobs/[id]/page.tsx Job detail + progress + download
│   ├── components/
│   │   ├── JobForm.tsx        Upload + pipeline selector
│   │   ├── ProgressBar.tsx    WebSocket-driven live progress
│   │   ├── CompareSlider.tsx  react-compare-slider before/after
│   │   └── JobCard.tsx        Job summary card
│   ├── lib/api.ts             Typed API client + WebSocket helper
│   └── tests/                 Vitest component + API lib tests (25 tests)
├── configs/presets/           YAML pipeline presets
├── tests/
│   ├── conftest.py            Shared fixtures (synthetic video, IdentityRestorer)
│   ├── unit/                  CPU-only, no real weights (~260 tests)
│   ├── integration/           FastAPI + SQLite in-process, Celery eager (22 tests)
│   └── system/                Full-app smoke tests (7 tests)
├── docs/                      MkDocs source
├── .github/
│   ├── workflows/             CI (ci.yml) + Release (release.yml)
│   ├── ISSUE_TEMPLATE/        Bug report + feature request templates
│   └── PULL_REQUEST_TEMPLATE.md
├── Dockerfile                 API image (Python, no CUDA)
├── Dockerfile.worker          GPU worker image (CUDA 12.1)
├── docker-compose.yml         Production stack (API + worker + Redis + Postgres + MinIO)
├── docker-compose.dev.yml     Dev stack (self-contained, hot-reload, CPU worker)
├── docker-compose.deps.yml    Deps only: Redis + Postgres for Procfile.dev workflow
├── Procfile               Production process definitions (api, worker, frontend, flower)
├── Procfile.dev           Development process definitions (hot-reload)
├── docker-compose.dev.yml     Dev stack (hot-reload, CPU worker)
├── pyproject.toml             Deps + entry points + tool config
└── alembic.ini / alembic/     Database migrations
```

---

## Architecture overview

### Sequential chunked pipeline

The `PipelineRunner` processes video in fixed-size chunks of frames:

```
VideoReader → [chunk 0] → Stage 1 → Stage 2 → ... → VideoWriter
             [chunk 1] → Stage 1 → Stage 2 → ... → VideoWriter
             ...
```

**Why not process the full video at once?**  
A 2-hour film at 24fps = 86,400 frames. No model can hold all of these in VRAM. Chunked execution keeps memory constant regardless of duration.

**Chunk overlap**: `chunk_overlap=2` means the last 2 frames of each chunk appear at the start of the next. Temporal models (BasicVSR++, RIFE) use this context to avoid seam artifacts at chunk boundaries.

**Color-space conversion**: Each restorer declares `input_color_space` / `output_color_space` (`"rgb"` or `"bgr"`). The runner converts automatically at each stage boundary — no manual conversion in restorer code.

### LRU model registry

Each Celery worker keeps a `ModelRegistry(max_loaded=2)`. When a pipeline stage requests a model that isn't loaded, the registry evicts the least-recently-used model (calling `unload()` to free VRAM) before loading the new one.

```
Pipeline: Real-ESRGAN → CodeFormer

Step 1: load Real-ESRGAN (6 GB)            → cache: [Real-ESRGAN]
Step 2: process all chunks with Real-ESRGAN
Step 3: load CodeFormer (4 GB)             → cache: [Real-ESRGAN, CodeFormer]
        (both fit; no eviction needed)
Step 4: process all chunks with CodeFormer

Next job: load BasicVSR++ (8 GB)
          → evict Real-ESRGAN (LRU)         → cache: [CodeFormer, BasicVSR++]
```

### Progress reporting

```
Celery worker
  └── PipelineRunner.run(progress_cb=...)
        └── on each chunk: ProgressReporter.update(progress)
              └── redis.publish("restorax:job_progress:{id}", json)

FastAPI WebSocket handler
  └── aioredis.subscribe("restorax:job_progress:{id}")
        └── ws.send_json(event) → browser
```

### Gaussian tile merging

For inputs larger than `tile_size` pixels, the restorer tiles the frame
and calls `process_frame` on each tile independently. Tiles are merged
with a 2D Gaussian weight window — the centre of each tile contributes
fully, edges blend with neighbours — eliminating visible seam lines.

---

## Running the stack locally

### Option A — Procfile (recommended)

The Procfile is the single source of truth for all development processes.
One command replaces four terminals.

```bash
# Start external services (Redis + Postgres)
docker compose -f docker-compose.deps.yml up -d

# Start all four processes at once
honcho start -f Procfile.dev
```

Start only what you need for a given task:

```bash
honcho start -f Procfile.dev api worker     # backend only — no UI
honcho start -f Procfile.dev api            # API smoke-testing
honcho start -f Procfile.dev frontend       # UI development against a running API
```

Logs from all processes are printed with colour-coded prefixes:

```
09:41:03 api.1      | INFO:     Uvicorn running on http://0.0.0.0:8000
09:41:03 worker.1   | [2026-04-25 09:41:03,512: INFO] celery@host ready.
09:41:03 frontend.1 | ▲ Next.js 16  - Local: http://localhost:3000
09:41:03 flower.1   | [I 09:41:03.781 ...] Inspect method inspector.
```

Process URLs:

- API + docs: <http://localhost:8000/docs>
- Web UI: <http://localhost:3000>
- Celery monitor: <http://localhost:5555>

Stop all processes: `Ctrl-C` once in the honcho terminal.

### Option B — CLI only (no Redis, no DB)

```bash
restorax run --input input.mp4 --pipeline sr_x4 --device cuda
```

No services needed — the CLI runs the pipeline in-process.

### Option C — Full Docker

```bash
# Dev: hot-reload, CPU worker, SQLite (self-contained)
docker compose -f docker-compose.dev.yml up --build

# Prod: GPU worker, PostgreSQL, MinIO
docker compose up --build
```

---

## Running tests

```bash
# All tests (226 unit + 11 integration = 237 total)
python -m pytest tests/ -q

# Unit tests only (fastest, no services needed)
python -m pytest tests/unit/ -v

# Integration tests (uses SQLite, no Redis/worker needed)
python -m pytest tests/integration/ -v

# Single test file
python -m pytest tests/unit/test_registry.py -v

# Coverage report
python -m pytest tests/unit/ --cov=restorax --cov-report=term-missing

# GPU tests (requires CUDA GPU + real model weights)
python -m pytest tests/ -m gpu -v
```

### Test architecture

| Layer | Location | Requirements |
|---|---|---|
| Unit | `tests/unit/` | CPU only, no weights, no DB |
| Integration | `tests/integration/` | SQLite, no Redis/worker |
| GPU | `tests/unit/` marked `@pytest.mark.gpu` | CUDA GPU + downloaded weights |

Every restorer has a dedicated unit test file in `tests/unit/restorers/`.
All restorers use **stub models** (correct shape/dtype, no real weights)
so CI runs without any GPU or HuggingFace downloads.

---

## Adding a new restorer

### 1. Create the file

```
restorax/restorers/<category>/<name>.py
```

Pick the right category: `super_resolution`, `colorization`, `face_restoration`,
`frame_interpolation`, `artifact_removal`, `hdr`, `stabilization`, `deinterlacing`.

### 2. Implement BaseRestorer

```python
from restorax.core.restorer import BaseRestorer, RestorerCapabilities, RestorerCategory, RestorerParams
import numpy as np, torch

class MyRestorer(BaseRestorer):

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "my_restorer"               # unique registry key

    @property
    def capabilities(self) -> RestorerCapabilities:
        return RestorerCapabilities(
            category=RestorerCategory.SUPER_RESOLUTION,
            input_color_space="rgb",       # "rgb" or "bgr"
            output_color_space="rgb",
            scale_factor=4,
            min_vram_gb=4.0,
            requires_temporal=False,       # set True for sequence models
        )

    def load(self, device: torch.device) -> None:
        # Load model weights into VRAM
        self._model = ...
        self._device = device
        self._loaded = True

    def unload(self) -> None:
        del self._model
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def process_frame(self, frame: np.ndarray, params: RestorerParams) -> np.ndarray:
        # frame: HxWx3 uint8 in input_color_space
        # return: HxWx3 uint8 in output_color_space (scaled by scale_factor)
        ...
```

For **temporal models** (see BasicVSR++, RIFE), override `process_sequence` and set `requires_temporal=True`.

### 3. Register in worker

```python
# restorax/tasks/job_tasks.py — in _get_registry()
from restorax.restorers.<category>.<name> import MyRestorer
_registry.register(MyRestorer)
```

### 4. Register in models API

```python
# restorax/api/routers/models.py
from restorax.restorers.<category>.<name> import MyRestorer
_RESTORER_CLASSES = [..., MyRestorer]
```

### 5. Register in pyproject.toml entry points

```toml
[project.entry-points."restorax.restorers"]
my_restorer = "restorax.restorers.<category>.<name>:MyRestorer"
```

### 6. Write unit tests

```
tests/unit/restorers/test_<name>.py
```

Test checklist:
- `test_name()` — correct slug
- `test_capabilities()` — category, scale, color space, requires_temporal
- `test_process_frame_output_shape()` — output HxW = input × scale_factor
- `test_process_frame_output_dtype()` — uint8
- `test_is_loaded()` / `test_unload()` — lifecycle

Use mocked weights (no HuggingFace downloads in tests):

```python
@pytest.fixture
def restorer():
    from unittest.mock import patch
    r = MyRestorer()
    with patch.object(r, "load") as mock_load:
        def fake_load(device):
            r._model = MyStub()
            r._device = device
            r._loaded = True
        mock_load.side_effect = fake_load
        r.load(torch.device("cpu"))
        yield r
```

### 7. Create a preset (optional)

```yaml
# configs/presets/my_pipeline.yaml
name: my_pipeline
chunk_size: 16
chunk_overlap: 2
stages:
  - restorer: my_restorer
    scale: 4
    enabled: true
```

---

## Writing a plugin package

A plugin is a separate installable Python package. It only needs to:

1. Implement `BaseRestorer`
2. Register under the `restorax.restorers` entry-point group

```toml
# my_plugin/pyproject.toml
[project.entry-points."restorax.restorers"]
my_restorer = "my_plugin.restorer:MyRestorer"
```

After `pip install my_plugin`, the restorer appears automatically
in `restorax models` and the web UI's model list.

See [docs/guides/plugins.md](docs/guides/plugins.md) for the full walkthrough.

---

## Vendoring a model architecture

Several restorers include a **stub model** that produces geometrically
correct output (right shape, right dtype) but no real enhancement.
To activate production-quality output, vendor the real arch:

| Restorer | Source repo | Target path |
|---|---|---|
| `ddcolor` | [piddnad/DDColor](https://github.com/piddnad/DDColor) | `restorers/colorization/ddcolor_arch.py` |
| `rife_v4` | [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | `restorers/frame_interpolation/rife_arch/` |
| `upscale_a_video` | [sczhou/Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) | `restorers/super_resolution/upscale_a_video_arch.py` |
| `scratch_removal` | [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | `restorers/artifact_removal/propainter_arch.py` |
| `hdrtvdm` | [AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM) | `restorers/hdr/hdrtvdm_arch.py` |

The restorer's `_build_model()` uses a `try/except ImportError` to auto-detect
whether the arch is present:

```python
try:
    from restorax.restorers.colorization.ddcolor_arch import DDColorArch
    model = DDColorArch(...)
except ImportError:
    model = _DDColorStub()   # falls back silently
```

Model weights download automatically via HuggingFace Hub on first `load()`.

---

## ONNX export

Export a restorer to ONNX for ~30% faster inference:

```bash
python -c "
from restorax.utils.onnx_export import export_restorer_to_onnx
export_restorer_to_onnx('real_esrgan_x4plus', input_size=(1,3,256,256))
"
# Saved to: models/real_esrgan_x4plus/real_esrgan_x4plus.onnx
```

The ONNX file uses dynamic axes for height/width, so it works on any input resolution. Use `onnxruntime-gpu` for TensorRT-backed inference.

---

## Multi-GPU workers

One Celery worker per GPU, each bound to its own queue:

```bash
CUDA_VISIBLE_DEVICES=0 celery -A restorax.tasks.celery_app worker \
  --queues gpu_0 --concurrency=1 --hostname worker_0@%h

CUDA_VISIBLE_DEVICES=1 celery -A restorax.tasks.celery_app worker \
  --queues gpu_1 --concurrency=1 --hostname worker_1@%h
```

Set in the API environment:

```env
RESTORAX_GPU_QUEUES=gpu_0,gpu_1
```

The `gpu_router.py` round-robins new jobs across the configured queues.
See [docs/guides/multi_gpu.md](docs/guides/multi_gpu.md) for Docker Compose config.

---

## API reference

Interactive docs at <http://localhost:8000/docs> (Swagger) or <http://localhost:8000/redoc>.

### Key endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs` | Submit a job (multipart file upload) |
| `POST` | `/jobs/batch` | Submit multiple files (one job each) |
| `GET` | `/jobs` | List recent jobs |
| `GET` | `/jobs/{id}` | Get job status + metrics |
| `GET` | `/jobs/{id}/download` | Download output video |
| `DELETE` | `/jobs/{id}` | Delete job + output |
| `GET` | `/models` | List all registered restorers |
| `POST` | `/pipelines` | Create pipeline template |
| `GET` | `/pipelines` | List pipeline templates |
| `PUT` | `/pipelines/{id}` | Update pipeline template |
| `DELETE` | `/pipelines/{id}` | Delete pipeline template |
| `WS` | `/ws/jobs/{id}/progress` | Live progress stream |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### WebSocket progress events

```json
{"job_id": "abc", "progress": 0.45, "status": "running"}
{"job_id": "abc", "progress": 1.0,  "status": "completed", "output_path": "/data/outputs/abc/output.mp4"}
{"job_id": "abc", "progress": 0.0,  "status": "failed", "error": "CUDA OOM"}
```

---

## Frontend development

```bash
cd frontend
npm install
npm run dev        # hot-reload dev server → http://localhost:3000
npm run build      # production build
npm run lint       # ESLint
```

### Environment

```env
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Key files

| File | Purpose |
|---|---|
| `lib/api.ts` | Typed API client + `wsJobProgress()` WebSocket helper |
| `components/JobForm.tsx` | File upload + pipeline selector |
| `components/ProgressBar.tsx` | WebSocket-driven progress bar |
| `components/CompareSlider.tsx` | Before/after comparison (`react-compare-slider`) |
| `components/JobCard.tsx` | Job summary card with live progress |
| `app/page.tsx` | Job list dashboard |
| `app/jobs/[id]/page.tsx` | Job detail with metrics + download |

---

## Code style

| Tool | Config | Command |
|---|---|---|
| Ruff (lint + format) | `pyproject.toml [tool.ruff]` | `ruff check restorax/` / `ruff format restorax/` |
| Mypy (types) | `pyproject.toml [tool.mypy]` | `mypy restorax/ --ignore-missing-imports` |
| ESLint (frontend) | `frontend/.eslintrc` | `cd frontend && npm run lint` |

### Conventions

- **No comments on obvious code.** Only add a comment when the _why_ is non-obvious.
- **No speculative features.** Build exactly what was asked for.
- **Touch only what you must.** Don't improve adjacent code unless asked.
- Restorer `process_frame` must return `np.ndarray` with `dtype=np.uint8`.
- Output spatial size must be exactly `(H * scale_factor, W * scale_factor)`.
- Never call `torch.cuda.empty_cache()` except in `unload()`.
- Database migrations via Alembic only — never `create_all()` in production.

---

## Release process

### Version bump

```bash
# Edit version in pyproject.toml
vim pyproject.toml   # version = "0.2.0"

git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
git tag v0.2.0
git push origin main --tags
```

Pushing a `v*` tag triggers the [Release workflow](.github/workflows/release.yml):
- Builds and publishes to PyPI (via OIDC trusted publishing)
- Builds and pushes Docker images to GitHub Container Registry

### Changelog

Keep `CHANGELOG.md` with entries under `[Unreleased]` → move to a version section on release.

### Docs

```bash
mkdocs build          # static site in site/
mkdocs gh-deploy      # push to gh-pages branch
```

---

## Common issues

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: av` | PyAV not installed | `pip install av` |
| `av.add_stream rate error` | Float passed as fps | Already fixed — uses `Fraction(fps).limit_denominator(1001)` |
| `AssertionError: Status code 204 must not have response body` | FastAPI ≥0.111 | Return `Response(status_code=204)` explicitly |
| `Entry points group empty` | Package not installed | `pip install -e .` |
| CUDA OOM on 4K input | Tile size not set | Add `tile_size: 512` to preset stage |
| Worker not picking up jobs | Wrong queue name | Check `RESTORAX_GPU_QUEUES` and `--queues` match |
| WebSocket disconnects immediately | Redis pub/sub timeout | Ensure Redis is running; check `RESTORAX_REDIS_URL` |
