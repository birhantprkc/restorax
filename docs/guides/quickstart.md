# Quick Start

## 1. Install

```bash
git clone https://github.com/yourname/restorax && cd restorax
conda create -n restorax python=3.11 && conda activate restorax
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]" && pip install honcho
cp .env.example .env        # review device, model dir, etc.
cd frontend && npm install && cd ..
```

**No GPU?** Replace the torch line:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Then set `RESTORAX_DEVICE=cpu` in `.env`.

---

## 2. Start external services

```bash
# Recommended: Docker Compose (starts Redis + Postgres)
docker compose -f docker-compose.deps.yml up -d

# Alternative: Redis only (SQLite is used by default in dev)
docker run -d -p 6379:6379 --name restorax-redis redis:7-alpine
```

---

## 3. Start the full stack with honcho

```bash
# All processes: API + worker + frontend + Flower monitor
honcho start -f Procfile.dev

# Headless (no UI, no flower)
honcho start -f Procfile.dev api worker

# API only — submit jobs via curl or the Python SDK
honcho start -f Procfile.dev api
```

Processes:

| Process | URL | Description |
|---|---|---|
| `api` | <http://localhost:8000> | FastAPI REST API + WebSocket |
| `worker` | — | Celery GPU worker |
| `frontend` | <http://localhost:3000> | Vite + React 18 web UI |
| `flower` | <http://localhost:5555> | Celery task monitor (optional) |

API docs live at <http://localhost:8000/docs>

---

## 4. Submit your first job

### Via Web UI

Open <http://localhost:3000>, drag and drop a video, choose a pipeline, click **Restore Video**.

### Via CLI

```bash
# 4× super-resolution
restorax run --input old_film.mp4 --pipeline sr_x4

# Full classic film pipeline (upscale + colorize + face restore + frame interpolation)
restorax run --input film.mp4 --pipeline classic_film --device cuda

# VHS tape restoration
restorax run --input tape.mp4 --pipeline vhs_restoration --output restored.mp4

# B&W newsreel: scratch removal + SR + colorization
restorax run --input newsreel.mp4 --pipeline newsreel

# With audio restoration
restorax run --input film.mp4 --pipeline classic_film_audio

# List all available pipelines
restorax presets

# List all AI models and their capabilities
restorax models
```

### Via REST API (curl)

```bash
# Upload video and start job
curl -X POST http://localhost:8000/jobs \
  -F "file=@film.mp4" \
  -F "pipeline_id=sr_x4"
# → {"id": "abc123", "status": "queued", ...}

# Poll status
curl http://localhost:8000/jobs/abc123

# Download when complete
curl http://localhost:8000/jobs/abc123/download -o restored.mp4

# Submit multiple files at once (batch)
curl -X POST http://localhost:8000/jobs/batch \
  -F "files=@film1.mp4" -F "files=@film2.mp4" \
  -F "pipeline_id=sr_x4"
```

### Watch live progress (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/jobs/abc123/progress");
ws.onmessage = e => {
  const { progress, status, output_path } = JSON.parse(e.data);
  console.log(`${(progress * 100).toFixed(0)}%  [${status}]`);
  if (status === "completed") console.log("Output:", output_path);
};
```

---

## 5. Docker (no local Python setup)

```bash
# Development — hot-reload, CPU worker, SQLite
docker-compose -f docker-compose.dev.yml up

# Production — GPU worker, PostgreSQL, MinIO
docker-compose up --build
```

---

## Python API

```python
import torch
from restorax.core.registry import ModelRegistry
from restorax.core.pipeline import Pipeline, PipelineRunner, Stage, compute_output_fps
from restorax.core.restorer import RestorerParams
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from restorax.video.reader import VideoReader
from restorax.video.writer import VideoWriter

registry = ModelRegistry(max_loaded=2)
registry.register(RealESRGANx4Restorer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
restorer = registry.get("real_esrgan_x4plus", device)

pipeline = Pipeline(
    name="sr_x4",
    stages=[Stage(restorer=restorer, params=RestorerParams(scale=4))],
)

with VideoReader("input.mp4") as reader:
    meta = reader.meta
    out_fps = compute_output_fps(pipeline, meta.fps)    # handles RIFE 2× fps
    with VideoWriter(
        "output.mp4", meta=meta, fps=out_fps,
        out_width=meta.width * 4, out_height=meta.height * 4,
    ) as writer:
        PipelineRunner().run(pipeline, reader, writer,
                             progress_cb=lambda p: print(f"{p:.0%}"))
```
