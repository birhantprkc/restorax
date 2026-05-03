# Configuration

All settings are read from environment variables (prefix: `RESTORAX_`) or a `.env` file in the project root.

## Full reference

| Variable | Default | Description |
|---|---|---|
| `RESTORAX_DATABASE_URL` | `sqlite+aiosqlite:///./restorax.db` | Database URL. Use `postgresql+asyncpg://` for production. |
| `RESTORAX_REDIS_URL` | `redis://localhost:6379/0` | Redis connection for Celery broker + progress pub/sub. |
| `RESTORAX_STORAGE_BACKEND` | `local` | `local` or `s3` |
| `RESTORAX_STORAGE_LOCAL_ROOT` | `./data` | Root directory for local file storage. |
| `RESTORAX_S3_ENDPOINT_URL` | `http://localhost:9000` | S3/MinIO endpoint. |
| `RESTORAX_S3_BUCKET` | `restorax` | S3 bucket name. |
| `RESTORAX_S3_ACCESS_KEY` | `minioadmin` | S3 access key. |
| `RESTORAX_S3_SECRET_KEY` | `minioadmin` | S3 secret key. |
| `RESTORAX_DEVICE` | `cuda` | PyTorch device: `cpu`, `cuda`, `cuda:0`, `cuda:1` |
| `RESTORAX_MODEL_DIR` | `./models` | Directory for downloaded model weights. |
| `RESTORAX_REGISTRY_MAX_LOADED` | `2` | Max models in VRAM simultaneously per worker. |
| `RESTORAX_GPU_QUEUES` | _(unset → `gpu_default`)_ | Comma-separated Celery queue names for multi-GPU routing. |
| `APP_ENV` | `development` | `development` or `production` (affects CORS, log verbosity). |
| `LOG_LEVEL` | `INFO` | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## Example `.env`

```env
RESTORAX_DATABASE_URL=postgresql+asyncpg://restorax:password@localhost:5432/restorax
RESTORAX_REDIS_URL=redis://localhost:6379/0
RESTORAX_STORAGE_BACKEND=local
RESTORAX_STORAGE_LOCAL_ROOT=/data/restorax
RESTORAX_DEVICE=cuda
RESTORAX_MODEL_DIR=/data/models
RESTORAX_REGISTRY_MAX_LOADED=2
RESTORAX_GPU_QUEUES=gpu_0,gpu_1
APP_ENV=production
LOG_LEVEL=INFO
```

## VRAM budgeting

The `REGISTRY_MAX_LOADED` setting controls how many models stay warm in VRAM between jobs.
Set it based on your GPU memory:

| GPU VRAM | Recommended `REGISTRY_MAX_LOADED` |
|---|---|
| 8 GB | 1 |
| 12 GB | 1–2 |
| 16 GB | 2 |
| 24 GB | 3 |
| 40 GB+ | 4+ |

For multi-stage pipelines, the registry evicts the least-recently-used model before loading the next stage's model.
