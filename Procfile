# RestoraX — Production Procfile
#
# Requirements: honcho (pip install honcho) or foreman (gem install foreman)
#
# External services (Redis, Postgres, MinIO) must be running before starting.
# Use docker-compose.yml for a full production stack, or start only the
# infrastructure services:
#   docker compose up -d redis postgres minio
#
# Usage:
#   honcho start                        # all processes
#   honcho start api worker             # headless (no UI, no flower)
#   PORT=9000 honcho start api          # custom port
#
# GPU workers: set RESTORAX_DEVICE=cuda and CUDA_VISIBLE_DEVICES=0
# Multi-GPU:   RESTORAX_GPU_QUEUES=gpu_0,gpu_1 and start one worker per GPU

api:      uvicorn restorax.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${API_WORKERS:-1}
worker:   celery -A restorax.tasks.celery_app worker --queues ${RESTORAX_GPU_QUEUES:-gpu_default} --concurrency=1 --loglevel=info
frontend: cd frontend && npm start
flower:   celery -A restorax.tasks.celery_app flower --port=${FLOWER_PORT:-5555}
