# Multi-GPU Setup

RestoraX routes each job to a single GPU. Multiple GPUs are saturated
by running multiple jobs concurrently — one job per GPU.

## Architecture

```
API  ──POST /jobs──►  Redis queue  ──►  Worker GPU 0 (CUDA:0)
                                   └──►  Worker GPU 1 (CUDA:1)
```

Each worker is a separate Celery process with `--concurrency=1`.
The API dispatches jobs round-robin across GPU queues.

## Configuration

Set the environment variable in the API service:

```env
RESTORAX_GPU_QUEUES=gpu_0,gpu_1,gpu_2,gpu_3
```

## Start workers

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 celery -A restorax.tasks.celery_app worker \
  --queues gpu_0 --concurrency=1 --hostname worker_0@%h \
  --loglevel=info

# GPU 1 (separate terminal or screen/tmux pane)
CUDA_VISIBLE_DEVICES=1 celery -A restorax.tasks.celery_app worker \
  --queues gpu_1 --concurrency=1 --hostname worker_1@%h \
  --loglevel=info
```

## Docker Compose (multi-GPU)

```yaml
# docker-compose.override.yml
services:
  worker_gpu0:
    extends:
      service: worker
    environment:
      CUDA_VISIBLE_DEVICES: "0"
    command: >
      celery -A restorax.tasks.celery_app worker
      --queues gpu_0 --concurrency=1 --hostname worker_0@%h
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  worker_gpu1:
    extends:
      service: worker
    environment:
      CUDA_VISIBLE_DEVICES: "1"
    command: >
      celery -A restorax.tasks.celery_app worker
      --queues gpu_1 --concurrency=1 --hostname worker_1@%h
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

## Monitor with Celery Flower

```bash
pip install flower
celery -A restorax.tasks.celery_app flower
# Open http://localhost:5555
```
