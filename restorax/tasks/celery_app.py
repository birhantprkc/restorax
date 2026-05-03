from celery import Celery

from restorax.config import settings

celery_app = Celery(
    "restorax",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["restorax.tasks.job_tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Worker — one GPU task at a time per process
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Results
    result_expires=3600 * 24,
    # Routing: single-job tasks go to gpu_default (round-robin across GPU workers)
    # Multi-GPU setup: start separate workers with --queues gpu_0, gpu_1, etc.
    # and set RESTORAX_GPU_QUEUES=gpu_0,gpu_1 to enable round-robin dispatch.
    task_default_queue="gpu_default",
    task_routes={
        "restorax.tasks.job_tasks.run_job": {"queue": "gpu_default"},
        "restorax.tasks.job_tasks.run_job_on_gpu": {},  # queue set dynamically
    },
    # Queue declaration — all GPU queues share the same task definition
    task_queues={
        "gpu_default": {"exchange": "gpu_default", "routing_key": "gpu_default"},
        "gpu_0": {"exchange": "gpu_0", "routing_key": "gpu_0"},
        "gpu_1": {"exchange": "gpu_1", "routing_key": "gpu_1"},
        "gpu_2": {"exchange": "gpu_2", "routing_key": "gpu_2"},
        "gpu_3": {"exchange": "gpu_3", "routing_key": "gpu_3"},
    },
)
