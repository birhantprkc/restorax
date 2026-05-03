from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRequest:
    input_path: str
    pipeline_id: str
    output_format: str = "mp4"
    output_codec: str = "libx264"
    output_crf: int = 18
    preserve_audio: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    id: UUID = field(default_factory=uuid4)
    request: JobRequest = field(default_factory=lambda: JobRequest("", ""))
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0.0 → 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_path: str | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    celery_task_id: str | None = None
