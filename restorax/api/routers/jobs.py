"""
Job management endpoints.

POST   /jobs         — submit a new restoration job (file upload)
POST   /jobs/batch   — submit multiple files at once (one job per file)
GET    /jobs         — list jobs
GET    /jobs/{id}    — get job status
GET    /jobs/{id}/download — download output video
DELETE /jobs/{id}    — cancel / delete a job
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from restorax.api.deps import get_db
from restorax.api.schemas.job import JobCreateRequest, JobListResponse, JobResponse
from restorax.config import settings
from restorax.core.exceptions import JobNotFoundError
from restorax.db.models import JobModel
from restorax.db.repositories.job_repo import JobRepository

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    file: UploadFile = File(..., description="Input video file"),
    pipeline_id: str = Form(...),
    output_format: str = Form("mp4"),
    output_codec: str = Form("libx264"),
    output_crf: int = Form(18),
    preserve_audio: bool = Form(True),
    restore_audio: bool = Form(False, description="Run audio restoration pipeline"),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Upload a video and enqueue a restoration job."""
    job_id = str(uuid.uuid4())

    # Save uploaded file to local storage root
    storage_root = Path(settings.storage_local_root)
    input_dir = storage_root / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename or "input.mp4").name
    input_path = input_dir / safe_filename
    content = await file.read()
    input_path.write_bytes(content)

    output_path = str(storage_root / "outputs" / job_id / f"output.{output_format}")

    # Resolve preset YAML path
    preset_path = _resolve_preset(pipeline_id)

    # Persist job to database
    repo = JobRepository(db)
    job_model = JobModel(
        id=job_id,
        status="queued",
        input_path=str(input_path),
        pipeline_id=pipeline_id,
        output_format=output_format,
        output_codec=output_codec,
        output_crf=output_crf,
        preserve_audio=preserve_audio,
        output_path=output_path,
    )
    await repo.create(job_model)

    # Dispatch Celery task
    from restorax.tasks.job_tasks import run_job
    task = run_job.apply_async(
        kwargs={
            "job_id": job_id,
            "pipeline_preset_path": preset_path,
            "input_path": str(input_path),
            "output_path": output_path,
            "restore_audio": restore_audio,
        }
    )
    await repo.update_status(job_id, status="queued", celery_task_id=task.id)

    job_model = await repo.get(job_id)
    return _to_response(job_model)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    repo = JobRepository(db)
    jobs = await repo.list_all(limit=limit, offset=offset)
    return JobListResponse(jobs=[_to_response(j) for j in jobs], total=len(jobs))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)) -> JobResponse:
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _to_response(job)


@router.get("/{job_id}/download")
async def download_output(job_id: str, db: AsyncSession = Depends(get_db)) -> FileResponse:
    """Download the output video for a completed job (local storage only)."""
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=409, detail="Job output not ready")
    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_path.name,
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    repo = JobRepository(db)
    try:
        await repo.delete(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/batch", response_model=JobListResponse, status_code=status.HTTP_201_CREATED)
async def create_batch_jobs(
    files: list[UploadFile] = File(..., description="Multiple video files"),
    pipeline_id: str = Form(...),
    output_format: str = Form("mp4"),
    output_codec: str = Form("libx264"),
    output_crf: int = Form(18),
    preserve_audio: bool = Form(True),
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    """
    Submit multiple video files as separate jobs in one request.

    Each file becomes an independent job dispatched to the next available GPU
    queue via the round-robin router (Phase 5.4). Returns the list of created jobs.
    """
    from restorax.tasks.gpu_router import next_gpu_queue

    repo = JobRepository(db)
    preset_path = _resolve_preset(pipeline_id)
    storage_root = Path(settings.storage_local_root)
    created: list[JobModel] = []

    for file in files:
        job_id = str(uuid.uuid4())
        input_dir = storage_root / "inputs" / job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "input.mp4").name
        input_path = input_dir / safe_name
        input_path.write_bytes(await file.read())
        output_path = str(storage_root / "outputs" / job_id / f"output.{output_format}")

        job_model = JobModel(
            id=job_id,
            status="queued",
            input_path=str(input_path),
            pipeline_id=pipeline_id,
            output_format=output_format,
            output_codec=output_codec,
            output_crf=output_crf,
            preserve_audio=preserve_audio,
            output_path=output_path,
        )
        await repo.create(job_model)

        from restorax.tasks.job_tasks import run_job
        gpu_queue = next_gpu_queue()
        task = run_job.apply_async(
            kwargs={
                "job_id": job_id,
                "pipeline_preset_path": preset_path,
                "input_path": str(input_path),
                "output_path": output_path,
            },
            queue=gpu_queue,
        )
        await repo.update_status(job_id, status="queued", celery_task_id=task.id)
        created.append(await repo.get(job_id))

    return JobListResponse(jobs=[_to_response(j) for j in created], total=len(created))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_preset(pipeline_id: str) -> str:
    """Return the absolute path to a YAML preset file."""
    candidates = [
        Path(f"configs/presets/{pipeline_id}.yaml"),
        Path(f"configs/presets/{pipeline_id}"),
        Path(pipeline_id),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise HTTPException(
        status_code=400,
        detail=f"Pipeline preset '{pipeline_id}' not found. "
               f"Available presets: {[p.stem for p in Path('configs/presets').glob('*.yaml')]}",
    )


def _to_response(job: JobModel) -> JobResponse:
    return JobResponse(
        id=str(job.id),
        status=job.status,
        progress=job.progress,
        pipeline_id=job.pipeline_id,
        input_path=job.input_path,
        output_path=job.output_path,
        error=job.error,
        metrics=job.metrics or {},
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        celery_task_id=job.celery_task_id,
    )
