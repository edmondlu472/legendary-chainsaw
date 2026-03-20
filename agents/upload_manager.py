"""Upload Manager Agent — handles the upload pipeline and scheduling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agents.base import BaseAgent, Task


@dataclass
class UploadJob:
    """Tracks an individual upload to a platform."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    asset_id: str = ""
    platform: str = ""
    status: str = "queued"  # queued | uploading | published | failed | scheduled
    scheduled_time: str | None = None
    publish_url: str | None = None
    retries: int = 0
    max_retries: int = 3
    error: str | None = None


class UploadManagerAgent(BaseAgent):
    """Manages the end-to-end upload pipeline for all platforms.

    Responsibilities:
    - Prepare files for platform-specific requirements (format, resolution)
    - Authenticate with platform APIs
    - Execute uploads with retry logic
    - Schedule uploads for optimal posting times
    - Track upload status and report results
    - Handle multi-platform simultaneous publishing
    """

    name = "upload_manager"
    role = "Upload & Publishing Manager"
    description = (
        "Manages file preparation, scheduling, and multi-platform "
        "uploading with retry logic and status tracking."
    )

    PLATFORM_SPECS = {
        "youtube": {
            "formats": ["mp4", "mov"],
            "max_size_mb": 256000,
            "max_duration_seconds": 43200,
            "thumbnail_required": True,
        },
        "tiktok": {
            "formats": ["mp4"],
            "max_size_mb": 287,
            "max_duration_seconds": 600,
            "thumbnail_required": False,
        },
        "instagram": {
            "formats": ["mp4", "mov"],
            "max_size_mb": 650,
            "max_duration_seconds": 5400,
            "thumbnail_required": False,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self.upload_queue: list[UploadJob] = []
        self.completed_uploads: list[UploadJob] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "queue_upload")
        if action == "queue_upload":
            return self._queue_upload(task)
        elif action == "validate_file":
            return self._validate_file(task)
        elif action == "execute_upload":
            return self._execute_upload(task)
        elif action == "get_status":
            return self._get_status(task)
        elif action == "schedule_upload":
            return self._schedule_upload(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _queue_upload(self, task: Task) -> dict[str, Any]:
        """Add an upload job to the queue."""
        asset_id = task.payload.get("asset_id", "")
        platforms = task.payload.get("platforms", ["youtube"])

        jobs = []
        for platform in platforms:
            job = UploadJob(
                asset_id=asset_id,
                platform=platform,
                status="queued",
            )
            self.upload_queue.append(job)
            jobs.append({"job_id": job.job_id, "platform": platform, "status": "queued"})

        return {"jobs": jobs, "queue_length": len(self.upload_queue)}

    def _validate_file(self, task: Task) -> dict[str, Any]:
        """Validate a file against platform requirements."""
        platform = task.payload.get("platform", "youtube")
        file_format = task.payload.get("format", "mp4")
        file_size_mb = task.payload.get("size_mb", 0)
        duration_seconds = task.payload.get("duration_seconds", 0)

        specs = self.PLATFORM_SPECS.get(platform, self.PLATFORM_SPECS["youtube"])

        issues = []
        if file_format not in specs["formats"]:
            issues.append(f"Format '{file_format}' not supported. Use: {specs['formats']}")
        if file_size_mb > specs["max_size_mb"]:
            issues.append(f"File too large ({file_size_mb}MB > {specs['max_size_mb']}MB)")
        if duration_seconds > specs["max_duration_seconds"]:
            issues.append(f"Video too long ({duration_seconds}s > {specs['max_duration_seconds']}s)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "platform": platform,
        }

    def _execute_upload(self, task: Task) -> dict[str, Any]:
        """Execute an upload (simulated — real implementation would call platform APIs)."""
        job_id = task.payload.get("job_id", "")

        job = next((j for j in self.upload_queue if j.job_id == job_id), None)
        if not job:
            return {"error": f"Job {job_id} not found", "status": "failed"}

        job.status = "uploading"

        # In production, this would call the actual platform API
        job.status = "published"
        job.publish_url = f"https://{job.platform}.com/watch/{job.asset_id}"

        self.upload_queue.remove(job)
        self.completed_uploads.append(job)

        return {
            "job_id": job.job_id,
            "status": "published",
            "publish_url": job.publish_url,
        }

    def _get_status(self, task: Task) -> dict[str, Any]:
        """Get status of all upload jobs or a specific one."""
        job_id = task.payload.get("job_id")

        if job_id:
            job = next(
                (j for j in self.upload_queue + self.completed_uploads if j.job_id == job_id),
                None,
            )
            if not job:
                return {"error": f"Job {job_id} not found"}
            return {"job_id": job.job_id, "status": job.status, "platform": job.platform}

        return {
            "queued": len(self.upload_queue),
            "completed": len(self.completed_uploads),
            "jobs": [
                {"job_id": j.job_id, "platform": j.platform, "status": j.status}
                for j in self.upload_queue + self.completed_uploads
            ],
        }

    def _schedule_upload(self, task: Task) -> dict[str, Any]:
        """Schedule an upload for a specific time."""
        job_id = task.payload.get("job_id", "")
        scheduled_time = task.payload.get("scheduled_time", "")

        job = next((j for j in self.upload_queue if j.job_id == job_id), None)
        if not job:
            return {"error": f"Job {job_id} not found"}

        job.status = "scheduled"
        job.scheduled_time = scheduled_time

        return {
            "job_id": job.job_id,
            "status": "scheduled",
            "scheduled_time": scheduled_time,
        }
