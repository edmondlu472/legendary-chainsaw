"""Base agent class that all agents inherit from."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A unit of work passed between agents."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_by: str = ""
    assigned_to: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None

    def complete(self, result: dict[str, Any]) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.result = {"error": error}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "payload": self.payload,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ContentBrief:
    """Structured output from ideation, input to content creation."""

    idea_id: str = ""
    topic: str = ""
    angle: str = ""
    target_audience: str = ""
    target_platforms: list[str] = field(default_factory=list)
    estimated_duration_seconds: int = 60
    monetization_score: float = 0.0
    trend_score: float = 0.0
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class VideoAsset:
    """Output from the content creator agent."""

    asset_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    brief_id: str = ""
    script: str = ""
    video_prompt: str = ""
    voiceover_text: str = ""
    thumbnail_prompt: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    duration_seconds: int = 0
    file_path: str | None = None
    status: str = "draft"


class BaseAgent:
    """Base class for all agents in the system."""

    name: str = "base"
    role: str = "Base Agent"
    description: str = ""

    def __init__(self) -> None:
        self.task_history: list[Task] = []

    def receive_task(self, task: Task) -> Task:
        """Process an incoming task and return the result."""
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_to = self.name
        try:
            result = self.execute(task)
            task.complete(result)
        except Exception as e:
            task.fail(str(e))
        self.task_history.append(task)
        return task

    def execute(self, task: Task) -> dict[str, Any]:
        """Override in subclasses to define agent behavior."""
        raise NotImplementedError

    def get_capabilities(self) -> dict[str, str]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
        }
