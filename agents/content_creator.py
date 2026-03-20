"""Content Creator Agent — produces video assets from content briefs."""

from __future__ import annotations

import uuid
from typing import Any

from agents.base import BaseAgent, Task, VideoAsset


class ContentCreatorAgent(BaseAgent):
    """Takes content briefs and produces all assets needed for a video.

    Responsibilities:
    - Generate video scripts from content briefs
    - Create AI video generation prompts
    - Write voiceover / narration text
    - Generate thumbnail descriptions and prompts
    - Produce title, description, and tag suggestions
    - Package everything into a VideoAsset for downstream agents
    """

    name = "content_creator"
    role = "Content Producer"
    description = (
        "Transforms content briefs into complete video asset packages "
        "including scripts, prompts, voiceover, and metadata."
    )

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[VideoAsset] = []
        self.supported_styles = [
            "educational",
            "entertainment",
            "tutorial",
            "listicle",
            "storytelling",
            "news_recap",
        ]

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "create_asset")
        if action == "create_asset":
            return self._create_asset(task)
        elif action == "generate_script":
            return self._generate_script(task)
        elif action == "generate_thumbnail":
            return self._generate_thumbnail(task)
        elif action == "revise_asset":
            return self._revise_asset(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _create_asset(self, task: Task) -> dict[str, Any]:
        """Create a full video asset package from a content brief."""
        brief = task.payload.get("brief", {})

        topic = brief.get("topic", "Untitled")
        angle = brief.get("angle", "")
        keywords = brief.get("keywords", [])
        duration = brief.get("estimated_duration_seconds", 60)

        asset = VideoAsset(
            asset_id=uuid.uuid4().hex[:12],
            brief_id=brief.get("idea_id", ""),
            script=f"[SCRIPT for: {topic}]\n\nHook: Grab attention in first 3 seconds.\nBody: Cover the main points.\nCTA: Like, subscribe, comment.",
            video_prompt=self._build_video_prompt(brief),
            voiceover_text=f"[VOICEOVER for: {topic}]",
            thumbnail_prompt=f"Eye-catching thumbnail for: {topic}",
            title=topic,
            description=angle if angle else f"Everything you need to know about {topic}.",
            tags=keywords if keywords else [topic.lower().replace(" ", "-")],
            duration_seconds=duration,
            status="draft",
        )

        self.assets.append(asset)
        return {"asset": asset.__dict__, "status": "asset_created"}

    def _generate_script(self, task: Task) -> dict[str, Any]:
        """Generate just the script portion of a video."""
        topic = task.payload.get("topic", "")
        style = task.payload.get("style", "educational")
        duration = task.payload.get("duration_seconds", 60)
        target_audience = task.payload.get("target_audience", "general")

        # Words per second estimate for pacing
        word_count = duration * 2.5  # ~150 words per minute

        return {
            "script": f"[GENERATED SCRIPT]\nTopic: {topic}\nStyle: {style}\nTarget: {target_audience}\nWord count target: {int(word_count)}",
            "style": style,
            "estimated_duration": duration,
        }

    def _generate_thumbnail(self, task: Task) -> dict[str, Any]:
        """Generate a thumbnail prompt for AI image generation."""
        title = task.payload.get("title", "")
        style = task.payload.get("style", "bold")

        prompt = (
            f"YouTube thumbnail, {style} style, high contrast, "
            f"engaging visual for: {title}"
        )

        return {"thumbnail_prompt": prompt, "recommended_size": "1280x720"}

    def _revise_asset(self, task: Task) -> dict[str, Any]:
        """Revise an existing asset based on QA or analytics feedback."""
        asset_id = task.payload.get("asset_id", "")
        revisions = task.payload.get("revisions", {})

        return {
            "asset_id": asset_id,
            "revisions_applied": list(revisions.keys()),
            "status": "revised",
        }

    def _build_video_prompt(self, brief: dict[str, Any]) -> str:
        """Build an AI video generation prompt from a brief."""
        topic = brief.get("topic", "")
        audience = brief.get("target_audience", "general")
        duration = brief.get("estimated_duration_seconds", 60)

        return (
            f"Create a {duration}-second video about '{topic}' "
            f"targeted at {audience} audience. "
            f"Style: engaging, professional, high-quality visuals."
        )
