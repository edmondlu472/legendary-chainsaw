"""Ideation Agent — generates and scores content ideas."""

from __future__ import annotations

import uuid
from typing import Any

from agents.base import BaseAgent, ContentBrief, Task


class IdeationAgent(BaseAgent):
    """Researches trends, generates content ideas, and produces content briefs.

    Responsibilities:
    - Monitor trending topics across platforms (YouTube, TikTok, Instagram)
    - Generate content ideas aligned with audience interests
    - Score ideas by monetization potential and trend relevance
    - Produce structured ContentBriefs for the Content Creator
    - Incorporate feedback from Analytics to refine future ideas
    """

    name = "ideation"
    role = "Ideation Specialist"
    description = (
        "Generates and evaluates content ideas based on trends, "
        "audience data, and monetization potential."
    )

    def __init__(self) -> None:
        super().__init__()
        self.idea_backlog: list[ContentBrief] = []
        self.trend_sources: list[str] = [
            "youtube_trending",
            "tiktok_discover",
            "google_trends",
            "reddit_popular",
            "twitter_trending",
        ]

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "generate_ideas")
        if action == "generate_ideas":
            return self._generate_ideas(task)
        elif action == "score_idea":
            return self._score_idea(task)
        elif action == "create_brief":
            return self._create_brief(task)
        elif action == "incorporate_feedback":
            return self._incorporate_feedback(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _generate_ideas(self, task: Task) -> dict[str, Any]:
        """Generate content ideas based on niche and trend data."""
        niche = task.payload.get("niche", "general")
        count = task.payload.get("count", 5)
        trend_data = task.payload.get("trend_data", {})

        ideas = []
        for i in range(count):
            idea = ContentBrief(
                idea_id=uuid.uuid4().hex[:12],
                topic=f"[PLACEHOLDER] Idea {i+1} for {niche}",
                angle="",
                target_audience=niche,
                target_platforms=["youtube", "tiktok"],
                estimated_duration_seconds=60,
                monetization_score=0.0,
                trend_score=0.0,
                keywords=[],
            )
            ideas.append(idea)

        self.idea_backlog.extend(ideas)
        return {
            "ideas": [
                {
                    "idea_id": idea.idea_id,
                    "topic": idea.topic,
                    "target_audience": idea.target_audience,
                }
                for idea in ideas
            ],
            "count": len(ideas),
        }

    def _score_idea(self, task: Task) -> dict[str, Any]:
        """Score an idea on monetization potential and trend alignment."""
        idea_id = task.payload.get("idea_id")
        metrics = task.payload.get("metrics", {})

        trend_score = metrics.get("trend_score", 0.0)
        competition_score = metrics.get("competition_score", 0.0)
        audience_fit = metrics.get("audience_fit", 0.0)

        # Weighted composite score
        monetization_score = (
            trend_score * 0.4 + audience_fit * 0.4 + (1 - competition_score) * 0.2
        )

        return {
            "idea_id": idea_id,
            "trend_score": trend_score,
            "monetization_score": round(monetization_score, 3),
            "recommendation": "proceed" if monetization_score > 0.5 else "reconsider",
        }

    def _create_brief(self, task: Task) -> dict[str, Any]:
        """Create a full content brief from an approved idea."""
        brief = ContentBrief(
            idea_id=task.payload.get("idea_id", uuid.uuid4().hex[:12]),
            topic=task.payload.get("topic", ""),
            angle=task.payload.get("angle", ""),
            target_audience=task.payload.get("target_audience", ""),
            target_platforms=task.payload.get("platforms", ["youtube"]),
            estimated_duration_seconds=task.payload.get("duration", 60),
            keywords=task.payload.get("keywords", []),
            notes=task.payload.get("notes", ""),
        )
        return {"brief": brief.__dict__, "status": "brief_created"}

    def _incorporate_feedback(self, task: Task) -> dict[str, Any]:
        """Use analytics feedback to adjust ideation strategy."""
        performance_data = task.payload.get("performance_data", {})
        top_topics = performance_data.get("top_topics", [])
        underperformers = performance_data.get("underperformers", [])

        return {
            "adjustments": {
                "prioritize_topics": top_topics,
                "deprioritize_topics": underperformers,
                "strategy_updated": True,
            }
        }
