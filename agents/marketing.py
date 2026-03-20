"""Marketing Agent — handles distribution, SEO, and audience growth."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, Task


class MarketingAgent(BaseAgent):
    """Optimizes content for discovery and manages distribution strategy.

    Responsibilities:
    - Optimize titles, descriptions, and tags for SEO
    - Plan cross-platform distribution schedules
    - Define audience targeting and growth strategies
    - A/B test metadata variations
    - Track and improve click-through rates (CTR)
    - Manage hashtag and keyword strategies per platform
    """

    name = "marketing"
    role = "Marketing & Distribution Strategist"
    description = (
        "Optimizes content metadata for discoverability and plans "
        "cross-platform distribution to grow audience and revenue."
    )

    PLATFORM_LIMITS = {
        "youtube": {"title_max": 100, "desc_max": 5000, "tags_max": 500},
        "tiktok": {"title_max": 150, "desc_max": 2200, "tags_max": 0},
        "instagram": {"title_max": 0, "desc_max": 2200, "tags_max": 0},
        "twitter": {"title_max": 0, "desc_max": 280, "tags_max": 0},
    }

    def __init__(self) -> None:
        super().__init__()
        self.distribution_queue: list[dict[str, Any]] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "optimize_metadata")
        if action == "optimize_metadata":
            return self._optimize_metadata(task)
        elif action == "plan_distribution":
            return self._plan_distribution(task)
        elif action == "ab_test":
            return self._create_ab_test(task)
        elif action == "growth_strategy":
            return self._growth_strategy(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _optimize_metadata(self, task: Task) -> dict[str, Any]:
        """Optimize title, description, and tags for a target platform."""
        platform = task.payload.get("platform", "youtube")
        title = task.payload.get("title", "")
        description = task.payload.get("description", "")
        tags = task.payload.get("tags", [])
        keywords = task.payload.get("keywords", [])

        limits = self.PLATFORM_LIMITS.get(platform, self.PLATFORM_LIMITS["youtube"])

        optimized_title = title[: limits["title_max"]] if limits["title_max"] else title
        optimized_desc = (
            description[: limits["desc_max"]] if limits["desc_max"] else description
        )

        # Prepend high-value keywords to tags
        optimized_tags = list(dict.fromkeys(keywords + tags))

        return {
            "platform": platform,
            "optimized_title": optimized_title,
            "optimized_description": optimized_desc,
            "optimized_tags": optimized_tags,
            "seo_score": self._estimate_seo_score(optimized_title, optimized_tags),
        }

    def _plan_distribution(self, task: Task) -> dict[str, Any]:
        """Plan when and where to publish content across platforms."""
        asset_id = task.payload.get("asset_id", "")
        platforms = task.payload.get("platforms", ["youtube"])
        preferred_times = task.payload.get("preferred_times", {})

        # Default optimal posting windows (UTC hours)
        optimal_windows = {
            "youtube": [14, 15, 16, 17],
            "tiktok": [11, 12, 19, 20],
            "instagram": [11, 13, 19],
            "twitter": [9, 12, 17],
        }

        schedule = []
        for platform in platforms:
            window = optimal_windows.get(platform, [12])
            entry = {
                "platform": platform,
                "recommended_hour_utc": preferred_times.get(platform, window[0]),
                "asset_id": asset_id,
                "status": "scheduled",
            }
            schedule.append(entry)
            self.distribution_queue.append(entry)

        return {"schedule": schedule, "total_platforms": len(platforms)}

    def _create_ab_test(self, task: Task) -> dict[str, Any]:
        """Create A/B test variants for titles or thumbnails."""
        original = task.payload.get("original", "")
        element = task.payload.get("element", "title")

        variants = [
            {"variant": "A", "value": original, "is_control": True},
            {"variant": "B", "value": f"[ALT] {original}", "is_control": False},
        ]

        return {
            "element_tested": element,
            "variants": variants,
            "test_duration_hours": 48,
        }

    def _growth_strategy(self, task: Task) -> dict[str, Any]:
        """Generate audience growth recommendations."""
        current_metrics = task.payload.get("current_metrics", {})
        niche = task.payload.get("niche", "general")

        return {
            "niche": niche,
            "strategies": [
                "consistent_posting_schedule",
                "cross_platform_promotion",
                "community_engagement",
                "collaboration_outreach",
                "seo_optimization",
                "trend_riding",
            ],
            "priority": "consistent_posting_schedule",
        }

    def _estimate_seo_score(self, title: str, tags: list[str]) -> float:
        """Rough heuristic SEO score (0-1)."""
        score = 0.0
        if len(title) > 20:
            score += 0.3
        if len(title) < 70:
            score += 0.2
        if len(tags) >= 5:
            score += 0.3
        if len(tags) >= 10:
            score += 0.2
        return min(score, 1.0)
