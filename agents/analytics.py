"""Analytics Agent — tracks performance and generates insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base import BaseAgent, Task


@dataclass
class VideoMetrics:
    """Performance metrics for a single video."""

    asset_id: str = ""
    platform: str = ""
    views: int = 0
    watch_time_hours: float = 0.0
    avg_view_duration_seconds: float = 0.0
    click_through_rate: float = 0.0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    subscribers_gained: int = 0
    estimated_revenue_usd: float = 0.0
    retention_curve: list[float] = field(default_factory=list)


class AnalyticsAgent(BaseAgent):
    """Tracks video performance and feeds insights back to the team.

    Responsibilities:
    - Collect and aggregate performance metrics across platforms
    - Identify top-performing and underperforming content
    - Calculate revenue metrics and projections
    - Detect audience behavior patterns
    - Generate actionable reports for Ideation and Marketing
    - Track channel-level growth trends
    """

    name = "analytics"
    role = "Performance Analyst"
    description = (
        "Tracks video and channel performance, identifies patterns, "
        "and provides data-driven insights to optimize content strategy."
    )

    def __init__(self) -> None:
        super().__init__()
        self.metrics_store: list[VideoMetrics] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "analyze_performance")
        if action == "analyze_performance":
            return self._analyze_performance(task)
        elif action == "revenue_report":
            return self._revenue_report(task)
        elif action == "top_performers":
            return self._top_performers(task)
        elif action == "audience_insights":
            return self._audience_insights(task)
        elif action == "feedback_for_ideation":
            return self._feedback_for_ideation(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _analyze_performance(self, task: Task) -> dict[str, Any]:
        """Analyze performance of a specific video or set of videos."""
        metrics_data = task.payload.get("metrics", {})

        metrics = VideoMetrics(
            asset_id=metrics_data.get("asset_id", ""),
            platform=metrics_data.get("platform", ""),
            views=metrics_data.get("views", 0),
            watch_time_hours=metrics_data.get("watch_time_hours", 0.0),
            avg_view_duration_seconds=metrics_data.get("avg_view_duration_seconds", 0.0),
            click_through_rate=metrics_data.get("click_through_rate", 0.0),
            likes=metrics_data.get("likes", 0),
            comments=metrics_data.get("comments", 0),
            shares=metrics_data.get("shares", 0),
            subscribers_gained=metrics_data.get("subscribers_gained", 0),
            estimated_revenue_usd=metrics_data.get("estimated_revenue_usd", 0.0),
        )

        self.metrics_store.append(metrics)

        engagement_rate = (
            (metrics.likes + metrics.comments + metrics.shares) / max(metrics.views, 1)
        )

        return {
            "asset_id": metrics.asset_id,
            "engagement_rate": round(engagement_rate, 4),
            "revenue_per_view": round(
                metrics.estimated_revenue_usd / max(metrics.views, 1), 6
            ),
            "performance_tier": self._classify_performance(metrics),
        }

    def _revenue_report(self, task: Task) -> dict[str, Any]:
        """Generate a revenue summary across all tracked content."""
        period = task.payload.get("period", "all_time")

        total_revenue = sum(m.estimated_revenue_usd for m in self.metrics_store)
        total_views = sum(m.views for m in self.metrics_store)
        video_count = len(self.metrics_store)

        return {
            "period": period,
            "total_revenue_usd": round(total_revenue, 2),
            "total_views": total_views,
            "videos_tracked": video_count,
            "avg_revenue_per_video": round(
                total_revenue / max(video_count, 1), 2
            ),
            "rpm": round(
                (total_revenue / max(total_views, 1)) * 1000, 2
            ),
        }

    def _top_performers(self, task: Task) -> dict[str, Any]:
        """Identify the top-performing videos by a given metric."""
        metric = task.payload.get("metric", "views")
        limit = task.payload.get("limit", 5)

        sorted_metrics = sorted(
            self.metrics_store,
            key=lambda m: getattr(m, metric, 0),
            reverse=True,
        )

        return {
            "ranked_by": metric,
            "top": [
                {
                    "asset_id": m.asset_id,
                    "platform": m.platform,
                    metric: getattr(m, metric, 0),
                }
                for m in sorted_metrics[:limit]
            ],
        }

    def _audience_insights(self, task: Task) -> dict[str, Any]:
        """Derive audience behavior insights from aggregate data."""
        if not self.metrics_store:
            return {"insights": [], "message": "No data available yet"}

        avg_view_duration = sum(
            m.avg_view_duration_seconds for m in self.metrics_store
        ) / len(self.metrics_store)

        avg_ctr = sum(
            m.click_through_rate for m in self.metrics_store
        ) / len(self.metrics_store)

        return {
            "avg_view_duration_seconds": round(avg_view_duration, 1),
            "avg_ctr": round(avg_ctr, 4),
            "total_subscribers_gained": sum(
                m.subscribers_gained for m in self.metrics_store
            ),
            "insights": [
                f"Average viewer watches {avg_view_duration:.0f}s",
                f"CTR is {'above' if avg_ctr > 0.05 else 'below'} 5% benchmark",
            ],
        }

    def _feedback_for_ideation(self, task: Task) -> dict[str, Any]:
        """Generate structured feedback for the Ideation agent."""
        top = sorted(self.metrics_store, key=lambda m: m.views, reverse=True)[:3]
        bottom = sorted(self.metrics_store, key=lambda m: m.views)[:3]

        return {
            "performance_data": {
                "top_topics": [m.asset_id for m in top],
                "underperformers": [m.asset_id for m in bottom],
                "recommendation": "Focus on formats similar to top performers",
            }
        }

    def _classify_performance(self, metrics: VideoMetrics) -> str:
        """Classify video performance into tiers."""
        if metrics.views >= 100000:
            return "viral"
        elif metrics.views >= 10000:
            return "strong"
        elif metrics.views >= 1000:
            return "moderate"
        else:
            return "growing"
