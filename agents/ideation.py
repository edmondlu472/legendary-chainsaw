"""Ideation Agent — generates and scores content ideas using LLM + trend data."""

from __future__ import annotations

import uuid
from typing import Any

from agents.base import (
    BaseAgent,
    ContentBrief,
    DEFAULT_TREND_SOURCES,
    LLMConfig,
    PLATFORM_SPECS,
    Task,
    TrendSourceConfig,
)


class IdeationAgent(BaseAgent):
    """Researches trends, generates content ideas, and produces content briefs.

    Responsibilities:
    - Monitor trending topics across platforms (YouTube, TikTok, Instagram)
    - Generate content ideas aligned with audience interests via Claude LLM
    - Score ideas by monetization potential and trend relevance
    - Produce structured ContentBriefs for the Content Creator
    - Incorporate feedback from Analytics to refine future ideas

    Tools employed:
    - Claude API          — idea generation, angle brainstorming, brief writing
    - Google Trends       — search volume & rising queries (pytrends)
    - YouTube Data API v3 — trending videos, keyword search volumes
    - TikTok Creative Ctr — trending sounds, hashtags, topics
    - Reddit API (PRAW)   — hot posts in target niches
    """

    name = "ideation"
    role = "Ideation Specialist"
    description = (
        "Generates and evaluates content ideas based on trends, "
        "audience data, and monetization potential across YouTube, TikTok, and Instagram."
    )

    # ----- LLM Prompt Templates -----

    SYSTEM_PROMPT = (
        "You are an expert content strategist for short-form and long-form video. "
        "You specialize in YouTube, TikTok, and Instagram Reels. Your goal is to "
        "generate viral, monetizable content ideas backed by current trends.\n\n"
        "When generating ideas, always include:\n"
        "1. A compelling topic with a unique angle\n"
        "2. A hook (the first sentence that grabs attention)\n"
        "3. Target audience description\n"
        "4. Why this will perform well right now (trend justification)\n"
        "5. Suggested content style (educational, entertainment, tutorial, listicle, storytelling, news_recap, shorts_hook)\n"
        "6. Estimated ideal duration in seconds\n"
        "7. 5-10 SEO keywords\n"
        "8. Monetization angle (ad-friendly? sponsor-friendly? affiliate potential?)"
    )

    IDEA_GENERATION_PROMPT = (
        "Generate {count} unique video content ideas for the niche: '{niche}'.\n\n"
        "Current trend signals:\n{trend_signals}\n\n"
        "Target platforms: {platforms}\n\n"
        "For each idea, provide:\n"
        "- topic: A specific, compelling title\n"
        "- angle: What makes this take unique\n"
        "- hook: Opening line to grab attention in <3 seconds\n"
        "- target_audience: Who watches this\n"
        "- content_style: One of (educational, entertainment, tutorial, listicle, storytelling, news_recap, shorts_hook)\n"
        "- duration_seconds: Ideal length\n"
        "- keywords: 5-10 SEO keywords\n"
        "- monetization_notes: Why this is monetizable\n"
        "- trend_justification: Why this works right now\n\n"
        "Prioritize ideas that work across multiple platforms. "
        "For YouTube, think searchable + evergreen. "
        "For TikTok, think hook-driven + shareable. "
        "For Instagram Reels, think visually stunning + relatable."
    )

    SCORING_PROMPT = (
        "Score this content idea on a 0-1 scale for each factor.\n\n"
        "Idea: {topic}\nAngle: {angle}\nNiche: {niche}\n"
        "Target platforms: {platforms}\n\n"
        "Score these factors:\n"
        "1. trend_alignment (0-1): How well does this align with current trends?\n"
        "2. audience_demand (0-1): Is there proven demand for this content?\n"
        "3. competition_gap (0-1): Is the space underserved? (1 = big gap)\n"
        "4. monetization_potential (0-1): Ad-friendly? Sponsor-friendly?\n"
        "5. cross_platform_viability (0-1): Will this work on YouTube + TikTok + Instagram?\n"
        "6. production_feasibility (0-1): Can AI tools produce this at quality?\n\n"
        "Return a JSON object with these scores and a brief justification for each."
    )

    BRIEF_PROMPT = (
        "Create a detailed content brief for this approved idea.\n\n"
        "Topic: {topic}\nAngle: {angle}\nTarget audience: {audience}\n"
        "Platforms: {platforms}\nKeywords: {keywords}\n\n"
        "The brief should include:\n"
        "1. Refined topic and angle\n"
        "2. A powerful hook (first 3 seconds)\n"
        "3. Key talking points (3-5 bullet points)\n"
        "4. Suggested visual style and pacing\n"
        "5. Call-to-action strategy\n"
        "6. Platform-specific notes:\n"
        "   - YouTube: SEO title strategy, end screen plan\n"
        "   - TikTok: Trend sound suggestion, duet/stitch potential\n"
        "   - Instagram: Hashtag clusters, caption strategy\n"
        "7. Monetization strategy (ads, affiliate links, sponsor integration points)"
    )

    # ----- Platform-specific scoring weights -----
    # Each platform values different content attributes
    PLATFORM_WEIGHTS = {
        "youtube": {
            "trend_alignment": 0.15,
            "audience_demand": 0.25,   # searchability matters most
            "competition_gap": 0.20,
            "monetization_potential": 0.25,  # highest RPM
            "cross_platform_viability": 0.05,
            "production_feasibility": 0.10,
        },
        "tiktok": {
            "trend_alignment": 0.30,   # virality is king
            "audience_demand": 0.10,
            "competition_gap": 0.10,
            "monetization_potential": 0.10,
            "cross_platform_viability": 0.15,
            "production_feasibility": 0.25,  # must be fast to produce
        },
        "instagram": {
            "trend_alignment": 0.20,
            "audience_demand": 0.15,
            "competition_gap": 0.15,
            "monetization_potential": 0.15,
            "cross_platform_viability": 0.20,  # cross-post value
            "production_feasibility": 0.15,
        },
    }

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        trend_sources: list[TrendSourceConfig] | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config)
        self.idea_backlog: list[ContentBrief] = []
        self.trend_sources = trend_sources or DEFAULT_TREND_SOURCES
        self._feedback_history: list[dict[str, Any]] = []

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

    # ----- Trend Data Fetching (stubs wired to real API endpoints) -----

    def _fetch_trend_signals(self, niche: str) -> dict[str, Any]:
        """Fetch current trend data from all configured sources.

        In production each source calls its respective API.  Returns a merged
        dict of trend signals that gets injected into the LLM prompt.
        """
        signals: dict[str, Any] = {}

        for source in self.trend_sources:
            if not source.enabled:
                continue

            if source.name == "google_trends":
                # TODO: from pytrends.request import TrendReq
                # pytrends = TrendReq(); pytrends.build_payload([niche])
                # signals["google_trends"] = pytrends.interest_over_time()
                signals["google_trends"] = {
                    "rising_queries": [f"{niche} 2026", f"best {niche}", f"{niche} tips"],
                    "interest_score": 78,
                }

            elif source.name == "youtube_trending":
                # TODO: call YouTube Data API v3
                # GET /youtube/v3/search?part=snippet&q={niche}&type=video&order=viewCount
                signals["youtube_trending"] = {
                    "top_videos": [f"Top {niche} video #1", f"Top {niche} video #2"],
                    "avg_views": 150_000,
                    "common_keywords": [niche, f"{niche} tutorial", f"{niche} explained"],
                }

            elif source.name == "tiktok_creative_center":
                # TODO: call TikTok Creative Center API
                # GET /open_api/v1.3/creative/trending
                signals["tiktok_trends"] = {
                    "trending_hashtags": [f"#{niche}", f"#{niche}tok", "#fyp"],
                    "trending_sounds": ["original_sound_trending_1"],
                    "avg_engagement_rate": 0.08,
                }

            elif source.name == "reddit_popular":
                # TODO: call Reddit API via PRAW
                # reddit.subreddit(niche).hot(limit=10)
                signals["reddit_hot"] = {
                    "hot_topics": [f"Discussion about {niche}", f"New {niche} discovery"],
                    "sentiment": "positive",
                }

        return signals

    def _format_trend_signals(self, signals: dict[str, Any]) -> str:
        """Format trend data into a readable string for the LLM prompt."""
        lines = []
        for source, data in signals.items():
            lines.append(f"[{source}]")
            for key, value in data.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "No trend data available."

    # ----- Core Actions -----

    def _generate_ideas(self, task: Task) -> dict[str, Any]:
        """Generate content ideas using LLM + trend data."""
        niche = task.payload.get("niche", "general")
        count = task.payload.get("count", 5)
        platforms = task.payload.get(
            "platforms", ["youtube", "tiktok", "instagram"]
        )
        trend_data = task.payload.get("trend_data", {})

        # Fetch live trend signals if not provided
        if not trend_data:
            trend_data = self._fetch_trend_signals(niche)

        trend_text = self._format_trend_signals(trend_data)

        # Build the LLM prompt
        user_prompt = self.IDEA_GENERATION_PROMPT.format(
            count=count,
            niche=niche,
            trend_signals=trend_text,
            platforms=", ".join(platforms),
        )

        # Call LLM for idea generation
        llm_response = self._llm_generate(self.SYSTEM_PROMPT, user_prompt)

        # Build structured briefs from LLM output
        # In production, parse the JSON response from Claude
        ideas = []
        for i in range(count):
            idea = ContentBrief(
                idea_id=uuid.uuid4().hex[:12],
                topic=f"[AI-Generated] Idea {i + 1} for {niche}",
                angle=f"Unique angle based on {list(trend_data.keys())} trends",
                target_audience=niche,
                target_platforms=platforms,
                estimated_duration_seconds=self._optimal_duration(platforms),
                monetization_score=0.0,  # scored separately
                trend_score=0.0,          # scored separately
                keywords=self._extract_keywords_from_trends(trend_data, niche),
                content_style="educational",
                hook=f"You won't believe what's happening in {niche} right now...",
                notes=f"LLM prompt used: {user_prompt[:100]}...",
            )
            ideas.append(idea)

        self.idea_backlog.extend(ideas)

        return {
            "ideas": [
                {
                    "idea_id": idea.idea_id,
                    "topic": idea.topic,
                    "angle": idea.angle,
                    "hook": idea.hook,
                    "target_audience": idea.target_audience,
                    "target_platforms": idea.target_platforms,
                    "keywords": idea.keywords,
                    "content_style": idea.content_style,
                    "estimated_duration": idea.estimated_duration_seconds,
                }
                for idea in ideas
            ],
            "count": len(ideas),
            "trend_sources_used": list(trend_data.keys()),
            "llm_model": self.llm.model,
        }

    def _score_idea(self, task: Task) -> dict[str, Any]:
        """Score an idea using LLM analysis + platform-specific weights."""
        idea_id = task.payload.get("idea_id")
        topic = task.payload.get("topic", "")
        angle = task.payload.get("angle", "")
        niche = task.payload.get("niche", "general")
        platforms = task.payload.get("platforms", ["youtube", "tiktok", "instagram"])
        metrics = task.payload.get("metrics", {})

        # If raw scores provided, use them; otherwise ask LLM
        if metrics:
            raw_scores = metrics
        else:
            scoring_prompt = self.SCORING_PROMPT.format(
                topic=topic, angle=angle, niche=niche,
                platforms=", ".join(platforms),
            )
            _llm_response = self._llm_generate(self.SYSTEM_PROMPT, scoring_prompt)
            # In production, parse JSON from LLM response
            raw_scores = {
                "trend_alignment": 0.7,
                "audience_demand": 0.6,
                "competition_gap": 0.5,
                "monetization_potential": 0.7,
                "cross_platform_viability": 0.8,
                "production_feasibility": 0.9,
            }

        # Calculate per-platform weighted scores
        platform_scores = {}
        for platform in platforms:
            weights = self.PLATFORM_WEIGHTS.get(
                platform, self.PLATFORM_WEIGHTS["youtube"]
            )
            score = sum(
                raw_scores.get(factor, 0.0) * weight
                for factor, weight in weights.items()
            )
            platform_scores[platform] = round(score, 3)

        # Overall composite (average across target platforms)
        overall = round(
            sum(platform_scores.values()) / max(len(platform_scores), 1), 3
        )

        return {
            "idea_id": idea_id,
            "raw_scores": raw_scores,
            "platform_scores": platform_scores,
            "overall_score": overall,
            "recommendation": "proceed" if overall > 0.5 else "reconsider",
            "best_platform": max(platform_scores, key=platform_scores.get),
        }

    def _create_brief(self, task: Task) -> dict[str, Any]:
        """Create a full content brief from an approved idea using LLM."""
        topic = task.payload.get("topic", "")
        angle = task.payload.get("angle", "")
        audience = task.payload.get("target_audience", "")
        platforms = task.payload.get("platforms", ["youtube", "tiktok", "instagram"])
        keywords = task.payload.get("keywords", [])

        # Ask LLM for a detailed brief
        brief_prompt = self.BRIEF_PROMPT.format(
            topic=topic, angle=angle, audience=audience,
            platforms=", ".join(platforms),
            keywords=", ".join(keywords),
        )
        llm_response = self._llm_generate(self.SYSTEM_PROMPT, brief_prompt)

        brief = ContentBrief(
            idea_id=task.payload.get("idea_id", uuid.uuid4().hex[:12]),
            topic=topic,
            angle=angle,
            target_audience=audience,
            target_platforms=platforms,
            estimated_duration_seconds=task.payload.get(
                "duration", self._optimal_duration(platforms)
            ),
            keywords=keywords,
            content_style=task.payload.get("content_style", "educational"),
            hook=task.payload.get("hook", ""),
            notes=f"Brief generated by {self.llm.model}. {llm_response[:200]}",
        )

        return {"brief": brief.__dict__, "status": "brief_created"}

    def _incorporate_feedback(self, task: Task) -> dict[str, Any]:
        """Use analytics feedback to adjust ideation strategy via LLM."""
        performance_data = task.payload.get("performance_data", {})
        top_topics = performance_data.get("top_topics", [])
        underperformers = performance_data.get("underperformers", [])

        self._feedback_history.append(performance_data)

        # Build a feedback-aware prompt adjustment
        feedback_prompt = (
            f"Based on recent performance data:\n"
            f"- Top performing content IDs: {top_topics}\n"
            f"- Underperforming content IDs: {underperformers}\n"
            f"- Historical feedback rounds: {len(self._feedback_history)}\n\n"
            f"Adjust the content strategy. What topics/styles should we "
            f"prioritize and what should we avoid? Be specific."
        )
        llm_response = self._llm_generate(self.SYSTEM_PROMPT, feedback_prompt)

        return {
            "adjustments": {
                "prioritize_topics": top_topics,
                "deprioritize_topics": underperformers,
                "strategy_updated": True,
                "llm_strategy_notes": llm_response[:500],
                "feedback_rounds_total": len(self._feedback_history),
            }
        }

    # ----- Helpers -----

    def _optimal_duration(self, platforms: list[str]) -> int:
        """Pick an optimal video duration that works across target platforms."""
        if "tiktok" in platforms and "youtube" not in platforms:
            return 45  # pure short-form
        if "youtube" in platforms and "tiktok" not in platforms:
            return 480  # 8 min long-form
        # Cross-platform: aim for the sweet spot that works as both
        # a YouTube Short AND a TikTok (≤60s)
        return 55

    def _extract_keywords_from_trends(
        self, trend_data: dict[str, Any], niche: str
    ) -> list[str]:
        """Pull keywords from trend data to seed the brief."""
        keywords = {niche}
        for _source, data in trend_data.items():
            if isinstance(data, dict):
                for key in ("rising_queries", "common_keywords", "trending_hashtags", "hot_topics"):
                    for item in data.get(key, []):
                        keywords.add(str(item).strip("#").lower())
        return list(keywords)[:10]
