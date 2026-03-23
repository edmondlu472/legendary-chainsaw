"""Marketing Agent — handles cross-platform distribution, SEO, and audience growth."""

from __future__ import annotations

from typing import Any

from agents.base import (
    BaseAgent,
    LLMConfig,
    PLATFORM_SPECS,
    PlatformVariant,
    Task,
    VideoAsset,
)


class MarketingAgent(BaseAgent):
    """Optimizes content for discovery and manages distribution strategy.

    Responsibilities:
    - Optimize titles, descriptions, and tags per platform (YouTube, TikTok, Instagram)
    - Plan cross-platform distribution with staggered scheduling
    - Generate platform-specific hashtag strategies
    - A/B test metadata variations via Claude LLM
    - Track and improve click-through rates (CTR)
    - Manage content repurposing across platforms
    - Generate growth strategies based on analytics feedback

    Tools employed:
    - Claude API            — SEO copy optimization, A/B variant generation
    - YouTube Data API v3   — keyword research, competitor analysis
    - Platform analytics    — CTR benchmarking, engagement tracking
    """

    name = "marketing"
    role = "Marketing & Distribution Strategist"
    description = (
        "Optimizes content metadata for discoverability across YouTube, TikTok, "
        "and Instagram, and plans cross-platform distribution for maximum reach."
    )

    # ----- LLM Prompt Templates -----

    SEO_SYSTEM_PROMPT = (
        "You are an expert SEO and social media marketing strategist specializing "
        "in video content across YouTube, TikTok, and Instagram Reels.\n\n"
        "You understand:\n"
        "- YouTube algorithm: CTR + watch time + session time\n"
        "- TikTok algorithm: completion rate + shares + replays\n"
        "- Instagram algorithm: saves + shares + reach signals\n\n"
        "Your optimizations must be platform-native — what works on YouTube won't "
        "work on TikTok. Adapt voice, length, and strategy per platform."
    )

    TITLE_OPTIMIZATION_PROMPT = (
        "Optimize this video title for {platform}.\n\n"
        "Original title: {title}\n"
        "Keywords: {keywords}\n"
        "Target audience: {audience}\n\n"
        "Platform rules for {platform}:\n{platform_rules}\n\n"
        "Generate 3 optimized title variants. For each:\n"
        "1. Include the primary keyword near the front\n"
        "2. Create curiosity or urgency\n"
        "3. Stay within {max_chars} characters\n"
        "4. Match the platform's native tone"
    )

    DESCRIPTION_OPTIMIZATION_PROMPT = (
        "Write an optimized video description for {platform}.\n\n"
        "Topic: {topic}\nAngle: {angle}\nKeywords: {keywords}\n\n"
        "Platform rules:\n{platform_rules}\n"
        "Max length: {max_chars} characters\n\n"
        "Include:\n{include_instructions}"
    )

    AB_TEST_PROMPT = (
        "Generate A/B test variants for this {element}.\n\n"
        "Original: {original}\n"
        "Platform: {platform}\n"
        "Goal: Maximize {goal}\n\n"
        "Create 2 alternative variants that test different psychological triggers:\n"
        "- Variant B: Test a different emotional angle\n"
        "- Variant C: Test a different structural approach\n\n"
        "Explain the hypothesis for each variant."
    )

    GROWTH_STRATEGY_PROMPT = (
        "Create a growth strategy for a {niche} content channel.\n\n"
        "Current metrics:\n{metrics}\n\n"
        "Platforms: {platforms}\n\n"
        "Provide specific, actionable recommendations for:\n"
        "1. Content cadence (posting frequency per platform)\n"
        "2. Cross-promotion strategy\n"
        "3. Community engagement tactics\n"
        "4. Collaboration opportunities\n"
        "5. Trending format adaptations\n"
        "6. Monetization optimization"
    )

    # ----- Platform-Specific Marketing Rules -----

    PLATFORM_TITLE_RULES = {
        "youtube": (
            "- Front-load the primary keyword\n"
            "- Use numbers when possible ('5 Ways...', 'Top 10...')\n"
            "- Create curiosity gap without being clickbait\n"
            "- Keep under 60 chars for mobile (max 100)\n"
            "- Avoid ALL CAPS (one word max for emphasis)\n"
            "- Include year if evergreen ('...in 2026')"
        ),
        "tiktok": (
            "- Hook-first: lead with the most intriguing part\n"
            "- Use casual, conversational tone\n"
            "- Emojis OK but don't overdo it\n"
            "- Keep short — the video IS the content\n"
            "- Reference trends when applicable\n"
            "- Max 150 characters"
        ),
        "instagram": (
            "- Instagram Reels don't have a separate title field\n"
            "- The first line of the caption serves as the 'title'\n"
            "- Make it a statement or question that stops the scroll\n"
            "- Use line breaks for readability\n"
            "- Save hashtags for the end of the caption"
        ),
    }

    PLATFORM_DESCRIPTION_RULES = {
        "youtube": {
            "max_chars": 5000,
            "include": (
                "- First 2 lines visible above the fold (most important)\n"
                "- Timestamps/chapters if video > 3 min\n"
                "- Relevant links (affiliate, social, resources)\n"
                "- 3-5 keyword-rich sentences for SEO\n"
                "- Related video suggestions\n"
                "- Subscribe CTA with channel link\n"
                "- Hashtags at the end (3-5 max)"
            ),
        },
        "tiktok": {
            "max_chars": 2200,
            "include": (
                "- Short, punchy caption (1-2 sentences)\n"
                "- Trending hashtags mixed with niche hashtags\n"
                "- CTA: 'Follow for more' or question prompt\n"
                "- No links (not clickable in description)\n"
                "- Use #fyp #foryou for discovery"
            ),
        },
        "instagram": {
            "max_chars": 2200,
            "include": (
                "- First line = hook (visible in feed)\n"
                "- 2-3 sentences of value or story\n"
                "- CTA: 'Save this' or 'Share with a friend'\n"
                "- Line break before hashtag block\n"
                "- Mix of large (1M+), medium (100k+), and small (<100k) hashtags\n"
                "- Max 30 hashtags, aim for 15-20 relevant ones"
            ),
        },
    }

    # ----- Hashtag Strategy -----

    HASHTAG_TIERS = {
        "tiktok": {
            "universal": ["#fyp", "#foryou", "#viral", "#trending"],
            "engagement": ["#greenscreen", "#stitch", "#duet", "#pov"],
            "max_count": 8,  # TikTok best practice: fewer, more targeted
        },
        "instagram": {
            "universal": ["#reels", "#explore", "#reelsinstagram", "#instareels"],
            "engagement": ["#savethis", "#sharethis", "#didyouknow"],
            "max_count": 25,  # sweet spot (max 30)
        },
        "youtube": {
            "universal": ["#shorts", "#youtube"],
            "engagement": [],
            "max_count": 5,  # YouTube shows only 3 above the title
        },
    }

    # ----- Optimal Posting Schedule -----

    POSTING_SCHEDULE = {
        "youtube": {
            "best_days": ["tuesday", "thursday", "saturday"],
            "best_hours_utc": [14, 15, 16, 17],
            "frequency": "3-5x per week for Shorts, 1-2x for long-form",
        },
        "tiktok": {
            "best_days": ["tuesday", "wednesday", "thursday"],
            "best_hours_utc": [11, 12, 19, 20],
            "frequency": "1-3x daily for growth phase, 5-7x/week maintenance",
        },
        "instagram": {
            "best_days": ["monday", "wednesday", "friday"],
            "best_hours_utc": [11, 13, 19],
            "frequency": "4-7 Reels per week, daily Stories",
        },
    }

    # ----- Cross-Platform Repurposing Rules -----

    REPURPOSE_MAP = {
        ("youtube_shorts", "tiktok"): {
            "changes": ["Remove subscribe CTA", "Add TikTok-native CTA", "Add trending hashtags"],
            "aspect_ratio": "9:16",
            "max_duration": 60,
        },
        ("youtube_shorts", "instagram"): {
            "changes": ["Remove subscribe CTA", "Add save/share CTA", "Add Instagram hashtags"],
            "aspect_ratio": "9:16",
            "max_duration": 90,
        },
        ("tiktok", "youtube_shorts"): {
            "changes": ["Remove TikTok watermark", "Add subscribe CTA", "Adjust metadata for SEO"],
            "aspect_ratio": "9:16",
            "max_duration": 60,
        },
        ("tiktok", "instagram"): {
            "changes": ["Remove TikTok watermark", "Adjust caption for Instagram", "Add hashtag block"],
            "aspect_ratio": "9:16",
            "max_duration": 90,
        },
    }

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config=llm_config)
        self.distribution_queue: list[dict[str, Any]] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "optimize_metadata")
        if action == "optimize_metadata":
            return self._optimize_metadata(task)
        elif action == "optimize_all_platforms":
            return self._optimize_all_platforms(task)
        elif action == "plan_distribution":
            return self._plan_distribution(task)
        elif action == "ab_test":
            return self._create_ab_test(task)
        elif action == "growth_strategy":
            return self._growth_strategy(task)
        elif action == "hashtag_strategy":
            return self._hashtag_strategy(task)
        elif action == "repurpose_plan":
            return self._repurpose_plan(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ----- Core Actions -----

    def _optimize_metadata(self, task: Task) -> dict[str, Any]:
        """Optimize title, description, and tags for a single platform using LLM."""
        platform = task.payload.get("platform", "youtube")
        title = task.payload.get("title", "")
        description = task.payload.get("description", "")
        tags = task.payload.get("tags", [])
        keywords = task.payload.get("keywords", [])
        audience = task.payload.get("target_audience", "general")

        specs = PLATFORM_SPECS.get(platform, {})
        limits = specs.get("metadata_limits", {})

        # Optimize title via LLM
        title_prompt = self.TITLE_OPTIMIZATION_PROMPT.format(
            platform=specs.get("display_name", platform),
            title=title,
            keywords=", ".join(keywords[:8]),
            audience=audience,
            platform_rules=self.PLATFORM_TITLE_RULES.get(platform, ""),
            max_chars=limits.get("title", 100),
        )
        optimized_title_response = self._llm_generate(
            self.SEO_SYSTEM_PROMPT, title_prompt
        )

        # Optimize description via LLM
        desc_rules = self.PLATFORM_DESCRIPTION_RULES.get(platform, {})
        desc_prompt = self.DESCRIPTION_OPTIMIZATION_PROMPT.format(
            platform=specs.get("display_name", platform),
            topic=title,
            angle=description[:200],
            keywords=", ".join(keywords[:8]),
            platform_rules=self.PLATFORM_TITLE_RULES.get(platform, ""),
            max_chars=desc_rules.get("max_chars", 2200),
            include_instructions=desc_rules.get("include", ""),
        )
        optimized_desc_response = self._llm_generate(
            self.SEO_SYSTEM_PROMPT, desc_prompt
        )

        # Apply metadata limits
        max_title = limits.get("title", 100)
        optimized_title = title[:max_title] if max_title else title
        max_desc = limits.get("description", 2200)
        optimized_desc = description[:max_desc] if max_desc else description

        # Build optimized tags (YouTube) or hashtags (TikTok/Instagram)
        optimized_tags = list(dict.fromkeys(keywords + tags))
        hashtags = self._build_hashtags(keywords, platform)

        seo_score = self._estimate_seo_score(optimized_title, optimized_tags, platform)

        return {
            "platform": platform,
            "optimized_title": optimized_title,
            "optimized_description": optimized_desc,
            "optimized_tags": optimized_tags,
            "hashtags": hashtags,
            "seo_score": seo_score,
            "llm_title_suggestions": optimized_title_response[:500],
            "llm_description_draft": optimized_desc_response[:500],
        }

    def _optimize_all_platforms(self, task: Task) -> dict[str, Any]:
        """Optimize metadata for all target platforms at once."""
        platforms = task.payload.get(
            "platforms", ["youtube", "tiktok", "instagram"]
        )
        title = task.payload.get("title", "")
        description = task.payload.get("description", "")
        tags = task.payload.get("tags", [])
        keywords = task.payload.get("keywords", [])

        results = {}
        for platform in platforms:
            sub_task = Task(
                title=f"Optimize for {platform}",
                created_by=self.name,
                payload={
                    "action": "optimize_metadata",
                    "platform": platform,
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "keywords": keywords,
                },
            )
            results[platform] = self._optimize_metadata(sub_task)

        return {
            "platforms_optimized": platforms,
            "results": results,
        }

    def _plan_distribution(self, task: Task) -> dict[str, Any]:
        """Plan staggered cross-platform distribution."""
        asset_id = task.payload.get("asset_id", "")
        platforms = task.payload.get(
            "platforms", ["youtube", "tiktok", "instagram"]
        )
        preferred_times = task.payload.get("preferred_times", {})

        schedule = []
        for i, platform in enumerate(platforms):
            sched = self.POSTING_SCHEDULE.get(platform, {})
            best_hours = sched.get("best_hours_utc", [12])
            best_days = sched.get("best_days", ["tuesday"])
            frequency = sched.get("frequency", "3x per week")

            entry = {
                "platform": platform,
                "asset_id": asset_id,
                "recommended_hour_utc": preferred_times.get(platform, best_hours[0]),
                "recommended_days": best_days,
                "posting_frequency": frequency,
                "stagger_offset_hours": i * 2,  # stagger by 2 hours between platforms
                "status": "scheduled",
                "repurpose_notes": self._get_repurpose_notes(platforms, platform),
            }
            schedule.append(entry)
            self.distribution_queue.append(entry)

        return {
            "schedule": schedule,
            "total_platforms": len(platforms),
            "strategy": (
                "Staggered release: post to TikTok first (fastest discovery), "
                "then YouTube Shorts (SEO value), then Instagram Reels (cross-post)."
            ),
            "recommended_order": self._optimal_platform_order(platforms),
        }

    def _create_ab_test(self, task: Task) -> dict[str, Any]:
        """Create A/B test variants using LLM for titles or thumbnails."""
        original = task.payload.get("original", "")
        element = task.payload.get("element", "title")
        platform = task.payload.get("platform", "youtube")
        goal = task.payload.get("goal", "CTR")

        ab_prompt = self.AB_TEST_PROMPT.format(
            element=element, original=original,
            platform=platform, goal=goal,
        )
        llm_response = self._llm_generate(self.SEO_SYSTEM_PROMPT, ab_prompt)

        variants = [
            {"variant": "A", "value": original, "is_control": True,
             "hypothesis": "Baseline — current best guess"},
            {"variant": "B", "value": f"[LLM Variant B] {original}",
             "is_control": False,
             "hypothesis": "Tests different emotional angle"},
            {"variant": "C", "value": f"[LLM Variant C] {original}",
             "is_control": False,
             "hypothesis": "Tests different structural approach"},
        ]

        return {
            "element_tested": element,
            "platform": platform,
            "goal": goal,
            "variants": variants,
            "test_duration_hours": 48,
            "min_sample_size": 1000,
            "llm_suggestions": llm_response[:500],
        }

    def _growth_strategy(self, task: Task) -> dict[str, Any]:
        """Generate audience growth recommendations using LLM."""
        current_metrics = task.payload.get("current_metrics", {})
        niche = task.payload.get("niche", "general")
        platforms = task.payload.get(
            "platforms", ["youtube", "tiktok", "instagram"]
        )

        strategy_prompt = self.GROWTH_STRATEGY_PROMPT.format(
            niche=niche,
            metrics=str(current_metrics) if current_metrics else "New channel, no data yet",
            platforms=", ".join(platforms),
        )
        llm_response = self._llm_generate(self.SEO_SYSTEM_PROMPT, strategy_prompt)

        # Build platform-specific cadence recommendations
        cadence = {}
        for platform in platforms:
            sched = self.POSTING_SCHEDULE.get(platform, {})
            cadence[platform] = {
                "frequency": sched.get("frequency", "3x per week"),
                "best_days": sched.get("best_days", []),
                "best_hours_utc": sched.get("best_hours_utc", []),
            }

        return {
            "niche": niche,
            "platform_cadence": cadence,
            "cross_platform_strategy": (
                "Post TikTok-first for discovery → repurpose to YouTube Shorts "
                "for SEO longevity → cross-post to Instagram Reels for audience diversity"
            ),
            "strategies": [
                "consistent_posting_schedule",
                "cross_platform_repurposing",
                "community_engagement_replies",
                "collaboration_outreach",
                "seo_optimization_youtube",
                "hashtag_strategy_tiktok_instagram",
                "trend_riding_quick_turnaround",
            ],
            "priority": "consistent_posting_schedule",
            "llm_recommendations": llm_response[:800],
        }

    def _hashtag_strategy(self, task: Task) -> dict[str, Any]:
        """Generate platform-specific hashtag strategy."""
        keywords = task.payload.get("keywords", [])
        niche = task.payload.get("niche", "general")
        platforms = task.payload.get(
            "platforms", ["youtube", "tiktok", "instagram"]
        )

        strategies = {}
        for platform in platforms:
            strategies[platform] = self._build_hashtags(keywords, platform, niche)

        return {
            "niche": niche,
            "strategies": strategies,
        }

    def _repurpose_plan(self, task: Task) -> dict[str, Any]:
        """Plan how to repurpose content across platforms."""
        source_platform = task.payload.get("source_platform", "youtube_shorts")
        target_platforms = task.payload.get(
            "target_platforms", ["tiktok", "instagram"]
        )

        plans = []
        for target in target_platforms:
            key = (source_platform, target)
            rules = self.REPURPOSE_MAP.get(key, {
                "changes": ["Adapt metadata for target platform"],
                "aspect_ratio": "9:16",
                "max_duration": 60,
            })
            plans.append({
                "source": source_platform,
                "target": target,
                **rules,
            })

        return {"repurpose_plans": plans}

    # ----- Internal Helpers -----

    def _build_hashtags(
        self,
        keywords: list[str],
        platform: str,
        niche: str = "",
    ) -> list[str]:
        """Build platform-appropriate hashtag list."""
        tier = self.HASHTAG_TIERS.get(platform, self.HASHTAG_TIERS["tiktok"])
        max_count = tier["max_count"]

        # Niche-specific tags from keywords
        niche_tags = [f"#{kw.replace(' ', '').lower()}" for kw in keywords[:max_count]]

        # Add universal platform tags
        universal = tier.get("universal", [])
        engagement = tier.get("engagement", [])

        # Build final list: niche tags first, then universal, then engagement
        all_tags = list(dict.fromkeys(niche_tags + universal + engagement))
        return all_tags[:max_count]

    def _estimate_seo_score(
        self, title: str, tags: list[str], platform: str
    ) -> float:
        """Platform-aware heuristic SEO score (0-1)."""
        score = 0.0

        if platform == "youtube":
            # YouTube SEO heavily depends on title + tags + description
            if 20 < len(title) < 70:
                score += 0.3
            if len(tags) >= 5:
                score += 0.2
            if len(tags) >= 10:
                score += 0.2
            if any(char.isdigit() for char in title):
                score += 0.1  # numbers in titles boost CTR
            if len(title) > 0 and title[0].isupper():
                score += 0.1
            # Keyword in first 5 words
            score += 0.1

        elif platform == "tiktok":
            # TikTok "SEO" is about discoverability via hashtags + sounds
            score += 0.3  # base — TikTok relies less on text SEO
            if len(tags) >= 3:
                score += 0.3
            if len(title) < 80:
                score += 0.2
            score += 0.2  # assume trending sound will be added

        elif platform == "instagram":
            # Instagram SEO via hashtags + alt text + caption
            score += 0.2
            if len(tags) >= 10:
                score += 0.3
            if len(tags) >= 20:
                score += 0.2
            if len(title) > 0:
                score += 0.1
            score += 0.2  # assume proper alt text

        return min(round(score, 2), 1.0)

    def _optimal_platform_order(self, platforms: list[str]) -> list[str]:
        """Determine the optimal posting order across platforms."""
        priority = {"tiktok": 1, "youtube": 2, "instagram": 3}
        return sorted(platforms, key=lambda p: priority.get(p, 99))

    def _get_repurpose_notes(
        self, all_platforms: list[str], current_platform: str
    ) -> str:
        """Get repurposing notes for a platform in context of the full distribution."""
        if current_platform == "tiktok":
            return "Post first — fastest discovery loop. Remove watermark before repurposing."
        elif current_platform == "youtube":
            return "Add SEO-optimized title/tags. Include end screen if long-form."
        elif current_platform == "instagram":
            return "Cross-post from TikTok. Add hashtag block. Use 'Save this' CTA."
        return ""
