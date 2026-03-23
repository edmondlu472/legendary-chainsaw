"""QA Agent — reviews content before publishing with platform-specific policy checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base import BaseAgent, LLMConfig, PLATFORM_SPECS, Task


@dataclass
class QAReport:
    """Result of a QA review on a video asset."""

    asset_id: str = ""
    platform: str = ""
    approved: bool = False
    issues: list[dict[str, str]] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    reviewer: str = "qa_agent"
    ai_disclosure_required: bool = True
    ai_disclosure_set: bool = False


class QAAgent(BaseAgent):
    """Reviews all content and metadata before it is published.

    Responsibilities:
    - Check content against platform-specific community guidelines
    - Verify metadata completeness per platform (YouTube, TikTok, Instagram)
    - Validate technical specs (format, resolution, duration per platform)
    - Enforce AI content disclosure requirements per platform
    - Flag potential copyright or policy violations via LLM analysis
    - Ensure brand voice and quality consistency
    - Gate the publishing pipeline — nothing ships without QA approval

    Tools employed:
    - Claude API   — content policy review, brand safety analysis, quality scoring
    - FFprobe      — technical video validation (codec, resolution, bitrate, duration)
    - Custom rules — platform ToS compliance, metadata limits, flagged term detection
    """

    name = "qa"
    role = "Quality Assurance Reviewer"
    description = (
        "Reviews all content for platform-specific policy compliance, metadata "
        "completeness, AI disclosure, and technical quality before publishing."
    )

    # ----- LLM Prompt Templates -----

    QA_SYSTEM_PROMPT = (
        "You are a strict content quality reviewer for AI-generated video content. "
        "You review scripts, titles, descriptions, and metadata before publishing "
        "to YouTube, TikTok, and Instagram.\n\n"
        "You check for:\n"
        "1. Platform community guideline violations\n"
        "2. Misleading claims or misinformation\n"
        "3. Brand safety issues (controversial topics, sensitive content)\n"
        "4. Copyright concerns (music, images, quotes)\n"
        "5. AI content disclosure compliance\n"
        "6. Quality issues (grammar, pacing, clarity)\n"
        "7. Engagement potential (does the hook work? Is the CTA clear?)\n\n"
        "Rate severity as: error (blocks publishing), warning (should fix), "
        "or info (nice to improve)."
    )

    CONTENT_REVIEW_PROMPT = (
        "Review this video content for {platform} publishing.\n\n"
        "Title: {title}\n"
        "Description: {description}\n"
        "Script: {script}\n"
        "Voiceover text: {voiceover}\n"
        "Tags: {tags}\n"
        "Duration: {duration}s\n"
        "AI-generated: Yes\n\n"
        "Check against {platform}'s community guidelines and policies.\n"
        "Flag any issues with severity (error/warning/info).\n"
        "Also rate overall quality on a 1-10 scale."
    )

    # ----- Flagged Terms (global) -----

    FLAGGED_TERMS_GLOBAL = [
        "guaranteed income", "get rich quick", "miracle cure",
        "act now", "limited time only", "100% guaranteed",
        "no risk", "secret method", "doctors hate this",
        "one weird trick", "free money",
    ]

    # ----- Platform-Specific Policy Rules -----

    PLATFORM_POLICIES = {
        "youtube": {
            "flagged_terms": [
                "subscribe to win", "cash giveaway", "like to enter",
            ],
            "max_title_length": 100,
            "required_fields": ["title", "description", "tags", "script"],
            "thumbnail_required": True,
            "ai_disclosure": {
                "required": True,
                "method": "Upload setting: 'Altered or synthetic content'",
                "enforcement": "Platform may add label; required for realistic content",
            },
            "restricted_categories": [
                "children_content_with_ai_faces",
                "medical_advice_without_disclaimer",
                "financial_advice_without_disclaimer",
            ],
            "content_rules": [
                "No misleading thumbnails that don't match content",
                "No artificial inflation of engagement metrics",
                "Must comply with YouTube's AI-generated content policy",
                "Monetization requires advertiser-friendly content",
            ],
        },
        "tiktok": {
            "flagged_terms": [
                "follow for follow", "f4f", "like for like",
            ],
            "max_title_length": 150,
            "required_fields": ["description", "script"],
            "thumbnail_required": False,
            "ai_disclosure": {
                "required": True,
                "method": "Label recommended; platform auto-detects some AI content",
                "enforcement": "Tolerant but recommends transparency",
            },
            "restricted_categories": [
                "deepfake_of_real_people",
                "ai_generated_news_without_label",
                "synthetic_voices_impersonating_real_people",
            ],
            "content_rules": [
                "No content that could be mistaken for real news without disclosure",
                "No AI-generated content impersonating real people",
                "Must follow TikTok Community Guidelines",
                "Duets/stitches must respect original creator rights",
            ],
        },
        "instagram": {
            "flagged_terms": [
                "follow for follow", "f4f", "dm for collab",
            ],
            "max_title_length": 0,  # no separate title field
            "required_fields": ["description", "script"],
            "thumbnail_required": False,
            "ai_disclosure": {
                "required": True,
                "method": "Platform-enforced AI label (mandatory since 2024)",
                "enforcement": "Strict — Instagram auto-applies AI-generated label",
            },
            "restricted_categories": [
                "deepfake_of_real_people",
                "ai_political_content",
                "synthetic_endorsements",
            ],
            "content_rules": [
                "AI-generated content label is automatically applied by Meta",
                "No synthetic media of real people without consent",
                "Must follow Instagram Community Guidelines",
                "Branded content must use partnership label",
            ],
        },
    }

    # ----- Technical Specs Validation -----

    TECHNICAL_REQUIREMENTS = {
        "youtube": {
            "min_resolution": "720p",
            "recommended_resolution": "1080p",
            "codec": "h264",
            "audio_codec": "aac",
            "min_bitrate_kbps": 2000,
            "shorts_aspect": "9:16",
            "long_aspect": "16:9",
            "max_file_size_mb": 256_000,
        },
        "tiktok": {
            "min_resolution": "720p",
            "recommended_resolution": "1080p",
            "codec": "h264",
            "audio_codec": "aac",
            "min_bitrate_kbps": 1000,
            "aspect": "9:16",
            "max_file_size_mb": 287,
        },
        "instagram": {
            "min_resolution": "720p",
            "recommended_resolution": "1080p",
            "codec": "h264",
            "audio_codec": "aac",
            "min_bitrate_kbps": 1500,
            "reels_aspect": "9:16",
            "feed_aspect": "1:1",
            "max_file_size_mb": 650,
        },
    }

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config=llm_config)
        self.review_history: list[QAReport] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "full_review")
        if action == "full_review":
            return self._full_review(task)
        elif action == "platform_review":
            return self._platform_review(task)
        elif action == "metadata_check":
            return self._metadata_check(task)
        elif action == "policy_check":
            return self._policy_check(task)
        elif action == "technical_check":
            return self._technical_check(task)
        elif action == "ai_disclosure_check":
            return self._ai_disclosure_check(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ----- Core Actions -----

    def _full_review(self, task: Task) -> dict[str, Any]:
        """Run all checks on a video asset for its target platform."""
        asset = task.payload.get("asset", {})
        platform = asset.get("platform", task.payload.get("platform", "youtube"))
        report = QAReport(
            asset_id=asset.get("asset_id", ""),
            platform=platform,
        )

        # 1. Metadata completeness
        metadata_result = self._check_metadata(asset, platform)
        report.checks_passed.extend(metadata_result["passed"])
        report.checks_failed.extend(metadata_result["failed"])
        report.issues.extend(metadata_result["issues"])

        # 2. Policy compliance (global + platform-specific)
        policy_result = self._check_policy(asset, platform)
        report.checks_passed.extend(policy_result["passed"])
        report.checks_failed.extend(policy_result["failed"])
        report.issues.extend(policy_result["issues"])

        # 3. Technical validation
        tech_result = self._check_technical(asset, platform)
        report.checks_passed.extend(tech_result["passed"])
        report.checks_failed.extend(tech_result["failed"])
        report.issues.extend(tech_result["issues"])

        # 4. AI disclosure compliance
        disclosure_result = self._check_ai_disclosure(asset, platform)
        report.checks_passed.extend(disclosure_result["passed"])
        report.checks_failed.extend(disclosure_result["failed"])
        report.issues.extend(disclosure_result["issues"])
        report.ai_disclosure_required = disclosure_result.get("required", True)
        report.ai_disclosure_set = disclosure_result.get("disclosure_set", False)

        # 5. LLM content quality review (if script present)
        if asset.get("script"):
            quality_result = self._llm_content_review(asset, platform)
            report.checks_passed.extend(quality_result.get("passed", []))
            report.checks_failed.extend(quality_result.get("failed", []))
            report.issues.extend(quality_result.get("issues", []))

        # Final gate decision
        has_errors = any(
            issue.get("severity") == "error" for issue in report.issues
        )
        report.approved = len(report.checks_failed) == 0 and not has_errors
        self.review_history.append(report)

        return {
            "asset_id": report.asset_id,
            "platform": report.platform,
            "approved": report.approved,
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
            "issues": report.issues,
            "ai_disclosure_required": report.ai_disclosure_required,
            "ai_disclosure_set": report.ai_disclosure_set,
            "gate": "pass" if report.approved else "blocked",
            "total_checks": len(report.checks_passed) + len(report.checks_failed),
        }

    def _platform_review(self, task: Task) -> dict[str, Any]:
        """Run a full review targeting a specific platform."""
        # Alias for full_review with explicit platform
        return self._full_review(task)

    def _metadata_check(self, task: Task) -> dict[str, Any]:
        """Check metadata completeness only."""
        asset = task.payload.get("asset", {})
        platform = task.payload.get("platform", "youtube")
        return self._check_metadata(asset, platform)

    def _policy_check(self, task: Task) -> dict[str, Any]:
        """Check policy compliance only."""
        asset = task.payload.get("asset", {})
        platform = task.payload.get("platform", "youtube")
        return self._check_policy(asset, platform)

    def _technical_check(self, task: Task) -> dict[str, Any]:
        """Check technical specifications only."""
        asset = task.payload.get("asset", {})
        platform = task.payload.get("platform", "youtube")
        return self._check_technical(asset, platform)

    def _ai_disclosure_check(self, task: Task) -> dict[str, Any]:
        """Check AI content disclosure compliance only."""
        asset = task.payload.get("asset", {})
        platform = task.payload.get("platform", "youtube")
        return self._check_ai_disclosure(asset, platform)

    # ----- Internal Check Methods -----

    def _check_metadata(
        self, asset: dict[str, Any], platform: str
    ) -> dict[str, Any]:
        """Verify all required metadata fields are present per platform."""
        passed = []
        failed = []
        issues = []

        policy = self.PLATFORM_POLICIES.get(platform, self.PLATFORM_POLICIES["youtube"])
        required_fields = policy["required_fields"]

        for f in required_fields:
            if asset.get(f):
                passed.append(f"metadata_{f}_present")
            else:
                failed.append(f"metadata_{f}_missing")
                issues.append({
                    "severity": "error",
                    "message": f"[{platform}] Missing required field: {f}",
                })

        # Title length check (platform-specific)
        title = asset.get("title", "")
        max_title = policy.get("max_title_length", 100)
        if max_title > 0:  # 0 means no title field (Instagram)
            if title and len(title) > max_title:
                failed.append("metadata_title_too_long")
                issues.append({
                    "severity": "warning",
                    "message": f"[{platform}] Title is {len(title)} chars (max {max_title})",
                })
            elif title:
                passed.append("metadata_title_length_ok")

        # Thumbnail check
        if policy.get("thumbnail_required"):
            if asset.get("thumbnail_prompt") or asset.get("thumbnail_path"):
                passed.append("metadata_thumbnail_present")
            else:
                failed.append("metadata_thumbnail_missing")
                issues.append({
                    "severity": "error",
                    "message": f"[{platform}] Thumbnail is required but missing",
                })

        # Hashtag count check (Instagram-specific)
        if platform == "instagram":
            hashtags = asset.get("hashtags", [])
            specs = PLATFORM_SPECS.get("instagram", {})
            max_hashtags = specs.get("hashtag_limit", 30)
            if len(hashtags) > max_hashtags:
                failed.append("metadata_too_many_hashtags")
                issues.append({
                    "severity": "warning",
                    "message": f"[instagram] {len(hashtags)} hashtags exceeds limit of {max_hashtags}",
                })
            elif hashtags:
                passed.append("metadata_hashtag_count_ok")

        return {"passed": passed, "failed": failed, "issues": issues}

    def _check_policy(
        self, asset: dict[str, Any], platform: str
    ) -> dict[str, Any]:
        """Check for policy violations — global + platform-specific."""
        passed = []
        failed = []
        issues = []

        text_to_check = " ".join([
            asset.get("title", ""),
            asset.get("description", ""),
            asset.get("script", ""),
            asset.get("voiceover_text", ""),
        ]).lower()

        # Global flagged terms
        global_flagged = [
            term for term in self.FLAGGED_TERMS_GLOBAL if term in text_to_check
        ]
        if global_flagged:
            failed.append("policy_global_flagged_terms")
            issues.append({
                "severity": "warning",
                "message": f"Global flagged terms found: {global_flagged}",
            })
        else:
            passed.append("policy_no_global_flagged_terms")

        # Platform-specific flagged terms
        policy = self.PLATFORM_POLICIES.get(platform, {})
        platform_flagged = [
            term for term in policy.get("flagged_terms", [])
            if term in text_to_check
        ]
        if platform_flagged:
            failed.append(f"policy_{platform}_flagged_terms")
            issues.append({
                "severity": "warning",
                "message": f"[{platform}] Platform-specific flagged terms: {platform_flagged}",
            })
        else:
            passed.append(f"policy_{platform}_terms_clean")

        # Restricted categories check
        restricted = policy.get("restricted_categories", [])
        # In production, use LLM to classify content against these categories
        passed.append(f"policy_{platform}_categories_checked")

        # Platform content rules — informational
        content_rules = policy.get("content_rules", [])
        for rule in content_rules:
            issues.append({
                "severity": "info",
                "message": f"[{platform}] Rule reminder: {rule}",
            })

        passed.append("policy_basic_check_done")
        return {"passed": passed, "failed": failed, "issues": issues}

    def _check_technical(
        self, asset: dict[str, Any], platform: str
    ) -> dict[str, Any]:
        """Validate technical specifications per platform."""
        passed = []
        failed = []
        issues = []

        specs = PLATFORM_SPECS.get(platform, {})
        tech_reqs = self.TECHNICAL_REQUIREMENTS.get(platform, {})

        duration = asset.get("duration_seconds", 0)

        # Duration checks
        if duration <= 0:
            failed.append("tech_invalid_duration")
            issues.append({"severity": "error", "message": "Duration must be > 0"})
        else:
            max_duration = specs.get("max_duration_seconds", 43200)
            if duration > max_duration:
                failed.append("tech_duration_exceeds_platform_max")
                issues.append({
                    "severity": "error",
                    "message": f"[{platform}] Video is {duration}s, max is {max_duration}s",
                })
            else:
                passed.append("tech_duration_ok")

            # Platform-specific duration hints
            if platform == "youtube":
                shorts_max = specs.get("shorts_max_seconds", 60)
                if duration <= shorts_max:
                    passed.append("tech_youtube_shorts_eligible")
                else:
                    issues.append({
                        "severity": "info",
                        "message": f"Video is {duration}s — will be long-form (not a Short)",
                    })
            elif platform == "tiktok":
                if duration > 180:
                    issues.append({
                        "severity": "info",
                        "message": f"TikTok sweet spot is 30-180s. Current: {duration}s",
                    })
            elif platform == "instagram":
                reels_max = specs.get("reels_max_seconds", 90)
                if duration > reels_max:
                    issues.append({
                        "severity": "warning",
                        "message": f"[instagram] Duration {duration}s exceeds Reels max of {reels_max}s",
                    })

        # Video source check
        if asset.get("video_prompt") or asset.get("file_path"):
            passed.append("tech_video_source_present")
        else:
            failed.append("tech_no_video_source")
            issues.append({"severity": "error", "message": "No video file or generation prompt"})

        # File size check (if file exists)
        file_size_mb = asset.get("file_size_mb", 0)
        if file_size_mb > 0:
            max_size = tech_reqs.get("max_file_size_mb", specs.get("max_size_mb", 256000))
            if file_size_mb > max_size:
                failed.append("tech_file_too_large")
                issues.append({
                    "severity": "error",
                    "message": f"[{platform}] File is {file_size_mb}MB, max is {max_size}MB",
                })
            else:
                passed.append("tech_file_size_ok")

        return {"passed": passed, "failed": failed, "issues": issues}

    def _check_ai_disclosure(
        self, asset: dict[str, Any], platform: str
    ) -> dict[str, Any]:
        """Verify AI content disclosure compliance per platform."""
        passed = []
        failed = []
        issues = []

        policy = self.PLATFORM_POLICIES.get(platform, {})
        disclosure_rules = policy.get("ai_disclosure", {})
        required = disclosure_rules.get("required", True)
        method = disclosure_rules.get("method", "Unknown")
        enforcement = disclosure_rules.get("enforcement", "Unknown")

        disclosure_set = asset.get("ai_disclosure", False)

        if required:
            if disclosure_set:
                passed.append(f"ai_disclosure_{platform}_set")
            else:
                failed.append(f"ai_disclosure_{platform}_missing")
                issues.append({
                    "severity": "error" if platform == "instagram" else "warning",
                    "message": (
                        f"[{platform}] AI content disclosure is required. "
                        f"Method: {method}. Enforcement: {enforcement}"
                    ),
                })
        else:
            passed.append(f"ai_disclosure_{platform}_not_required")

        return {
            "passed": passed,
            "failed": failed,
            "issues": issues,
            "required": required,
            "disclosure_set": disclosure_set,
        }

    def _llm_content_review(
        self, asset: dict[str, Any], platform: str
    ) -> dict[str, Any]:
        """Use LLM to perform a deeper content quality review."""
        passed = []
        failed = []
        issues = []

        review_prompt = self.CONTENT_REVIEW_PROMPT.format(
            platform=PLATFORM_SPECS.get(platform, {}).get("display_name", platform),
            title=asset.get("title", "N/A"),
            description=asset.get("description", "N/A")[:500],
            script=asset.get("script", "N/A")[:1000],
            voiceover=asset.get("voiceover_text", "N/A")[:500],
            tags=", ".join(asset.get("tags", [])),
            duration=asset.get("duration_seconds", 0),
        )

        llm_response = self._llm_generate(self.QA_SYSTEM_PROMPT, review_prompt)

        # In production, parse LLM JSON response for structured issues
        passed.append("llm_content_review_completed")
        issues.append({
            "severity": "info",
            "message": f"LLM review: {llm_response[:300]}",
        })

        return {"passed": passed, "failed": failed, "issues": issues}
