"""Base agent class and shared data structures for the multi-agent content system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Platform(Enum):
    """Supported publishing platforms."""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


class ContentStyle(Enum):
    """Video content styles the system can produce."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    TUTORIAL = "tutorial"
    LISTICLE = "listicle"
    STORYTELLING = "storytelling"
    NEWS_RECAP = "news_recap"
    SHORTS_HOOK = "shorts_hook"  # optimised for ≤60s vertical video


class VideoGenProvider(Enum):
    """AI video generation services."""
    KLING = "kling"
    RUNWAY = "runway"
    PIKA = "pika"


class ImageGenProvider(Enum):
    """AI image generation services (thumbnails, overlays)."""
    DALLE3 = "dall-e-3"
    FLUX = "flux"
    MIDJOURNEY = "midjourney"


class TTSProvider(Enum):
    """Text-to-speech / voiceover services."""
    ELEVENLABS = "elevenlabs"
    GOOGLE_TTS = "google_tts"
    AZURE_TTS = "azure_tts"


# ---------------------------------------------------------------------------
# Tool / service configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for the LLM used across agents (script writing, SEO, QA)."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key_env: str = "ANTHROPIC_API_KEY"  # env var name, never store raw keys


@dataclass
class VideoGenConfig:
    """Configuration for the AI video generation service."""
    provider: VideoGenProvider = VideoGenProvider.KLING
    api_key_env: str = "VIDEO_GEN_API_KEY"
    default_resolution: str = "1080x1920"  # 9:16 vertical (Shorts/Reels/TikTok)
    default_fps: int = 30
    max_duration_seconds: int = 60
    output_format: str = "mp4"
    # Provider-specific settings
    kling_mode: str = "standard"  # standard | pro
    runway_model: str = "gen-3"


@dataclass
class ImageGenConfig:
    """Configuration for AI thumbnail / image generation."""
    provider: ImageGenProvider = ImageGenProvider.DALLE3
    api_key_env: str = "IMAGE_GEN_API_KEY"
    thumbnail_size: str = "1280x720"       # YouTube standard
    square_size: str = "1080x1080"         # Instagram grid
    vertical_size: str = "1080x1920"       # TikTok / Reels cover
    quality: str = "hd"
    style: str = "vivid"


@dataclass
class TTSConfig:
    """Configuration for text-to-speech voiceover generation."""
    provider: TTSProvider = TTSProvider.ELEVENLABS
    api_key_env: str = "TTS_API_KEY"
    voice_id: str = "default"
    model: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    output_format: str = "mp3_44100_128"
    words_per_minute: int = 150  # pacing reference


@dataclass
class TrendSourceConfig:
    """Configuration for a single trend data source."""
    name: str = ""
    api_key_env: str = ""
    base_url: str = ""
    enabled: bool = True
    rate_limit_per_min: int = 60


# ---------------------------------------------------------------------------
# Platform specifications — publishing requirements per platform
# ---------------------------------------------------------------------------

PLATFORM_SPECS: dict[str, dict[str, Any]] = {
    "youtube": {
        "display_name": "YouTube",
        "formats": ["mp4", "mov"],
        "max_size_mb": 256_000,
        "max_duration_seconds": 43_200,  # 12 hours
        "shorts_max_seconds": 60,
        "thumbnail_required": True,
        "thumbnail_size": "1280x720",
        "aspect_ratios": {"long": "16:9", "shorts": "9:16"},
        "metadata_limits": {"title": 100, "description": 5000, "tags_total_chars": 500},
        "optimal_posting_hours_utc": [14, 15, 16, 17],
        "monetization": {
            "program": "YouTube Partner Program (YPP)",
            "requirements": "1000 subs + 4000 watch hours OR 10M Shorts views",
            "avg_rpm_usd": {"long": 5.0, "shorts": 0.06},
            "revenue_streams": ["ad_revenue", "memberships", "super_chat", "merch_shelf"],
        },
        "ai_content_policy": "Allowed; must disclose AI-generated content via upload settings",
        "api": {
            "upload": "YouTube Data API v3",
            "analytics": "YouTube Analytics API",
            "auth": "OAuth 2.0",
            "scopes": [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/yt-analytics.readonly",
            ],
        },
    },
    "tiktok": {
        "display_name": "TikTok",
        "formats": ["mp4"],
        "max_size_mb": 287,
        "max_duration_seconds": 600,  # 10 minutes
        "thumbnail_required": False,
        "thumbnail_size": "1080x1920",
        "aspect_ratios": {"default": "9:16"},
        "metadata_limits": {"title": 150, "description": 2200, "tags_total_chars": 0},
        "optimal_posting_hours_utc": [11, 12, 19, 20],
        "monetization": {
            "program": "TikTok Creativity Program",
            "requirements": "10k followers + 100k views in 30 days",
            "avg_rpm_usd": {"short": 1.0, "creativity_program": 4.0},
            "revenue_streams": ["creativity_program", "live_gifts", "tiktok_shop"],
        },
        "ai_content_policy": "Tolerant; AI-generated content widely accepted, label recommended",
        "api": {
            "upload": "TikTok Content Posting API",
            "analytics": "TikTok Analytics API",
            "auth": "OAuth 2.0",
            "scopes": ["video.upload", "video.list", "user.info.basic"],
        },
    },
    "instagram": {
        "display_name": "Instagram",
        "formats": ["mp4", "mov"],
        "max_size_mb": 650,
        "max_duration_seconds": 5400,  # 90 minutes
        "reels_max_seconds": 90,
        "thumbnail_required": False,
        "thumbnail_size": "1080x1920",
        "aspect_ratios": {"reels": "9:16", "feed": "1:1", "landscape": "16:9"},
        "metadata_limits": {"title": 0, "description": 2200, "tags_total_chars": 0},
        "optimal_posting_hours_utc": [11, 13, 19],
        "hashtag_limit": 30,
        "monetization": {
            "program": "Instagram Reels Bonuses (invite-only)",
            "requirements": "Professional/Creator account",
            "avg_rpm_usd": {"reels": 2.0},
            "revenue_streams": ["reels_bonuses", "brand_partnerships", "shopping"],
        },
        "ai_content_policy": "Must use AI-generated content label (platform enforced)",
        "api": {
            "upload": "Instagram Graph API (via Facebook Business Suite)",
            "analytics": "Instagram Insights API",
            "auth": "OAuth 2.0 (Facebook Login)",
            "scopes": [
                "instagram_basic",
                "instagram_content_publish",
                "instagram_manage_insights",
            ],
        },
    },
}

# Convenience: default trend sources with API integration points
DEFAULT_TREND_SOURCES: list[TrendSourceConfig] = [
    TrendSourceConfig(
        name="google_trends",
        api_key_env="",  # no key required for pytrends
        base_url="https://trends.google.com",
        rate_limit_per_min=30,
    ),
    TrendSourceConfig(
        name="youtube_trending",
        api_key_env="YOUTUBE_API_KEY",
        base_url="https://www.googleapis.com/youtube/v3",
        rate_limit_per_min=60,
    ),
    TrendSourceConfig(
        name="tiktok_creative_center",
        api_key_env="TIKTOK_API_KEY",
        base_url="https://business-api.tiktok.com/open_api/v1.3",
        rate_limit_per_min=60,
    ),
    TrendSourceConfig(
        name="reddit_popular",
        api_key_env="REDDIT_CLIENT_SECRET",
        base_url="https://oauth.reddit.com",
        rate_limit_per_min=60,
    ),
]

# Default tool stack
DEFAULT_LLM = LLMConfig()
DEFAULT_VIDEO_GEN = VideoGenConfig()
DEFAULT_IMAGE_GEN = ImageGenConfig()
DEFAULT_TTS = TTSConfig()


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

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
    content_style: str = "educational"
    hook: str = ""  # attention-grabbing opening line


@dataclass
class VideoAsset:
    """Output from the content creator agent."""

    asset_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    brief_id: str = ""
    platform: str = ""  # which platform this variant targets
    script: str = ""
    video_prompt: str = ""
    voiceover_text: str = ""
    thumbnail_prompt: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)  # for TikTok/Instagram
    duration_seconds: int = 0
    aspect_ratio: str = "9:16"
    file_path: str | None = None
    thumbnail_path: str | None = None
    voiceover_path: str | None = None
    status: str = "draft"
    ai_disclosure: bool = True  # flag for platform AI content labels


@dataclass
class PlatformVariant:
    """A platform-specific adaptation of a single piece of content."""

    platform: str = ""
    asset: VideoAsset = field(default_factory=VideoAsset)
    optimized_title: str = ""
    optimized_description: str = ""
    optimized_tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    scheduled_time: str | None = None
    seo_score: float = 0.0


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Base class for all agents in the system."""

    name: str = "base"
    role: str = "Base Agent"
    description: str = ""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.task_history: list[Task] = []
        self.llm = llm_config or DEFAULT_LLM

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

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "llm_model": self.llm.model,
        }

    # ----- LLM helper (stub — wire to real Anthropic SDK in production) -----

    def _llm_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM and return the text response.

        In production this calls the Anthropic messages API.  For now it
        returns a structured placeholder so downstream logic can run.
        """
        # TODO: Replace with real API call:
        # from anthropic import Anthropic
        # client = Anthropic()
        # response = client.messages.create(
        #     model=self.llm.model,
        #     max_tokens=self.llm.max_tokens,
        #     temperature=self.llm.temperature,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_prompt}],
        # )
        # return response.content[0].text
        return f"[LLM:{self.llm.model}] {user_prompt[:200]}"
