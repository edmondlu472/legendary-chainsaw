"""Upload Manager Agent — handles multi-platform upload pipeline with OAuth and scheduling."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent, LLMConfig, PLATFORM_SPECS, Task


@dataclass
class UploadJob:
    """Tracks an individual upload to a platform."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    asset_id: str = ""
    platform: str = ""
    status: str = "queued"  # queued | validating | uploading | published | failed | scheduled
    scheduled_time: str | None = None
    publish_url: str | None = None
    retries: int = 0
    max_retries: int = 3
    error: str | None = None
    file_path: str | None = None
    thumbnail_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None


@dataclass
class OAuthCredentials:
    """OAuth 2.0 credentials for a platform (loaded from secure storage)."""

    platform: str = ""
    client_id_env: str = ""       # env var name for client ID
    client_secret_env: str = ""   # env var name for client secret
    access_token: str | None = None
    refresh_token: str | None = None
    token_expiry: str | None = None
    scopes: list[str] = field(default_factory=list)


class UploadManagerAgent(BaseAgent):
    """Manages the end-to-end upload pipeline for YouTube, TikTok, and Instagram.

    Responsibilities:
    - Prepare files for platform-specific requirements (format, resolution, codec)
    - Authenticate with platform APIs via OAuth 2.0
    - Execute uploads with exponential backoff retry logic
    - Schedule uploads for optimal posting times per platform
    - Track upload status and report results across all platforms
    - Handle multi-platform simultaneous publishing
    - Manage thumbnail uploads (required for YouTube)

    Tools employed:
    - YouTube Data API v3          — video upload, metadata, thumbnail, scheduling
    - TikTok Content Posting API   — video upload, description, scheduling
    - Instagram Graph API          — Reels publishing via Facebook Business Suite
    - FFmpeg / FFprobe             — format validation, transcoding if needed
    """

    name = "upload_manager"
    role = "Upload & Publishing Manager"
    description = (
        "Manages file preparation, OAuth authentication, scheduling, and "
        "multi-platform uploading with retry logic and status tracking."
    )

    # ----- OAuth Configuration per Platform -----

    OAUTH_CONFIGS: dict[str, dict[str, Any]] = {
        "youtube": {
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "client_id_env": "YOUTUBE_CLIENT_ID",
            "client_secret_env": "YOUTUBE_CLIENT_SECRET",
            "scopes": [
                "https://www.googleapis.com/auth/youtube.upload",
                "https://www.googleapis.com/auth/youtube",
                "https://www.googleapis.com/auth/youtube.readonly",
            ],
            "redirect_uri": "http://localhost:8080/callback",
        },
        "tiktok": {
            "auth_url": "https://www.tiktok.com/v2/auth/authorize/",
            "token_url": "https://open.tiktokapis.com/v2/oauth/token/",
            "client_id_env": "TIKTOK_CLIENT_KEY",
            "client_secret_env": "TIKTOK_CLIENT_SECRET",
            "scopes": ["video.upload", "video.list", "user.info.basic"],
            "redirect_uri": "http://localhost:8080/callback",
        },
        "instagram": {
            "auth_url": "https://www.facebook.com/v18.0/dialog/oauth",
            "token_url": "https://graph.facebook.com/v18.0/oauth/access_token",
            "client_id_env": "FACEBOOK_APP_ID",
            "client_secret_env": "FACEBOOK_APP_SECRET",
            "scopes": [
                "instagram_basic",
                "instagram_content_publish",
                "pages_read_engagement",
            ],
            "redirect_uri": "http://localhost:8080/callback",
            "notes": "Requires Facebook Business Suite and linked Instagram Professional account",
        },
    }

    # ----- Upload API Endpoints -----

    API_ENDPOINTS: dict[str, dict[str, str]] = {
        "youtube": {
            "upload": "https://www.googleapis.com/upload/youtube/v3/videos",
            "metadata": "https://www.googleapis.com/youtube/v3/videos",
            "thumbnails": "https://www.googleapis.com/youtube/v3/thumbnails/set",
            "method": "resumable_upload",
            "docs": "https://developers.google.com/youtube/v3/docs/videos/insert",
        },
        "tiktok": {
            "init_upload": "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
            "upload_video": "https://open.tiktokapis.com/v2/post/publish/video/init/",
            "status": "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            "method": "chunked_upload",
            "docs": "https://developers.tiktok.com/doc/content-posting-api",
        },
        "instagram": {
            "create_container": "https://graph.facebook.com/v18.0/{ig_user_id}/media",
            "publish": "https://graph.facebook.com/v18.0/{ig_user_id}/media_publish",
            "status": "https://graph.facebook.com/v18.0/{container_id}",
            "method": "container_publish",
            "docs": "https://developers.facebook.com/docs/instagram-api/guides/content-publishing",
            "notes": "Instagram requires video to be hosted at a public URL first",
        },
    }

    # ----- FFmpeg Transcoding Presets -----

    TRANSCODE_PRESETS: dict[str, dict[str, Any]] = {
        "youtube_shorts": {
            "resolution": "1080x1920",
            "codec": "h264",
            "audio_codec": "aac",
            "bitrate": "8M",
            "fps": 30,
            "aspect": "9:16",
            "ffmpeg_args": [
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1",
            ],
        },
        "youtube_long": {
            "resolution": "1920x1080",
            "codec": "h264",
            "audio_codec": "aac",
            "bitrate": "12M",
            "fps": 30,
            "aspect": "16:9",
            "ffmpeg_args": [
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1",
            ],
        },
        "tiktok": {
            "resolution": "1080x1920",
            "codec": "h264",
            "audio_codec": "aac",
            "bitrate": "6M",
            "fps": 30,
            "aspect": "9:16",
            "ffmpeg_args": [
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1",
            ],
        },
        "instagram_reels": {
            "resolution": "1080x1920",
            "codec": "h264",
            "audio_codec": "aac",
            "bitrate": "6M",
            "fps": 30,
            "aspect": "9:16",
            "ffmpeg_args": [
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1",
            ],
        },
    }

    # ----- Retry Configuration -----

    RETRY_CONFIG = {
        "max_retries": 3,
        "backoff_base_seconds": 2,
        "backoff_multiplier": 2,  # exponential: 2s, 4s, 8s
        "retryable_errors": [
            "rate_limit_exceeded",
            "server_error",
            "timeout",
            "network_error",
            "upload_interrupted",
        ],
        "non_retryable_errors": [
            "invalid_credentials",
            "insufficient_permissions",
            "invalid_video_format",
            "content_policy_violation",
        ],
    }

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config=llm_config)
        self.upload_queue: list[UploadJob] = []
        self.completed_uploads: list[UploadJob] = []
        self._credentials: dict[str, OAuthCredentials] = {}

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "queue_upload")
        if action == "queue_upload":
            return self._queue_upload(task)
        elif action == "validate_file":
            return self._validate_file(task)
        elif action == "execute_upload":
            return self._execute_upload(task)
        elif action == "execute_all":
            return self._execute_all_queued(task)
        elif action == "get_status":
            return self._get_status(task)
        elif action == "schedule_upload":
            return self._schedule_upload(task)
        elif action == "get_transcode_config":
            return self._get_transcode_config(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ----- Core Actions -----

    def _queue_upload(self, task: Task) -> dict[str, Any]:
        """Add upload jobs to the queue for one or more platforms."""
        asset_id = task.payload.get("asset_id", "")
        platforms = task.payload.get("platforms", ["youtube"])
        metadata = task.payload.get("metadata", {})
        file_path = task.payload.get("file_path")
        thumbnail_path = task.payload.get("thumbnail_path")

        jobs = []
        for platform in platforms:
            # Validate before queuing
            specs = PLATFORM_SPECS.get(platform)
            if not specs:
                jobs.append({
                    "platform": platform, "status": "failed",
                    "error": f"Unsupported platform: {platform}",
                })
                continue

            job = UploadJob(
                asset_id=asset_id,
                platform=platform,
                status="queued",
                file_path=file_path,
                thumbnail_path=thumbnail_path,
                metadata={
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "hashtags": metadata.get("hashtags", []),
                    "ai_disclosure": metadata.get("ai_disclosure", True),
                    **self._platform_upload_metadata(platform, metadata),
                },
            )
            self.upload_queue.append(job)
            jobs.append({
                "job_id": job.job_id,
                "platform": platform,
                "status": "queued",
                "api_endpoint": self.API_ENDPOINTS.get(platform, {}).get("upload", "unknown"),
                "auth_required": True,
                "oauth_config": self.OAUTH_CONFIGS.get(platform, {}).get("client_id_env", ""),
            })

        return {"jobs": jobs, "queue_length": len(self.upload_queue)}

    def _validate_file(self, task: Task) -> dict[str, Any]:
        """Validate a file against platform requirements."""
        platform = task.payload.get("platform", "youtube")
        file_format = task.payload.get("format", "mp4")
        file_size_mb = task.payload.get("size_mb", 0)
        duration_seconds = task.payload.get("duration_seconds", 0)

        specs = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["youtube"])

        issues = []
        warnings = []

        # Format check
        if file_format not in specs["formats"]:
            issues.append(f"Format '{file_format}' not supported. Use: {specs['formats']}")

        # Size check
        if file_size_mb > specs["max_size_mb"]:
            issues.append(f"File too large ({file_size_mb}MB > {specs['max_size_mb']}MB)")

        # Duration check
        if duration_seconds > specs["max_duration_seconds"]:
            issues.append(
                f"Video too long ({duration_seconds}s > {specs['max_duration_seconds']}s)"
            )

        # Platform-specific warnings
        if platform == "youtube" and duration_seconds <= 60:
            warnings.append("Video is ≤60s — will be treated as a YouTube Short")
        elif platform == "tiktok" and duration_seconds > 180:
            warnings.append(
                f"TikTok sweet spot is 30-180s. Current: {duration_seconds}s"
            )
        elif platform == "instagram":
            reels_max = specs.get("reels_max_seconds", 90)
            if duration_seconds > reels_max:
                warnings.append(
                    f"Duration exceeds Instagram Reels max ({reels_max}s)"
                )

        # Recommend transcode preset if needed
        transcode_needed = file_format != "mp4" or file_size_mb > specs["max_size_mb"]
        preset = self._recommend_preset(platform, duration_seconds)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "platform": platform,
            "transcode_needed": transcode_needed,
            "recommended_preset": preset,
        }

    def _execute_upload(self, task: Task) -> dict[str, Any]:
        """Execute an upload for a specific job.

        In production, this calls the actual platform API.
        Currently simulated with realistic response structure.
        """
        job_id = task.payload.get("job_id", "")

        job = next((j for j in self.upload_queue if j.job_id == job_id), None)
        if not job:
            return {"error": f"Job {job_id} not found", "status": "failed"}

        # Step 1: Verify OAuth credentials
        creds = self._get_credentials(job.platform)
        if not creds:
            job.error = f"No OAuth credentials for {job.platform}"
            return {
                "job_id": job.job_id,
                "status": "failed",
                "error": job.error,
                "auth_setup_required": True,
                "oauth_config": self.OAUTH_CONFIGS.get(job.platform, {}),
            }

        # Step 2: Mark as uploading
        job.status = "uploading"

        # Step 3: Execute platform-specific upload
        # In production, this would call the actual API:
        endpoint = self.API_ENDPOINTS.get(job.platform, {})
        result = self._simulate_platform_upload(job, endpoint)

        if result["success"]:
            job.status = "published"
            job.publish_url = result["publish_url"]
            job.completed_at = datetime.now(timezone.utc).isoformat()
            self.upload_queue.remove(job)
            self.completed_uploads.append(job)
        else:
            # Retry logic
            job.retries += 1
            if (
                job.retries < job.max_retries
                and result.get("error_type") in self.RETRY_CONFIG["retryable_errors"]
            ):
                job.status = "queued"  # re-queue for retry
                backoff = (
                    self.RETRY_CONFIG["backoff_base_seconds"]
                    * (self.RETRY_CONFIG["backoff_multiplier"] ** (job.retries - 1))
                )
                job.error = f"Retry {job.retries}/{job.max_retries} after {backoff}s"
            else:
                job.status = "failed"
                job.error = result.get("error", "Unknown upload error")
                self.upload_queue.remove(job)
                self.completed_uploads.append(job)

        return {
            "job_id": job.job_id,
            "platform": job.platform,
            "status": job.status,
            "publish_url": job.publish_url,
            "retries": job.retries,
            "error": job.error,
            "api_method": endpoint.get("method", "unknown"),
        }

    def _execute_all_queued(self, task: Task) -> dict[str, Any]:
        """Execute all queued uploads in sequence."""
        results = []
        # Snapshot the queue since it mutates during iteration
        jobs_to_process = list(self.upload_queue)

        for job in jobs_to_process:
            sub_task = Task(
                title=f"Upload to {job.platform}",
                created_by=self.name,
                payload={"action": "execute_upload", "job_id": job.job_id},
            )
            result = self._execute_upload(sub_task)
            results.append(result)

        published = sum(1 for r in results if r.get("status") == "published")
        failed = sum(1 for r in results if r.get("status") == "failed")

        return {
            "total_processed": len(results),
            "published": published,
            "failed": failed,
            "remaining_in_queue": len(self.upload_queue),
            "results": results,
        }

    def _get_status(self, task: Task) -> dict[str, Any]:
        """Get status of all upload jobs or a specific one."""
        job_id = task.payload.get("job_id")

        if job_id:
            all_jobs = self.upload_queue + self.completed_uploads
            job = next((j for j in all_jobs if j.job_id == job_id), None)
            if not job:
                return {"error": f"Job {job_id} not found"}
            return {
                "job_id": job.job_id,
                "status": job.status,
                "platform": job.platform,
                "publish_url": job.publish_url,
                "retries": job.retries,
                "error": job.error,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
            }

        return {
            "queued": len(self.upload_queue),
            "completed": len(self.completed_uploads),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "platform": j.platform,
                    "status": j.status,
                    "publish_url": j.publish_url,
                }
                for j in self.upload_queue + self.completed_uploads
            ],
        }

    def _schedule_upload(self, task: Task) -> dict[str, Any]:
        """Schedule an upload for a specific time with timezone awareness."""
        job_id = task.payload.get("job_id", "")
        scheduled_time = task.payload.get("scheduled_time", "")
        timezone_str = task.payload.get("timezone", "UTC")

        job = next((j for j in self.upload_queue if j.job_id == job_id), None)
        if not job:
            return {"error": f"Job {job_id} not found"}

        job.status = "scheduled"
        job.scheduled_time = scheduled_time

        # Check if platform supports scheduled publishing
        platform_supports_scheduling = {
            "youtube": True,   # YouTube supports scheduled uploads natively
            "tiktok": True,    # TikTok supports scheduled posting
            "instagram": True, # Instagram supports scheduling via Business Suite
        }

        return {
            "job_id": job.job_id,
            "status": "scheduled",
            "scheduled_time": scheduled_time,
            "timezone": timezone_str,
            "platform": job.platform,
            "native_scheduling": platform_supports_scheduling.get(job.platform, False),
        }

    def _get_transcode_config(self, task: Task) -> dict[str, Any]:
        """Get the FFmpeg transcode preset for a platform."""
        platform = task.payload.get("platform", "youtube")
        duration = task.payload.get("duration_seconds", 60)

        preset_key = self._recommend_preset(platform, duration)
        preset = self.TRANSCODE_PRESETS.get(preset_key, {})

        return {
            "platform": platform,
            "preset_key": preset_key,
            "preset": preset,
            "ffmpeg_command_template": self._build_ffmpeg_command(preset),
        }

    # ----- Internal Helpers -----

    def _platform_upload_metadata(
        self, platform: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Build platform-specific upload metadata."""
        if platform == "youtube":
            return {
                "category_id": metadata.get("category_id", "22"),  # 22 = People & Blogs
                "privacy_status": metadata.get("privacy_status", "public"),
                "made_for_kids": False,
                "self_declared_made_for_kids": False,
                "notify_subscribers": metadata.get("notify_subscribers", True),
                "embeddable": True,
                "license": "youtube",
                "recording_details": {},
            }
        elif platform == "tiktok":
            return {
                "privacy_level": metadata.get("privacy_level", "SELF_ONLY"),
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "brand_content_toggle": metadata.get("brand_content", False),
                "brand_organic_toggle": False,
            }
        elif platform == "instagram":
            return {
                "media_type": "REELS",
                "share_to_feed": metadata.get("share_to_feed", True),
                "caption": metadata.get("description", ""),
                "location_id": metadata.get("location_id"),
                "user_tags": metadata.get("user_tags", []),
            }
        return {}

    def _get_credentials(self, platform: str) -> OAuthCredentials | None:
        """Get or refresh OAuth credentials for a platform.

        In production, this would:
        1. Load tokens from secure storage (e.g., encrypted DB or vault)
        2. Check token expiry
        3. Refresh if expired using the refresh token
        4. Return valid credentials
        """
        if platform in self._credentials:
            return self._credentials[platform]

        oauth_config = self.OAUTH_CONFIGS.get(platform)
        if not oauth_config:
            return None

        # Simulated: in production, load from secure storage
        creds = OAuthCredentials(
            platform=platform,
            client_id_env=oauth_config["client_id_env"],
            client_secret_env=oauth_config["client_secret_env"],
            scopes=oauth_config["scopes"],
            # access_token would be loaded from storage
            access_token="[SIMULATED_TOKEN]",
        )
        self._credentials[platform] = creds
        return creds

    def _simulate_platform_upload(
        self, job: UploadJob, endpoint: dict[str, str]
    ) -> dict[str, Any]:
        """Simulate a platform upload. Replace with real API calls in production."""
        # In production:
        # if job.platform == "youtube":
        #     return self._youtube_resumable_upload(job, endpoint)
        # elif job.platform == "tiktok":
        #     return self._tiktok_chunked_upload(job, endpoint)
        # elif job.platform == "instagram":
        #     return self._instagram_container_publish(job, endpoint)

        publish_urls = {
            "youtube": f"https://youtube.com/shorts/{job.asset_id}",
            "tiktok": f"https://tiktok.com/@creator/video/{job.asset_id}",
            "instagram": f"https://instagram.com/reel/{job.asset_id}",
        }

        return {
            "success": True,
            "publish_url": publish_urls.get(
                job.platform, f"https://{job.platform}.com/video/{job.asset_id}"
            ),
            "platform_video_id": f"{job.platform}_{job.asset_id}",
            "upload_method": endpoint.get("method", "unknown"),
        }

    def _recommend_preset(self, platform: str, duration: int) -> str:
        """Recommend a transcode preset based on platform and duration."""
        if platform == "youtube":
            return "youtube_shorts" if duration <= 60 else "youtube_long"
        elif platform == "tiktok":
            return "tiktok"
        elif platform == "instagram":
            return "instagram_reels"
        return "tiktok"  # safe default for vertical video

    def _build_ffmpeg_command(self, preset: dict[str, Any]) -> str:
        """Build an FFmpeg command template from a preset."""
        if not preset:
            return ""
        args = " ".join(preset.get("ffmpeg_args", []))
        return f"ffmpeg -i INPUT {args} OUTPUT.mp4"
