"""Content Creator Agent — produces video assets from content briefs using AI tools."""

from __future__ import annotations

import uuid
from typing import Any

from agents.base import (
    BaseAgent,
    DEFAULT_IMAGE_GEN,
    DEFAULT_TTS,
    DEFAULT_VIDEO_GEN,
    ImageGenConfig,
    LLMConfig,
    PLATFORM_SPECS,
    Task,
    TTSConfig,
    VideoAsset,
    VideoGenConfig,
)


class ContentCreatorAgent(BaseAgent):
    """Takes content briefs and produces all assets needed for a video.

    Responsibilities:
    - Generate video scripts from content briefs via Claude LLM
    - Create AI video generation prompts (Kling AI / Runway / Pika)
    - Write voiceover / narration text and send to ElevenLabs TTS
    - Generate thumbnail prompts for DALL-E 3 / Flux
    - Produce platform-specific asset variants (YouTube vs TikTok vs Instagram)
    - Package everything into VideoAssets for downstream agents

    Tools employed:
    - Claude API       — script writing, title/description copywriting
    - Kling AI / Runway — AI video generation from text prompts
    - ElevenLabs       — text-to-speech voiceover generation
    - DALL-E 3 / Flux  — thumbnail and cover image generation
    - FFmpeg           — video assembly, audio overlay, format conversion
    """

    name = "content_creator"
    role = "Content Producer"
    description = (
        "Transforms content briefs into complete video asset packages "
        "with platform-specific variants for YouTube, TikTok, and Instagram."
    )

    # ----- LLM Prompt Templates -----

    SCRIPT_SYSTEM_PROMPT = (
        "You are an expert video scriptwriter who creates engaging scripts for "
        "YouTube, TikTok, and Instagram Reels. You understand pacing, hooks, "
        "retention patterns, and platform-specific audience expectations.\n\n"
        "Rules:\n"
        "- Always start with a powerful hook (first 3 seconds)\n"
        "- Use short, punchy sentences for short-form (<60s)\n"
        "- Include pattern interrupts every 15-20 seconds\n"
        "- End with a clear CTA appropriate to the platform\n"
        "- Mark visual cues with [VISUAL: description]\n"
        "- Mark text overlays with [TEXT: content]\n"
        "- Mark transitions with [CUT] or [TRANSITION: type]"
    )

    SCRIPT_PROMPT_TEMPLATES = {
        "educational": (
            "Write a {duration}-second educational video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → Problem → Key insight → Evidence → Takeaway → CTA\n"
            "Word count target: ~{word_count} words ({wpm} words/min pacing)\n"
            "Keywords to weave in naturally: {keywords}"
        ),
        "entertainment": (
            "Write a {duration}-second entertaining video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → Setup → Escalation → Payoff → CTA\n"
            "Tone: Fun, energetic, shareable. Think reaction-worthy.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
        "tutorial": (
            "Write a {duration}-second tutorial video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → What we're building → Step 1 → Step 2 → Step 3 → Result → CTA\n"
            "Include [SCREEN: description] markers for screen recordings.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
        "listicle": (
            "Write a {duration}-second listicle video script: '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → Item N (start from last) → ... → Item 1 (best) → CTA\n"
            "Each item: 1 sentence intro + key point + [VISUAL] cue.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
        "storytelling": (
            "Write a {duration}-second storytelling video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → Context → Rising action → Climax → Resolution → CTA\n"
            "Use conversational tone. Build suspense. Make it personal.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
        "news_recap": (
            "Write a {duration}-second news recap video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "Structure: Hook → What happened → Why it matters → What's next → CTA\n"
            "Be factual but engaging. Use urgency without being clickbait.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
        "shorts_hook": (
            "Write a {duration}-second short-form video script about '{topic}'.\n"
            "Angle: {angle}\nTarget audience: {audience}\nHook: {hook}\n\n"
            "This is for YouTube Shorts / TikTok / Instagram Reels.\n"
            "Structure: Instant hook (1s) → Fast value (3-5 points) → Loop/CTA\n"
            "Keep it FAST. No fluff. Every second counts.\n"
            "Word count target: ~{word_count} words\nKeywords: {keywords}"
        ),
    }

    # ----- Video Generation Prompt Templates -----

    VIDEO_GEN_TEMPLATES = {
        "kling": {
            "template": (
                "Create a {duration}-second {aspect_ratio} video.\n"
                "Scene: {scene_description}\n"
                "Style: {visual_style}\n"
                "Camera: {camera_motion}\n"
                "Mood: {mood}\n"
                "Quality: cinematic, professional, high-detail"
            ),
            "camera_options": [
                "static wide shot", "slow zoom in", "smooth pan left to right",
                "dolly forward", "aerial establishing shot", "close-up with shallow DOF",
            ],
        },
        "runway": {
            "template": (
                "{scene_description}. {visual_style} aesthetic. "
                "{camera_motion}. {mood} atmosphere. "
                "Professional quality, {aspect_ratio} vertical video."
            ),
            "camera_options": [
                "static", "slow push in", "tracking shot",
                "orbit", "crane up", "handheld",
            ],
        },
    }

    # ----- Thumbnail Prompt Templates -----

    THUMBNAIL_TEMPLATES = {
        "youtube": (
            "Professional YouTube thumbnail, 1280x720, {style} style.\n"
            "Subject: {subject}\n"
            "Text overlay space on the right third for '{title_text}'.\n"
            "Colors: high contrast, saturated. Facial expression: {expression}.\n"
            "Background: {background}. Clean, no clutter. Eye-catching at small sizes."
        ),
        "tiktok": (
            "TikTok video cover, 1080x1920 vertical.\n"
            "Subject: {subject}\n"
            "Bold text overlay: '{title_text}'.\n"
            "Style: trendy, vibrant, scroll-stopping.\n"
            "Background: {background}."
        ),
        "instagram": (
            "Instagram Reels cover, 1080x1920 vertical.\n"
            "Subject: {subject}\n"
            "Aesthetic: clean, polished, Instagram-native.\n"
            "Text overlay: '{title_text}'.\n"
            "Color palette: {color_palette}.\n"
            "Background: {background}."
        ),
    }

    # ----- Platform CTA Templates -----

    PLATFORM_CTAS = {
        "youtube": "If you found this helpful, smash that like button and subscribe for more!",
        "tiktok": "Follow for part 2! Drop a comment if you want more.",
        "instagram": "Save this for later and share it with someone who needs to see this!",
    }

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        video_gen_config: VideoGenConfig | None = None,
        image_gen_config: ImageGenConfig | None = None,
        tts_config: TTSConfig | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config)
        self.video_gen = video_gen_config or DEFAULT_VIDEO_GEN
        self.image_gen = image_gen_config or DEFAULT_IMAGE_GEN
        self.tts = tts_config or DEFAULT_TTS
        self.assets: list[VideoAsset] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "create_asset")
        if action == "create_asset":
            return self._create_asset(task)
        elif action == "create_platform_variants":
            return self._create_platform_variants(task)
        elif action == "generate_script":
            return self._generate_script(task)
        elif action == "generate_thumbnail":
            return self._generate_thumbnail(task)
        elif action == "generate_voiceover":
            return self._generate_voiceover(task)
        elif action == "revise_asset":
            return self._revise_asset(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ----- Core Actions -----

    def _create_asset(self, task: Task) -> dict[str, Any]:
        """Create a full video asset package from a content brief."""
        brief = task.payload.get("brief", {})

        topic = brief.get("topic", "Untitled")
        angle = brief.get("angle", "")
        audience = brief.get("target_audience", "general")
        keywords = brief.get("keywords", [])
        duration = brief.get("estimated_duration_seconds", 60)
        style = brief.get("content_style", "educational")
        hook = brief.get("hook", "")
        platforms = brief.get("target_platforms", ["youtube", "tiktok", "instagram"])

        # 1. Generate script via LLM
        script = self._build_script(topic, angle, audience, keywords, duration, style, hook)

        # 2. Build video generation prompt
        video_prompt = self._build_video_prompt_for_provider(
            topic=topic, style=style, duration=duration,
        )

        # 3. Build voiceover text (cleaned script without visual cues)
        voiceover_text = self._build_voiceover_text(script)

        # 4. Build thumbnail prompt (default to YouTube as primary)
        thumbnail_prompt = self._build_thumbnail_prompt(
            platform="youtube", topic=topic, title_text=topic[:40],
        )

        # 5. Generate platform-appropriate description
        description = self._build_description(topic, angle, keywords, "youtube")

        asset = VideoAsset(
            asset_id=uuid.uuid4().hex[:12],
            brief_id=brief.get("idea_id", ""),
            platform=platforms[0] if platforms else "youtube",
            script=script,
            video_prompt=video_prompt,
            voiceover_text=voiceover_text,
            thumbnail_prompt=thumbnail_prompt,
            title=topic,
            description=description,
            tags=keywords[:15] if keywords else [topic.lower().replace(" ", "-")],
            hashtags=self._generate_hashtags(keywords, platforms),
            duration_seconds=duration,
            aspect_ratio="9:16" if duration <= 60 else "16:9",
            status="draft",
            ai_disclosure=True,
        )

        self.assets.append(asset)
        return {"asset": asset.__dict__, "status": "asset_created"}

    def _create_platform_variants(self, task: Task) -> dict[str, Any]:
        """Create platform-specific variants of a single piece of content."""
        brief = task.payload.get("brief", {})
        platforms = task.payload.get(
            "platforms", brief.get("target_platforms", ["youtube", "tiktok", "instagram"])
        )
        topic = brief.get("topic", "Untitled")
        angle = brief.get("angle", "")
        audience = brief.get("target_audience", "general")
        keywords = brief.get("keywords", [])
        style = brief.get("content_style", "educational")
        hook = brief.get("hook", "")

        variants = []
        for platform in platforms:
            specs = PLATFORM_SPECS.get(platform, {})
            duration = self._platform_optimal_duration(platform, brief)
            aspect = list(specs.get("aspect_ratios", {}).values())[0] if specs.get("aspect_ratios") else "9:16"

            script = self._build_script(
                topic, angle, audience, keywords, duration, style, hook, platform
            )
            video_prompt = self._build_video_prompt_for_provider(
                topic=topic, style=style, duration=duration, aspect_ratio=aspect,
            )
            voiceover_text = self._build_voiceover_text(script)
            thumbnail_prompt = self._build_thumbnail_prompt(
                platform=platform, topic=topic, title_text=topic[:40],
            )
            description = self._build_description(topic, angle, keywords, platform)

            asset = VideoAsset(
                asset_id=uuid.uuid4().hex[:12],
                brief_id=brief.get("idea_id", ""),
                platform=platform,
                script=script,
                video_prompt=video_prompt,
                voiceover_text=voiceover_text,
                thumbnail_prompt=thumbnail_prompt,
                title=topic if platform != "instagram" else "",
                description=description,
                tags=keywords[:15] if platform == "youtube" else [],
                hashtags=self._generate_hashtags(keywords, [platform]),
                duration_seconds=duration,
                aspect_ratio=aspect,
                status="draft",
                ai_disclosure=True,
            )
            self.assets.append(asset)
            variants.append(asset.__dict__)

        return {
            "variants": variants,
            "platforms": platforms,
            "count": len(variants),
        }

    def _generate_script(self, task: Task) -> dict[str, Any]:
        """Generate just the script portion of a video."""
        topic = task.payload.get("topic", "")
        style = task.payload.get("style", "educational")
        duration = task.payload.get("duration_seconds", 60)
        audience = task.payload.get("target_audience", "general")
        platform = task.payload.get("platform", "youtube")
        hook = task.payload.get("hook", "")
        keywords = task.payload.get("keywords", [])

        script = self._build_script(
            topic, "", audience, keywords, duration, style, hook, platform
        )

        return {
            "script": script,
            "style": style,
            "platform": platform,
            "estimated_duration": duration,
            "word_count": len(script.split()),
        }

    def _generate_thumbnail(self, task: Task) -> dict[str, Any]:
        """Generate thumbnail prompts for each target platform."""
        title = task.payload.get("title", "")
        platforms = task.payload.get("platforms", ["youtube"])

        thumbnails = {}
        for platform in platforms:
            prompt = self._build_thumbnail_prompt(
                platform=platform,
                topic=title,
                title_text=title[:40],
            )
            specs = PLATFORM_SPECS.get(platform, {})
            thumbnails[platform] = {
                "prompt": prompt,
                "size": specs.get("thumbnail_size", "1280x720"),
                "provider": self.image_gen.provider.value,
                "required": specs.get("thumbnail_required", False),
            }

        return {"thumbnails": thumbnails}

    def _generate_voiceover(self, task: Task) -> dict[str, Any]:
        """Generate voiceover config for TTS service."""
        text = task.payload.get("voiceover_text", "")
        duration = task.payload.get("duration_seconds", 60)

        # Calculate optimal pacing
        word_count = len(text.split())
        estimated_duration = word_count / (self.tts.words_per_minute / 60)

        return {
            "voiceover_text": text,
            "word_count": word_count,
            "estimated_duration_seconds": round(estimated_duration, 1),
            "target_duration_seconds": duration,
            "tts_config": {
                "provider": self.tts.provider.value,
                "voice_id": self.tts.voice_id,
                "model": self.tts.model,
                "stability": self.tts.stability,
                "similarity_boost": self.tts.similarity_boost,
                "output_format": self.tts.output_format,
            },
            "pacing_ok": abs(estimated_duration - duration) < duration * 0.15,
        }

    def _revise_asset(self, task: Task) -> dict[str, Any]:
        """Revise an existing asset based on QA or analytics feedback."""
        asset_id = task.payload.get("asset_id", "")
        revisions = task.payload.get("revisions", {})
        feedback = task.payload.get("feedback", "")

        # Use LLM to apply revisions intelligently
        if feedback:
            revision_prompt = (
                f"Revise this content based on the following feedback:\n"
                f"Feedback: {feedback}\n"
                f"Current title: {revisions.get('title', 'N/A')}\n"
                f"Current description: {revisions.get('description', 'N/A')}\n\n"
                f"Provide revised title and description that address the feedback."
            )
            _llm_response = self._llm_generate(
                self.SCRIPT_SYSTEM_PROMPT, revision_prompt
            )

        return {
            "asset_id": asset_id,
            "revisions_applied": list(revisions.keys()),
            "status": "revised",
            "llm_assisted": bool(feedback),
        }

    # ----- Internal Builders -----

    def _build_script(
        self,
        topic: str,
        angle: str,
        audience: str,
        keywords: list[str],
        duration: int,
        style: str,
        hook: str,
        platform: str = "youtube",
    ) -> str:
        """Generate a video script using the LLM with style-specific templates."""
        wpm = self.tts.words_per_minute
        word_count = int(duration * (wpm / 60))

        template = self.SCRIPT_PROMPT_TEMPLATES.get(
            style, self.SCRIPT_PROMPT_TEMPLATES["educational"]
        )

        user_prompt = template.format(
            duration=duration,
            topic=topic,
            angle=angle or "general overview",
            audience=audience,
            hook=hook or f"Here's something about {topic} you need to know...",
            word_count=word_count,
            wpm=wpm,
            keywords=", ".join(keywords[:8]) if keywords else topic,
        )

        # Add platform-specific CTA instruction
        cta = self.PLATFORM_CTAS.get(platform, self.PLATFORM_CTAS["youtube"])
        user_prompt += f"\n\nEnd with this CTA style (adapt naturally): {cta}"

        script = self._llm_generate(self.SCRIPT_SYSTEM_PROMPT, user_prompt)
        return script

    def _build_video_prompt_for_provider(
        self,
        topic: str,
        style: str,
        duration: int,
        aspect_ratio: str = "9:16",
    ) -> str:
        """Build a video generation prompt tuned for the configured provider."""
        provider_key = self.video_gen.provider.value
        template_config = self.VIDEO_GEN_TEMPLATES.get(
            provider_key, self.VIDEO_GEN_TEMPLATES["kling"]
        )

        visual_styles = {
            "educational": "clean, modern, professional with infographic elements",
            "entertainment": "dynamic, colorful, fast-paced with bold visuals",
            "tutorial": "clean screen recording style with step-by-step visuals",
            "listicle": "numbered segments with distinct visual transitions",
            "storytelling": "cinematic, atmospheric, emotionally engaging",
            "news_recap": "news broadcast style, clean graphics, urgent tone",
            "shorts_hook": "eye-catching, fast cuts, bold text overlays",
        }

        moods = {
            "educational": "informative and trustworthy",
            "entertainment": "exciting and fun",
            "tutorial": "calm and focused",
            "listicle": "energetic and organized",
            "storytelling": "dramatic and immersive",
            "news_recap": "urgent and professional",
            "shorts_hook": "high-energy and attention-grabbing",
        }

        camera = template_config["camera_options"][0]  # default to first option

        prompt = template_config["template"].format(
            duration=min(duration, self.video_gen.max_duration_seconds),
            aspect_ratio=aspect_ratio,
            scene_description=f"Visual representation of: {topic}",
            visual_style=visual_styles.get(style, visual_styles["educational"]),
            camera_motion=camera,
            mood=moods.get(style, moods["educational"]),
        )

        return prompt

    def _build_thumbnail_prompt(
        self,
        platform: str,
        topic: str,
        title_text: str,
    ) -> str:
        """Build a thumbnail/cover image generation prompt for the target platform."""
        template = self.THUMBNAIL_TEMPLATES.get(
            platform, self.THUMBNAIL_TEMPLATES["youtube"]
        )

        return template.format(
            style="bold and colorful",
            subject=f"Visual metaphor for: {topic}",
            title_text=title_text,
            expression="surprised or excited" if platform == "youtube" else "confident",
            background="gradient or abstract tech pattern",
            color_palette="warm earth tones" if platform == "instagram" else "vibrant",
        )

    def _build_voiceover_text(self, script: str) -> str:
        """Strip visual cues and stage directions from script for TTS."""
        lines = []
        for line in script.split("\n"):
            stripped = line.strip()
            # Remove visual cues, transitions, and stage directions
            if stripped.startswith(("[VISUAL:", "[TEXT:", "[CUT]", "[TRANSITION:", "[SCREEN:")):
                continue
            if stripped:
                lines.append(stripped)
        return " ".join(lines)

    def _build_description(
        self,
        topic: str,
        angle: str,
        keywords: list[str],
        platform: str,
    ) -> str:
        """Build a platform-appropriate video description."""
        specs = PLATFORM_SPECS.get(platform, {})
        max_len = specs.get("metadata_limits", {}).get("description", 2200)

        if platform == "youtube":
            desc = (
                f"{angle if angle else f'Everything you need to know about {topic}.'}\n\n"
                f"In this video:\n"
                f"- Key insights about {topic}\n"
                f"- Practical takeaways you can use today\n"
                f"- Why this matters right now\n\n"
                f"🔔 Subscribe for more content like this!\n\n"
                f"Keywords: {', '.join(keywords[:10])}\n\n"
                f"#{'  #'.join(kw.replace(' ', '') for kw in keywords[:5])}"
            )
        elif platform == "tiktok":
            hashtags = " ".join(f"#{kw.replace(' ', '')}" for kw in keywords[:8])
            desc = f"{topic} {hashtags} #fyp #viral"
        elif platform == "instagram":
            hashtags = " ".join(f"#{kw.replace(' ', '')}" for kw in keywords[:15])
            desc = (
                f"{angle if angle else topic}\n\n"
                f"Save this for later! 📌\n\n"
                f"{hashtags}"
            )
        else:
            desc = f"{angle if angle else topic}"

        return desc[:max_len]

    def _generate_hashtags(
        self, keywords: list[str], platforms: list[str]
    ) -> list[str]:
        """Generate platform-appropriate hashtags from keywords."""
        base = [f"#{kw.replace(' ', '').lower()}" for kw in keywords[:10]]

        platform_tags = {
            "tiktok": ["#fyp", "#viral", "#foryou", "#trending"],
            "instagram": ["#reels", "#explore", "#instagood"],
            "youtube": ["#shorts"],
        }

        tags = list(base)
        for platform in platforms:
            tags.extend(platform_tags.get(platform, []))

        # Instagram has a 30 hashtag limit
        if "instagram" in platforms:
            return tags[:30]
        return tags[:20]

    def _platform_optimal_duration(
        self, platform: str, brief: dict[str, Any]
    ) -> int:
        """Get optimal duration for a specific platform."""
        requested = brief.get("estimated_duration_seconds", 60)
        specs = PLATFORM_SPECS.get(platform, {})

        if platform == "youtube":
            shorts_max = specs.get("shorts_max_seconds", 60)
            if requested <= shorts_max:
                return min(requested, shorts_max)
            return requested  # long-form
        elif platform == "tiktok":
            return min(requested, specs.get("max_duration_seconds", 600))
        elif platform == "instagram":
            reels_max = specs.get("reels_max_seconds", 90)
            return min(requested, reels_max)
        return requested
