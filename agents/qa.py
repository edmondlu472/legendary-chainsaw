"""QA Agent — reviews content before publishing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base import BaseAgent, Task


@dataclass
class QAReport:
    """Result of a QA review on a video asset."""

    asset_id: str = ""
    approved: bool = False
    issues: list[dict[str, str]] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    reviewer: str = "qa_agent"


class QAAgent(BaseAgent):
    """Reviews all content and metadata before it is published.

    Responsibilities:
    - Check content against platform community guidelines
    - Verify metadata completeness (title, description, tags, thumbnail)
    - Validate technical specs (format, resolution, duration limits)
    - Flag potential copyright or policy violations
    - Ensure brand voice and quality consistency
    - Gate the publishing pipeline — nothing ships without QA approval
    """

    name = "qa"
    role = "Quality Assurance Reviewer"
    description = (
        "Reviews all content for policy compliance, metadata completeness, "
        "technical quality, and brand consistency before publishing."
    )

    # Words/phrases that may trigger platform policy issues
    FLAGGED_TERMS = [
        "guaranteed income",
        "get rich quick",
        "miracle cure",
        "act now",
        "limited time only",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.review_history: list[QAReport] = []

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "full_review")
        if action == "full_review":
            return self._full_review(task)
        elif action == "metadata_check":
            return self._metadata_check(task)
        elif action == "policy_check":
            return self._policy_check(task)
        elif action == "technical_check":
            return self._technical_check(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _full_review(self, task: Task) -> dict[str, Any]:
        """Run all checks on a video asset."""
        asset = task.payload.get("asset", {})
        report = QAReport(asset_id=asset.get("asset_id", ""))

        # Metadata completeness
        metadata_result = self._check_metadata(asset)
        report.checks_passed.extend(metadata_result["passed"])
        report.checks_failed.extend(metadata_result["failed"])
        report.issues.extend(metadata_result["issues"])

        # Policy compliance
        policy_result = self._check_policy(asset)
        report.checks_passed.extend(policy_result["passed"])
        report.checks_failed.extend(policy_result["failed"])
        report.issues.extend(policy_result["issues"])

        # Technical validation
        tech_result = self._check_technical(asset)
        report.checks_passed.extend(tech_result["passed"])
        report.checks_failed.extend(tech_result["failed"])
        report.issues.extend(tech_result["issues"])

        report.approved = len(report.checks_failed) == 0
        self.review_history.append(report)

        return {
            "asset_id": report.asset_id,
            "approved": report.approved,
            "checks_passed": report.checks_passed,
            "checks_failed": report.checks_failed,
            "issues": report.issues,
            "gate": "pass" if report.approved else "blocked",
        }

    def _metadata_check(self, task: Task) -> dict[str, Any]:
        """Check metadata completeness only."""
        asset = task.payload.get("asset", {})
        result = self._check_metadata(asset)
        return {
            "passed": result["passed"],
            "failed": result["failed"],
            "issues": result["issues"],
        }

    def _policy_check(self, task: Task) -> dict[str, Any]:
        """Check policy compliance only."""
        asset = task.payload.get("asset", {})
        result = self._check_policy(asset)
        return {
            "passed": result["passed"],
            "failed": result["failed"],
            "issues": result["issues"],
        }

    def _technical_check(self, task: Task) -> dict[str, Any]:
        """Check technical specifications only."""
        asset = task.payload.get("asset", {})
        result = self._check_technical(asset)
        return {
            "passed": result["passed"],
            "failed": result["failed"],
            "issues": result["issues"],
        }

    def _check_metadata(self, asset: dict[str, Any]) -> dict[str, Any]:
        """Verify all required metadata fields are present and valid."""
        passed = []
        failed = []
        issues = []

        required_fields = ["title", "description", "tags", "script"]
        for f in required_fields:
            if asset.get(f):
                passed.append(f"metadata_{f}_present")
            else:
                failed.append(f"metadata_{f}_missing")
                issues.append({"severity": "error", "message": f"Missing required field: {f}"})

        title = asset.get("title", "")
        if title and len(title) > 100:
            failed.append("metadata_title_too_long")
            issues.append({"severity": "warning", "message": f"Title is {len(title)} chars (max 100)"})
        elif title:
            passed.append("metadata_title_length_ok")

        return {"passed": passed, "failed": failed, "issues": issues}

    def _check_policy(self, asset: dict[str, Any]) -> dict[str, Any]:
        """Check for potential policy violations."""
        passed = []
        failed = []
        issues = []

        text_to_check = " ".join([
            asset.get("title", ""),
            asset.get("description", ""),
            asset.get("script", ""),
            asset.get("voiceover_text", ""),
        ]).lower()

        flagged = [term for term in self.FLAGGED_TERMS if term in text_to_check]

        if flagged:
            failed.append("policy_flagged_terms")
            issues.append({
                "severity": "warning",
                "message": f"Flagged terms found: {flagged}",
            })
        else:
            passed.append("policy_no_flagged_terms")

        passed.append("policy_basic_check_done")
        return {"passed": passed, "failed": failed, "issues": issues}

    def _check_technical(self, asset: dict[str, Any]) -> dict[str, Any]:
        """Validate technical specifications."""
        passed = []
        failed = []
        issues = []

        duration = asset.get("duration_seconds", 0)
        if duration <= 0:
            failed.append("tech_invalid_duration")
            issues.append({"severity": "error", "message": "Duration must be > 0"})
        elif duration > 3600:
            failed.append("tech_duration_too_long")
            issues.append({"severity": "warning", "message": "Video exceeds 1 hour"})
        else:
            passed.append("tech_duration_ok")

        if asset.get("video_prompt") or asset.get("file_path"):
            passed.append("tech_video_source_present")
        else:
            failed.append("tech_no_video_source")
            issues.append({"severity": "error", "message": "No video file or generation prompt"})

        return {"passed": passed, "failed": failed, "issues": issues}
