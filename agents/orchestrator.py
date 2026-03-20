"""Orchestrator Agent — coordinates the entire content pipeline."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, Task, TaskStatus


class OrchestratorAgent(BaseAgent):
    """Central coordinator that manages the full content production pipeline.

    Responsibilities:
    - Receive high-level content requests and break them into agent tasks
    - Route tasks to the appropriate agent in the correct order
    - Track pipeline state from ideation through publishing
    - Handle failures, retries, and escalations
    - Coordinate feedback loops (Analytics → Ideation)
    - Provide pipeline-level status reporting
    """

    name = "orchestrator"
    role = "Pipeline Coordinator"
    description = (
        "Coordinates the full content pipeline — routes tasks between agents, "
        "manages workflow state, and ensures end-to-end execution."
    )

    # The standard pipeline order
    PIPELINE_STAGES = [
        "ideation",
        "content_creation",
        "marketing_optimization",
        "qa_review",
        "upload",
        "analytics_tracking",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agents: dict[str, BaseAgent] = {}
        self.pipelines: dict[str, list[Task]] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent

    def register_team(self, agents: list[BaseAgent]) -> None:
        """Register a list of agents."""
        for agent in agents:
            self.register_agent(agent)

    def execute(self, task: Task) -> dict[str, Any]:
        action = task.payload.get("action", "run_pipeline")
        if action == "run_pipeline":
            return self._run_pipeline(task)
        elif action == "route_task":
            return self._route_task(task)
        elif action == "pipeline_status":
            return self._pipeline_status(task)
        elif action == "feedback_loop":
            return self._feedback_loop(task)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _run_pipeline(self, task: Task) -> dict[str, Any]:
        """Execute the full content pipeline for a given request."""
        niche = task.payload.get("niche", "general")
        topic = task.payload.get("topic")
        pipeline_id = task.id
        pipeline_tasks: list[Task] = []

        # Stage 1: Ideation
        ideation_task = Task(
            title="Generate content idea",
            created_by=self.name,
            assigned_to="ideation",
            payload={
                "action": "create_brief" if topic else "generate_ideas",
                "niche": niche,
                "topic": topic or "",
                "count": 1,
            },
        )
        ideation_result = self._dispatch("ideation", ideation_task)
        pipeline_tasks.append(ideation_task)

        if ideation_result is None:
            return {"pipeline_id": pipeline_id, "status": "failed", "stage": "ideation"}

        # Stage 2: Content Creation
        brief = ideation_result.get("brief", ideation_result)
        creation_task = Task(
            title="Create video asset",
            created_by=self.name,
            assigned_to="content_creator",
            payload={"action": "create_asset", "brief": brief},
        )
        creation_result = self._dispatch("content_creator", creation_task)
        pipeline_tasks.append(creation_task)

        if creation_result is None:
            return {"pipeline_id": pipeline_id, "status": "failed", "stage": "content_creation"}

        asset = creation_result.get("asset", {})

        # Stage 3: Marketing Optimization
        marketing_task = Task(
            title="Optimize metadata",
            created_by=self.name,
            assigned_to="marketing",
            payload={
                "action": "optimize_metadata",
                "platform": "youtube",
                "title": asset.get("title", ""),
                "description": asset.get("description", ""),
                "tags": asset.get("tags", []),
                "keywords": brief.get("keywords", []),
            },
        )
        marketing_result = self._dispatch("marketing", marketing_task)
        pipeline_tasks.append(marketing_task)

        # Stage 4: QA Review
        qa_task = Task(
            title="QA review",
            created_by=self.name,
            assigned_to="qa",
            payload={"action": "full_review", "asset": asset},
        )
        qa_result = self._dispatch("qa", qa_task)
        pipeline_tasks.append(qa_task)

        if qa_result and not qa_result.get("approved", False):
            return {
                "pipeline_id": pipeline_id,
                "status": "blocked_by_qa",
                "qa_issues": qa_result.get("issues", []),
                "stages_completed": len(pipeline_tasks),
            }

        # Stage 5: Upload
        upload_task = Task(
            title="Queue upload",
            created_by=self.name,
            assigned_to="upload_manager",
            payload={
                "action": "queue_upload",
                "asset_id": asset.get("asset_id", ""),
                "platforms": brief.get("target_platforms", ["youtube"]),
            },
        )
        upload_result = self._dispatch("upload_manager", upload_task)
        pipeline_tasks.append(upload_task)

        self.pipelines[pipeline_id] = pipeline_tasks

        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "stages_completed": len(pipeline_tasks),
            "asset_id": asset.get("asset_id", ""),
            "upload_jobs": upload_result.get("jobs", []) if upload_result else [],
        }

    def _route_task(self, task: Task) -> dict[str, Any]:
        """Route a single task to the specified agent."""
        target_agent = task.payload.get("target_agent", "")
        inner_payload = task.payload.get("inner_payload", {})

        inner_task = Task(
            title=task.title,
            created_by=self.name,
            assigned_to=target_agent,
            payload=inner_payload,
        )

        result = self._dispatch(target_agent, inner_task)
        return {"routed_to": target_agent, "result": result}

    def _pipeline_status(self, task: Task) -> dict[str, Any]:
        """Report status of a pipeline."""
        pipeline_id = task.payload.get("pipeline_id", "")
        tasks = self.pipelines.get(pipeline_id, [])

        return {
            "pipeline_id": pipeline_id,
            "total_tasks": len(tasks),
            "tasks": [t.to_dict() for t in tasks],
        }

    def _feedback_loop(self, task: Task) -> dict[str, Any]:
        """Run the analytics → ideation feedback loop."""
        # Get feedback from analytics
        analytics_task = Task(
            title="Generate ideation feedback",
            created_by=self.name,
            assigned_to="analytics",
            payload={"action": "feedback_for_ideation"},
        )
        analytics_result = self._dispatch("analytics", analytics_task)

        if not analytics_result:
            return {"status": "no_analytics_data"}

        # Feed it to ideation
        feedback_task = Task(
            title="Incorporate analytics feedback",
            created_by=self.name,
            assigned_to="ideation",
            payload={
                "action": "incorporate_feedback",
                "performance_data": analytics_result.get("performance_data", {}),
            },
        )
        ideation_result = self._dispatch("ideation", feedback_task)

        return {
            "feedback_delivered": True,
            "analytics_summary": analytics_result,
            "ideation_adjustments": ideation_result,
        }

    def _dispatch(self, agent_name: str, task: Task) -> dict[str, Any] | None:
        """Send a task to a registered agent and return the result."""
        agent = self.agents.get(agent_name)
        if not agent:
            task.fail(f"Agent '{agent_name}' not registered")
            return None

        completed_task = agent.receive_task(task)
        if completed_task.status == TaskStatus.COMPLETED:
            return completed_task.result
        return None
