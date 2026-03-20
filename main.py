"""Main entry point — assembles the agent team and runs a demo pipeline."""

from agents import (
    AnalyticsAgent,
    ContentCreatorAgent,
    IdeationAgent,
    MarketingAgent,
    OrchestratorAgent,
    QAAgent,
    UploadManagerAgent,
)
from agents.base import Task


def build_team() -> OrchestratorAgent:
    """Assemble the full agent team under the orchestrator."""
    orchestrator = OrchestratorAgent()
    orchestrator.register_team([
        IdeationAgent(),
        ContentCreatorAgent(),
        MarketingAgent(),
        UploadManagerAgent(),
        AnalyticsAgent(),
        QAAgent(),
    ])
    return orchestrator


def run_pipeline(orchestrator: OrchestratorAgent, topic: str, niche: str = "general") -> dict:
    """Run the full content pipeline for a given topic."""
    task = Task(
        title=f"Full pipeline: {topic}",
        created_by="user",
        payload={
            "action": "run_pipeline",
            "topic": topic,
            "niche": niche,
        },
    )
    result_task = orchestrator.receive_task(task)
    return result_task.result


def main() -> None:
    orchestrator = build_team()

    print("=== Agent Team ===")
    for name, agent in orchestrator.agents.items():
        info = agent.get_capabilities()
        print(f"  [{info['name']}] {info['role']}")
        print(f"    {info['description']}")
        print()

    print("=== Running Demo Pipeline ===")
    result = run_pipeline(
        orchestrator,
        topic="5 AI Tools That Will Change How You Work in 2026",
        niche="tech/productivity",
    )

    print(f"Pipeline status: {result.get('status')}")
    print(f"Stages completed: {result.get('stages_completed')}")
    print(f"Asset ID: {result.get('asset_id')}")

    if result.get("status") == "blocked_by_qa":
        print(f"QA Issues: {result.get('qa_issues')}")

    if result.get("upload_jobs"):
        print("Upload jobs:")
        for job in result["upload_jobs"]:
            print(f"  {job['platform']}: {job['status']}")


if __name__ == "__main__":
    main()
