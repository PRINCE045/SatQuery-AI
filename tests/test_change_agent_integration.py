from pathlib import Path

from agent.orchestrator import SatQueryAgent


SAMPLE_IMAGE = Path("data/sample/sentinel2_sample.tif")


def test_change_analysis_is_routed_to_dcff(monkeypatch):
    assert SAMPLE_IMAGE.is_file()

    monkeypatch.setenv(
        "SATQUERY_CHANGE_BACKEND",
        "unavailable",
    )

    agent = SatQueryAgent()

    result = agent.run(
        query="What changed between these two images?",
        inputs={
            "image_t1": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
            "image_t2": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
        },
    )

    assert result.plan["task"] == "bi_temporal_change_analysis"
    assert result.plan["selected_tool"] == "change_analysis"

    assert result.result["tool_name"] == "change_analysis"

    assert result.result["outputs"]["model"] == "DCFF-Net"
    assert result.result["outputs"]["status"] == "unavailable"

    assert result.result["confidence"] == 0.0

    execution_steps = [
        step
        for step in result.execution_trace
        if step["stage"] == "tool_execution"
    ]

    assert len(execution_steps) == 1
    assert execution_steps[0]["tool"] == "change_analysis"
    assert execution_steps[0]["status"] == "unavailable"

def test_change_agent_preserves_execution_metadata(
    monkeypatch,
):

    assert SAMPLE_IMAGE.is_file()

    monkeypatch.setenv(
        "SATQUERY_CHANGE_BACKEND",
        "unavailable",
    )

    agent = SatQueryAgent()

    result = agent.run(
        query="What changed between these two images?",
        inputs={
            "image_t1": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
            "image_t2": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
        },
    )

    execution_steps = [
        step
        for step in result.execution_trace
        if step["stage"] == "tool_execution"
    ]

    assert len(execution_steps) == 1

    step = execution_steps[0]

    assert step["model"] == "DCFF-Net"
    assert step["backend"] == "unavailable"
    assert isinstance(
        step["runtime_seconds"],
        float,
    )