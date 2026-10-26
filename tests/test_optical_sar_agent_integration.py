from pathlib import Path

from agent.orchestrator import SatQueryAgent


SAMPLE_IMAGE = Path("data/sample/sentinel2_sample.tif")


def test_optical_sar_analysis_reaches_mmovseg_adapter(monkeypatch):
    assert SAMPLE_IMAGE.is_file()

    monkeypatch.setenv(
        "SATQUERY_OPTICAL_SAR_BACKEND",
        "unavailable",
    )

    agent = SatQueryAgent()

    result = agent.run(
        query="Analyze the optical and SAR images together.",
        inputs={
            "optical_image": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
            "sar_image": {
                "path": str(SAMPLE_IMAGE),
                "modality": "sar",
            },
        },
    )

    assert result.status == "unavailable"

    assert result.plan["task"] == "optical_sar_analysis"
    assert result.plan["selected_tool"] == "optical_sar_fusion"

    assert result.result["tool_name"] == "optical_sar_fusion"

    assert result.result["outputs"]["model"] == "MM-OVSeg B/16"
    assert result.result["outputs"]["status"] == "unavailable"

    assert result.result["confidence"] == 0.0

    execution_steps = [
        step
        for step in result.execution_trace
        if step["stage"] == "tool_execution"
    ]

    assert len(execution_steps) == 1

    step = execution_steps[0]

    assert step["tool"] == "optical_sar_fusion"
    assert step["status"] == "unavailable"
    assert step["model"] == "MM-OVSeg B/16"
    assert step["backend"] == "unavailable"
    assert isinstance(
        step["runtime_seconds"],
        float,
    )


def test_optical_sar_rejects_two_optical_inputs(monkeypatch):
    monkeypatch.setenv(
        "SATQUERY_OPTICAL_SAR_BACKEND",
        "unavailable",
    )

    agent = SatQueryAgent()

    result = agent.run(
        query="Analyze the optical and SAR images together.",
        inputs={
            "optical_image": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
            "sar_image": {
                "path": str(SAMPLE_IMAGE),
                "modality": "optical",
            },
        },
    )

    assert result.status == "input_error"

    assert result.plan["task"] == "optical_sar_analysis"

    pair_steps = [
        step
        for step in result.execution_trace
        if step["stage"] == "pair_validation"
    ]

    assert len(pair_steps) == 1
    assert pair_steps[0]["status"] == "rejected"

    execution_steps = [
        step
        for step in result.execution_trace
        if step["stage"] == "tool_execution"
    ]

    assert len(execution_steps) == 0