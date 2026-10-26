from agent.orchestrator import SatQueryAgent


SAMPLE_IMAGE = "data/sample/sample.tif"


def test_vqa_agent_routes_to_rs_vqa_specialist():
    agent = SatQueryAgent()

    result = agent.run(
        query="Describe this satellite image.",
        inputs={
            "single_image": {
                "path": SAMPLE_IMAGE,
                "modality": "optical",
            }
        },
    )

    assert result.status == "unavailable"

    assert result.plan["task"] == "single_image_vqa"
    assert result.plan["selected_tool"] == "remote_sensing_vqa"

    outputs = result.result["outputs"]

    assert outputs["task"] == "single_image_vqa"
    assert outputs["model"] == (
        "Qwen3-VL-2B-Instruct + RSVQA-HR LoRA"
    )
    assert outputs["execution"]["backend"] == "unavailable"


def test_vqa_execution_trace_contains_specialist_details():
    agent = SatQueryAgent()

    result = agent.run(
        query="What is visible in this satellite image?",
        inputs={
            "single_image": {
                "path": SAMPLE_IMAGE,
                "modality": "optical",
            }
        },
    )

    execution_steps = [
        step
        for step in result.execution_trace
        if step.get("stage") == "tool_execution"
    ]

    assert len(execution_steps) == 1

    execution = execution_steps[0]

    assert execution["tool"] == "remote_sensing_vqa"
    assert execution["status"] == "unavailable"
    assert execution["model"] == (
        "Qwen3-VL-2B-Instruct + RSVQA-HR LoRA"
    )
    assert execution["backend"] == "unavailable"