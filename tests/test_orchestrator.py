from agent.orchestrator import SatQueryAgent


SAMPLE_IMAGE = "data/sample/sample.tif"


def test_single_image_agent_pipeline():
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

    outputs = result.result["outputs"]

    assert outputs["model"] == (
        "Qwen3-VL-2B-Instruct + RSVQA-HR LoRA"
    )
    assert outputs["task"] == "single_image_vqa"
    assert outputs["execution"]["backend"] == "unavailable"