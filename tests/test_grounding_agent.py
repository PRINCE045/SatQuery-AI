from agent.orchestrator import SatQueryAgent


def test_agent_routes_grounding_query():
    agent = SatQueryAgent()

    result = agent.run(
        query="Locate the airport in this remote sensing image.",
        inputs={
            "single_image": {
                "path": "data/sample/sentinel2_sample.tif",
                "modality": "optical",
            }
        },
    )

    # The agent successfully executed the registered tool.
    assert result.status == "development"

    # The actual Grounding DINO specialist is not connected yet.
    assert result.result["outputs"]["status"] == "development"

    # Query routing
    assert result.plan["task"] == "text_guided_region_grounding"
    assert result.plan["selected_tool"] == "grounding"

    # Correct tool execution
    assert result.result["tool_name"] == "grounding"

    # Auditable execution trace
    assert result.execution_trace[0]["stage"] == "query_interpretation"
    assert result.execution_trace[1]["stage"] == "tool_selection"
    assert result.execution_trace[2]["stage"] == "input_validation"
    assert result.execution_trace[3]["stage"] == "tool_execution"
    assert result.execution_trace[4]["stage"] == "evidence"

    # The natural-language query reaches the grounding specialist.
    assert result.result["outputs"]["query"] == (
        "Locate the airport in this remote sensing image."
    )

    # No fake detections.
    assert result.result["outputs"]["objects"] == []

    # No fake evidence.
    assert result.result["outputs"]["evidence"] == []