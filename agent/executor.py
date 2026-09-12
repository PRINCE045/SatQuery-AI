from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.tool_registry import ToolRegistry, ToolDefinition


@dataclass
class ExecutionResult:
    tool_name: str
    status: str
    message: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)


class ToolExecutor:
    """
    Executes tools selected by the SatQuery agent.

    At this stage, specialist models are represented by controlled
    development executors. Real AI models will be connected later.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
    ) -> ExecutionResult:

        tool = self.registry.get(tool_name)

        if tool is None:
            return ExecutionResult(
                tool_name=tool_name,
                status="error",
                message=f"Unknown tool: {tool_name}",
            )

        missing_inputs = [
            required
            for required in tool.required_inputs
            if required not in inputs
        ]

        if missing_inputs:
            return ExecutionResult(
                tool_name=tool_name,
                status="input_error",
                message="Required inputs are missing.",
                outputs={
                    "missing_inputs": missing_inputs,
                    "required_inputs": tool.required_inputs,
                },
            )

        if tool.execute is not None:
            result = tool.execute(inputs)
            return ExecutionResult(
                tool_name=tool_name,
                status="success",
                message="Tool executed successfully.",
                outputs=result,
            )

        return self._development_executor(tool, inputs)

    def _development_executor(
        self,
        tool: ToolDefinition,
        inputs: Dict[str, Any],
    ) -> ExecutionResult:

        return ExecutionResult(
            tool_name=tool.name,
            status="ready",
            message=(
                f"Tool '{tool.name}' passed input validation. "
                "Specialist model execution will be connected in the model integration phase."
            ),
            outputs={
                "model": tool.model_name,
                "received_inputs": list(inputs.keys()),
                "execution_mode": "development",
            },
            evidence=[
                {
                    "type": "execution_trace",
                    "tool": tool.name,
                    "model": tool.model_name,
                    "status": "validated",
                }
            ],
        )