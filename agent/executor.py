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
    Executes specialists selected by the SatQuery agent.
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
            try:
                result = tool.execute(inputs)

                if not isinstance(result, dict):
                    result = {
                        "status": "error",
                        "message": (
                            "Specialist returned an invalid result type."
                        ),
                    }

                status = result.get("status", "success")

                message = result.get(
                    "message",
                    "Tool executed successfully."
                    if status == "success"
                    else "Tool execution completed.",
                )

                evidence = result.get("evidence", [])

                return ExecutionResult(
                    tool_name=tool_name,
                    status=status,
                    message=message,
                    outputs=result,
                    evidence=evidence,
                )

            except Exception as exc:
                return ExecutionResult(
                    tool_name=tool_name,
                    status="error",
                    message=f"Tool execution failed: {exc}",
                    outputs={
                        "exception_type": type(exc).__name__,
                    },
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
                f"Tool '{tool.name}' passed input validation, "
                "but its specialist runtime is not connected."
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