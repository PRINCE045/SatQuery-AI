from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.executor import ToolExecutor
from agent.query_engine import QueryEngine
from agent.schemas import QueryPlan
from agent.tool_registry import ToolRegistry
from services.input_validator import (
    inspect_input,
    validate_pair_compatibility,
)


@dataclass
class AgentResult:
    """
    Final result produced by the SatQuery orchestration layer.
    """

    status: str
    query: str
    plan: Dict[str, Any]
    result: Dict[str, Any]
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


class SatQueryAgent:
    """
    Central agentic controller for SatQuery AI.

    Responsibilities:
    1. Interpret the natural-language query.
    2. Determine the required specialist task.
    3. Validate supplied inputs.
    4. Validate pair compatibility when required.
    5. Select the appropriate registered tool.
    6. Execute the tool.
    7. Return evidence and an auditable execution trace.
    """

    def __init__(self):
        self.query_engine = QueryEngine()
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    def run(
        self,
        query: str,
        inputs: Dict[str, Dict[str, Any]],
    ) -> AgentResult:

        trace: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # STEP 1 — Query interpretation
        # ---------------------------------------------------------

        plan: QueryPlan = self.query_engine.plan(query)

        trace.append(
            {
                "step": 1,
                "stage": "query_interpretation",
                "task": plan.task,
                "selected_tool": plan.selected_tool,
                "confidence": plan.confidence,
            }
        )

        # ---------------------------------------------------------
        # STEP 2 — Tool lookup
        # ---------------------------------------------------------

        tool = self.registry.get(plan.selected_tool)

        if tool is None:
            trace.append(
                {
                    "step": 2,
                    "stage": "tool_selection",
                    "status": "error",
                    "reason": "Selected tool is not registered.",
                }
            )

            return AgentResult(
                status="error",
                query=query,
                plan=self._plan_to_dict(plan),
                result={
                    "error": f"Tool not registered: {plan.selected_tool}"
                },
                execution_trace=trace,
            )

        trace.append(
            {
                "step": 2,
                "stage": "tool_selection",
                "status": "success",
                "tool": tool.name,
                "model": tool.model_name,
            }
        )

        # ---------------------------------------------------------
        # STEP 3 — Validate required inputs exist
        # ---------------------------------------------------------

        missing_inputs = [
            required
            for required in plan.required_inputs
            if required not in inputs
        ]

        if missing_inputs:
            trace.append(
                {
                    "step": 3,
                    "stage": "input_validation",
                    "status": "rejected",
                    "missing_inputs": missing_inputs,
                }
            )

            return AgentResult(
                status="input_error",
                query=query,
                plan=self._plan_to_dict(plan),
                result={
                    "error": "Required inputs are missing.",
                    "missing_inputs": missing_inputs,
                },
                execution_trace=trace,
            )

        validated_inputs: Dict[str, Any] = {}
        inspected_inputs: Dict[str, Dict[str, Any]] = {}

        # ---------------------------------------------------------
        # STEP 4 — Inspect each input
        # ---------------------------------------------------------

        for input_name in plan.required_inputs:

            input_definition = inputs[input_name]

            file_path = input_definition.get("path")
            modality = input_definition.get("modality")

            if not file_path:
                trace.append(
                    {
                        "step": 4,
                        "stage": "input_validation",
                        "status": "rejected",
                        "input": input_name,
                        "reason": "Input path is missing.",
                    }
                )

                return AgentResult(
                    status="input_error",
                    query=query,
                    plan=self._plan_to_dict(plan),
                    result={
                        "error": f"Path missing for input: {input_name}"
                    },
                    execution_trace=trace,
                )

            validation = inspect_input(
                file_path=file_path,
                modality=modality,
            )

            trace.append(
                {
                    "step": 4,
                    "stage": "input_validation",
                    "input": input_name,
                    "status": (
                        "success"
                        if validation.get("valid")
                        else "rejected"
                    ),
                    "filename": validation.get("filename"),
                    "modality": validation.get("modality"),
                    "width": validation.get("width"),
                    "height": validation.get("height"),
                    "band_count": validation.get("band_count"),
                    "crs": validation.get("crs"),
                }
            )

            if not validation.get("valid"):
                return AgentResult(
                    status="input_error",
                    query=query,
                    plan=self._plan_to_dict(plan),
                    result={
                        "error": (
                            f"Input validation failed: {input_name}"
                        ),
                        "details": validation,
                    },
                    execution_trace=trace,
                )

            validated_inputs[input_name] = file_path
            inspected_inputs[input_name] = validation

        # ---------------------------------------------------------
        # STEP 5 — Pair compatibility
        # ---------------------------------------------------------

        if len(plan.required_inputs) == 2:

            first_name = plan.required_inputs[0]
            second_name = plan.required_inputs[1]

            pair_validation = validate_pair_compatibility(
                inspected_inputs[first_name],
                inspected_inputs[second_name],
                require_optical_sar=(
                    plan.task == "optical_sar_analysis"
                ),
            )

            trace.append(
                {
                    "step": 5,
                    "stage": "pair_validation",
                    "status": (
                        "success"
                        if pair_validation.get("compatible")
                        else "rejected"
                    ),
                    "is_optical_sar_pair": pair_validation.get(
                        "is_optical_sar_pair",
                        False,
                    ),
                    "issues": pair_validation.get("issues", []),
                }
            )

            if not pair_validation.get("compatible"):
                return AgentResult(
                    status="input_error",
                    query=query,
                    plan=self._plan_to_dict(plan),
                    result={
                        "error": "Input pair compatibility validation failed.",
                        "details": pair_validation,
                    },
                    execution_trace=trace,
                )

        # ---------------------------------------------------------
        # STEP 6 — Pass query to language-driven specialists
        # ---------------------------------------------------------

        if plan.task in {
            "single_image_vqa",
            "text_guided_region_grounding",
        }:
            validated_inputs["question"] = query

        # ---------------------------------------------------------
        # STEP 7 — Execute specialist
        # ---------------------------------------------------------

        execution = self.executor.execute(
            tool_name=plan.selected_tool,
            inputs=validated_inputs,
        )

        trace.append(
            {
                "step": 7,
                "stage": "tool_execution",
                "tool": execution.tool_name,
                "status": execution.status,
                "message": execution.message,
                "model": execution.outputs.get("model"),
                "backend": execution.outputs.get(
                    "execution",
                    {},
                ).get("backend"),
                "runtime_seconds": execution.outputs.get(
                    "execution",
                    {},
                ).get("runtime_seconds"),
            }
        )

        # ---------------------------------------------------------
        # STEP 8 — Evidence
        # ---------------------------------------------------------

        trace.append(
            {
                "step": 8,
                "stage": "evidence",
                "status": "success",
                "evidence_count": len(execution.evidence),
            }
        )

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------

        return AgentResult(
            status=execution.status,
            query=query,
            plan=self._plan_to_dict(plan),
            result={
                "tool_name": execution.tool_name,
                "message": execution.message,
                "outputs": execution.outputs,
                "evidence": execution.evidence,
                "confidence": execution.outputs.get(
                    "confidence",
                    0.0,
                ),
            },
            execution_trace=trace,
        )

    @staticmethod
    def _plan_to_dict(
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        return {
            "task": plan.task,
            "description": plan.description,
            "required_inputs": plan.required_inputs,
            "selected_tool": plan.selected_tool,
            "confidence": plan.confidence,
        }