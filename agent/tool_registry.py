from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from models.change_service import DCFFChangeService
from models.grounding_service import DevelopmentGroundingService
from models.optical_sar_service import OpticalSARService
from models.vqa_service import RemoteSensingVQAService


@dataclass
class ToolDefinition:
    name: str
    description: str
    supported_tasks: List[str]
    required_inputs: List[str]
    model_name: str
    execute: Optional[Callable] = None
    parameters: Dict = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry for all SatQuery specialist tools.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self.optical_sar_service = OpticalSARService()
        self.vqa_service = RemoteSensingVQAService()
        self.grounding_service = DevelopmentGroundingService()
        self.change_service = DCFFChangeService()

        self._register_default_tools()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        return self._tools.get(tool_name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def _execute_vqa(self, inputs):
        """
        Execute remote-sensing VQA.

        main.py stores uploaded image information as:
            {
                "path": "...",
                "modality": "optical"
            }

        The VQA service itself expects the actual filesystem path.
        """

        image_input = inputs.get("single_image")

        if not image_input:
            return {
                "status": "input_error",
                "message": "No image was provided for VQA.",
            }

        # main.py sends single_image as a dictionary.
        # Keep compatibility with a plain string as well.
        if isinstance(image_input, dict):
            image_path = image_input.get("path")
        else:
            image_path = image_input

        if not image_path:
            return {
                "status": "input_error",
                "message": "The uploaded image path is missing.",
            }

        # The orchestrator may provide the natural-language question
        # under either "question" or "query".
        question = (
            inputs.get("question")
            or inputs.get("query")
            or ""
        )

        if not question.strip():
            return {
                "status": "input_error",
                "message": "No VQA question was provided.",
            }

        return self.vqa_service.answer(
            image_path=image_path,
            question=question,
            context={
                "modality": (
                    image_input.get("modality")
                    if isinstance(image_input, dict)
                    else None
                )
            },
        )

    def _execute_grounding(self, inputs):
        image_path = inputs["single_image"]
        query = inputs.get("question", "")

        return self.grounding_service.answer(
            image_path=image_path,
            query=query,
        )

    def _execute_change_analysis(self, inputs):
        return self.change_service.analyze(
            image_t1=inputs["image_t1"],
            image_t2=inputs["image_t2"],
        )

    def _execute_optical_sar(self, inputs):
        return self.optical_sar_service.analyze(
            optical_image=inputs["optical_image"],
            sar_image=inputs["sar_image"],
        )

    def _register_default_tools(self) -> None:
        self.register(
            ToolDefinition(
                name="remote_sensing_vqa",
                description=(
                    "Answers natural-language questions about "
                    "remote-sensing imagery."
                ),
                supported_tasks=["single_image_vqa"],
                required_inputs=["single_image"],
                model_name="remote-sensing-vqa",
                execute=self._execute_vqa,
            )
        )

        self.register(
            ToolDefinition(
                name="grounding",
                description=(
                    "Locates and highlights regions referred to "
                    "by a natural-language query."
                ),
                supported_tasks=["text_guided_region_grounding"],
                required_inputs=["single_image"],
                model_name="remote-sensing-grounding",
                execute=self._execute_grounding,
            )
        )

        self.register(
            ToolDefinition(
                name="change_analysis",
                description=(
                    "Performs pixel-level bi-temporal change detection "
                    "using the DCFF-Net remote-sensing specialist."
                ),
                supported_tasks=["bi_temporal_change_analysis"],
                required_inputs=["image_t1", "image_t2"],
                model_name="DCFF-Net",
                execute=self._execute_change_analysis,
                parameters={
                    "benchmark": "LEVIR-CD",
                    "best_iou": 0.8296,
                    "best_f1": 0.9069,
                    "best_configuration": "MS-TTA 1/1.25/1.5",
                },
            )
        )

        self.register(
            ToolDefinition(
                name="optical_sar_fusion",
                description=(
                    "Performs joint analysis using optical/multispectral "
                    "and SAR imagery using MM-OVSeg B/16."
                ),
                supported_tasks=["optical_sar_analysis"],
                required_inputs=["optical_image", "sar_image"],
                model_name="MM-OVSeg B/16",
                execute=self._execute_optical_sar,
            )
        )