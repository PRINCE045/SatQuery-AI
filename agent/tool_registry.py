from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


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

    The registry allows the agent to discover which specialist
    components are available and what inputs they require.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        return self._tools.get(tool_name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def _register_default_tools(self) -> None:

        self.register(
            ToolDefinition(
                name="remote_sensing_vqa",
                description="Answers natural-language questions about remote-sensing imagery.",
                supported_tasks=["single_image_vqa"],
                required_inputs=["single_image"],
                model_name="remote-sensing-vqa",
            )
        )

        self.register(
            ToolDefinition(
                name="grounding",
                description="Locates and highlights regions referred to by a natural-language query.",
                supported_tasks=["text_guided_region_grounding"],
                required_inputs=["single_image"],
                model_name="remote-sensing-grounding",
            )
        )

        self.register(
            ToolDefinition(
                name="change_analysis",
                description="Analyzes changes between spatially corresponding images acquired at different times.",
                supported_tasks=["bi_temporal_change_analysis"],
                required_inputs=["image_t1", "image_t2"],
                model_name="remote-sensing-change-model",
            )
        )

        self.register(
            ToolDefinition(
                name="optical_sar_fusion",
                description="Performs joint analysis using optical/multispectral and SAR imagery.",
                supported_tasks=["optical_sar_analysis"],
                required_inputs=["optical_image", "sar_image"],
                model_name="optical-sar-analysis-model",
            )
        )