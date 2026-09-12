from agent.schemas import QueryPlan


class QueryEngine:
    """
    First-generation SatQuery query planner.

    This lightweight rule-based planner is intentionally used during
    the architecture-validation phase. It will later be replaced or
    augmented with an intelligent query-routing component.
    """

    def plan(self, query: str) -> QueryPlan:
        text = query.lower().strip()

        if any(word in text for word in ["changed", "change", "between", "difference"]):
            return QueryPlan(
                task="bi_temporal_change_analysis",
                description="Analyze changes between two corresponding observations.",
                required_inputs=["image_t1", "image_t2"],
                selected_tool="change_analysis",
                confidence=0.90,
            )

        if any(word in text for word in ["optical", "sar", "radar"]):
            return QueryPlan(
                task="optical_sar_analysis",
                description="Perform joint analysis of optical/multispectral and SAR imagery.",
                required_inputs=["optical_image", "sar_image"],
                selected_tool="optical_sar_fusion",
                confidence=0.90,
            )

        if any(word in text for word in ["highlight", "where", "locate", "region"]):
            return QueryPlan(
                task="text_guided_region_grounding",
                description="Locate and highlight the region referred to by the query.",
                required_inputs=["single_image"],
                selected_tool="grounding",
                confidence=0.85,
            )

        return QueryPlan(
            task="single_image_vqa",
            description="Answer a natural-language question about a remote-sensing image.",
            required_inputs=["single_image"],
            selected_tool="remote_sensing_vqa",
            confidence=0.75,
        )