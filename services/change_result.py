from __future__ import annotations

from typing import Any, Dict

import numpy as np

from services.change_processor import ChangeProcessor


class ChangeResultBuilder:
    """
    Converts a DCFF probability map into the standardized
    SatQuery change-analysis evidence contract.

    This class is deliberately model-independent.

    DCFF is responsible for producing the probability map.
    ChangeProcessor is responsible for interpreting that map.
    This class packages the result for the agent.
    """

    def __init__(
        self,
        processor: ChangeProcessor | None = None,
    ) -> None:

        self.processor = processor or ChangeProcessor()

    def build(
        self,
        probability_map: np.ndarray,
    ) -> Dict[str, Any]:

        processed = self.processor.process(
            probability_map
        )

        # The binary NumPy mask is useful internally but should
        # never be placed directly into the JSON-facing result.
        processed.pop("mask", None)

        change_detected = processed[
            "change_detected"
        ]

        return {
            "change_detected": change_detected,
            "changed_area_percent": processed[
                "changed_area_percent"
            ],
            "changed_pixels": processed[
                "changed_pixels"
            ],
            "total_pixels": processed[
                "total_pixels"
            ],
            "change_regions": processed[
                "change_regions"
            ],
            "threshold": processed[
                "threshold"
            ],
            "mask_available": True,
        }