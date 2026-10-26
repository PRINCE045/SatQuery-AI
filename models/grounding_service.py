from pathlib import Path
from typing import Any, Dict, List, Optional


class DevelopmentGroundingService:
    """
    Development interface for the remote-sensing grounding specialist.

    The actual Grounding DINO/cloud model will be connected later.
    This service deliberately does NOT generate fake bounding boxes.
    """

    model_name = "remote-sensing-grounding"

    def answer(
        self,
        image_path: str,
        query: str,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path)

        if not path.exists():
            return {
                "status": "input_error",
                "answer": None,
                "message": f"Image not found: {image_path}",
                "model": self.model_name,
            }

        return {
            "status": "development",
            "answer": None,
            "message": (
                "Remote-sensing grounding specialist is not connected yet. "
                "The service interface is ready for the cloud Grounding DINO "
                "specialist."
            ),
            "model": self.model_name,
            "query": query,
            "target": target,
            "objects": [],
            "evidence": [],
        }