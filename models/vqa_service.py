from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from services.inference_client import InferenceClient
from services.inference_contract import InferenceRequest


class VQAService(ABC):

    @abstractmethod
    def answer(
        self,
        image_path: str,
        question: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RemoteSensingVQAService(VQAService):

    MODEL_NAME = "Qwen3-VL-2B-Instruct + VRSBench LoRA"
    ADAPTATION_DATASET = "VRSBench"

    def __init__(self):
        import os

        self.backend = os.getenv(
            "SATQUERY_VQA_BACKEND",
            "unavailable",
        )

        self.inference_client = InferenceClient(
            backend=self.backend
        )

    def answer(
        self,
        image_path: str,
        question: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        image = Path(image_path)

        if not image.is_file():
            return {
                "task": "single_image_vqa",
                "status": "input_error",
                "answer": None,
                "message": f"Image not found: {image_path}",
                "model": self.MODEL_NAME,
                "evidence": [],
            }

        if not question.strip():
            return {
                "task": "single_image_vqa",
                "status": "input_error",
                "answer": None,
                "message": "A natural-language question is required.",
                "model": self.MODEL_NAME,
                "evidence": [],
            }

        request = InferenceRequest(
            task="single_image_vqa",
            model=self.MODEL_NAME,
            image_paths=[str(image.resolve())],
            question=question,
            metadata={
                "adaptation_dataset": self.ADAPTATION_DATASET,
                "adaptation_method": "Language-side LoRA",
            },
        )

        response = self.inference_client.predict(request)

        return {
            "task": response.task,
            "status": response.status,
            "answer": response.answer,
            "message": (
                response.error
                if response.error
                else "VQA inference completed."
            ),
            "model": response.model,
            "confidence": response.confidence,
            "evidence": response.evidence,
            "adaptation": {
                "dataset": self.ADAPTATION_DATASET,
                "method": "Language-side LoRA",
            },
            "question": question,
            "execution": {
                "backend": self.backend,
            },
            "metadata": response.metadata,
        }


# Preserve compatibility with the existing registry/tests.
DevelopmentVQAService = RemoteSensingVQAService