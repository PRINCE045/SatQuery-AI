import os
import re
from html import unescape
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

from services.inference_contract import (
    InferenceRequest,
    InferenceResponse,
)


class InferenceClient:
    """
    Remote inference client for SatQuery AI.

    The local backend does NOT load the heavy remote-sensing model.
    Instead, it sends the image + question to the configured
    Hugging Face Gradio Space.
    """

    def __init__(self, backend: str = "unavailable"):
        self.backend = (backend or "unavailable").strip().lower()

        self.space = os.getenv(
            "SATQUERY_HF_SPACE",
            "Kingprince24/satquery-ai-vqa",
        )

        self.api_name = os.getenv(
            "SATQUERY_HF_API",
            "/analyze_image",
        )

        self.hf_token = os.getenv("HF_TOKEN")

        self._client: Client | None = None

    def _get_client(self) -> Client:
        """
        Lazily create and cache the Gradio client.

        This prevents reconnecting to the Hugging Face Space
        for every inference request.
        """

        if self._client is None:
            client_kwargs: dict[str, Any] = {}

            if self.hf_token:
                client_kwargs["token"] = self.hf_token

            self._client = Client(
                self.space,
                httpx_kwargs={
                    "timeout": 300.0,
                },
                **client_kwargs,
            )

        return self._client

    @staticmethod
    def _strip_html(value: Any) -> str:
        """
        Convert the HTML answer returned by the Gradio UI
        into clean plain text for the local API.
        """

        if value is None:
            return ""

        text = str(value)

        # Preserve basic line breaks.
        text = re.sub(
            r"<br\s*/?>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"</p\s*>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        # Remove remaining HTML tags.
        text = re.sub(
            r"<[^>]+>",
            "",
            text,
        )

        # Decode entities such as &amp; and &lt;.
        text = unescape(text)

        return text.strip()

    def _unavailable_response(
        self,
        request: InferenceRequest,
        message: str,
    ) -> InferenceResponse:
        return InferenceResponse(
            status="unavailable",
            task=request.task,
            model=request.model,
            error=message,
            metadata={
                "backend": self.backend,
                "space": self.space,
            },
        )

    def predict(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:

        # ---------------------------------------------------------
        # Backend validation
        # ---------------------------------------------------------

        if self.backend == "unavailable":
            return self._unavailable_response(
                request,
                "Remote GPU inference backend is not configured.",
            )

        if self.backend not in {"huggingface", "hf"}:
            return InferenceResponse(
                status="error",
                task=request.task,
                model=request.model,
                error=f"Unsupported inference backend: {self.backend}",
                metadata={
                    "backend": self.backend,
                },
            )

        # ---------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------

        if not request.image_paths:
            return InferenceResponse(
                status="input_error",
                task=request.task,
                model=request.model,
                error="No image was provided for inference.",
                metadata={
                    "backend": self.backend,
                },
            )

        image_path = Path(request.image_paths[0])

        if not image_path.is_file():
            return InferenceResponse(
                status="input_error",
                task=request.task,
                model=request.model,
                error=f"Image file does not exist: {image_path}",
                metadata={
                    "backend": self.backend,
                },
            )

        if not request.question or not request.question.strip():
            return InferenceResponse(
                status="input_error",
                task=request.task,
                model=request.model,
                error="A question is required for VQA inference.",
                metadata={
                    "backend": self.backend,
                },
            )

        # ---------------------------------------------------------
        # Remote Hugging Face inference
        # ---------------------------------------------------------

        try:
            client = self._get_client()

            result = client.predict(
                handle_file(str(image_path)),
                request.question.strip(),
                api_name=self.api_name,
            )

            # Gradio normally returns multiple outputs because
            # the Space's analyze_image() function returns:
            #
            #   1. processed image
            #   2. metadata HTML
            #   3. answer HTML
            #
            if not isinstance(result, (list, tuple)):
                result = [result]

            if not result:
                return InferenceResponse(
                    status="error",
                    task=request.task,
                    model=request.model,
                    error="Hugging Face inference returned no output.",
                    metadata={
                        "backend": self.backend,
                        "space": self.space,
                        "api_name": self.api_name,
                    },
                )

            # Current SatQuery HF Space returns the answer
            # as its third output.
            if len(result) >= 3:
                answer_raw = result[2]
            else:
                answer_raw = result[-1]

            answer = self._strip_html(answer_raw)

            answer = re.sub(
                r"^##\s*🧠\s*SatQuery AI Analysis\s*",
                "",
                answer,
                flags=re.IGNORECASE,
            )

            answer = re.sub(
                r"Qwen3-VL-2B-Instruct\s*[·•]\s*VRSBench Adaptation\s*$",
                "",
                answer,
                flags=re.IGNORECASE,
            )

            answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

            if not answer:
                ...

            if not answer:
                return InferenceResponse(
                    status="error",
                    task=request.task,
                    model=request.model,
                    error="Hugging Face inference returned an empty answer.",
                    metadata={
                        "backend": self.backend,
                        "space": self.space,
                        "api_name": self.api_name,
                        "raw_outputs": len(result),
                    },
                )

            # -----------------------------------------------------
            # Successful response
            # -----------------------------------------------------

            return InferenceResponse(
                status="success",
                task=request.task,
                model=request.model,
                answer=answer,
                confidence=0.0,
                evidence=[],
                metadata={
                    "backend": "huggingface",
                    "space": self.space,
                    "api_name": self.api_name,
                    "remote_outputs": len(result),
                    "confidence_available": False,
                    "evidence_available": False,
                },
            )

        except Exception as exc:
            return InferenceResponse(
                status="error",
                task=request.task,
                model=request.model,
                error=f"Hugging Face inference failed: {exc}",
                metadata={
                    "backend": self.backend,
                    "space": self.space,
                    "api_name": self.api_name,
                    "exception_type": type(exc).__name__,
                },
            )