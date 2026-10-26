from pathlib import Path

from models.grounding_service import DevelopmentGroundingService


def test_grounding_service_development_response():
    image_path = Path("data/sample/sentinel2_sample.tif")

    service = DevelopmentGroundingService()

    result = service.answer(
        image_path=str(image_path),
        query="Locate the airport in this remote sensing image.",
    )

    assert result["status"] == "development"
    assert result["model"] == "remote-sensing-grounding"
    assert result["objects"] == []
    assert result["answer"] is None