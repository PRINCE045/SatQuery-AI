from pathlib import Path
import numpy as np

from models.change_service import DCFFChangeService


SAMPLE_IMAGE = Path(
    "data/sample/sentinel2_sample.tif"
)


def test_change_service_contract_without_backend(monkeypatch):

    assert SAMPLE_IMAGE.is_file()

    monkeypatch.setenv(
        "SATQUERY_CHANGE_BACKEND",
        "unavailable",
    )

    service = DCFFChangeService()

    result = service.analyze(
        image_t1=str(SAMPLE_IMAGE),
        image_t2=str(SAMPLE_IMAGE),
    )

    assert result["task"] == (
        "bi_temporal_change_analysis"
    )

    assert result["status"] == "unavailable"

    assert result["model"] == "DCFF-Net"

    assert result["change_detected"] is None

    assert result["changed_area_percent"] is None

    assert result["change_regions"] == []

    assert result["confidence"] == 0.0

    assert len(result["evidence"]) == 2

    assert result["evidence"][0]["role"] == "time_1"
    assert result["evidence"][1]["role"] == "time_2"

    assert "backend" in result["execution"]
    assert "runtime_seconds" in result["execution"]


def test_change_service_rejects_missing_image(tmp_path):

    image_t1 = tmp_path / "image_t1.tif"
    image_t1.write_bytes(b"placeholder")

    service = DCFFChangeService()

    result = service.analyze(
        image_t1=str(image_t1),
        image_t2=str(tmp_path / "missing.tif"),
    )

    assert result["status"] == "input_error"
    assert result["confidence"] == 0.0
    assert result["change_detected"] is None


def test_change_benchmark_metadata():

    assert DCFFChangeService.BENCHMARK["dataset"] == (
        "LEVIR-CD"
    )

    assert (
        DCFFChangeService.BENCHMARK["best_iou"]
        == 0.8296
    )

    assert (
        DCFFChangeService.BENCHMARK["best_f1"]
        == 0.9069
    )

    assert (
        DCFFChangeService.BENCHMARK[
            "best_configuration"
        ]
        == "MS-TTA 1/1.25/1.5"
    )


def test_change_service_builds_real_prediction_contract(
    monkeypatch,
):

    monkeypatch.setenv(
        "SATQUERY_CHANGE_BACKEND",
        "local",
    )

    service = DCFFChangeService()

    probability = np.zeros(
        (20, 20),
        dtype=np.float32,
    )

    probability[5:10, 6:14] = 0.95

    result = service.build_prediction_result(
        probability_map=probability,
        image_t1=str(SAMPLE_IMAGE),
        image_t2=str(SAMPLE_IMAGE),
    )

    assert result["status"] == "success"

    assert result["model"] == "DCFF-Net"

    assert result["change_detected"] is True

    assert result["changed_area_percent"] == 10.0

    assert len(result["change_regions"]) == 1

    assert result["confidence"] is None

    assert len(result["evidence"]) == 3

    assert result["metrics"]["changed_pixels"] == 40

    assert result["metrics"]["total_pixels"] == 400