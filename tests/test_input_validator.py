from services.input_validator import (
    inspect_input,
    validate_file_format,
    validate_pair_compatibility,
)


def test_valid_sentinel2_input():
    result = inspect_input(
        "data/sample/sentinel2_sample.tif",
        "optical",
    )

    assert result["valid"] is True
    assert result["modality"] == "optical"
    assert result["band_count"] == 3


def test_invalid_modality():
    result = inspect_input(
        "data/sample/sentinel2_sample.tif",
        "thermal",
    )

    assert result["valid"] is False


def test_missing_file():
    result = inspect_input(
        "data/sample/missing.tif",
        "optical",
    )

    assert result["valid"] is False


def test_optical_sar_pair():
    optical = {
        "width": 1000,
        "height": 1000,
        "crs": "EPSG:32631",
        "modality": "optical",
    }

    sar = {
        "width": 1000,
        "height": 1000,
        "crs": "EPSG:32631",
        "modality": "sar",
    }

    result = validate_pair_compatibility(
        optical,
        sar,
    )

    assert result["compatible"] is True
    assert result["is_optical_sar_pair"] is True


def test_incompatible_pair():
    optical = {
        "width": 1000,
        "height": 1000,
        "crs": "EPSG:32631",
        "modality": "optical",
    }

    sar = {
        "width": 800,
        "height": 800,
        "crs": "EPSG:32631",
        "modality": "sar",
    }

    result = validate_pair_compatibility(
        optical,
        sar,
    )

    assert result["compatible"] is False
    assert len(result["issues"]) > 0