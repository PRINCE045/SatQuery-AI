import numpy as np
import pytest

from services.change_processor import ChangeProcessor


def test_no_change():

    probability = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    processor = ChangeProcessor()

    result = processor.process(probability)

    assert result["change_detected"] is False
    assert result["changed_pixels"] == 0
    assert result["changed_area_percent"] == 0.0
    assert result["change_regions"] == []


def test_single_change_region():

    probability = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    probability[2:5, 3:7] = 0.9

    processor = ChangeProcessor(
        threshold=0.5,
        min_region_pixels=1,
    )

    result = processor.process(probability)

    assert result["change_detected"] is True
    assert result["changed_pixels"] == 12
    assert result["changed_area_percent"] == 12.0

    assert len(result["change_regions"]) == 1

    region = result["change_regions"][0]

    assert region["x"] == 0.3
    assert region["y"] == 0.2
    assert region["width"] == 0.4
    assert region["height"] == 0.3
    assert region["area_pixels"] == 12


def test_separate_regions():

    probability = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    probability[1:3, 1:3] = 0.9
    probability[7:9, 7:9] = 0.9

    processor = ChangeProcessor(
        threshold=0.5,
        min_region_pixels=1,
    )

    result = processor.process(probability)

    assert result["change_detected"] is True
    assert result["changed_pixels"] == 8
    assert len(result["change_regions"]) == 2


def test_small_regions_are_filtered():

    probability = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    probability[1, 1] = 0.9
    probability[5:8, 5:8] = 0.9

    processor = ChangeProcessor(
        threshold=0.5,
        min_region_pixels=4,
    )

    result = processor.process(probability)

    assert result["change_detected"] is True
    assert len(result["change_regions"]) == 1
    assert result["change_regions"][0]["area_pixels"] == 9


def test_invalid_dimensions():

    processor = ChangeProcessor()

    with pytest.raises(ValueError):
        processor.process(
            np.zeros((2, 2, 1))
        )


def test_invalid_probability_range():

    processor = ChangeProcessor()

    probability = np.ones(
        (5, 5),
        dtype=np.float32,
    ) * 2.0

    with pytest.raises(ValueError):
        processor.process(probability)