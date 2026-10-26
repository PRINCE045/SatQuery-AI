import json
import numpy as np

from services.change_processor import ChangeProcessor
from services.change_result import ChangeResultBuilder


def test_change_result_builder():

    probability = np.zeros(
        (10, 10),
        dtype=np.float32,
    )

    probability[2:5, 3:7] = 0.9

    builder = ChangeResultBuilder(
        processor=ChangeProcessor(
            threshold=0.5,
            min_region_pixels=1,
        )
    )

    result = builder.build(probability)

    assert result["change_detected"] is True

    assert result["changed_pixels"] == 12

    assert result["total_pixels"] == 100

    assert result["changed_area_percent"] == 12.0

    assert len(result["change_regions"]) == 1

    assert result["mask_available"] is True

    # The public result must be JSON-friendly.
    assert "mask" not in result


def test_change_result_builder_no_change():

    probability = np.zeros(
        (20, 20),
        dtype=np.float32,
    )

    builder = ChangeResultBuilder()

    result = builder.build(probability)

    assert result["change_detected"] is False

    assert result["changed_pixels"] == 0

    assert result["changed_area_percent"] == 0.0

    assert result["change_regions"] == []

    assert result["mask_available"] is True


def test_change_result_is_json_serializable():

    probability = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    probability[2:5, 2:5] = 0.9

    builder = ChangeResultBuilder()

    result = builder.build(probability)

    serialized = json.dumps(result)

    assert isinstance(serialized, str)
    assert len(serialized) > 0