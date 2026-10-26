import numpy as np
from PIL import Image

from services.change_artifacts import ChangeArtifactWriter


def test_write_binary_change_mask(tmp_path):

    mask = np.zeros(
        (10, 12),
        dtype=np.uint8,
    )

    mask[2:5, 3:7] = 1

    writer = ChangeArtifactWriter(
        tmp_path
    )

    result = writer.write_mask(mask)

    output = tmp_path / "change_mask.png"

    assert output.is_file()

    assert result["type"] == "change_mask"
    assert result["format"] == "PNG"
    assert result["width"] == 12
    assert result["height"] == 10
    assert result["changed_pixels"] == 12

    saved = np.array(
        Image.open(output)
    )

    assert saved.shape == (10, 12)

    assert saved[3, 4] == 255
    assert saved[0, 0] == 0


def test_write_change_overlay(tmp_path):

    image = np.zeros(
        (10, 12, 3),
        dtype=np.uint8,
    )

    image[:, :] = 100

    mask = np.zeros(
        (10, 12),
        dtype=np.uint8,
    )

    mask[2:5, 3:7] = 1

    writer = ChangeArtifactWriter(
        tmp_path
    )

    result = writer.write_overlay(
        image=image,
        mask=mask,
    )

    output = tmp_path / "change_overlay.png"

    assert output.is_file()

    assert result["type"] == "change_overlay"

    saved = np.array(
        Image.open(output)
    )

    assert saved.shape == (10, 12, 3)

    assert tuple(saved[3, 4]) == (
        255,
        0,
        0,
    )

    assert tuple(saved[0, 0]) == (
        100,
        100,
        100,
    )


def test_reject_invalid_mask(tmp_path):

    mask = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    writer = ChangeArtifactWriter(
        tmp_path
    )

    try:
        writer.write_mask(mask)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )