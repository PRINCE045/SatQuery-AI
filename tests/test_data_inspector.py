from services.data_inspector import inspect_raster


def test_inspect_raster():
    info = inspect_raster("data/sample/sample.tif")

    assert info["width"] > 0
    assert info["height"] > 0
    assert info["band_count"] > 0