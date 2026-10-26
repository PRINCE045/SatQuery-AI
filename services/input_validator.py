from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio


SUPPORTED_RASTER_FORMATS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}

SUPPORTED_MODALITIES = {
    "optical",
    "multispectral",
    "sar",
}


def validate_file_format(file_path: str) -> Dict[str, Any]:
    """
    Validate whether the uploaded file uses a supported format.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "valid": False,
            "reason": "File does not exist.",
        }

    extension = path.suffix.lower()

    if extension not in SUPPORTED_RASTER_FORMATS:
        return {
            "valid": False,
            "reason": f"Unsupported file format: {extension}",
            "supported_formats": sorted(SUPPORTED_RASTER_FORMATS),
        }

    return {
        "valid": True,
        "format": extension.lstrip("."),
    }


def inspect_input(
    file_path: str,
    modality: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate and inspect a remote-sensing image.

    Modality is deliberately supplied explicitly rather than guessed
    from the filename. This prevents the agent from making unsafe
    assumptions about whether an image is optical, multispectral, or SAR.
    """

    format_result = validate_file_format(file_path)

    if not format_result["valid"]:
        return {
            "valid": False,
            "stage": "format_validation",
            "error": format_result["reason"],
        }

    if modality is not None:
        modality = modality.lower().strip()

        if modality not in SUPPORTED_MODALITIES:
            return {
                "valid": False,
                "stage": "modality_validation",
                "error": f"Unsupported modality: {modality}",
                "supported_modalities": sorted(SUPPORTED_MODALITIES),
            }

    path = Path(file_path)

    try:
        with rasterio.open(path) as src:
            return {
                "valid": True,
                "filename": path.name,
                "format": src.driver,
                "extension": path.suffix.lower(),
                "modality": modality,
                "width": src.width,
                "height": src.height,
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else None,
                "resolution": {
                    "x": src.res[0],
                    "y": src.res[1],
                },
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
            }

    except Exception as exc:
        return {
            "valid": False,
            "stage": "raster_inspection",
            "error": str(exc),
        }


def validate_pair_compatibility(
    image_a: Dict[str, Any],
    image_b: Dict[str, Any],
    require_optical_sar: bool = False,
) -> Dict[str, Any]:
    """
    Check whether two remote-sensing images are compatible
    for paired analysis.

    This is intentionally conservative. A pair is not considered
    compatible merely because both files can be opened.
    """

    issues: List[str] = []

    if image_a.get("width") != image_b.get("width"):
        issues.append("Image widths do not match.")

    if image_a.get("height") != image_b.get("height"):
        issues.append("Image heights do not match.")

    crs_a = image_a.get("crs")
    crs_b = image_b.get("crs")

    if crs_a and crs_b and crs_a != crs_b:
        issues.append("Coordinate reference systems do not match.")

    modality_a = image_a.get("modality")
    modality_b = image_b.get("modality")

    optical_modalities = {"optical", "multispectral"}

    is_optical_sar = (
        modality_a in optical_modalities
        and modality_b == "sar"
    ) or (
        modality_b in optical_modalities
        and modality_a == "sar"
    )

    # Cross-modal validation is required only when the selected
    # task is specifically optical + SAR analysis.
    #
    # Bi-temporal change detection may legitimately use:
    # optical + optical
    # SAR + SAR
    # multispectral + multispectral
    if require_optical_sar and not is_optical_sar:
        issues.append(
            "An optical-SAR analysis requires one "
            "optical/multispectral input and one SAR input."
        )

    return {
        "compatible": len(issues) == 0,
        "is_optical_sar_pair": is_optical_sar,
        "issues": issues,
    }