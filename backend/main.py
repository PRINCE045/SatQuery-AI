import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from agent.executor import ToolExecutor
from agent.orchestrator import SatQueryAgent
from agent.query_engine import QueryEngine
from agent.tool_registry import ToolRegistry


app = FastAPI(
    title="SatQuery AI",
    description="Agentic Vision-Language Assistant for Remote-Sensing Image Analysis",
    version="0.3.0",
)

query_engine = QueryEngine()
tool_registry = ToolRegistry()
tool_executor = ToolExecutor(tool_registry)
agent = SatQueryAgent()

RASTER_ROOT = Path("data/raster_previews")
RASTER_ROOT.mkdir(parents=True, exist_ok=True)

app.mount(
    "/raster-previews",
    StaticFiles(directory=str(RASTER_ROOT)),
    name="raster-previews",
)


@app.get("/")
def root():
    return {
        "name": "SatQuery AI",
        "problem_statement": "SIH26167",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "satquery-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/query/plan")
def plan_query(request: Dict[str, Any]):
    query = request.get("query", "")
    plan = query_engine.plan(query)

    return {
        "query": query,
        "plan": {
            "task": plan.task,
            "description": plan.description,
            "required_inputs": plan.required_inputs,
            "selected_tool": plan.selected_tool,
            "confidence": plan.confidence,
        },
    }


@app.get("/api/tools")
def list_tools():
    tools = tool_registry.list_tools()

    return {
        "count": len(tools),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "supported_tasks": tool.supported_tasks,
                "required_inputs": tool.required_inputs,
                "model_name": tool.model_name,
                "parameters": tool.parameters,
            }
            for tool in tools
        ],
    }


@app.post("/api/tools/{tool_name}/execute")
def execute_tool(tool_name: str, inputs: dict):
    result = tool_executor.execute(tool_name, inputs)

    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "message": result.message,
        "outputs": result.outputs,
        "evidence": result.evidence,
    }


@app.post("/api/agent/run")
def run_agent(payload: dict):
    query = payload.get("query")
    inputs = payload.get("inputs", {})

    if not query:
        return {
            "status": "input_error",
            "error": "Query is required.",
        }

    result = agent.run(
        query=query,
        inputs=inputs,
    )

    return {
        "status": result.status,
        "query": result.query,
        "plan": result.plan,
        "result": result.result,
        "execution_trace": result.execution_trace,
    }


def _public_base_url() -> str:
    """
    Base URL used by the React app to load generated raster overlays.

    For local development this is the FastAPI server. In deployment,
    set SATQUERY_PUBLIC_BASE_URL to the public backend URL.
    """
    return os.getenv(
        "SATQUERY_PUBLIC_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")


def _stretch_band(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    finite = np.isfinite(band)

    if not finite.any():
        return np.zeros(band.shape, dtype=np.uint8)

    values = band[finite]
    low, high = np.percentile(values, [2, 98])

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = (band - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _read_raster_preview(
    source_path: str,
    output_name: str,
) -> Dict[str, Any]:
    """
    Render an uploaded raster into a browser-viewable PNG and return
    geographic bounds in WGS84 for Leaflet ImageOverlay.

    This is a visualization/evidence operation only. It does not alter
    the underlying scientific raster.
    """
    source = Path(source_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Raster source does not exist: {source_path}"
        )

    suffix = source.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with rasterio.open(source) as dataset:
            width = dataset.width
            height = dataset.height
            count = dataset.count
            crs = dataset.crs

            if count >= 3:
                descriptions = [
                    (dataset.descriptions[i - 1] or "").upper()
                    for i in range(1, count + 1)
                ]

                red_idx = next(
                    (
                        i + 1
                        for i, name in enumerate(descriptions)
                        if name in {"B4", "RED", "R"}
                    ),
                    1,
                )
                green_idx = next(
                    (
                        i + 1
                        for i, name in enumerate(descriptions)
                        if name in {"B3", "GREEN", "G"}
                    ),
                    2,
                )
                blue_idx = next(
                    (
                        i + 1
                        for i, name in enumerate(descriptions)
                        if name in {"B2", "BLUE", "B"}
                    ),
                    3,
                )

                indexes = [
                    min(red_idx, count),
                    min(green_idx, count),
                    min(blue_idx, count),
                ]
            else:
                indexes = [1]

            # Keep preview generation lightweight for large scenes.
            max_dimension = 1600
            scale = min(
                1.0,
                max_dimension / max(width, height),
            )
            out_width = max(1, round(width * scale))
            out_height = max(1, round(height * scale))

            data = dataset.read(
                indexes,
                out_shape=(len(indexes), out_height, out_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )

            if len(indexes) == 1:
                gray = _stretch_band(data[0])
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                rgb = np.stack(
                    [_stretch_band(data[i]) for i in range(3)],
                    axis=-1,
                )

            # Keep the raster itself opaque. Leaflet handles the geographic
            # placement through the transformed bounds returned below.
            image = Image.fromarray(rgb, mode="RGB")

            output_file = RASTER_ROOT / output_name
            image.save(
                output_file,
                format="PNG",
                optimize=True,
            )

            bounds = dataset.bounds

            if crs is not None:
                from rasterio.warp import transform_bounds

                west, south, east, north = transform_bounds(
                    crs,
                    "EPSG:4326",
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                    densify_pts=21,
                )
            else:
                west, south, east, north = (
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )

            center_lon = (west + east) / 2.0
            center_lat = (south + north) / 2.0

            return {
                "url": (
                    f"{_public_base_url()}/raster-previews/"
                    f"{output_file.name}"
                ),
                "bounds": [
                    [south, west],
                    [north, east],
                ],
                "width": width,
                "height": height,
                "bands": count,
                "crs": str(crs) if crs else None,
                "center": {
                    "lat": center_lat,
                    "lon": center_lon,
                },
                "bounds_order": "south,west,north,east",
                "source": source.name,
                "format": "GeoTIFF",
            }

    # Standard image fallback. No geographic transform is possible,
    # so use the current map center as a small visual fallback only.
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1400, 1400))
        output_file = RASTER_ROOT / output_name
        image.save(output_file, format="PNG", optimize=True)

    return {
        "url": (
            f"{_public_base_url()}/raster-previews/"
            f"{output_file.name}"
        ),
        "bounds": None,
        "width": image.width,
        "height": image.height,
        "bands": 3,
        "crs": None,
        "center": None,
        "source": source.name,
        "format": "Image",
    }


def _normalize_evidence(
    outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert specialist evidence into the object-shaped evidence contract
    expected by the frontend while retaining the original records.
    """
    raw = outputs.get("evidence")

    if isinstance(raw, dict):
        return dict(raw)

    if isinstance(raw, list):
        outputs["evidence_records"] = raw

    return {}


def _attach_raster_evidence(
    outputs: Dict[str, Any],
    inputs: Dict[str, Any],
) -> None:
    """
    Attach browser-viewable raster evidence to the specialist output.

    This is what makes the Leaflet map show the actual uploaded remote-
    sensing scene instead of only the OpenStreetMap basemap.
    """
    evidence = _normalize_evidence(outputs)

    request_context = inputs.get("request_context", {})
    task = str(request_context.get("task") or "").lower()

    single = inputs.get("single_image")
    image_t2 = inputs.get("image_t2")
    optical = inputs.get("optical_image")
    sar = inputs.get("sar_image")

    try:
        if single and single.get("path"):
            preview = _read_raster_preview(
                single["path"],
                f"{uuid.uuid4().hex}_single.png",
            )
            evidence["raster"] = preview

            if preview.get("center"):
                outputs.setdefault("metadata", {})
                outputs["metadata"]["center"] = preview["center"]

        if image_t2 and image_t2.get("path"):
            preview_t2 = _read_raster_preview(
                image_t2["path"],
                f"{uuid.uuid4().hex}_t2.png",
            )
            evidence["raster_t2"] = preview_t2

        if optical and optical.get("path"):
            preview_optical = _read_raster_preview(
                optical["path"],
                f"{uuid.uuid4().hex}_optical.png",
            )
            evidence["raster_optical"] = preview_optical

            if preview_optical.get("center"):
                outputs.setdefault("metadata", {})
                outputs["metadata"]["center"] = preview_optical["center"]

        if sar and sar.get("path"):
            preview_sar = _read_raster_preview(
                sar["path"],
                f"{uuid.uuid4().hex}_sar.png",
            )
            evidence["raster_sar"] = preview_sar

        outputs["evidence"] = evidence

    except Exception as exc:
        outputs.setdefault("metadata", {})
        outputs["metadata"]["raster_preview_error"] = str(exc)


def _attach_change_mask(
    outputs: Dict[str, Any],
    image_t1: str,
    image_t2: str,
) -> None:
    """
    Create a visual change overlay from the two uploaded rasters.

    This is intentionally a deterministic raster-difference visualization.
    It is NOT claimed as DCFF-Net inference.
    """
    if not image_t1 or not image_t2:
        return

    t1 = Path(image_t1)
    t2 = Path(image_t2)

    if t1.suffix.lower() not in {".tif", ".tiff"}:
        return
    if t2.suffix.lower() not in {".tif", ".tiff"}:
        return

    with rasterio.open(t1) as a, rasterio.open(t2) as b:
        out_h = min(a.height, b.height, 1400)
        out_w = min(a.width, b.width, 1400)

        def read_gray(dataset):
            data = dataset.read(
                1,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            return _stretch_float(data)

        def _stretch_float(data):
            finite = np.isfinite(data)
            if not finite.any():
                return np.zeros_like(data, dtype=np.float32)

            vals = data[finite]
            low, high = np.percentile(vals, [2, 98])
            if high <= low:
                return np.zeros_like(data, dtype=np.float32)

            return np.clip(
                (data - low) / (high - low),
                0.0,
                1.0,
            )

        g1 = read_gray(a)
        g2 = read_gray(b)

        diff = np.abs(g2 - g1)
        threshold = float(np.percentile(diff, 92))
        mask = diff >= threshold

        # Transparent overlay encoded as RGBA:
        # changed pixels = amber/white, unchanged = transparent.
        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        rgba[..., 0] = 240
        rgba[..., 1] = 160
        rgba[..., 2] = 60
        rgba[..., 3] = np.where(mask, 190, 0).astype(np.uint8)

        output_file = RASTER_ROOT / f"{uuid.uuid4().hex}_change.png"
        Image.fromarray(rgba, mode="RGBA").save(
            output_file,
            format="PNG",
            optimize=True,
        )

        bounds = a.bounds

        if a.crs is not None:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(
                a.crs,
                "EPSG:4326",
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )
        else:
            west, south, east, north = (
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

        evidence = _normalize_evidence(outputs)
        evidence["change_mask"] = {
            "url": (
                f"{_public_base_url()}/raster-previews/"
                f"{output_file.name}"
            ),
            "bounds": [
                [south, west],
                [north, east],
            ],
            "threshold": threshold,
            "method": "deterministic raster-difference visualization",
        }

        outputs["evidence"] = evidence
        outputs.setdefault("metadata", {})
        outputs["metadata"]["change_visualization"] = (
            "Deterministic raster-difference overlay; "
            "not DCFF-Net inference."
        )


@app.post("/api/agent/analyze")
async def analyze_agent(
    query: str = Form(...),
    modality: str = Form("optical"),
    task: str = Form(""),
    aoi: str = Form(""),
    layer: str = Form("True colour"),
    source: str = Form("Optical"),
    image: UploadFile | None = File(None),
    image_t1: UploadFile | None = File(None),
    image_t2: UploadFile | None = File(None),
    optical_image: UploadFile | None = File(None),
    sar_image: UploadFile | None = File(None),
):
    upload_root = Path("data/uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    inputs: Dict[str, Any] = {}

    async def save_upload(upload: UploadFile | None):
        if upload is None:
            return None

        suffix = Path(upload.filename or "").suffix.lower()

        allowed_extensions = {
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
        }

        if suffix not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {suffix or 'unknown'}"
            )

        destination = upload_root / f"{uuid.uuid4().hex}{suffix}"

        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        return str(destination)

    try:
        parsed_aoi = None

        if aoi.strip():
            try:
                parsed_aoi = json.loads(aoi)

                if parsed_aoi.get("type") != "Polygon":
                    raise ValueError("AOI must be a GeoJSON Polygon.")

                if not parsed_aoi.get("bbox"):
                    raise ValueError("AOI bbox is required.")

                if not parsed_aoi.get("geometry"):
                    raise ValueError("AOI geometry is required.")

            except json.JSONDecodeError:
                raise ValueError("AOI must be valid JSON.")

        inputs["request_context"] = {
            "task": task or None,
            "modality": modality,
            "aoi": parsed_aoi,
            "layer": layer,
            "source": source,
        }

        # Single-image VQA / grounding.
        if image is not None:
            image_path = await save_upload(image)
            inputs["single_image"] = {
                "path": image_path,
                "modality": modality,
            }

        # Change detection: explicitly populate image_t1 and image_t2.
        if image_t1 is not None:
            image_t1_path = await save_upload(image_t1)
            inputs["image_t1"] = {
                "path": image_t1_path,
                "modality": modality,
            }

        if image_t2 is not None:
            image_t2_path = await save_upload(image_t2)
            inputs["image_t2"] = {
                "path": image_t2_path,
                "modality": modality,
            }

        # Backward compatibility: if an older frontend sends image as
        # the first temporal scene, expose it as image_t1 too.
        if image is not None and image_t2 is not None:
            inputs["image_t1"] = {
                "path": inputs["single_image"]["path"],
                "modality": modality,
            }

        # Optical + SAR.
        if optical_image is not None:
            optical_path = await save_upload(optical_image)
            inputs["optical_image"] = {
                "path": optical_path,
                "modality": "optical",
            }

        if sar_image is not None:
            sar_path = await save_upload(sar_image)
            inputs["sar_image"] = {
                "path": sar_path,
                "modality": "sar",
            }

        result = agent.run(
            query=query,
            inputs=inputs,
        )

        # The executor returns a result envelope whose specialist output
        # lives under "outputs". Attach real raster evidence there.
        if isinstance(result.result, dict):
            outputs = result.result.get("outputs")

            if isinstance(outputs, dict):
                _attach_raster_evidence(
                    outputs,
                    inputs,
                )

                if (
                    inputs.get("image_t1")
                    and inputs.get("image_t2")
                ):
                    try:
                        _attach_change_mask(
                            outputs,
                            inputs["image_t1"]["path"],
                            inputs["image_t2"]["path"],
                        )
                    except Exception as exc:
                        outputs.setdefault("metadata", {})
                        outputs["metadata"]["change_mask_error"] = str(exc)

        return {
            "status": result.status,
            "query": result.query,
            "plan": result.plan,
            "result": result.result,
            "execution_trace": result.execution_trace,
        }

    except ValueError as exc:
        return {
            "status": "input_error",
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }