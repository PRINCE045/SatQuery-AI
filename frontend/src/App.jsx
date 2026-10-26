import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Rectangle,
  Polygon,
  Polyline,
  ImageOverlay,
  useMap,
  useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

/* ------------------------------------------------------------------
    SatQuery AI — task-aware remote-sensing workstation

    npm install leaflet react-leaflet

    BACKEND CONTRACT
    -----------------
    POST /api/agent/analyze   (multipart/form-data)
      query         string
      task          "vqa" | "grounding" | "change" | "fusion"   (inferred client-side, backend may override)
      aoi           JSON string: { type:"Polygon", bbox:[w,s,e,n], geometry:GeoJSON, label }  |  ""
      layer         "True colour" | "NDVI" | "NDWI"
      source        "Optical" | "SAR"
      image         file   — vqa / grounding, optional
      image_t1      file   — change
      image_t2      file   — change
      image_optical  file   — fusion
      image_sar     file   — fusion

    Response 200
      {
        status,
        query,
        plan,
        result,
        execution_trace
      }

    Nothing below renders a field unless the response (or the drawn
    geometry) actually supplied it.
------------------------------------------------------------------ */

const QUICK = [
  "What crops are grown here?",
  "Map built-up expansion",
  "Show water bodies in this area",
  "What changed here?",
  "Analyze optical + SAR",
];

const LAYERS = ["True colour", "NDVI", "NDWI"];
const SOURCES = ["Optical", "SAR"];

const TRACE_STEPS = [
  { id: "query_interpreted", label: "Query interpreted" },
  { id: "inputs_validated", label: "Inputs validated" },
  { id: "specialist_selected", label: "Selecting specialist" },
  { id: "inference", label: "Running inference" },
  { id: "evidence_ready", label: "Preparing evidence" },
];

const EVIDENCE_LABELS = {
  raster: "Raster",
  raster_t2: "T2 raster",
  raster_optical: "Optical raster",
  raster_sar: "SAR raster",
  boxes: "Bounding box",
  change_mask: "Change mask",
  segmentation_mask: "Segmentation mask",
};

function inferTask(q) {
  const s = q.toLowerCase();
  if (/\bsar\b|optical\s*\+\s*sar|fusion/.test(s)) return { id: "fusion", label: "optical + SAR fusion", view: "dual" };
  if (/change|before and after|compare|difference/.test(s)) return { id: "change", label: "change detection", view: "split" };
  if (/where is|locate|boundary|outline|bounding box|find the/.test(s)) return { id: "grounding", label: "grounding", view: "single" };
  if (s.trim()) return { id: "vqa", label: "visual question answering", view: "single" };
  return null;
}

const DEFAULT_CENTER = [22.9734, 78.6569];
const DEFAULT_ZOOM = 4.4;
const R_EARTH = 6371;
const toRad = (d) => (d * Math.PI) / 180;

function bboxAreaKm2([w, s, e, n]) {
  const dLat = toRad(n - s);
  const dLon = toRad(e - w);
  const midLat = toRad((n + s) / 2);
  return Math.abs(R_EARTH * dLon * Math.cos(midLat) * (R_EARTH * dLat));
}

function ringToBounds(ring) {
  // ring: [[lon,lat], ...]
  const lats = ring.map((p) => p[1]);
  const lons = ring.map((p) => p[0]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

export default function App() {
  const [query, setQuery] = useState("");
  const [center, setCenter] = useState(DEFAULT_CENTER);
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  const [layer, setLayer] = useState("True colour");
  const [source, setSource] = useState("Optical");
  const [opacity, setOpacity] = useState(80);

  const [aoi, setAoi] = useState(null); // { label, bbox, ring:[[lon,lat],...], boundsArr }
  const [shapeMode, setShapeMode] = useState("rectangle"); // rectangle | polygon
  const [drawing, setDrawing] = useState(false);
  const [draftPoints, setDraftPoints] = useState([]); // live polygon/rect preview, latlng objects

  const [place, setPlace] = useState("");
  const [placeBusy, setPlaceBusy] = useState(false);

  const [files, setFiles] = useState({}); // { image, image_t1, image_t2, image_optical, image_sar }

  const [status, setStatus] = useState("idle"); // idle | running | done | offline
  const [doneTrace, setDoneTrace] = useState([]);
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);
  const [activeBox, setActiveBox] = useState(null);
  const [note, setNote] = useState("");

  const inferred = inferTask(query);

  const taskFromId = (id) => {
    if (id === "fusion") {
      return { id: "fusion", label: "optical + SAR fusion", view: "dual" };
    }
    if (id === "change") {
      return { id: "change", label: "change detection", view: "split" };
    }
    if (id === "grounding") {
      return { id: "grounding", label: "grounding", view: "single" };
    }
    if (id === "vqa") {
      return { id: "vqa", label: "visual question answering", view: "single" };
    }
    return { id: null, label: null, view: "single" };
  };

  const task =
    typeof selected?.task_view === "string"
      ? taskFromId(selected.task_view)
      : selected?.task_view || inferred || taskFromId(null);

  /* ---- request-driven trace: the backend's own progress if it streams one,
     otherwise a local placeholder that advances while waiting ---- */
  useEffect(() => {
    if (status !== "running") return;
    setDoneTrace([]);
    let i = 0;
    const id = setInterval(() => {
      i = Math.min(i + 1, TRACE_STEPS.length - 1);
      setDoneTrace(TRACE_STEPS.slice(0, i).map((s) => s.id));
    }, 450);
    return () => clearInterval(id);
  }, [status]);

  /* ---- AOI: rectangle or polygon, always real geometry ---- */
  const finalizeAoi = useCallback((latlngs) => {
    if (latlngs.length < (shapeMode === "rectangle" ? 2 : 3)) {
      setNote("Draw a larger area.");
      return;
    }
    let ring;
    if (shapeMode === "rectangle") {
      const lats = latlngs.map((p) => p.lat);
      const lngs = latlngs.map((p) => p.lng);
      const s = Math.min(...lats), n = Math.max(...lats);
      const w = Math.min(...lngs), e = Math.max(...lngs);
      if (n - s < 0.0008 || e - w < 0.0008) {
        setNote("Area too small — drag a larger box.");
        return;
      }
      ring = [[w, s], [e, s], [e, n], [w, n], [w, s]];
    } else {
      ring = latlngs.map((p) => [p.lng, p.lat]);
      ring.push(ring[0]);
    }
    const bbox = ringToBounds(ring);
    setAoi({
      label: "Custom area",
      bbox,
      ring,
      boundsArr: ring.map(([lon, lat]) => [lat, lon]),
    });
    setNote("");
    setDrawing(false);
    setDraftPoints([]);
  }, [shapeMode]);

  const clearAoi = () => {
    setAoi(null);
    setResults([]);
    setSelected(null);
    setActiveBox(null);
    setStatus("idle");
  };

  /* ---- place search moves the map only — it never sets the AOI ---- */
  const goToPlace = async (e) => {
    e.preventDefault();
    if (!place.trim()) return;
    setPlaceBusy(true);
    setNote("");
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(place)}`
      );
      const data = await res.json();
      if (!data.length) setNote(`No match for “${place}”.`);
      else {
        setCenter([+data[0].lat, +data[0].lon]);
        setZoom(11);
      }
    } catch {
      setNote("Couldn't reach the geocoder — draw an area on the map instead.");
    } finally {
      setPlaceBusy(false);
    }
  };

  /* ---- file inputs, adaptive to the inferred task ---- */
  const setFile = (key, f) => setFiles((prev) => ({ ...prev, [key]: f || undefined }));

  const uploadSlots = useMemo(() => {
    if (!inferred) return [];
    if (inferred.id === "change") {
      return [
        { key: "image_t1", label: "T1 image", required: true },
        { key: "image_t2", label: "T2 image", required: true },
      ];
    }
    if (inferred.id === "fusion") {
      return [
        { key: "image_optical", label: "Optical image", required: true },
        { key: "image_sar", label: "SAR image", required: true },
      ];
    }
    return [{ key: "image", label: "Reference image", required: true }];
  }, [inferred]);

  /* ---- run ---- */
  const run = async () => {
    if (!query.trim()) {
      setNote("Type a question first.");
      return;
    }

    if (inferred?.id === "vqa" && !files.image) {
      setNote("Upload a reference image before running visual question answering.");
      return;
    }

    if (inferred?.id === "change" && (!files.image_t1 || !files.image_t2)) {
      setNote("Upload both T1 and T2 images before running change detection.");
      return;
    }

    if (inferred?.id === "fusion" && (!files.image_optical || !files.image_sar)) {
      setNote("Upload both Optical and SAR images before running fusion analysis.");
      return;
    }

    setStatus("running");
    setNote("");
    setActiveBox(null);
    setResults([]);
    setSelected(null);
    setDoneTrace([]);

    const form = new FormData();

    form.append("query", query);
    form.append("modality", source.toLowerCase());

    if (aoi) {
      form.append(
        "aoi",
        JSON.stringify({
          type: "Polygon",
          bbox: aoi.bbox,
          geometry: {
            type: "Polygon",
            coordinates: [aoi.ring],
          },
          label: aoi.label,
        })
      );
    } else {
      form.append("aoi", "");
    }

    form.append("layer", layer);
    form.append("source", source);

    Object.entries(files).forEach(([key, file]) => {
      if (file) {
        form.append(key, file);
      }
    });

    try {
      const response = await fetch("/api/agent/analyze", {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      const backendResult = data.result;
      const outputs = backendResult?.outputs;

      if (!backendResult || !outputs) {
        throw new Error("Invalid analysis response.");
      }

      const taskTitles = {
        vqa: "Visual Question Answering",
        grounding: "Text-Guided Region Grounding",
        change: "Bi-Temporal Change Analysis",
        fusion: "Optical + SAR Fusion",
      };

      const taskDescriptions = {
        vqa: "Natural-language understanding of the uploaded remote-sensing scene.",
        grounding: "Locating the region described by the user's query.",
        change: "Comparing corresponding observations across two dates.",
        fusion: "Joint interpretation of optical and SAR observations.",
      };

      const currentTaskId = inferred?.id ?? "vqa";

      const uiResult = {
        id: `${Date.now()}`,
        task: backendResult.tool_name,
        task_view: currentTaskId,
        model: outputs.model ?? null,
        title: taskTitles[currentTaskId] ?? data.plan?.description ?? "Satellite image analysis",
        description: taskDescriptions[currentTaskId] ?? "Remote-sensing analysis completed.",
        question: query,
        answer: outputs.answer ?? outputs.message ?? "No answer was returned.",
        confidence:
          typeof outputs.confidence === "number" && outputs.confidence > 0
            ? `${Math.round(outputs.confidence * 100)}%`
            : null,
        satellite: outputs.metadata?.satellite ?? null,
        acquisition: outputs.metadata?.acquisition ?? null,
        cloud: outputs.metadata?.cloud ?? null,
        bands: outputs.metadata?.bands ?? null,
        tile: outputs.metadata?.tile ?? null,
        center: outputs.metadata?.center ?? null,
        zoom: outputs.metadata?.zoom ?? null,
        evidence: outputs.evidence ?? {},
        artifacts: outputs.metadata?.artifacts ?? [],
        changeDetected: outputs.change_detected ?? null,
        changedAreaPercent: outputs.changed_area_percent ?? null,
        metrics: outputs.metrics ?? {},
      };

      const traceStages = new Set();

      for (const step of data.execution_trace ?? []) {
        if (step.stage === "query_interpretation") {
          traceStages.add("query_interpreted");
        }

        if (step.stage === "input_validation") {
          traceStages.add("inputs_validated");
        }

        if (step.stage === "tool_selection") {
          traceStages.add("specialist_selected");
        }

        if (step.stage === "tool_execution") {
          traceStages.add("inference");
        }

        if (step.stage === "evidence") {
          traceStages.add("evidence_ready");
        }
      }

      setDoneTrace([...traceStages]);

      if (outputs.status === "unavailable") {
        setNote(outputs.message ?? "Inference backend unavailable.");
      }

      setResults([uiResult]);
      setSelected(uiResult);
      setStatus("done");
    } catch (error) {
      console.error("SatQuery analysis failed:", error);

      setResults([]);
      setSelected(null);
      setStatus("offline");
      setNote(
        error instanceof Error
          ? `Analysis failed: ${error.message}`
          : "Analysis service unreachable."
      );
    }
  };

  const viewOnMap = (r) => {
    setSelected(r);
    setActiveBox(null);
    if (r.center) {
      setCenter([r.center.lat, r.center.lon]);
      setZoom(r.zoom ?? 12);
    }
  };

  const aoiArea = aoi ? bboxAreaKm2(aoi.bbox) : null;
  const aoiValue = aoi ? aoi.label : status === "running" ? "resolving" : "not selected";

  const evidenceChips =
    selected?.evidence && !Array.isArray(selected.evidence)
      ? Object.keys(selected.evidence).filter(
          (k) => EVIDENCE_LABELS[k] && selected.evidence[k]
        )
      : [];
  if (selected?.tile || selected?.acquisition) evidenceChips.push("metadata");

  return (
    <div className="app">
      {/* ------------ TOP BAR ------------ */}
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <span className="brand-name">SatQuery<em>AI</em></span>
        </div>

        <dl className="readout">
          <div className={`field ${aoi ? "set" : ""}`}>
            <dt>Area of interest</dt>
            <dd>{aoiValue}</dd>
          </div>
          <div className={`field ${aoi ? "set" : ""}`}>
            <dt>Bounding box</dt>
            <dd>{aoi ? aoi.bbox.map((v) => v.toFixed(3)).join(", ") : "—"}</dd>
          </div>
          <div className="field">
            <dt>Date range</dt>
            <dd>{aoi ? "last 30 days" : "—"}</dd>
          </div>
          <div className="field">
            <dt>Satellite</dt>
            <dd>{selected?.satellite ?? "—"}</dd>
          </div>
        </dl>

        <div className="sys">
          <span className={`dot ${status}`} />
          <span className="sys-label">
            {status === "offline" ? "backend offline" : status === "running" ? "working" : status === "done" ? "complete" : "ready"}
          </span>
          <span className="sih">SIH26167</span>
        </div>
      </header>

      {/* ------------ WORKSPACE ------------ */}
      <main className="workspace">
        {/* ---- query panel ---- */}
        <section className="panel query-panel">
          <h2>Ask</h2>

          <div className="query-box">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
              placeholder={aoi ? `Ask about ${aoi.label.toLowerCase()} — “what changed here?”` : "Ask about anywhere, or draw an area on the map first"}
              spellCheck="false"
            />
            <div className="query-foot">
              <span className="hint">
                {inferred ? `Detected — ${inferred.label}` : aoi ? `“here” = ${aoi.label.toLowerCase()}` : "no area bound"}
              </span>
              <button className="run" onClick={run} disabled={status === "running"}>
                {status === "running" ? "Analysing" : "Analyse"}
              </button>
            </div>
          </div>

          {uploadSlots.length > 0 && (
            <div className="uploads">
              <div className="data-input-heading">
                <span className="cg-label">Data input</span>
                {inferred?.id === "vqa" && (
                  <span className="input-requirement">REQUIRED</span>
                )}
              </div>
              <div className="upload-row">
                {uploadSlots.map((slot) => (
                  <UploadSlot
                    key={slot.key}
                    slot={slot}
                    file={files[slot.key]}
                    onChange={(f) => setFile(slot.key, f)}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="quick">
            {QUICK.map((q) => <button key={q} onClick={() => setQuery(q)}>{q}</button>)}
          </div>

          {note && <p className="note">{note}</p>}

          {(status === "running" || status === "done") && (
            <ol className="trace">
              {TRACE_STEPS.map((s, i) => {
                const doneIx = doneTrace.includes(s.id);
                const activeIx = status === "running" && !doneIx && (i === 0 || doneTrace.includes(TRACE_STEPS[i - 1].id));
                return (
                  <li key={s.id} className={doneIx ? "done" : activeIx ? "active" : ""}>
                    <span className="mark-icon">{doneIx ? "✓" : activeIx ? "●" : "○"}</span>
                    {s.label}
                  </li>
                );
              })}
              {status === "done" && <li className="complete">Analysis complete</li>}
            </ol>
          )}

          <div className="results">
            <div className="results-head">
              <h3>Results</h3>
              <span className="count">{results.length} result{results.length === 1 ? "" : "s"}</span>
            </div>

            {results.length === 0 ? (
              <div className="empty">
                <svg viewBox="0 0 32 32" className="empty-icon" aria-hidden="true"><circle cx="16" cy="16" r="10" /><circle cx="16" cy="16" r="2.4" /></svg>
                <p>No analysis yet</p>
                <p className="empty-sub">Ask a question, or draw an area on the map, to begin.</p>
              </div>
            ) : (
              results.map((r, i) => (
                <article key={i} className={`result ${selected === r ? "on" : ""}`}>
                  <header className="result-header">
                    <div className="result-heading">
                      <span className="result-kicker">ANALYSIS COMPLETE</span>
                      <h4 className="r-title">{r.title}</h4>
                      <p className="r-description">{r.description}</p>
                    </div>
                    {r.confidence && <span className="r-conf">{r.confidence}</span>}
                  </header>
                  <div className="question-context">
                    <span className="question-label">QUERY</span>
                    <span>{r.question}</span>
                  </div>
                  <div className="answer-block">
                    <span className="answer-label">SATQUERY RESPONSE</span>
                    <p className="r-answer">{r.answer}</p>
                  </div>
                  <div className="result-meta">
                    <span>
                      MODEL
                      <strong>{r.model || "—"}</strong>
                    </span>
                    <span>
                      TASK
                      <strong>
                        {{
                          vqa: "Visual Question Answering",
                          grounding: "Region Grounding",
                          change: "Change Analysis",
                          fusion: "Optical + SAR Fusion",
                        }[r.task_view] || "Analysis"}
                      </strong>
                    </span>
                  </div>
                  <footer>
                    <span>{[r.satellite, r.acquisition].filter(Boolean).join(" · ") || "Remote sensing scene"}</span>
                    <button className="view-btn" onClick={() => viewOnMap(r)}>View on map →</button>
                  </footer>
                </article>
              ))
            )}
          </div>
        </section>

        {/* ---- spatial panel ---- */}
        <section className="panel spatial-panel">
          <div className="map-shell">
            <div className="map-top">
              <form className="place" onSubmit={goToPlace}>
                <input value={place} onChange={(e) => setPlace(e.target.value)} placeholder="Search a place" aria-label="Search a place" />
                <button type="submit" disabled={placeBusy}>{placeBusy ? "…" : "Go"}</button>
              </form>
              <span className="place-note">moves the map only — draw to set AOI</span>

              {inferred && <span className="task-pill">{inferred.label}</span>}

              <span className={`aoi-strip ${aoi ? "on" : ""}`}>
                {aoi ? `AOI bound · ${aoiArea < 1 ? (aoiArea * 100).toFixed(1) + " ha" : aoiArea.toFixed(1) + " km²"} · bbox available` : "no area bound"}
              </span>
            </div>

            <div className={`map-grid view-${task.view}`}>
              <div className="map-frame">
                <MapContainer center={center} zoom={zoom} className="leaflet-box" zoomControl={false} scrollWheelZoom>
                  <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <ViewSync
                    center={center}
                    zoom={zoom}
                    bounds={
                      selected?.evidence?.raster?.bounds ||
                      selected?.evidence?.raster_optical?.bounds ||
                      null
                    }
                  />
                  <DrawLayer drawing={drawing} shapeMode={shapeMode} onFinalize={finalizeAoi} onDraft={setDraftPoints} />

                  {aoi && <Polygon positions={aoi.boundsArr} pathOptions={{ color: "#f0a03c", weight: 1.4, fillOpacity: 0.08 }} />}

                  {drawing && shapeMode === "rectangle" && draftPoints.length === 2 && (
                    <Rectangle
                      bounds={[[Math.min(draftPoints[0].lat, draftPoints[1].lat), Math.min(draftPoints[0].lng, draftPoints[1].lng)],
                               [Math.max(draftPoints[0].lat, draftPoints[1].lat), Math.max(draftPoints[0].lng, draftPoints[1].lng)]]}
                      pathOptions={{ color: "#f0a03c", dashArray: "4 5", weight: 1, fillOpacity: 0.04 }}
                    />
                  )}
                  {drawing && shapeMode === "polygon" && draftPoints.length > 0 && (
                    <Polyline positions={draftPoints.map((p) => [p.lat, p.lng])} pathOptions={{ color: "#f0a03c", dashArray: "4 5", weight: 1.4 }} />
                  )}

                  {selected?.evidence?.raster?.url &&
                    selected.evidence.raster.bounds && (
                      <ImageOverlay
                        url={selected.evidence.raster.url}
                        bounds={selected.evidence.raster.bounds}
                        opacity={opacity / 100}
                        zIndex={300}
                      />
                    )}

                  {selected?.evidence?.raster_optical?.url &&
                    selected.evidence.raster_optical.bounds && (
                      <ImageOverlay
                        url={selected.evidence.raster_optical.url}
                        bounds={selected.evidence.raster_optical.bounds}
                        opacity={opacity / 100}
                        zIndex={300}
                      />
                    )}

                  {selected?.evidence?.change_mask?.url &&
                    selected.evidence.change_mask.bounds && (
                      <ImageOverlay
                        url={selected.evidence.change_mask.url}
                        bounds={selected.evidence.change_mask.bounds}
                        opacity={opacity / 100}
                        zIndex={500}
                      />
                    )}

                  {selected?.evidence?.segmentation_mask?.url &&
                    selected.evidence.segmentation_mask.bounds && (
                      <ImageOverlay
                        url={selected.evidence.segmentation_mask.url}
                        bounds={selected.evidence.segmentation_mask.bounds}
                        opacity={opacity / 100}
                        zIndex={500}
                      />
                    )}
                  {selected?.evidence?.boxes?.map((b, i) => (
                    <Rectangle
                      key={i}
                      bounds={b.bounds}
                      pathOptions={{ color: activeBox === i ? "#5fd3c4" : "#f0a03c", weight: activeBox === i ? 2.2 : 1.3, fillOpacity: activeBox === i ? 0.14 : 0.04 }}
                      eventHandlers={{ click: () => setActiveBox(i) }}
                    />
                  ))}
                </MapContainer>
                <div className="overlay-scan" style={{ opacity: opacity / 220 }} />
                <span className="corner tl">
                  {selected?.evidence?.raster?.url ||
                  selected?.evidence?.raster_optical?.url
                    ? `${layer} · ${source} · scene`
                    : `${layer} · ${source}`}
                </span>
              </div>

              {task.view !== "single" && (
                <div className="map-frame secondary">
                  <MapContainer
                    center={center}
                    zoom={zoom}
                    className="leaflet-box"
                    zoomControl={false}
                    dragging={false}
                    scrollWheelZoom={false}
                    doubleClickZoom={false}
                  >
                    <TileLayer
                      attribution="&copy; OpenStreetMap contributors"
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <ViewSync
                      center={center}
                      zoom={zoom}
                      bounds={
                        task.view === "split"
                          ? selected?.evidence?.raster_t2?.bounds || null
                          : task.view === "dual"
                            ? selected?.evidence?.raster_sar?.bounds || null
                            : null
                      }
                    />

                    {task.view === "split" &&
                      selected?.evidence?.raster_t2?.url &&
                      selected.evidence.raster_t2.bounds && (
                        <ImageOverlay
                          url={selected.evidence.raster_t2.url}
                          bounds={selected.evidence.raster_t2.bounds}
                          opacity={opacity / 100}
                          zIndex={300}
                        />
                      )}

                    {task.view === "dual" &&
                      selected?.evidence?.raster_sar?.url &&
                      selected.evidence.raster_sar.bounds && (
                        <ImageOverlay
                          url={selected.evidence.raster_sar.url}
                          bounds={selected.evidence.raster_sar.bounds}
                          opacity={opacity / 100}
                          zIndex={300}
                        />
                      )}
                  </MapContainer>

                  <span className="corner tl">
                    {task.view === "split" ? "T2 · Optical" : "SAR"}
                  </span>

                  <span className="corner br">
                    {task.view === "split"
                      ? selected?.evidence?.raster_t2
                        ? "scene loaded"
                        : "awaiting scene"
                      : selected?.evidence?.raster_sar
                        ? "scene loaded"
                        : "awaiting scene"}
                  </span>
                </div>
              )}
            </div>

            {selected?.evidence?.boxes?.length > 0 && (
              <div className="box-list">
                <span className="cg-label">Detections</span>
                <div className="box-chips">
                  {selected.evidence.boxes.map((b, i) => (
                    <button key={i} className={activeBox === i ? "on" : ""} onClick={() => setActiveBox(i)}>
                      {b.label ?? `region ${i + 1}`}{b.score ? ` · ${Math.round(b.score * 100)}%` : ""}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {evidenceChips.length > 0 && (
              <div className="evidence-row">
                <span className="cg-label">Evidence</span>
                <div className="evidence-chips">
                  {evidenceChips.map((k) => (
                    <span key={k} className="chip">{EVIDENCE_LABELS[k] ?? "Source metadata"}</span>
                  ))}
                </div>
              </div>
            )}

            {selected?.task_view === "change" &&
              (selected.changedAreaPercent != null ||
                selected.changeDetected != null) && (
                <div className="change-summary">
                  <span className="cg-label">Change summary</span>
                  <div className="change-summary-values">
                    <span>
                      DETECTED
                      <strong>
                        {selected.changeDetected ? "YES" : "NO"}
                      </strong>
                    </span>
                    <span>
                      CHANGED AREA
                      <strong>
                        {selected.changedAreaPercent != null
                          ? `${Number(selected.changedAreaPercent).toFixed(2)}%`
                          : "—"}
                      </strong>
                    </span>
                  </div>
                </div>
              )}

            <div className="map-controls">
              <div className="control-group">
                <span className="cg-label">Layer</span>
                <div className="seg">{LAYERS.map((l) => <button key={l} className={layer === l ? "on" : ""} onClick={() => setLayer(l)}>{l}</button>)}</div>
              </div>
              <div className="control-group">
                <span className="cg-label">Source</span>
                <div className="seg">{SOURCES.map((s) => <button key={s} className={source === s ? "on" : ""} onClick={() => setSource(s)}>{s}</button>)}</div>
              </div>
              <label className="opacity">
                Overlay
                <input type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
                <span>{opacity}%</span>
              </label>
              <div className="control-group">
                <span className="cg-label">Shape</span>
                <div className="seg">
                  <button className={shapeMode === "rectangle" ? "on" : ""} onClick={() => setShapeMode("rectangle")}>Rectangle</button>
                  <button className={shapeMode === "polygon" ? "on" : ""} onClick={() => setShapeMode("polygon")}>Polygon</button>
                </div>
              </div>
              <div className="tools">
                <button className={drawing ? "on" : ""} onClick={() => { setDrawing((d) => !d); setDraftPoints([]); }}>
                  {drawing ? (shapeMode === "polygon" ? "Double-click to finish" : "Drawing…") : "Draw area"}
                </button>
                <button onClick={clearAoi} disabled={!aoi}>Clear</button>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ------------ EVIDENCE CONTRACT ------------ */}
      <section className="contract">
        <div className="contract-head">
          <h2>Evidence contract</h2>
          <p>Every answer carries its provenance. Fields stay empty until analysis fills them.</p>
        </div>
        <table>
          <thead>
            <tr className="group-row"><th colSpan={5}>Source</th><th colSpan={3}>Analysis</th></tr>
            <tr>
              <th>Tile</th><th>Bounding box</th><th>Acquired</th><th>Cloud</th><th>Bands</th>
              <th>Model</th><th>Confidence</th><th>Artefacts</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{selected?.tile ?? "—"}</td>
              <td className="mono">{aoi ? aoi.bbox.map((v) => v.toFixed(3)).join(", ") : "—"}</td>
              <td>{selected?.acquisition ?? "—"}</td>
              <td>{selected?.cloud ?? "—"}</td>
              <td>{selected?.bands ?? "—"}</td>
              <td>{selected?.model ?? "—"}</td>
              <td>{selected?.confidence ?? "—"}</td>
              <td>{selected?.artifacts?.length ? selected.artifacts.map((a, i) => (
                <a key={i} className="link" href={a.url} target="_blank" rel="noreferrer">{a.name}</a>
              )) : "—"}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <footer className="statusbar">
        <span>SatQuery core</span><span>Query engine</span><span>Analysis</span><span>Evidence pipeline</span>
        <span className="ver">v0.3.0</span>
      </footer>
    </div>
  );
}

function UploadSlot({ slot, file, onChange }) {
  const isTiff = file && /\.(tif|tiff)$/i.test(file.name);
  const isBrowserPreviewable =
    file && /^image\/(png|jpeg|jpg|webp|gif)$/i.test(file.type);

  const url = useMemo(
    () => (file && isBrowserPreviewable ? URL.createObjectURL(file) : null),
    [file, isBrowserPreviewable]
  );

  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url]
  );

  return (
    <label className={`upload-slot ${file ? "has-file" : ""}`}>
      {!file && <span className="upload-plus">+</span>}
      {file && url && <img src={url} alt="" className="upload-preview" />}
      {file && isTiff && (
        <div className="raster-file-card">
          <div className="raster-icon">TIFF</div>
          <div className="raster-info">
            <strong>GeoTIFF</strong>
            <span title={file.name}>
              {file.name.length > 22 ? `${file.name.slice(0, 19)}...` : file.name}
            </span>
          </div>
        </div>
      )}
      {file && !url && !isTiff && (
        <div className="raster-file-card">
          <div className="raster-icon">IMG</div>
          <div className="raster-info">
            <strong>Image</strong>
            <span title={file.name}>
              {file.name.length > 22 ? `${file.name.slice(0, 19)}...` : file.name}
            </span>
          </div>
        </div>
      )}
      {!file && <span className="upload-label">{slot.label}</span>}
      {file && <div className="upload-file-name">{slot.label}</div>}
      <input
        type="file"
        accept="image/*,.tif,.tiff"
        hidden
        onChange={(e) => onChange(e.target.files?.[0])}
      />
      {file && (
        <button
          type="button"
          className="upload-remove"
          aria-label="Remove uploaded file"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onChange(null);
          }}
        >
          ×
        </button>
      )}
    </label>
  );
}

function FitToRasterBounds({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (!bounds || !Array.isArray(bounds) || bounds.length !== 2) {
      return;
    }

    const [[south, west], [north, east]] = bounds;

    if (
      !Number.isFinite(south) ||
      !Number.isFinite(west) ||
      !Number.isFinite(north) ||
      !Number.isFinite(east)
    ) {
      return;
    }

    if (north <= south || east <= west) {
      return;
    }

    const leafletBounds = [
      [south, west],
      [north, east],
    ];

    map.fitBounds(leafletBounds, {
      padding: [18, 18],
      maxZoom: 15,
      animate: false,
    });
  }, [map, JSON.stringify(bounds)]);

  return null;
}

function ViewSync({ center, zoom, bounds }) {
  const map = useMap();

  useEffect(() => {
    if (bounds && Array.isArray(bounds) && bounds.length === 2) {
      const [[south, west], [north, east]] = bounds;

      if (
        [south, west, north, east].every(Number.isFinite) &&
        north > south &&
        east > west
      ) {
        const fit = () => {
          map.invalidateSize({ pan: false });

          map.fitBounds(
            [
              [south, west],
              [north, east],
            ],
            {
              padding: [4, 4],
              animate: false,
            },
          );
        };

        // The result panel changes the map layout after inference.
        // Recalculate the Leaflet viewport after the container has its
        // final dimensions, otherwise fitBounds can leave large empty
        // areas around a small raster overlay.
        requestAnimationFrame(fit);
        window.setTimeout(fit, 80);
        return;
      }
    }

    map.setView(center, zoom, { animate: false });
  }, [center, zoom, bounds, map]);

  return null;
}

/* rectangle: drag. polygon: click to add vertices, double-click to close. */
function DrawLayer({ drawing, shapeMode, onFinalize, onDraft }) {
  const map = useMap();
  const dragStart = useRef(null);
  const points = useRef([]);

  useEffect(() => {
    map.dragging[drawing && shapeMode === "rectangle" ? "disable" : "enable"]();
    map.getContainer().style.cursor = drawing ? "crosshair" : "";
    if (!drawing) { points.current = []; dragStart.current = null; }
  }, [drawing, shapeMode, map]);

  useMapEvents({
    mousedown(e) {
      if (!drawing || shapeMode !== "rectangle") return;
      dragStart.current = e.latlng;
    },
    mousemove(e) {
      if (!drawing || shapeMode !== "rectangle" || !dragStart.current) return;
      onDraft([dragStart.current, e.latlng]);
    },
    mouseup(e) {
      if (!drawing || shapeMode !== "rectangle" || !dragStart.current) return;
      onFinalize([dragStart.current, e.latlng]);
      dragStart.current = null;
    },
    click(e) {
      if (!drawing || shapeMode !== "polygon") return;
      points.current = [...points.current, e.latlng];
      onDraft(points.current);
    },
    dblclick(e) {
      if (!drawing || shapeMode !== "polygon") return;
      onFinalize(points.current);
      points.current = [];
    },
  });

  return null;
}