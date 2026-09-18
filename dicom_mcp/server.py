"""dicom-mcp server: exposes local DICOM studies to LLM clients over MCP.

Transports: stdio (default). Run with `python -m dicom_mcp.server`.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Annotated, Any

import pydicom
from pydantic import Field

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer

from . import __version__
from .config import Config
from .imaging import (geometry, pixel_stats, redact, render_png, sample_pixels)
from .indexer import Index, InstanceRec, Series, Study

config = Config.from_env()
index = Index(config)

INSTRUCTIONS = (
    "This server exposes local DICOM imaging studies. Workflow: "
    "1) dicom_status / dicom_list_studies to discover studies; "
    "2) dicom_study / dicom_series to pick a series; "
    "3) dicom_render or dicom_montage to LOOK at slices (returns PNG image "
    "content), dicom_pixel_stats / dicom_sample_pixels for quantitative "
    "values (e.g. Hounsfield units), dicom_instance_header for raw tags. "
    "Instances are referenced by 'ref' = file name, SOP Instance UID, "
    "absolute path, or '#<zero-based index>' within the series. "
    "Never invent UIDs; always copy them from a previous tool result. "
    "Studies may contain PHI: prefer anonymize=true in dicom_export."
)

mcp = MCPServer(
    "dicom-mcp",
    instructions=INSTRUCTIONS,
    version=__version__,
)

# --------------------------------------------------------------------- utils
Err = dict


def _j(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)


def _need_scan() -> None:
    index.maybe_rescan_if_env()
    index.ensure()


def _bad(msg: str) -> Err:
    return {"error": msg}


def _resolve_study(study_uid: str) -> Study | Err:
    _need_scan()
    hits = index.find_study(study_uid)
    if not hits:
        return _bad(f"No study matching '{study_uid}'. Use dicom_list_studies.")
    if len(hits) > 1:
        return _bad(
            f"'{study_uid}' is ambiguous across {len(hits)} studies: "
            + ", ".join(h.uid for h in hits)
        )
    return hits[0]


def _resolve_series(series_uid: str) -> tuple[Study, Series] | Err:
    _need_scan()
    hits = index.find_series(series_uid)
    if not hits:
        return _bad(
            f"No series matching '{series_uid}'. Use dicom_study to list series."
        )
    if len(hits) > 1:
        return _bad(
            f"'{series_uid}' is ambiguous across {len(hits)} studies: "
            + ", ".join(s.uid for _, s in hits)
        )
    return hits[0]


def _resolve_instance(study: Study, series: Series,
                      ref: str) -> tuple[InstanceRec, dict] | Err:
    rec = index.instance_by_path_or_uid(ref, series)
    if rec is None:
        ins = series.sorted_instances()
        return _bad(
            f"Instance '{ref}' not found in series {series.uid} "
            f"({len(ins)} instances). Try ref like '#0', a file name from "
            "dicom_series, or an SOP Instance UID."
        )
    return rec, _load(rec)


def _load(rec: InstanceRec, pixels: bool = True) -> dict:
    """Full dataset load with one retry; returns {'ds': Dataset} or {'error':...}."""
    try:
        ds = pydicom.dcmread(rec.path, force=True)
        return {"ds": ds}
    except Exception as e:
        return {"error": f"Failed to read {rec.path}: {type(e).__name__}: {e}"}


def _ds_or_err(loaded) -> Any:
    if "error" in loaded:
        return _bad(loaded["error"])
    return loaded["ds"]


# ---------------------------------------------------------------------- tools
@mcp.tool(annotations={"title": "DICOM server status"})
def dicom_status() -> str:
    """Report configured roots, scan freshness, study/series/instance counts,
    and any unreadable files. Call this first if you are unsure data exists."""
    index.maybe_rescan_if_env()
    index.ensure()
    mods: dict[str, int] = {}
    for st in index.studies.values():
        for s in st.series.values():
            mods[s.modality or "?"] = mods.get(s.modality or "?", 0) + len(s.instances)
    return _j({
        "roots": [str(p) for p in config.roots],
        "workdir": str(config.workdir),
        "scanned_at": index.scanned_at,
        "files_seen": index.files_seen,
        "studies": len(index.studies),
        "series": sum(len(st.series) for st in index.studies.values()),
        "instances": sum(st.instance_count for st in index.studies.values()),
        "instances_by_modality": mods,
        "unreadable_files": index.errors[:20],
        "unreadable_count": len(index.errors),
    })


@mcp.tool(annotations={"title": "Rescan DICOM roots"})
def dicom_rescan() -> str:
    """Force a fresh filesystem scan of the DICOM roots (use after dropping
    new files or calling dicom_import_zip)."""
    index.ensure(force=True)
    return _j({"studies": len(index.studies),
               "instances": sum(st.instance_count for st in index.studies.values()),
               "unreadable_count": len(index.errors)})


@mcp.tool(annotations={"title": "Import a DICOM zip archive"})
def dicom_import_zip(
    zip_path: Annotated[str, Field(description="Absolute path to a .zip containing DICOM files")],
    extract_subdir: Annotated[str | None, Field(description="Optional subdir name inside the work area")] = None,
) -> str:
    """Extract a zip of DICOM files into the server work area and index it.
    Safe: entries with path traversal are skipped."""
    zp = Path(zip_path)
    if not zp.is_file():
        return _j(_bad(f"Not a file: {zip_path}"))
    sub = extract_subdir or zp.stem
    dest = (config.workdir / "imports" / sub).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if not str(dest).startswith(str(config.workdir.resolve())):
        return _j(_bad("extract_subdir escapes the work directory"))
    n_ok = n_bad = 0
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.startswith(("/", "..")) or ":/" in name or ":" in name.split("/")[0]:
                n_bad += 1
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                n_bad += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n_ok += 1
    index.ensure(force=True)
    return _j({"extracted": n_ok, "skipped_unsafe": n_bad,
               "destination": str(dest),
               "studies_now": len(index.studies)})


@mcp.tool(annotations={"title": "List DICOM studies"})
def dicom_list_studies(
    patient_query: Annotated[str | None, Field(description="Substring match on patient name or ID (case-insensitive)")] = None,
    modality: Annotated[str | None, Field(description="e.g. CT, MR, US, DX")] = None,
    date_from: Annotated[str | None, Field(description="YYYYMMDD inclusive")] = None,
    date_to: Annotated[str | None, Field(description="YYYYMMDD inclusive")] = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List indexed studies (newest first) with compact summaries and series
    counts. Filter by patient text, modality, or study date range."""
    _need_scan()
    rows = sorted(index.studies.values(),
                  key=lambda s: (s.date, s.time, s.uid), reverse=True)
    if patient_query:
        q = patient_query.lower()
        rows = [s for s in rows
                if q in s.patient_name.lower() or q in s.patient_id.lower()
                or q in s.description.lower()]
    if modality:
        m = modality.upper()
        rows = [s for s in rows
                if any(x.modality.upper() == m for x in s.series.values())]
    if date_from:
        rows = [s for s in rows if s.date >= date_from]
    if date_to:
        rows = [s for s in rows if s.date <= date_to]
    total = len(rows)
    rows = rows[offset:offset + max(1, limit)]
    return _j({
        "total": total,
        "offset": offset,
        "returned": len(rows),
        "studies": [{
            "study_uid": s.uid, "date": s.date, "time": s.time,
            "description": s.description,
            "patient_id": s.patient_id, "patient_name": s.patient_name,
            "modalities": sorted({x.modality for x in s.series.values() if x.modality}),
            "series_count": len(s.series),
            "instance_count": s.instance_count,
        } for s in rows],
    })


@mcp.tool(annotations={"title": "One study in detail"})
def dicom_study(
    study_uid: Annotated[str, Field(description="Study Instance UID (prefixes allowed if unique)")],
) -> str:
    """Full study summary: patient, dates, and every series with modality,
    description, instance count and example instance refs."""
    study = _resolve_study(study_uid)
    if isinstance(study, dict):
        return _j(study)
    return _j(index.study_summary(study))


@mcp.tool(annotations={"title": "One series in detail"})
def dicom_series(
    series_uid: Annotated[str, Field(description="Series Instance UID (prefix if unique)")],
    include_instances: Annotated[bool, Field(description="Include every instance (file name, instance number, path)")] = True,
) -> str:
    """Series detail: geometry, spacing, and the instance list you pass to
    render/stats/header tools (use '#index' refs)."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    ins = series.sorted_instances()
    out = index.series_summary(series)
    out["study_uid"] = study.uid
    out["study_description"] = study.description
    if include_instances:
        out["instances"] = [
            {"index": i,
             "ref": f"#{i}",
             "file": Path(r.path).name,
             "instance_number": r.instance_number,
             "sop_instance_uid": r.sop_class_uid,
             "rows": r.rows, "columns": r.columns,
             "frames": r.number_of_frames,
             "bytes": r.size,
             "path": r.path}
            for i, r in enumerate(ins)
        ]
    # geometry from the middle instance
    mid = ins[len(ins) // 2] if ins else None
    if mid:
        loaded = _load(mid)
        if "error" not in loaded:
            out["geometry_sample"] = geometry(loaded["ds"])
    return _j(out)


@mcp.tool(annotations={"title": "Full DICOM header of one instance"})
def dicom_instance_header(
    series_uid: str,
    ref: Annotated[str, Field(description="'#<index>', file name, SOP UID, or absolute path")],
    tag_filter: Annotated[str | None, Field(description="Only tags whose keyword/name contains this text (case-insensitive)")] = None,
    include_private: bool = False,
    max_elements: int = 600,
) -> str:
    """Dump the DICOM data dictionary tags for one instance (values, VRs).
    Use tag_filter to keep output small."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    res = _resolve_instance(study, series, ref)
    if isinstance(res, dict):
        return _j(res)
    rec, loaded = res
    ds = _ds_or_err(loaded)
    if isinstance(ds, dict):
        return _j(ds)
    elems = []
    seen = 0
    for elem in ds:
        if elem.tag.is_private and not include_private:
            continue
        kw = elem.keyword or ""
        try:
            from pydicom.datadict import dictionary_keyword, dictionary_name
            name = dictionary_name(elem.tag)
        except Exception:
            name = str(elem.tag)
        key = f"{kw} {name}"
        if tag_filter and tag_filter.lower() not in key.lower():
            continue
        seen += 1
        if seen > max_elements:
            break
        val = elem.value
        if elem.VR in ("OB", "OW", "OF", "UN") and elem.tag.group != 0x0002:
            val = f"<binary {len(val) if hasattr(val, '__len__') else '?'} bytes>"
        elif isinstance(val, list) and len(val) > 16:
            val = list(val[:16]) + [f"... {len(val)} total"]
        elif isinstance(val, str) and len(val) > 500:
            val = val[:500] + "..."
        elems.append({"tag": str(elem.tag), "vr": elem.VR, "keyword": kw,
                      "name": name, "value": val})
    return _j({"path": rec.path, "transfer_syntax": rec.transfer_syntax,
               "element_count_shown": len(elems), "elements": elems})


@mcp.tool(annotations={"title": "Render a DICOM slice as PNG"})
def dicom_render(
    series_uid: str,
    ref: Annotated[str, Field(description="'#<index>', file name, SOP UID, or path")],
    frame: Annotated[int, Field(description="Frame index for multi-frame objects")] = 0,
    window_center: float | None = None,
    window_width: float | None = None,
    max_px: int | None = None,
) -> list:
    """Return a PNG image of one slice/frame with window/level applied.
    Without wc/ww uses DICOM default windows, else modality fallbacks
    (CT: soft tissue). Pass the returned metadata when discussing appearance.
    Common CT windows: lung C-600/W1500, bone C400/W1800, brain C40/W80.
    Result content: [image, metadata-json]."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return [_j(hit)]
    study, series = hit
    res = _resolve_instance(study, series, ref)
    if isinstance(res, dict):
        return [_j(res)]
    rec, loaded = res
    ds = _ds_or_err(loaded)
    if isinstance(ds, dict):
        return [_j(ds)]
    try:
        img, meta = render_png(ds, frame=frame, window_center=window_center,
                               window_width=window_width,
                               max_px=max_px or config.max_image_px)
    except ValueError as e:
        return [_j({"error": str(e)})]
    meta.update({"series_uid": series.uid, "ref": ref,
                 "file": Path(rec.path).name,
                 "modality": series.modality})
    return [img, _j(meta)]


@mcp.tool(annotations={"title": "Montage of multiple slices"})
def dicom_montage(
    series_uid: str,
    tiles: Annotated[int, Field(description="Number of tiles (1-36)")] = 9,
    refs: Annotated[list[str] | None, Field(description="Explicit list of refs; if omitted, sample evenly across the series")] = None,
    frame: int = 0,
    window_center: float | None = None,
    window_width: float | None = None,
    tile_px: int = 256,
) -> list:
    """Compose an N-tile montage PNG (great for overview of a CT/MR stack).
    Result content: [image, metadata-json]."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return [_j(hit)]
    study, series = hit
    ins = series.sorted_instances()
    if not ins:
        return [_j(_bad("empty series"))]
    if refs:
        chosen = []
        for r in refs:
            rr = index.instance_by_path_or_uid(r, series)
            if rr is None:
                return [_j(_bad(f"ref not found: {r}"))]
            chosen.append(rr)
    else:
        tiles = max(1, min(int(tiles), 36, len(ins)))
        idxs = [round(i * (len(ins) - 1) / (tiles - 1)) if tiles > 1 else len(ins) // 2
                for i in range(tiles)]
        chosen = [ins[i] for i in sorted(set(idxs))]
    from PIL import Image as PILImage
    imgs = []
    used = []
    err = None
    for rec in chosen:
        loaded = _load(rec)
        ds = _ds_or_err(loaded)
        if isinstance(ds, dict):
            err = ds["error"]
            continue
        try:
            mimg, meta = render_png(ds, frame=frame,
                                    window_center=window_center,
                                    window_width=window_width, max_px=tile_px)
        except ValueError as e:
            err = str(e)
            continue
        imgs.append(PILImage.open(__import__("io").BytesIO(mimg.data)))
        used.append({"ref": f"#{ins.index(rec)}",
                     "file": Path(rec.path).name,
                     "window": [meta["window_center"], meta["window_width"]]})
    if not imgs:
        return [_j(_bad(f"no tiles rendered; last error: {err}"))]
    cols = min(len(imgs), int(len(imgs) ** 0.5 + 0.999))
    rows = (len(imgs) + cols - 1) // cols
    tile_w = max(i.width for i in imgs)
    tile_h = max(i.height for i in imgs)
    sheet = PILImage.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    for k, im in enumerate(imgs):
        x = (k % cols) * tile_w + (tile_w - im.width) // 2
        y = (k // cols) * tile_h + (tile_h - im.height) // 2
        sheet.paste(im, (x, y))
    import io as _io
    buf = _io.BytesIO()
    sheet.save(buf, format="PNG")
    return [MCPImage(data=buf.getvalue(), format="png"), _j({
        "series_uid": series.uid, "tiles": len(imgs),
        "grid": {"cols": cols, "rows": rows},
        "tile_source": used,
        "partial_error": err,
    })]


@mcp.tool(annotations={"title": "Pixel statistics (HU) for an instance"})
def dicom_pixel_stats(
    series_uid: str,
    ref: str,
    frame: int = 0,
) -> str:
    """Histogram-style stats of stored pixel values; 'rescaled' block holds
    calibrated units (Hounsfield for CT: air -1000, water 0, cortical bone
    +700..+3000, acute blood +40..+80, fat -120..-60)."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    res = _resolve_instance(study, series, ref)
    if isinstance(res, dict):
        return _j(res)
    rec, loaded = res
    ds = _ds_or_err(loaded)
    if isinstance(ds, dict):
        return _j(ds)
    try:
        st = pixel_stats(ds, frame=frame)
    except ValueError as e:
        return _j({"error": str(e)})
    st["geometry"] = geometry(ds)
    return _j(st)


@mcp.tool(annotations={"title": "Sample pixel values at points"})
def dicom_sample_pixels(
    series_uid: str,
    ref: str,
    points: Annotated[list[list[int]], Field(description="List of [row, col] pixel coordinates (origin top-left)", min_length=1)],
    frame: int = 0,
) -> str:
    """Read stored+rescaled values (HU on CT) at specific coordinates — use
    to quantify a lesion the user points at in a rendered image."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    res = _resolve_instance(study, series, ref)
    if isinstance(res, dict):
        return _j(res)
    rec, loaded = res
    ds = _ds_or_err(loaded)
    if isinstance(ds, dict):
        return _j(ds)
    pts = [(int(p[0]), int(p[1])) for p in points]
    try:
        out = sample_pixels(ds, pts, frame=frame)
    except ValueError as e:
        return _j({"error": str(e)})
    return _j({"file": Path(rec.path).name, "samples": out})


@mcp.tool(annotations={"title": "Export instance/series to work area"})
def dicom_export(
    series_uid: str,
    ref: Annotated[str | None, Field(description="Single instance ref; omit to export the whole series")] = None,
    anonymize: Annotated[bool, Field(description="Strip PHI tags before writing (recommended for cloud LLMs)")] = False,
    as_zip: Annotated[bool, Field(description="Zip the output when exporting a series")] = True,
) -> str:
    """Copy original (or anonymized) DICOM files into the server work dir and
    return their absolute paths, so other local tools can consume them."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    outdir = config.workdir / "export" / f"{stamp}_{series.uid[:16]}"
    outdir.mkdir(parents=True, exist_ok=True)
    recs: list[InstanceRec]
    if ref is not None:
        res = _resolve_instance(study, series, ref)
        if isinstance(res, dict):
            return _j(res)
        recs = [res[0]]
    else:
        recs = series.sorted_instances()
    written = []
    for rec in recs:
        dest = outdir / Path(rec.path).name
        if anonymize:
            try:
                ds = pydicom.dcmread(rec.path, force=True)
            except Exception as e:
                written.append({"from": rec.path, "error": str(e)})
                continue
            try:
                rd = redact(ds)
                dest = dest.with_name(Path(rec.path).stem + "_anon.dcm")
                rd.save_as(dest, write_like_original=False)
            except Exception as e:
                written.append({"from": rec.path, "error": f"anon failed: {e}"})
                continue
        else:
            shutil.copy2(rec.path, dest)
        written.append({"path": str(dest), "bytes": dest.stat().st_size})
    archive = None
    if len(recs) > 1 and as_zip:
        archive = outdir.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for w in written:
                if "path" in w:
                    zf.write(w["path"], Path(w["path"]).name)
    return _j({"series_uid": series.uid, "anonymized": anonymize,
               "files": written, "zip": str(archive) if archive else None})


@mcp.tool(annotations={"title": "Full-text search across DICOM headers"})
def dicom_find_text(
    query: Annotated[str, Field(description="Case-insensitive substring, e.g. a description word or finding")],
    deep: Annotated[bool, Field(description="Scan every header tag (slower); default scans only common descriptive tags")] = False,
    limit: int = 40,
) -> str:
    """Search study/series/instance header text (descriptions, laterality,
    body part, SR text) and return matching series/instances with context."""
    _need_scan()
    q = query.lower()
    hits = []
    for st in index.studies.values():
        for se in st.series.values():
            for rec in se.instances:
                fields = {
                    "study_description": st.description,
                    "series_description": rec.series_desc,
                    "modality": rec.modality,
                }
                if deep:
                    try:
                        ds = pydicom.dcmread(rec.path, stop_before_pixels=True, force=True)
                    except Exception:
                        continue
                    for elem in ds:
                        if elem.VR in ("OB", "OW", "OF", "UN", "AT", "SL"):
                            continue
                        kw = elem.keyword or str(elem.tag)
                        sval = str(elem.value)
                        if len(sval) < 3000:
                            fields[kw] = sval
                for fld, val in fields.items():
                    if q in str(val).lower():
                        hits.append({
                            "study_uid": st.uid, "series_uid": se.uid,
                            "series_index_hint": se.description,
                            "field": fld, "value": str(val)[:300],
                            "file": Path(rec.path).name,
                            "path": rec.path,
                        })
                        break
                if len(hits) >= limit:
                    return _j({"query": query, "truncated": True, "hits": hits})
    return _j({"query": query, "truncated": False, "hits": hits,
               "note": "empty: try deep=True"})


@mcp.tool(annotations={"title": "Read a DICOM SR as text"})
def dicom_read_sr(
    series_uid: str,
    ref: Annotated[str, Field(description="'#0' for the first (usually only) SR instance")] = "#0",
    max_chars: int = 8000,
) -> str:
    """Flatten a Structured Report (SR) content tree into readable indented
    text with value types — the fastest way for an LLM to read radiology
    reports stored as DICOM."""
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    res = _resolve_instance(study, series, ref)
    if isinstance(res, dict):
        return _j(res)
    rec, loaded = res
    ds = _ds_or_err(loaded)
    if isinstance(ds, dict):
        return _j(ds)
    if not hasattr(ds, "ContentSequence"):
        return _j({"error": "This instance has no ContentSequence — it is not a "
                            "DICOM Structured Report."})
    lines: list[str] = []
    total = 0

    def walk(items, depth: int) -> None:
        nonlocal total
        for it in items:
            vt = str(getattr(it, "ValueType", ""))
            name = str(getattr(it, "ConceptNameCodeSequence", [None])[0]
                       .get("CodeMeaning", "") if getattr(it, "ConceptNameCodeSequence", None)
                       else "")
            txt = ""
            if vt == "TEXT":
                txt = str(getattr(it, "TextValue", ""))
            elif vt == "CODE":
                cs = getattr(it, "ConceptCodeSequence", None) or []
                txt = "; ".join(str(getattr(c, "CodeMeaning", "")) for c in cs)
            elif vt in ("NUM",):
                nv = getattr(it, "NumericValue", None)
                units = getattr(it, "MeasurementUnitsCodeSequence", None) or []
                utxt = str(getattr(units[0], "CodeMeaning", "")) if units else ""
                txt = f"{nv} {utxt}".strip()
            elif vt == "UIDREF":
                txt = str(getattr(it, "UID", ""))
            line = "  " * depth + (f"{name}: " if name else "") + txt
            total += len(line) + 1
            if total > max_chars:
                lines.append("... [truncated]")
                return
            lines.append(line)
            sub = getattr(it, "ContentSequence", None)
            if sub:
                walk(sub, depth + 1)

    walk(ds.ContentSequence, 0)
    return _j({
        "instance": Path(rec.path).name,
        "institution_note": "PHI may be present in report text",
        "title": str(getattr(ds, "ContentDescription", "")),
        "text": "\n".join(lines),
    })


# ----------------------------------------------------------------- resources
@mcp.resource("dicom://index/summary", mime_type="application/json",
              description="Whole-archive summary: roots, counts, studies")
def resource_index_summary() -> str:
    _need_scan()
    return _j({
        "roots": [str(p) for p in config.roots],
        "studies": [index.study_summary(s) for s in
                    sorted(index.studies.values(),
                           key=lambda s: s.date, reverse=True)],
    })


@mcp.resource("dicom://study/{study_uid}/summary", mime_type="application/json")
def resource_study_summary(study_uid: str) -> str:
    study = _resolve_study(study_uid)
    if isinstance(study, dict):
        return _j(study)
    return _j(index.study_summary(study))


@mcp.resource("dicom://series/{series_uid}/instances", mime_type="application/json")
def resource_series_instances(series_uid: str) -> str:
    hit = _resolve_series(series_uid)
    if isinstance(hit, dict):
        return _j(hit)
    study, series = hit
    return _j(index.series_summary(series) | {
        "instances": [{"index": i, "ref": f"#{i}",
                       "file": Path(r.path).name,
                       "instance_number": r.instance_number}
                      for i, r in enumerate(series.sorted_instances())]})


# ------------------------------------------------------------------- prompts
@mcp.prompt(description="Guided radiology overview of one study")
def study_overview(study_uid: str) -> str:
    return (
        f"Review DICOM study {study_uid} step by step: "
        "1) call dicom_study and summarize patient/study context; "
        "2) for each series call dicom_series and note modality/body part; "
        "3) for the most informative CT/MR series call dicom_montage "
        "(9 tiles) and describe what you see per slice; "
        "4) if an SR series exists, call dicom_read_sr and reconcile with "
        "the images; 5) end with findings, uncertainties, and what a human "
        "radiologist should verify. State clearly that this is not a "
        "medical diagnosis."
    )


def main() -> None:
    config.log(f"roots={[str(p) for p in config.roots]} workdir={config.workdir}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
