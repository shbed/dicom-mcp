"""Pixel rendering: window/level, colormap, PNG encoding for LLM display,
and numeric pixel access for quantitative questions."""
from __future__ import annotations

import io
import math

import numpy as np
import pydicom
from PIL import Image as PILImage
from pydicom.dataset import Dataset

from mcp.server.mcpserver import Image as MCPImage

PHI_TAGS = [
    # patient
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime",
    "PatientAge", "PatientSex", "PatientSize", "PatientWeight",
    "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
    "OtherPatientNames", "PatientComments", "EthnicGroup", "Occupation",
    # provider / institution
    "ReferringPhysicianName", "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "NameOfPhysiciansReadingStudy", "OperatorsName", "PerformingPhysicianName",
    "RequestingPhysician", "InstitutionName", "InstitutionAddress",
    "InstitutionalDepartmentName",
    # dates/times/ids
    "AccessionNumber", "OtherAccessionNumbers",
    "StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
    "AcquisitionDate", "AcquisitionTime", "ContentDate", "ContentTime",
    "InstanceCreationDate", "InstanceCreationTime", "VerificationDateTime",
    "ObservationDateTime", "StartDate", "CompletionDate", "AdmissionDate",
    "DischargeDate", "ScheduleingPhysiciansName", "ReferencedStudySequence",
    # free text
    "PhysiciansOfRecord", "ReadingPhysician", "RequestAttributesSequence",
    "CommentsOnThePerformedProcedureStep", "ReasonForTheImagingServiceRequest",
    "Indications", "PatientTransportArrangements",
]

_PHI_KEYWORDS = (
    "name", "physician", "address", "telephone", "birth", "accession",
    "institution", "operator", "ethnic", "occupation", "comments",
    "indications", "patientid", "patientsex", "patientage", "patientweight",
    "patientsize", "otherpatient", "reasonfor", "transportarrangements",
    "admissiondate", "dischargedate", "verified",
)
_KEEP_NOT_WITH_NAME = ("uid", "orientation", "position", "spacing")


def redact(ds: Dataset) -> Dataset:
    """Return a copy with obvious PHI removed; keeps clinical attributes."""
    out = ds.copy()
    for kw in PHI_TAGS:
        if kw in out:
            del out[kw]
    # second pass by keyword heuristic on top-level elements
    for elem in list(out):
        kw = (elem.keyword or "").lower()
        if not kw:
            continue
        if elem.tag.is_private:
            del out[elem.tag]
            continue
        if kw.endswith("uid"):  # UIDs are pseudonymous link keys, keep them
            continue
        if any(w in kw for w in _KEEP_NOT_WITH_NAME):
            continue
        if any(w in kw for w in _PHI_KEYWORDS):
            try:
                del out[elem.tag]
            except KeyError:
                pass
    return out


def render_png(ds: Dataset, frame: int = 0,
               window_center: float | None = None,
               window_width: float | None = None,
               max_px: int = 1024) -> tuple[MCPImage, dict]:
    """Render one frame to a PNG MCPImage. Returns (image, info)."""
    arr, info = _pixel_array(ds, frame)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    is_color = not photometric.startswith("MONOCHROME") and arr.ndim == 3 \
        and arr.shape[-1] in (3, 4)
    if is_color:
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            a = rgb.astype(np.float64)
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            a = (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)
            rgb = (a * 255).astype(np.uint8)
        wc, ww, wc_src = None, None, "none_color_image"
    else:
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        wc, ww, wc_src = _resolve_window(ds, window_center, window_width)
        if ww and ww > 0:
            lo = wc - ww / 2.0
            arr = np.clip((arr - lo) / ww, 0.0, 1.0)
        else:
            lo = float(np.nanmin(arr)) if arr.size else 0.0
            hi = float(np.nanmax(arr)) if arr.size else 1.0
            if hi - lo < 1e-9:
                arr = np.zeros_like(arr, dtype=np.float64)
            else:
                arr = (arr - lo) / (hi - lo)
        g = _window_lut(photometric)(arr)
        rgb = np.stack([g, g, g], axis=-1)
    img = PILImage.fromarray(rgb, mode="RGB")
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)),
                          max(1, int(img.height * ratio))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = {
        "rows": info["rows"], "columns": info["columns"],
        "window_center": wc, "window_width": ww,
        "window_center_source": wc_src,
        "resized_to": {"width": img.width, "height": img.height},
    }
    return MCPImage(data=buf.getvalue(), format="png"), meta


def pixel_stats(ds: Dataset, frame: int = 0) -> dict:
    """Quantitative summary of stored values (HU when rescale applies)."""
    arr, _ = _pixel_array(ds, frame, raw=True)
    a = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr.ravel()
    if a.size == 0:
        return {"error": "no pixels"}
    out = {
        "dtype": str(arr.dtype),
        "count": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": round(float(np.mean(a)), 3),
        "p05": round(float(np.percentile(a, 5)), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
    }
    intercept = _num(getattr(ds, "RescaleIntercept", None))
    slope = _num(getattr(ds, "RescaleSlope", None))
    units = str(getattr(ds, "RescaleType", "") or "")
    if slope is not None and intercept is not None and (slope != 1 or intercept != 0):
        out["rescaled"] = {
            "slope": slope, "intercept": intercept, "units": units or None,
            "min": out["min"] * slope + intercept,
            "max": out["max"] * slope + intercept,
            "mean": out["mean"] * slope + intercept,
            "p05": out["p05"] * slope + intercept,
            "p95": out["p95"] * slope + intercept,
        }
    return out


def sample_pixels(ds: Dataset, points: list[tuple[int, int]],
                  frame: int = 0) -> list[dict]:
    """Sample stored values at (row, col) points; rescale if present."""
    arr, _ = _pixel_array(ds, frame, raw=True)
    intercept = _num(getattr(ds, "RescaleIntercept", None)) or 0.0
    slope = _num(getattr(ds, "RescaleSlope", None)) or 1.0
    units = str(getattr(ds, "RescaleType", "") or "")
    out = []
    h = arr.shape[0] if arr.ndim >= 2 else 1
    w = arr.shape[1] if arr.ndim >= 2 else arr.shape[0]
    for (r, c) in points:
        if not (0 <= r < h and 0 <= c < w):
            out.append({"row": r, "col": c, "error": f"out of bounds (h={h}, w={w})"})
            continue
        v = float(arr[r, c]) if arr.ndim >= 2 else float(arr[c])
        sv = v * slope + intercept
        out.append({"row": r, "col": c, "stored": v,
                    "rescaled": round(sv, 3), "units": units or None})
    return out


def geometry(ds: Dataset) -> dict:
    """Spatial geometry when present (for multi-slice series)."""
    out: dict = {}
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None:
        out["image_position_patient"] = [float(x) for x in ipp]
    if iop is not None:
        out["image_orientation_patient"] = [float(x) for x in iop]
    sp = _num(getattr(ds, "SliceThickness", None))
    if sp is not None:
        out["slice_thickness_mm"] = sp
    pd = _num(getattr(ds, "PixelSpacing", None))
    if pd is not None:
        out["pixel_spacing_mm"] = pd
    elif getattr(ds, "ImagerPixelSpacing", None) is not None:
        out["imager_pixel_spacing_mm"] = [
            float(x) for x in ds.ImagerPixelSpacing]
    if "image_position_patient" in out and "image_orientation_patient" in out:
        row = np.array(out["image_orientation_patient"][:3])
        col = np.array(out["image_orientation_patient"][3:])
        n = np.cross(row, col)
        out["slice_normal"] = [round(float(x), 6) for x in n]
    sps = _num(getattr(ds, "SpacingBetweenSlices", None))
    if sps is not None:
        out["spacing_between_slices_mm"] = sps
    return out
def _num(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pixel_array(ds: Dataset, frame: int = 0,
                 raw: bool = False) -> tuple[np.ndarray, dict]:
    """Decode pixels. raw=True returns stored values (no modality LUT).

    pydicom 3.x: `ds.pixel_array` is a property returning *stored* values;
    the modality LUT (RescaleSlope/Intercept -> e.g. HU) is applied here.
    """
    if ds.get("PixelData") is None:
        raise ValueError(
            f"No PixelData in this instance (SOP class "
            f"{getattr(ds, 'SOPClassUID', '?')}). It is likely structured "
            "report / raw data / an overlay-only object.")
    try:
        arr = ds.pixel_array
    except Exception as e:
        raise ValueError(
            f"Could not decode pixel data (transfer syntax "
            f"{ds.file_meta.TransferSyntaxUID if ds.file_meta else '?'}): "
            f"{type(e).__name__}: {e}. Try `dicom_export` to get the file "
            "in native form.") from e
    if not raw:
        intercept = _num(getattr(ds, "RescaleIntercept", None)) or 0.0
        slope = _num(getattr(ds, "RescaleSlope", None))
        if slope is None:
            slope = 1.0
        if slope != 1.0 or intercept != 0.0:
            arr = arr.astype(np.float64) * slope + intercept
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    if arr.ndim == 3 and rows and cols:
        if arr.shape[-2:] == (rows, cols):
            arr = arr[min(frame, arr.shape[0] - 1)]  # multi-frame, frame first
    if arr.ndim == 2:
        pass
    info = {
        "rows": rows or arr.shape[-2],
        "columns": cols or arr.shape[-1],
        "frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
    }
    return np.asarray(arr), info


def _resolve_window(ds: Dataset, wc, ww) -> tuple[float | None, float | None, str]:
    def _pair(tag):
        v = getattr(ds, tag, None)
        if v is None:
            return None, None
        try:
            if isinstance(v, pydicom.multival.MultiValue):
                return float(v[0]), float(v[1]) if len(v) > 1 else None
            parts = [p.strip() for p in str(v).split("\\")]
            return float(parts[0]), float(parts[1]) if len(parts) > 1 else None
        except (ValueError, IndexError):
            return None, None

    if wc is not None or ww is not None:
        c = float(wc) if wc is not None else 0.0
        w = float(ww) if ww is not None else 4.0 * max(abs(c), 1.0)
        return c, (w if w > 0 else 1.0), "caller"
    wcs = getattr(ds, "WindowCenter", None)
    wws = getattr(ds, "WindowWidth", None)
    if wcs is not None and wws is not None:
        try:
            centers = wcs if isinstance(wcs, pydicom.multival.MultiValue) else str(wcs).split("\\")
            widths = wws if isinstance(wws, pydicom.multival.MultiValue) else str(wws).split("\\")
            # prefer a named explanation that matches an index
            expl = getattr(ds, "WindowCenterWidthExplanation", None)
            idx = 0
            if expl is not None:
                names = expl if isinstance(expl, pydicom.multival.MultiValue) else [expl]
                for i, n in enumerate(names):
                    if "mediastin" in str(n).lower() or "soft" in str(n).lower():
                        idx = i
                        break
            c = float(centers[min(idx, len(centers) - 1)])
            w = float(widths[min(idx, len(widths) - 1)])
            return c, (w if w > 0 else 1.0), "dicom_default"
        except (ValueError, IndexError):
            pass
    # sensible fallbacks by modality
    mod = str(getattr(ds, "Modality", "")).upper()
    if mod == "CT":
        return 40.0, 400.0, "fallback_ct_soft_tissue"
    return None, None, "auto_minmax"


def _window_lut(photometric: str):
    invert = "1" in photometric[-1]

    def f(a01: np.ndarray) -> np.uint8:
        a = 1.0 - a01 if invert else a01
        return (a * 255.0 + 0.5).astype(np.uint8)

    return f
