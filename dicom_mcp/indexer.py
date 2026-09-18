"""Filesystem scan and study/series/instance index.

The index is built lazily on first use and refreshed when DICOM_RESCAN=1 or
when files discovered on a rescan are new/changed. All lookups are served
from memory so MCP tool calls stay fast.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

from .config import Config

# Read only the header (stop before large pixel data) during the scan.
_SCAN_TAGS = [
    "SOPInstanceUID", "SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID",
    "Modality", "StudyDate", "StudyTime", "StudyDescription", "SeriesDescription",
    "InstanceNumber", "SeriesNumber", "PatientID", "PatientName", "PatientBirthDate",
    "NumberOfFrames", "Rows", "Columns", "TransferSyntaxUID",
]


@dataclass
class InstanceRec:
    path: str
    sop_class_uid: str = ""
    study_uid: str = ""
    series_uid: str = ""
    modality: str = ""
    study_date: str = ""
    study_time: str = ""
    study_desc: str = ""
    series_desc: str = ""
    instance_number: str = ""
    series_number: str = ""
    patient_id: str = ""
    patient_name: str = ""
    patient_birth_date: str = ""
    rows: int | None = None
    columns: int | None = None
    number_of_frames: int | None = None
    transfer_syntax: str = ""
    size: int = 0
    mtime: float = 0.0
    # populated lazily after first full load
    _ds: Dataset | None = field(default=None, repr=False)


@dataclass
class Series:
    uid: str
    study_uid: str
    modality: str = ""
    description: str = ""
    series_number: str = ""
    instances: list[InstanceRec] = field(default_factory=list)

    def sorted_instances(self) -> list[InstanceRec]:
        def key(r: InstanceRec):
            try:
                n = float(r.instance_number or "nan")
            except ValueError:
                n = float("nan")
            return (n if n == n else float("inf"), r.path)
        return sorted(self.instances, key=key)


@dataclass
class Study:
    uid: str
    date: str = ""
    time: str = ""
    description: str = ""
    patient_id: str = ""
    patient_name: str = ""
    patient_birth_date: str = ""
    series: dict[str, Series] = field(default_factory=dict)

    @property
    def instance_count(self) -> int:
        return sum(len(s.instances) for s in self.series.values())


def _s(ds: Dataset, kw: str) -> str:
    try:
        v = ds.get(kw)
    except Exception:
        return ""
    if v is None:
        return ""
    return str(v).strip()


def _i(ds: Dataset, kw: str) -> int | None:
    try:
        v = ds.get(kw)
        if v is None:
            return None
        return int(ds[kw])
    except Exception:
        return None


class Index:
    def __init__(self, config: Config):
        self.config = config
        self.studies: dict[str, Study] = {}
        self.by_sop: dict[str, InstanceRec] = {}
        self.errors: list[dict] = []
        self.scanned_at: float | None = None
        self.files_seen: int = 0
        self._file_mtimes: dict[str, float] = {}
        self._by_path_rec: dict[str, InstanceRec] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ scan
    def ensure(self, force: bool = False) -> None:
        with self._lock:
            if self.scanned_at is not None and not force:
                return
            self._scan_locked()

    def maybe_rescan_if_env(self) -> None:
        if os.environ.get("DICOM_RESCAN") == "1":
            with self._lock:
                self._scan_locked()
            os.environ["DICOM_RESCAN"] = "0"

    def _iter_candidates(self):
        for root in self.config.roots:
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    yield os.path.join(dirpath, fn)

    def _scan_locked(self) -> None:
        t0 = time.time()
        self.errors = []
        self.files_seen = 0
        seen_files: dict[str, float] = {}
        by_path: dict[str, InstanceRec] = {}
        prev_recs = self._by_path_rec

        for path in self._iter_candidates():
            if len(seen_files) >= self.config.max_files:
                break
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_size == 0:
                continue
            seen_files[path] = st.st_mtime
            cached = prev_recs.get(path)
            if cached is not None and self._file_mtimes.get(path) == st.st_mtime:
                by_path[path] = cached
                continue
            rec = self._read_header(path, st)
            if rec is not None:
                by_path[rec.path] = rec

        # rebuild structures
        self._by_path_rec = by_path
        self._file_mtimes = seen_files
        studies: dict[str, Study] = {}
        self.by_sop = {}
        for rec in by_path.values():
            study = studies.get(rec.study_uid)
            if study is None:
                study = Study(
                    uid=rec.study_uid, date=rec.study_date, time=rec.study_time,
                    description=rec.study_desc, patient_id=rec.patient_id,
                    patient_name=rec.patient_name,
                    patient_birth_date=rec.patient_birth_date,
                )
                studies[rec.study_uid] = study
            ser = study.series.get(rec.series_uid)
            if ser is None:
                ser = Series(uid=rec.series_uid, study_uid=rec.study_uid,
                             modality=rec.modality,
                             description=rec.series_desc,
                             series_number=rec.series_number)
                study.series[rec.series_uid] = ser
            ser.instances.append(rec)
            if rec.sop_class_uid:
                self.by_sop[rec.sop_class_uid + ":" + rec.path] = rec
        self.studies = studies
        self.scanned_at = time.time()
        self.files_seen = len(seen_files)
        self.config.log(
            f"scan: {len(seen_files)} files, {len(studies)} studies, "
            f"{sum(s.instance_count for s in studies.values())} instances, "
            f"{len(self.errors)} unreadable, {time.time()-t0:.2f}s"
        )

    _by_path_rec: dict[str, InstanceRec]

    def _read_header(self, path: str, st) -> InstanceRec | None:
        try:
            ds = pydicom.dcmread(
                path, stop_before_pixels=True, force=True,
                specific_tags=[t for t in _SCAN_TAGS],
            )
        except Exception as e:  # not DICOM, or broken
            self.errors.append({"path": path, "error": f"{type(e).__name__}: {e}"})
            return None
        sop = _s(ds, "SOPInstanceUID")
        if not sop:
            self.errors.append({"path": path, "error": "no SOPInstanceUID"})
            return None
        study_uid = _s(ds, "StudyInstanceUID") or f"UNSTUDIED:{path}"
        series_uid = _s(ds, "SeriesInstanceUID") or f"NOSERIES:{path}"
        rec = InstanceRec(
            path=path,
            sop_class_uid=sop,
            study_uid=study_uid,
            series_uid=series_uid,
            modality=_s(ds, "Modality"),
            study_date=_s(ds, "StudyDate"),
            study_time=_s(ds, "StudyTime"),
            study_desc=_s(ds, "StudyDescription"),
            series_desc=_s(ds, "SeriesDescription"),
            instance_number=_s(ds, "InstanceNumber"),
            series_number=_s(ds, "SeriesNumber"),
            patient_id=_s(ds, "PatientID"),
            patient_name=_s(ds, "PatientName"),
            patient_birth_date=_s(ds, "PatientBirthDate"),
            rows=_i(ds, "Rows"),
            columns=_i(ds, "Columns"),
            number_of_frames=_i(ds, "NumberOfFrames"),
            transfer_syntax=str(ds.file_meta.TransferSyntaxUID)
            if ds.file_meta is not None else "",
            size=st.st_size,
            mtime=st.st_mtime,
        )
        return rec

    # ---------------------------------------------------------------- lookup
    def find_study(self, uid_prefix: str) -> list[Study]:
        uid_prefix = uid_prefix.strip()
        exact = self.studies.get(uid_prefix)
        if exact:
            return [exact]
        return [s for u, s in self.studies.items() if u.startswith(uid_prefix)]

    def find_series(self, uid_prefix: str) -> list[tuple[Study, Series]]:
        out = []
        uid_prefix = uid_prefix.strip()
        for study in self.studies.values():
            if uid_prefix in study.series:
                out.append((study, study.series[uid_prefix]))
            else:
                out.extend(
                    (study, s) for u, s in study.series.items()
                    if u.startswith(uid_prefix)
                )
        return out

    def series_instances(self, series: Series) -> list[InstanceRec]:
        return series.sorted_instances()

    def instance_by_path_or_uid(self, ref: str,
                                series: Series | None = None) -> InstanceRec | None:
        """Resolve by absolute path, by file name, or by SOP Instance UID."""
        ref = ref.strip()
        if os.path.isfile(ref):
            return self._by_path_rec.get(os.path.abspath(ref))
        p = Path(ref)
        if p.name != ref:
            r = self._by_path_rec.get(ref)
            if r:
                return r
        pool = series.instances if series else list(self._by_path_rec.values())
        by_name = [r for r in pool if Path(r.path).name == ref]
        if len(by_name) >= 1:
            return by_name[0]
        by_uid = [r for r in self._by_path_rec.values() if r.sop_class_uid == ref]
        if by_uid:
            return by_uid[0]
        # allow "index" like "#12" against a series
        if ref.startswith("#") and series is not None:
            try:
                idx = int(ref[1:])
                ins = series.sorted_instances()
                if 0 <= idx < len(ins):
                    return ins[idx]
            except ValueError:
                return None
        return None

    # ---------------------------------------------------------------- views
    def study_summary(self, study: Study) -> dict:
        return {
            "study_uid": study.uid,
            "date": study.date,
            "time": study.time,
            "description": study.description,
            "patient_id": study.patient_id,
            "patient_name": study.patient_name,
            "patient_birth_date": study.patient_birth_date,
            "modality": sorted({s.modality for s in study.series.values() if s.modality}),
            "series_count": len(study.series),
            "instance_count": study.instance_count,
            "series": [self.series_summary(s) for s in study.series.values()],
        }

    def series_summary(self, series: Series) -> dict:
        ins = series.sorted_instances()
        sizes = [r.size for r in ins] or [0]
        return {
            "series_uid": series.uid,
            "modality": series.modality,
            "description": series.description,
            "series_number": series.series_number,
            "instance_count": len(ins),
            "first_instance_path": ins[0].path if ins else None,
            "bytes_avg": int(sum(sizes) / len(sizes)),
            "example_instances": [
                {"index": i, "instance_number": r.instance_number,
                 "path": r.path, "rows": r.rows, "columns": r.columns}
                for i, r in enumerate(ins[:5])
            ],
        }
