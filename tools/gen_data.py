"""Generate synthetic DICOM test data into ../dicom-data (and copy some
pydicom-bundled reference objects). Run: python tools/gen_data.py"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (CTImageStorage, ComprehensiveSRStorage,
                         generate_uid)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "dicom-data"
OUT.mkdir(exist_ok=True)

study_uid = generate_uid()
series_ct = generate_uid()
series_lung = generate_uid()


def _base(ds_cls=Dataset):
    ds = ds_cls()
    ds.PatientName = "TEST^PHANTOM"
    ds.PatientID = "PHANTOM-001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "O"
    ds.InstitutionName = "Synthetic General Hospital"
    ds.ReferringPhysicianName = "DOE^RADI"
    ds.AccessionNumber = "SYN20260001"
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20260901"
    ds.StudyTime = "101500.00"
    ds.StudyDescription = "CT ABDOMEN PELVIS W CONTRAST"
    ds.StudyID = "SYN1"
    ds.AccessionNumber = "SYN-ACC-1"
    return ds


def make_ct_slice(series_uid, series_desc, series_num, instance_num,
                  rows=128, cols=128, phantom="abdomen"):
    ds = _base()
    ds.SOPClassUID = CTImageStorage
    sop = generate_uid()
    ds.SOPInstanceUID = sop
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = series_desc
    ds.SeriesNumber = series_num
    ds.InstanceNumber = instance_num
    ds.Modality = "CT"
    ds.Manufacturer = "Synthetic Medical Systems"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.RescaleType = "HU"
    ds.WindowCenter = "40\\-600\\400"
    ds.WindowWidth = "400\\1500\\1800"
    ds.WindowCenterWidthExplanation = "Soft tissue\\Lung\\Bone"
    ds.SliceThickness = 5.0
    ds.PixelSpacing = [0.78, 0.78]
    ds.ImagePositionPatient = [-(cols * 0.78) / 2, -(rows * 0.78) / 2,
                               instance_num * 5.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = instance_num * 5.0
    # build phantom in HU then convert to stored = HU + 1024
    y, x = np.mgrid[0:rows, 0:cols]
    cy, cx = rows / 2, cols / 2
    hu = np.full((rows, cols), -1000.0)  # air
    # body ellipse ~ soft tissue
    body = ((x - cx) / (cols * 0.42)) ** 2 + ((y - cy) / (rows * 0.46)) ** 2 <= 1
    hu[body] = 45.0
    # spine posterior center: bone ~ 700 HU
    sp = ((x - cx) / 9) ** 2 + ((y - (cy + rows * 0.30)) / 7) ** 2 <= 1
    hu[sp] = 700.0
    # aorta: ~ 130 HU (post-contrast)
    ao = ((x - (cx + 10)) / 6) ** 2 + ((y - (cy + 6)) / 6) ** 2 <= 1
    hu[ao] = 130.0
    # liver-ish slab anterior right ~ 60 HU with a hypodense lesion ~ 15 HU
    lv = ((x - (cx - 25)) / 22) ** 2 + ((y - (cy - 22)) / 18) ** 2 <= 1
    hu[lv] = 60.0
    les = ((x - (cx - 20)) / 6) ** 2 + ((y - (cy - 20)) / 5) ** 2 <= 1
    hu[les & lv] = 15.0
    # lung windows only in a lung series variant
    if phantom == "lung":
        hu[:] = -850.0
        hu[body] = -820.0
        n = 12
        rng = np.random.default_rng(instance_num)
        for _ in range(n):  # vessels
            vx, vy = rng.normal(cx, 20), rng.normal(cy, 22)
            v = ((x - vx) / 2.2) ** 2 + ((y - vy) / 2.2) ** 2 <= 1
            hu[v & body] = -40.0
        # one 8mm nodule ~ -20 HU in right lower lobe
        nod = ((x - (cx - 28)) / 4) ** 2 + ((y - (cy + 18)) / 4) ** 2 <= 1
        hu[nod & body] = -20.0
    stored = (hu + 1024).astype(np.int16)
    # re-apply signed 16-bit: pydicom expects int16 for PixelRep=1
    ds.PixelData = stored.tobytes()
    return ds


def write(ds: Dataset, path: Path):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    fd = FileDataset(str(path), ds, file_meta=meta, preamble=b"\0" * 128)
    fd.save_as(path, write_like_original=False)
    return path


def code(scheme, meaning):
    c = Dataset()
    c.CodeValue = "123456"
    c.CodingSchemeDesignator = "SCT" if scheme == "SCT" else "99CM"
    c.CodeMeaning = meaning
    return c


def text_item(name, text, vt="TEXT"):
    it = Dataset()
    it.RelationshipType = "CONTAINS"
    it.ValueType = vt
    if name:
        it.ConceptNameCodeSequence = [code("99CM", name)]
    if vt == "TEXT":
        it.TextValue = text
    elif vt == "CODE":
        it.ConceptCodeSequence = [code("SCT", text)]
    return it


def num_item(name, value, units):
    it = Dataset()
    it.RelationshipType = "CONTAINS"
    it.ValueType = "NUM"
    it.ConceptNameCodeSequence = [code("SCT", name)]
    it.NumericValue = value
    mu = Dataset(); mu.CodeValue = "MM"; mu.CodingSchemeDesignator = "UCUM"
    mu.CodeMeaning = "mm"
    it.MeasurementUnitsCodeSequence = [mu]
    return it


def make_sr():
    ds = _base()
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 99
    ds.InstanceNumber = 1
    ds.Modality = "SR"
    ds.ContentDescription = "RADIOLOGY REPORT - CT ABDOMEN PELVIS"
    ds.CompletionFlag = "PARTIAL"
    ds.VerificationFlag = "UNVERIFIED"
    ds.InstanceAvailability = "ONLINE"
    ds.ContentSequence = [
        text_item("Findings", None, "TEXT") if False else text_item("", "Findings:"),
        text_item("Organ", "Liver normal in size and contour. "
                  "A 14 mm hypodense lesion in segment VII, "
                  "non-specific, recommend MRI for further characterization."),
        text_item("Organ", "Spleen, pancreas, adrenals unremarkable."),
        text_item("Organ", "Kidneys enhance symmetrically. No hydronephrosis."),
        text_item("Measurement", None) if False else num_item(
            "Lesion diameter", 14.0, "mm"),
        text_item("Impression", "1. 14 mm hypodense hepatic lesion, "
                  "indeterminate. 2. No acute abdominal process."),
        text_item("Attending", "DOE^RADI (electronically signed 20260901)"),
    ]
    return ds


def main():
    d_ct = OUT / "CT_ABDOMEN"
    d_lung = OUT / "CT_CHEST_LUNG"
    d_sr = OUT / "SR_REPORT"
    for d in (d_ct, d_lung, d_sr):
        d.mkdir(exist_ok=True)
    for i in range(1, 9):
        write(make_ct_slice(series_ct, "AXIAL SOFT TISSUE 5MM", 2, i),
              d_ct / f"I{i:03d}.dcm")
    for i in range(1, 7):
        write(make_ct_slice(series_lung, "AXIAL LUNG 5MM", 3, i,
                            phantom="lung"),
              d_lung / f"L{i:03d}.dcm")
    write(make_sr(), d_sr / "report.dcm")

    # pydicom bundled references (local, no download)
    from pydicom.data import get_testdata_file
    refs = ["MR_small.dcm", "US1.dcm", "rtdose.dcm", "emri_small.dcm",
            "workflow1_reordered.dcm"]
    n_refs = 0
    for r in refs:
        p = get_testdata_file(r)
        if p:
            shutil.copy(p, OUT / f"ref_{r}")
            n_refs += 1
    # a junk file + a zip of the lung series
    (OUT / "not_dicom.txt").write_text("this is not a dicom file\n")
    with zipfile.ZipFile(OUT / "lung_series.zip", "w") as zf:
        for f in sorted(d_lung.glob("*.dcm")):
            zf.write(f, f"lung/{f.name}")
    print(f"wrote {OUT}: 8 CT + 6 lung + 1 SR refs={n_refs} junk+zip")
    # also a zip for import testing (chest zip moved out of root so import
    # target is unique)


if __name__ == "__main__":
    main()
