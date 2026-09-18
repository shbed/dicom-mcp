"""End-to-end test: spawn the server over stdio, call every tool, validate
image output and JSON shapes. Run: python tools/e2e_test.py"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["DICOM_ROOTS"] = str(ROOT / "dicom-data")
os.environ["DICOM_WORKDIR"] = str(ROOT / "test-work")

from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402
from mcp import ClientSession  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def jload(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def text_of(res) -> str:
    parts = []
    for c in res.content:
        if c.type == "text":
            parts.append(c.text)
    return "\n".join(parts)


async def main():
    params = StdioServerParameters(
        command=PY, args=["-m", "dicom_mcp.server"],
        cwd=str(ROOT), env={**os.environ})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            init = await s.initialize()
            check("initialize", init.server_info.name == "dicom-mcp",
                  str(init.server_info))

            tools = (await s.list_tools()).tools
            names = sorted(t.name for t in tools)
            print("  tools:", ", ".join(names))
            check("tool_count>=12", len(tools) >= 12, str(len(tools)))

            async def call(name, **kw):
                return await s.call_tool(name, kw)

            # ---- status / rescan
            r = await call("dicom_status")
            d = jload(text_of(r))
            check("status json", d is not None, text_of(r)[:200])
            check("status has studies", d and d.get("studies", 0) >= 2,
                  str(d and d.get("studies")))
            check("status unreadable counts junk",
                  d and d.get("unreadable_count", 0) >= 1)

            # ---- list studies
            r = await call("dicom_list_studies")
            d = jload(text_of(r))
            check("list studies", d and d.get("total", 0) >= 2,
                  str(d)[:200])
            study = next((st for st in d["studies"]
                          if "CT ABDOMEN" in (st["description"] or "")),
                         d["studies"][0])
            study_uid = study["study_uid"]

            # filter by modality + text
            r = await call("dicom_list_studies", modality="SR")
            d2 = jload(text_of(r))
            check("filter modality=SR", d2 and all(
                "SR" in st["modalities"] for st in d2["studies"]), str(d2)[:200])
            r = await call("dicom_list_studies", patient_query="phantom")
            d2 = jload(text_of(r))
            check("filter patient", d2 and d2["total"] >= 1)

            # ---- study detail
            r = await call("dicom_study", study_uid=study_uid)
            d = jload(text_of(r))
            check("study detail", d and len(d.get("series", [])) >= 1)
            series = d["series"][0]
            series_uid = series["series_uid"]

            # prefix resolution
            r = await call("dicom_study", study_uid=study_uid[:12])
            check("study prefix match", jload(text_of(r)) is not None)

            # ---- series detail
            r = await call("dicom_series", series_uid=series_uid)
            d = jload(text_of(r))
            check("series instances list",
                  d and len(d.get("instances", [])) == 8, str(d)[:200])
            check("geometry present",
                  d and "image_position_patient" in d.get("geometry_sample", {}))

            # ---- header
            r = await call("dicom_instance_header", series_uid=series_uid,
                           ref="#0", tag_filter="window")
            d = jload(text_of(r))
            check("header tag_filter", d and any(
                "Window" in e["keyword"] for e in d["elements"]), str(d)[:300])

            # ---- render
            r = await call("dicom_render", series_uid=series_uid, ref="#3")
            kinds = [c.type for c in r.content]
            check("render returns image", "image" in kinds, str(kinds))
            img = next(c for c in r.content if c.type == "image")
            meta = next((c for c in r.content if c.type == "text"), None)
            md = jload(meta.text) if meta else None
            check("render meta window soft tissue",
                  md and md.get("window_center") == 40.0, str(md))
            raw = img.data if isinstance(img.data, bytes) else \
                __import__("base64").b64decode(img.data)
            check("png magic", raw[:4] == b"\x89PNG", str(img.data)[:40])
            out = Path(ROOT / "test-render.png")
            out.write_bytes(raw)
            print(f"        saved render: {out} ({len(img.data)} bytes)")

            # explicit lung window
            r = await call("dicom_render", series_uid=series_uid, ref="#3",
                           window_center=-600, window_width=1500)
            md = jload(next(c.text for c in r.content if c.type == "text"))
            check("render explicit window",
                  md and md["window_center"] == -600.0)

            # ---- montage
            r = await call("dicom_montage", series_uid=series_uid, tiles=6)
            kinds = [c.type for c in r.content]
            check("montage image", "image" in kinds, str(kinds))
            md = jload(next(c.text for c in r.content if c.type == "text"))
            check("montage 6 tiles", md and md.get("tiles") == 6, str(md)[:200])
            out = Path(ROOT / "test-montage.png")
            img = next(c for c in r.content if c.type == "image")
            raw = img.data if isinstance(img.data, bytes) else \
                __import__("base64").b64decode(img.data)
            out.write_bytes(raw)
            print(f"        saved montage: {out} ({len(img.data)} bytes)")

            # ---- pixel stats
            r = await call("dicom_pixel_stats", series_uid=series_uid, ref="#3")
            d = jload(text_of(r))
            rescaled = d.get("rescaled", {}) if d else {}
            check("stats air ~-1000HU", rescaled and rescaled["min"] < -900,
                  str(rescaled))
            check("stats bone ~700HU", rescaled and rescaled["max"] > 600,
                  str(rescaled))

            # ---- sample pixels: center of body ellipse in a 128x128 image
            r = await call("dicom_sample_pixels", series_uid=series_uid,
                           ref="#3", points=[[64, 64], [0, 0], [200, 200]])
            d = jload(text_of(r))
            check("sample 3 pts", d and len(d["samples"]) == 3, str(d))
            check("corner is air", d and abs(d["samples"][1]["rescaled"] + 1000) < 1)
            check("out-of-bounds reported", d and "error" in d["samples"][2])

            # ---- SR read
            r = await call("dicom_list_studies", modality="SR")
            d = jload(text_of(r))
            sr_study = d["studies"][0]["study_uid"]
            r = await call("dicom_study", study_uid=sr_study)
            d = jload(text_of(r))
            sr_series = next(s["series_uid"] for s in d["series"]
                             if s["modality"] == "SR")
            r = await call("dicom_read_sr", series_uid=sr_series, ref="#0")
            d = jload(text_of(r))
            check("SR text has findings",
                  d and "hypodense" in d.get("text", ""), str(d)[:200])
            check("SR numeric item", d and "14" in d.get("text", ""))

            # ---- text search
            r = await call("dicom_find_text", query="lung")
            d = jload(text_of(r))
            check("find_text lung", d and len(d["hits"]) >= 1, str(d)[:200])

            # ---- export + anonymize
            r = await call("dicom_export", series_uid=series_uid, ref="#0",
                           anonymize=True)
            d = jload(text_of(r))
            check("export anon file", d and "path" in d["files"][0], str(d)[:200])
            if d and "path" in d["files"][0]:
                import pydicom
                ds = pydicom.dcmread(d["files"][0]["path"])
                gone = all(not hasattr(ds, k) for k in
                           ("PatientName", "PatientID", "ReferringPhysicianName",
                            "InstitutionName"))
                kept = hasattr(ds, "RescaleSlope") and hasattr(ds, "Rows")
                check("anon removed PHI", gone)
                check("anon kept clinical", kept)
                check("anon keeps SOP UID", hasattr(ds, "SOPInstanceUID"))

            # ---- zip import
            zip_path = ROOT / "dicom-data" / "lung_series.zip"
            r = await call("dicom_import_zip", zip_path=str(zip_path),
                           extract_subdir="ziptest")
            d = jload(text_of(r))
            check("zip import extracted", d and d["extracted"] == 6, str(d))

            # ---- MR reference file renders via auto window
            r = await call("dicom_list_studies", modality="MR")
            d = jload(text_of(r))
            mr_study = d["studies"][0]["study_uid"]
            r = await call("dicom_study", study_uid=mr_study)
            mr_series = jload(text_of(r))["series"][0]["series_uid"]
            r = await call("dicom_render", series_uid=mr_series, ref="#0")
            check("MR renders", any(c.type == "image" for c in r.content),
                  text_of(r)[:250])

            # ---- error paths
            r = await call("dicom_study", study_uid="1.2.3.999")
            check("bad uid error", "error" in text_of(r), text_of(r)[:120])
            r = await call("dicom_render", series_uid=series_uid,
                           ref="nosuchfile.dcm")
            check("bad ref error", "error" in text_of(r), text_of(r)[:120])
            r = await call("dicom_read_sr", series_uid=series_uid, ref="#0")
            check("SR on non-SR errors", "error" in text_of(r), text_of(r)[:120])

            # ---- resources & prompts
            res = await s.list_resources()
            check("static resource", any(
                "index/summary" in str(x.uri) for x in res.resources),
                  str([str(x.uri) for x in res.resources]))
            rr = await s.read_resource(f"dicom://study/{study_uid}/summary")
            body = rr.contents[0].text if rr.contents else ""
            check("read study resource", "series" in str(body), str(body)[:120])
            prompts = await s.list_prompts()
            check("prompt listed", any(
                p.name == "study_overview" for p in prompts.prompts))

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


asyncio.run(main())
