# dicom-mcp

MCP server that exposes local DICOM imaging studies to a local (or any) LLM
over stdio: study/series discovery, raw headers, rendered slices and
montages as PNG image content, pixel statistics (Hounsfield units), point
sampling, DICOM-SR report reading, text search, zip import, and PHI-safe
export.

Built on the official Python MCP SDK 2.x (`MCPServer`) and pydicom 3.x.

## Layout

```
dicom_mcp/
  server.py    MCP tools, resources, prompts (entry point)
  indexer.py   filesystem scan + study/series/instance index (lazy, cached)
  imaging.py   window/level, PNG rendering, pixel stats, PHI redaction
  config.py    env-driven configuration
tools/
  gen_data.py  synthetic test phantom (CT abdomen + lung + SR + refs)
  e2e_test.py  full MCP client test over stdio (39 checks)
```

## Quick start

```bat
cd C:\Users\shbedbed\dicom-mcp
.venv\Scripts\python tools\gen_data.py      :: optional demo data
.venv\Scripts\python tools\e2e_test.py      :: verify everything
.venv\Scripts\dicom-mcp.exe                 :: run the server on stdio
```

## Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `DICOM_ROOTS` | `%LOCALAPPDATA%\hermes\dicom-inbox`, `./dicom-data` | path-sep separated dirs the server may read |
| `DICOM_WORKDIR` | `%LOCALAPPDATA%\hermes\dicom-mcp-work` | zip imports, exports |
| `DICOM_MAX_FILES` | 20000 | scan cap |
| `DICOM_MAX_IMAGE_PX` | 1024 | max PNG edge |
| `DICOM_RESCAN=1` | — | force rescan on next call |

## Tools

| Tool | Purpose |
|---|---|
| `dicom_status` | roots, counts, unreadable files — call first |
| `dicom_rescan` | force fresh scan after adding files |
| `dicom_list_studies` | list/filter studies (patient, modality, date) |
| `dicom_study` / `dicom_series` | detailed summaries, instance lists |
| `dicom_instance_header` | raw tags with `tag_filter` |
| `dicom_render` | one slice → PNG (window/level aware) + metadata |
| `dicom_montage` | up to 36-tile overview PNG |
| `dicom_pixel_stats` | stored + rescaled (HU) statistics |
| `dicom_sample_pixels` | HU at exact [row,col] points |
| `dicom_read_sr` | flatten DICOM SR tree to readable text |
| `dicom_find_text` | search header text (`deep=True` for all tags) |
| `dicom_export` | copy instance/series out, `anonymize=true` strips PHI |
| `dicom_import_zip` | safe zip extraction into work dir + index |

Resources: `dicom://index/summary`, `dicom://study/{uid}/summary`,
`dicom://series/{uid}/instances`. Prompt: `study_overview`.

Instance `ref` values accepted: `#<index>` (zero-based, from
`dicom_series`), file name, SOP Instance UID, or absolute path.

## Hermes integration

Already configured in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  dicom:
    command: C:/Users/shbedbed/dicom-mcp/.venv/Scripts/dicom-mcp.exe
    env:
      DICOM_ROOTS: C:/Users/shbedbed/dicom-mcp/dicom-data
      DICOM_WORKDIR: C:/Users/shbedbed/dicom-mcp/work
```

Restart Hermes; tools appear as `mcp_dicom_*`.

### Claude Desktop / Cursor (same pattern)

```json
{
  "mcpServers": {
    "dicom": {
      "command": "C:/Users/shbedbed/dicom-mcp/.venv/Scripts/dicom-mcp.exe",
      "env": { "DICOM_ROOTS": "D:/your/dicom/folder" }
    }
  }
}
```

## Privacy notes

- The server reads only from `DICOM_ROOTS` and writes only under
  `DICOM_WORKDIR`.
- Rendered PNGs show image content only, but pixel values and headers are
  exposed as text tools — treat those as PHI.
- Use `dicom_export` with `anonymize=true` before sharing with cloud LLMs.
  Redaction removes names, dates, physician/institution fields and private
  tags; UIDs and clinical attributes are kept. This is best-effort, not
  a DICOM PS3.15 E.01 de-identification.
- Not a medical device. Output is informational only.

## pydicom 3.x gotchas handled here

- `Dataset.pixel_array` is a property (stored values); modality LUT
  (`RescaleSlope/Intercept`) must be applied manually.
- `tag.is_private` is a property, not a method.
- `pydicom.deid` (with `Dataset.deidentify()`) no longer exists.
- MCP SDK 2.x: `FastMCP` → `MCPServer` (`mcp.server.mcpserver`); image
  content arrives base64-encoded on the wire.
