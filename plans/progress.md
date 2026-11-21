# V3R Progress

Newest entry first. One entry per task card or meaningful work chunk.

---

## 2026-11-21 — Solv3R pivot: CLI, worker and Blender add-on

**Status:** done. Video and stream modes verified end-to-end; CPU backend default on this
machine because the ROCm 7.2 wheel has no gfx1010 binaries (D-012).

**Files created / changed**

- `worker/` — `config.py`, `source.py`, `points.py`, `outputs.py`, `session.py`, `app.py`:
  JSON config validation, video/stream ingest, depth→points, artifacts, Solv3R streaming.
- `v3r_cli.py` — command-line entry; same worker package as the Blender plugin.
- `V3R/services/point_cloud.py` — native Point Cloud object with per-point `radius` and `Col`;
  EEVEE/Cycles render it without per-point meshes (D-014).
- `V3R/services/reconstruction_service.py` — spawns the worker, reads JSON lines in a thread,
  applies points on a `bpy.app.timers` tick, stops with timeout.
- `V3R/properties/v3r_settings.py` — source, output, worker paths, device, preset, resolution,
  sample fps, max frames, point stride/budget/radius, depth, stream timeout, full Solv3R knobs.
- `V3R/operators/reconstruction.py` — start/stop, open output, clear cloud.
- `V3R/panels/v3r_panel.py` — source, run controls, status, advanced sub-panel.
- `V3R/__init__.py`, `V3R/blender_manifest.toml` — v2.0.0.
- `tests/test_worker_config.py`, `tests/test_worker_points.py`,
  `tests/blender_smoke.py`, `tests/blender_stream_smoke.py`, `tests/mjpeg_server.py`.
- `README.md`, `docs/DECISIONS.md` (D-011…D-014), `AGENTS.md`, `pyproject.toml`, `.gitignore`.

**Verification (from repo root)**

| Command | Result |
|---|---|
| `.venv/bin/ruff format core worker V3R tests v3r_cli.py` | clean |
| `.venv/bin/ruff check core worker V3R tests v3r_cli.py` | All checks passed |
| `.venv/bin/mypy core worker tests v3r_cli.py` | Success: no issues |
| `.venv/bin/pytest -q` | 51 passed |
| `v3r_cli.py --source samples/drone1.mp4 --sample-fps 2 --max-frames 10` | 10 frames, 17 243 points, report.md written |
| `v3r_cli.py --source http://127.0.0.1:8768 --source-kind stream --max-frames 6` | 6 frames, 10 248 points, no errors |
| `blender -b --factory-startup --python tests/blender_smoke.py` | 6 frames, 10 295 points, no error |
| `blender -b --factory-startup --python tests/blender_stream_smoke.py` | 6 frames, 10 251 points, no error |
| `blender -b --factory-startup --python tests/blender_render_smoke.py` | EEVEE render of 13 718 points |
| `blender -b --factory-startup --python-expr "...addon_enable(module='V3R')"` | ENABLED True, panel registered |
| `v3r_cli.py --no-depth --max-frames 6` | 6 frames, 10 295 points, no depth files |

**Blockers / notes**

- gfx1010 is not supported by ROCm 7.2; `cuda` is a verbose error here. CPU at S4/252 is
  ~0.3–0.7 s per answered frame (D-012).
- vdo.ninja is WebRTC; use OBS Browser Source → RTMP/RTSP/HTTP, then give V3R that URL
  (D-013).
- Model weights live at `models/solv3r.safetensors` (SHA256
  `3cf9f215e6fee4639547cea482081be8103f7be5fdea57f58c795f552a624cb4`).

**Next**

- Demonstrate the Blender viewport update on the RX 5700 XT with a longer clip.
- If a CUDA/ROCm-supported GPU is available, measure `--device cuda` and record Gate A in
  `docs/DECISIONS.md`.
- Optional: per-point image colour is already wired; tune `point_radius` for the scene scale.

**Packaging**

- `V3R_addon.zip` — installable add-on folder (`V3R/`).
- `V3R_bundle.zip` — add-on + worker + CLI + docs + sample clips (no model, no venv).
- The model and worker venv are not in the archives; see `README.md` → Packaging.

---

## 2026-11-17 — T0 · Skeleton, contracts, debug IO

**Status:** done, with one blocked input (telemetry format — D-000).

**Files created**

- `core/settings.py` — 37-entry settings registry: `registry()`, `defaults()`, `describe()`,
  `validate()`; `SettingsError` on unknown keys, wrong types, or out-of-range values.
- `core/dataclasses.py` — frozen dataclasses `FlightSample`, `Frame`, `KeyframeResult`,
  `DepthResult`, `ScaleFit`, `Region`; six message types with `to_line` / `from_line` and the
  wire field `"class"` mapped to `class_name`.
- `core/geo.py` — `enu_from_llh`, `camera_rotation`, `build_rays`, `ray_ground_distance`
  (pure numpy; conventions in D-003).
- `core/debugio.py` — public surface: `RunFolder`, `create_run_folder`, `CsvAppender`,
  `TELEMETRY_COLUMNS`, `input_fingerprint`, `write_meta`, `png_gate`.
- `core/_run_folder.py` — private path helper split out to stay under the 200-line rule (D-009).
- `docs/DECISIONS.md` — D-000 (telemetry blocked) through D-009, plus Gate A/B/C definitions.
- `docs/TODO.md` — register seeded with TODO-101..118 and the standing debug rule.
- `pyproject.toml` — pytest (`[gpu]` marker, `pythonpath`), ruff, mypy config.
- `tests/test_contracts.py`, `tests/test_geo.py`, `tests/test_debugio.py`.
- `AGENTS.md` — progress-log rule.

**Verification (run from repo root, venv at `.venv/`)**

| Command | Result |
|---|---|
| `.venv/bin/ruff format core tests` | clean |
| `.venv/bin/ruff check core tests` | All checks passed |
| `.venv/bin/mypy core tests` | Success: no issues in 10 source files |
| `.venv/bin/pytest -q` | 40 passed in 0.10s |
| `grep -rn "import torch\|from torch" core tests` | clean |
| `wc -l core/*.py` | all files ≤ 200 lines |

**T0 checklist**

- [x] pytest green without GPU (ruff format + ruff check + mypy clean)
- [x] debugio creates the full §5.1 tree (test covers every directory)
- [x] ENU conversion round-trips a known coordinate (100 m north/east, 12.5 m up)
- [ ] telemetry format written in `docs/DECISIONS.md` **from the real sample** — BLOCKED
- [x] TODO register seeded with TODO-101..118
- [x] no torch import anywhere

**Blockers**

- No real telemetry CSV and no sample video exist in the repo or on the build machine. Per the
  T0 failure protocol and rule R8, no column names were invented. See `docs/DECISIONS.md`
  D-000 for the unblock action and the search evidence.

**Next**

- T1 (GPU proof) can start independently.
- T2 needs the drone team's real telemetry CSV plus the 10 s sample video; it records the
  format in D-000 before writing `ingest/telemetry.py`.