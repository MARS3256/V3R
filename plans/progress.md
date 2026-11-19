# V3R Progress

Newest entry first. One entry per task card or meaningful work chunk.

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