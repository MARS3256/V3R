# V3R Decisions

Append-only log of design decisions taken while building V3R. Each entry has an id, the
decision, the reason, and the evidence. A deviation from a task card threshold becomes an
entry here with its measured reason before work continues.

---

## D-000 Telemetry file format — BLOCKED, no sample supplied

**Status:** BLOCKED. This is the one T0 input that could not be produced.

The T0 card requires the exact telemetry format to be transcribed **from the real telemetry
CSV** and explicitly forbids inventing column names. No telemetry sample and no sample video
exist in the repository or on the build machine. Searched on 2026-11-17:

- repository tree (`/home/mars/Desktop/Study/cap/V3R`) — only source, plans, zips;
- the parent capstone tree `/home/mars/Desktop/Study/cap`;
- `/home/mars` up to depth 4 for `*.csv`, `*telemetry*`, `*flight*`, `*.mp4`, `*.srt`.

No file matched. Therefore this document deliberately records **no** column names, units,
angle conventions, altitude reference, or timestamp semantics. Inventing them would violate
the card's failure protocol and code rule R8 (no fake data).

**Unblock action:** the drone team supplies the real telemetry CSV together with the 10 s
sample video. The first task that reads it (T2, `ingest/telemetry.py`) must fill the
"Telemetry format (from the real sample)" section below **from the file**, then tick this
entry. Until then `FlightSample` in `core/dataclasses.py` is the only telemetry contract, and
it carries the field names fixed by the T0 card, not a file format.

### Telemetry format (from the real sample)

_Not yet recorded — blocked by D-000._

Required content when unblocked: column names in file order; units; angle convention
(degrees, 0–360 or ±180, yaw zero reference); altitude reference (ASL or relative); timestamp
semantics (video time vs UTC vs flight clock); expected frame rate.

---

## D-001 Settings registry is built fresh on every call

**Decision:** `core/settings.py` exposes `registry()` which constructs a new tuple of frozen
`Setting` entries on each call. `defaults()`, `describe()` and `validate()` all call it.
There is no module-level mutable registry.

**Reason:** code rule R6 forbids globals and singletons. A tuple built per call is immutable
and cheap (37 entries), so the rule is satisfied without a hidden global. The spec table
`_SPEC` is a module constant of plain tuples, not shared mutable state.

**Consequence:** callers must not cache `Setting` objects expecting identity; value equality
holds because `Setting` is a frozen dataclass.

---

## D-002 Worker-to-Blender wire format

**Decision:**

- Every message is one compact JSON object per line (`json.dumps(..., separators=(",", ":"))`).
- Each message carries a `type` tag (`status`, `origin`, `terrain`, `object_upsert`,
  `object_remove`, `error`). The plan's field tables do not list `type` as a field, but the
  Blender side cannot dispatch without it; it is part of the wire envelope, not the payload.
- `object_upsert` carries the wire field `"class"` (plan §4.9). Python reserves `class`, so
  the dataclass field is `class_name` and `wire_names = {"class_name": "class"}` maps it.
- Blender-to-worker traffic is only the bare line `stop` (not JSON), per plan §4.2.
- `from_line` rejects a missing or wrong `type`, unknown fields, and wrong field counts with
  `MessageError`; nothing is silently coerced.

**Reason:** the channel must be dispatchable, deterministic and round-trip-safe. The single
wire-name exception is documented here because it is visible to the Blender and worker code.

---

## D-003 Geometry conventions (shared contract for T4/T5)

**Decision:** `core/geo.py` fixes the following conventions. They may not change without a
new entry here.

- **World frame:** local East-North-Up, metres, X east, Y north, Z up, anchored at the first
  accepted telemetry fix (the launch point). Altitude is relative to that fix.
- **Geodetic conversion:** WGS84 `a = 6378137.0`, `e² = 6.69437999014e-3`, local
  equirectangular about the origin latitude. Accurate to millimetres over a few kilometres;
  this is a stated approximation, not a projection library.
- **Body frame:** x forward, y right, z down. At rest (level, yaw 0) body x is north, y east,
  z down.
- **Camera frame:** +Z optical axis, +X image right, +Y image down.
- **Mount tilt:** `mount_tilt_deg` is a forward tilt **from nadir** (0° = straight down).
  With a level body and 15° tilt the camera forward axis maps to `(0, sin 15°, -cos 15°)` in
  ENU (north and down). This matches the plan's lever-arm argument in §4.4.4.
- **Rotation composition:** `camera_rotation = B0 · Rx(roll) · Ry(pitch) · Rz(yaw) · Ry(tilt) · M0`,
  where `B0` maps body to ENU at rest and `M0` maps camera to body at rest. Positive yaw turns
  right/east, positive pitch is nose up, positive roll is right wing down.
- **Pixel centres:** pixel `(u, v)` samples at `(u + 0.5, v + 0.5)`; `fx = (W/2)/tan(fov_x/2)`,
  `cx = W/2`, `cy = H/2`, square pixels. The principal point is the image centre.
- **Ground intersection:** rays with `dz >= 0` (up or level) or a negative ray parameter
  (ground behind the camera) return `inf`, never a guessed distance.

**Reason:** T4 (back-projection) and T5 (scale fit) consume these four functions. Fixing the
conventions in one place makes their analytic unit tests meaningful and prevents an
unrecorded sign flip between modules.

---

## D-004 Run folder is owned solely by the debugio surface

**Decision:** `create_run_folder` is the only way to make a run folder, and `RunFolder` is the
only source of artifact paths. It refuses an existing run id with `RunFolderExistsError`.
The tree is created eagerly from `_TREE` so a run folder is complete before any stage runs.
Path construction physically lives in the private helper `core/_run_folder.py`; `core/debugio.py`
re-exports `RunFolder`, `create_run_folder`, `RunFolderExistsError` and `TELEMETRY_COLUMNS`
so every other module imports paths from `core.debugio` only (see D-009).

**Reason:** the plan (§4.2, §5.1) makes the worker the single writer of the run folder. One
path authority keeps the CLI and plugin artifact trees identical (plan §5.2).

---

## D-005 PNG write gate semantics

**Decision:** `png_gate(run, debug_mode, fingerprint)` returns `True` when:

- `debug_mode == "on"` — always; or
- `debug_mode == "always_first_run"` — when no **sibling** run folder has a `meta.json` with
  the same `input_fingerprint`.

`debug_mode == "off"` always returns `False`. Any other value raises `ValueError`. A sibling
`meta.json` that is missing or unreadable is skipped; that run simply does not count as a
previous run. CSVs and NPZ/PLY are always written regardless of this gate (plan §5.5).

**Reason:** implements the research-first rule without slowing demo runs. The gate reads
sibling `meta.json` files only; it never writes outside the current run folder.

---

## D-006 `meta.json` records a wall-clock timestamp

**Decision:** `meta.json` includes `created_at_utc`. This is metadata about the run, not
pipeline math.

**Reason:** the determinism rule (no wall-clock in pipeline math) is about reproducible
meshes. A provenance timestamp helps a human order runs. No geometry, depth, or mesh value
depends on it.

---

## D-007 Toolchain and Python version

**Decision:** `requires-python = ">=3.11"`; dev tooling is `pytest`, `ruff`, `mypy`, and the
runtime dependency is `numpy`. Pytest uses `pythonpath = ["."]` so tests import `core` without
installation. The `[gpu]` marker is registered and skipped when no GPU is present.

**Reason:** Blender 5.2 ships a Python new enough for `typing.Self` and modern dataclasses.
Keeping the core dependency-free (numpy only) is what makes the module importable in both the
worker venv and the test environment.

---

## D-008 No telemetry parser in T0

**Decision:** `ingest/telemetry.py` is not created in T0. `FlightSample` is defined, but no
code reads a telemetry file.

**Reason:** the card assigns the parser to T2 and forbids inventing a file format without the
real sample. Writing a parser now would be exactly the fake-data failure D-000 describes.

---

## D-009 debugio path code split to satisfy the 200-line rule

**Decision:** path construction lives in `core/_run_folder.py` (private helper).
`core/debugio.py` imports and re-exports `RunFolder`, `create_run_folder`,
`RunFolderExistsError` and `TELEMETRY_COLUMNS`, and owns the CSV appender, `meta.json`,
`input_fingerprint` and the PNG gate. No other module imports `core._run_folder`.

**Reason:** the T0 file list gave one file (`core/debugio.py`) for the whole §5 machinery.
Written whole it was 284 lines, over the 200-line limit in code rule R3. Splitting the pure
path helper out keeps both files under the limit while preserving the single public import
surface: other code still does `from core.debugio import ...`. The plan's intent (one path
authority, no stage building its own paths) is unchanged.

**Consequence:** a reviewer looking for path helpers should look in `core/debugio.py` first;
the implementation is one file down in `core/_run_folder.py`.

---

## Gate definitions (plan §9; copied here on Day 1)

- **Gate A (Day 1):** depth ONNX runs on the RX 5700 XT at ≤ 200 ms/frame, ≤ 2 GB VRAM, with
  visually sane output. Fail → same-day pivot (ncnn depth path, or `cpu_reference` with a
  smaller image), written here before other tasks continue.
- **Gate B (Day 3):** the CLI runs end-to-end on a real ≥ 3-min clip with telemetry, the
  §5.1 tree is fully populated, `report.md` has the required numbers, and the synthetic
  two-box test passes. Fail → Day 4 becomes a fix day; T11 drops to Day 6 and T13 is cut.
- **Gate C (Day 5):** §1.1 items 1–4 and 7 demonstrated live, and item 9 verified once on a
  real plugin run (artifact tree populated, `report.md` finalized on Stop). Fail → Days 6–7
  are fix-only; the demo shows CLI artifacts plus a partial plugin, honestly labelled.

A gate is never marked passed by an agent; an agent only produces the evidence.