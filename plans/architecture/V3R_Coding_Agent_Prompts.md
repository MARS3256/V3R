# V3R — Ready-to-Paste Coding-Agent Prompts (T0–T13)

*Companion to `V3R_System_Design_and_1_Week_Plan_v2.md` (Rev 2.1) · 14 self-contained prompts, one per task card · Nov 2026*

Each card below is a **complete prompt**: project facts, the full §8 code rules (verbatim), exact interfaces and data contracts, debug-artifact duties, tests, and a done-when checklist. Copy one entire fenced block into the coding agent — nothing else is needed.

> **Rev 3.0 (2026-11-21) — Solv3R pivot.** The inference engine is now [Solv3R](https://github.com/ubc-vision/solv3r) (PyTorch). The following are **removed** everywhere in this file and the design doc: the ban on `import torch` / `from torch` (R12), the AMD-only / DirectML / NCNN / no-CUDA lock (C1/C2, A1–A5), and the GPS/telemetry metric-scale requirement (including task T5 and decision D-000). PyTorch is allowed **inside the worker process and its own virtual environment**; Blender-side add-on code still must not import torch, because Blender's Python is not the worker environment. The backend (`cuda` / `rocm` / `cpu`) is a startup setting; an unavailable backend is a verbose error, never a silent fallback. All other §8 rules, non-blocking Blender, debug artifacts and honest-error rules remain binding.

## How to use

1. Work strictly in card order per day (plan §9): Day 1 = T0, then T1 ∥ T2 · Day 2 = T3 ∥ T4 ∥ T5 · Day 3 = T6 ∥ T7, then T8 · Day 4 = T9 ∥ T10 · Day 5 = T11, T12 starts · Day 6 = T12 finish, T13 stretch.
2. **Parallel-safety is defined in Appendix B.** Cards that share `core/debugio.py` extend it *append-only* (new functions at the end, no edits to existing lines) — that rule makes Day-2 parallelism merge-safe.
3. When an agent finishes, accept the output only if: the DONE WHEN checklist is ticked, test output is pasted and green, the artifacts really exist on disk (spot-check the run folder), and the §8 review checklist passes. No exceptions — this is what keeps a week of agent work mergeable.
4. If an agent stops with "a code rule blocks me", that is the protocol working (§8). Resolve the conflict in `docs/DECISIONS.md`, then re-paste the card with one added line stating the decision.
5. Gates A/B/C are **human decisions on the real machine** (Appendix A). An agent must never mark a gate as passed — it only produces the evidence.
6. Cards reference plan sections (§…) of the design doc. Keep that file in the repo under `docs/` so agents can consult it when a card tells them to.

Optional size cut: if your agent supports a repo rules file (`AGENTS.md` / `.cursorrules`), paste the shared block of any card there once and delete it from subsequent cards — every card still works standalone either way.

---

## T0 — Skeleton, contracts, debug IO

```text
==================================================================
V3R CODING-AGENT PROMPT — T0 · Skeleton, contracts, debug IO
Day 1, ~3 h. Blocks everything: every other card consumes your files.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement one task card of V3R, a Blender 5.2 add-on (repo
github.com/MARS3256/V3R, GPL-3.0-or-later) that turns drone video plus
per-frame flight telemetry (GPS lat/lon/alt + roll/pitch/yaw) into
metre-scaled 3D objects appearing live in the Blender viewport. You own
ONLY the files listed below. If you need a change outside them, STOP
and report — do not edit others' files.

PROJECT FACTS (fixed; do not violate, do not re-decide)
- Two processes: the Blender add-on (UI, bpy) writes a config file and
  spawns a worker process with its own Python env; the worker streams
  one JSON status object per line on stdout; Blender only polls,
  applies, returns — it never waits.
- Hardware: AMD RX 5700 XT 8 GB, Windows. Inference = ONNX Runtime +
  DirectML. "import torch" and CUDA calls are BANNED repo-wide.
- Depth model: Depth Anything V2 Small ONNX (Apache-2.0). It outputs
  RELATIVE depth; metric scale comes from a ground-plane fit against
  GPS height — never from the model.
- Debug output is a deliverable: every stage emits a machine-readable
  row (telemetry.csv / stage CSVs) and a human-viewable artifact
  (PNG/PLY). All run-folder paths are created only through
  core/debugio.py.
- Determinism: no RNG, no wall-clock in pipeline math; two runs on the
  same input produce the same meshes. Every rejection is counted with
  a reason — nothing faked, nothing silently dropped.

CODE RULES (obey all; if one blocks you, STOP and say which one)
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
core/settings.py, core/dataclasses.py, core/geo.py, core/debugio.py,
docs/DECISIONS.md, docs/TODO.md, pyproject.toml (ruff + mypy + pytest
config, [gpu] marker registered), tests/test_contracts.py,
tests/test_geo.py, tests/test_debugio.py.
Also create: core/__init__.py, tests/__init__.py (empty).

INPUT YOU NEED FIRST
The real 10-second sample video + the real telemetry CSV from the
drone team. Ask for their paths before writing any parser code.

YOUR TASK (do exactly this, in order)
1. pyproject.toml: pytest (markers: [gpu] skippable when no GPU),
   ruff (format + check), mypy basic. Keep it minimal.
2. Get the real sample. From the ACTUAL telemetry file write into
   docs/DECISIONS.md the exact format: column names in file order,
   units, angle convention (degrees? 0–360 or ±180? yaw zero
   reference), altitude reference (ASL or relative), timestamp
   semantics (video time vs UTC vs flight clock), expected frame
   rate. From the file, not from memory of what anyone said. If the
   sample is unavailable, STOP — never invent column names.
3. core/settings.py — one registry; every entry = key, label, type,
   default, unit, allowed range, provisional flag, one-line reason.
   Ship exactly these defaults:
   Camera: fov_x_deg 90 (spec assumption, Task 12 calibrates);
     mount_tilt_deg 15 / roll_offset_deg 0 / yaw_offset_deg 0 (team
     statement, flight check validates); image_long_side 1280 (VRAM
     headroom + speed; Gate A may raise).
   Keyframes: overlap_ratio 0.65 (standard mapping practice);
     max_time_gap_s 6.0 (safety net if GPS speed data missing);
     blur_margin_ratio 0.5 / max_tilt_deg 45 (conservative starts).
   Scale fit: min_ground_fraction 0.20 (below this the fit is
     untrustworthy); max_fit_mad_m 3.0 (generous vs GPS vertical
     noise); smoothness_window 9 / smoothness_max_var 0.01
     (provisional — measured in Task 12); outlier_mad_multiplier 3.0
     (provisional); ground_region_row_fraction 0.5 (provisional);
     ground_ref_settle_keyframes 10 (before the feedback loop
     engages); ground_filter_kernel 5 (provisional).
   Height map: cell_factor 4 / min_hits 3 / area_limit_km2 4 (plan
     §4.4.6; bounds memory).
   Objects: min_object_height_m 2.0 / min_cells 25 / keep_overlap 0.3
     / remove_after_updates 5 / min_region_confidence 0.5
     (provisional) / approx_poly_epsilon 1.5 (provisional).
   Perf: max_updates_per_tick 6 / tick_budget_ms 30 / stop_timeout_s
     10 (NFR-1).
   Stream: probe_timeout_s 5 / lost_timeout_s 10 / queue_len 8.
   Detect: diff_pixel_threshold 8 (provisional, 0–255 scale).
   Consistency: gps_consistency_tolerance_m 5.0 (provisional,
     consumer-GPS vertical noise bound; tuned in Task 12).
   Compute: compute_backend directml (Gate A decides; ncnn_vulkan
     optional); worker_python_path "python" (override to the
     ground-station venv interpreter).
   General: debug_artifacts always_first_run (on | off |
     always_first_run).
   API: defaults() -> dict; validate(overrides) -> dict (unknown key
     or out-of-range value raises SettingsError); describe() -> list
     of frozen Setting entries.
4. core/dataclasses.py — frozen dataclasses, full type hints:
   FlightSample(t_s, lat_deg, lon_deg, alt_m, roll_deg, pitch_deg,
     yaw_deg); Frame(index, t_s, image_bgr, flight);
   KeyframeResult(frame, is_keyframe, reason, d_since_last_m,
     footprint_w_m); DepthResult(relative_depth, depth_ms);
   ScaleFit(scale, mad_m, ground_frac, ok, reason);
   Region(object_id, cells, height_m, centroid_e, centroid_n,
     footprint_area_m2, first_t, last_t, updates).
   Messages (exact field contract, plan §4.9):
   status: state, frames_read, keyframes_used, keyframes_rejected,
     objects, coverage_pct, text
   origin: lat, lon, alt_m
   terrain: version, file
   object_upsert: object_id, version, file, height_m,
     footprint_area_m2, class   [class = "unknown_structure"]
   object_remove: object_id
   error: code, message, fatal
   Each message dataclass has to_line() -> str (compact JSON, one
   line) and from_line(text) -> Self. Round-trip must preserve all
   fields. Blender -> worker traffic is only the line "stop".
5. core/geo.py — pure functions (numpy in/out, R14). These four
   signatures are the shared geometry contract for T4/T5 — they may
   not change without a DECISIONS.md entry:
   enu_from_llh(origin, lat, lon, alt) -> (e, n, u) metres
     [WGS84; local equirectangular approximation is acceptable —
     state it in the docstring; origin = first accepted fix]
   camera_rotation(sample, mount_tilt_deg, roll_offset_deg,
     yaw_offset_deg) -> (3,3) mapping camera-frame directions to ENU
     [document the convention: R maps cam->ENU, axes X east Y north
     Z up; camera rotation = body roll/pitch/yaw composed with the
     fixed mount rotation, default 15 deg forward tilt]
   build_rays(width, height, fov_x_deg) -> (3,H,W) normalized
     camera-ray directions [fx = (W/2)/tan(fov_x/2), square pixels,
     principal point at image centre]
   ray_ground_distance(dir_cam, rotation, cam_pos_enu, ground_z) ->
     per-pixel slant distance where the ray hits the plane z =
     ground_z [inf where it never hits]
6. core/debugio.py — the ONLY module that touches run-folder paths.
   create_run_folder(runs_root, run_id) -> RunFolder (refuses an
   existing dir). RunFolder exposes path helpers per artifact family:
   telemetry_csv, skipped_csv, depth_stats_csv, diff_metrics_csv,
   keyframe_jpg(i), depth_raw_npz(i), depth_png(i), diff_png(i),
   ground_png(i), cloud_ply(i), cloud_text(i), heightmap_npz,
   heightmap_png(step), objects_dir(step), detections_csv, meta_json,
   report_md, graphs_dir. It creates this tree exactly:
   ingest/keyframes/  depth/raw/  depth/png/  diff/  ground/
   cloud/text/  heightmap/png/  objects/  detect/  graphs/
   CsvAppender(path, columns): header once, append rows, flush per
   row. TELEMETRY_COLUMNS, exact order (plan §5.3):
   frame_index, t_video_s, lat, lon, alt, roll, pitch, yaw, h_agl_m,
   footprint_w_m, d_since_last_m, blur_score, is_keyframe,
   skip_reason, depth_ms, fit_scale, fit_mad_m, ground_frac,
   cloud_pts, fuse_ms, regions, mesh_ms, total_ms, gpu_mem_mb
   write_meta(run, settings_snapshot, model_hashes,
   input_fingerprint, versions) -> meta.json (settings snapshot,
   versions, model SHA256, machine info). input_fingerprint(video,
   telemetry) = sha256 of first+last 1 MB + file size (identity
   fingerprint for the debug gate — document that it is not
   security). png_gate per §5.5: PNG-family artifacts are written
   when debug_artifacts == "on", or when this is the first run on a
   new input fingerprint (checked against sibling runs' meta.json;
   an unreadable sibling meta.json is skipped — that run just does
   not count). CSVs and npz/PLY are ALWAYS written.
7. docs/TODO.md — seed the register, one line each, status
   "not started": TODO-101 Texture projection · 102 Object
   classification · 103 Proxy models · 104 Water bodies · 105 Rubble
   · 106 Damage comparison · 107 ODM background refinement · 108
   Absolute georeferencing export · 109 Human markers + dwell
   tracking · 110 Multi-strip flights · 111 Height-map compute-shader
   fusion · 112 Resume interrupted runs · 113 Linux ground station ·
   114 Automatic camera calibration · 115 Offline batch quality pass
   · 116 Analog FPV intake (research-only) · 117 Visual pose
   refinement of telemetry · 118 Auto telemetry offset correction.
   Add the standing rule: every TODO ships its own debug artifacts +
   telemetry columns in the same commit as the feature.

TESTS (pytest, green without GPU)
- All six message types round-trip (to_line -> from_line -> equal).
- TELEMETRY_COLUMNS equals the §5.3 list above, exact order.
- settings: defaults() returns the table; validate raises
  SettingsError on unknown key and on out-of-range value.
- geo (analytic): ENU round-trip — a point 100 m north of origin maps
  to (0, 100, 0) ± 1e-6; camera_rotation for a level camera with 15
  deg mount tilt maps the camera forward axis to the expected ENU
  direction; build_rays centre ray matches the documented convention;
  ray_ground_distance for a level camera at 100 m altitude hits the
  ground at the analytically known slant range; rays that never hit
  (pointing up) return inf.
- debugio: the tree contains every §5.1 directory; CsvAppender writes
  the header exactly once across close/reopen; png_gate behaviour for
  first run / second run same fingerprint / mode on / mode off;
  meta.json round-trips the settings snapshot.

DEBUG ARTIFACT DUTIES
This card BUILDS the §5 machinery itself — no stage artifacts yet.
The tree, the CSV machinery, the PNG gate and meta.json must exist
and be tested, because every later card writes through them.

DONE WHEN (tick every box; paste the filled checklist at the end)
[ ] pytest green without GPU (ruff format + ruff check + mypy clean)
[ ] debugio creates the full §5.1 tree
[ ] ENU conversion round-trips a known coordinate
[ ] telemetry format written in docs/DECISIONS.md FROM the real sample
[ ] TODO register seeded with TODO-101..118
[ ] No torch import anywhere (grep clean)

FAILURE PROTOCOL
If the telemetry sample is missing, STOP and say so — do not invent
column names. If a code rule blocks you, name the rule. Any threshold
you must deviate from becomes a DECISIONS.md entry with the measured
reason, then proceed. Finish with the ticked checklist, the test
output, and a <= 5-line summary.
```

---

## T1 — GPU proof spike (Gate A owner)

```text
==================================================================
V3R CODING-AGENT PROMPT — T1 · GPU proof spike
Day 1, ~4 h incl. measuring. Gate A owner.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T1 of V3R: prove on the real machine that the
depth model runs on the AMD RX 5700 XT 8 GB via ONNX Runtime
DirectML, with numbers. You own ONLY the files listed below.

PROJECT FACTS (fixed; do not violate, do not re-decide)
- Target: RX 5700 XT 8 GB, Windows. Inference stack = ONNX Runtime +
  DirectML (AMD's documented recommendation for Radeon inference).
  There is NO ONNX Runtime Vulkan provider; NCNN-Vulkan is a measured
  option B, not the default. "import torch" is BANNED repo-wide.
- Depth model: Depth Anything V2 Small ONNX — 25 M params, 99 MB,
  Apache-2.0, from Hugging Face onnx-community/depth-anything-v2-small
  (alternative to measure: onnx-community/depth-anything-v3-small,
  also Apache-2.0). The model outputs RELATIVE depth; metric scale is
  fitted later against GPS height — never trusted from the model.
- Metric-depth ONNX checkpoints are EXCLUDED (documented broken
  conversions, wrong training domain) — do not try them.
- Budget context (plan §3.5): depth inference should land 30–100 ms;
  Gate A passes at <= 200 ms/frame and <= 2 GB VRAM.
- Every failure is measured and reported, never hidden.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
models/depth.py, models/weights/ (downloaded ONNX files,
git-ignored), scripts/gate_a_bench.py, docs/DECISIONS.md (append one
entry).

YOUR TASK (do exactly this, in order)
1. Create a clean venv — NOT Blender's Python. Install:
   onnxruntime-directml, opencv-python, numpy, scipy, pytest.
   Record exact package versions in the bench output (they go into
   every run's meta.json later).
2. Download the ONNX models (v2-small primary, v3-small alternative).
   Record the SHA256 of each file — later tasks write these hashes
   into meta.json. Do not commit the weight files; git-ignore them.
3. models/depth.py — thin ONNX wrapper, numpy in / numpy out only
   (zone R1: models/ uses only onnxruntime + numpy + cv2):
   DepthModel(onnx_path, compute_backend) creates an InferenceSession
   with the provider per backend (directml -> DmlExecutionProvider;
   cpu_reference -> CPUExecutionProvider).
   preprocess(image_bgr, long_side): resize long side, letterbox to
   the model's input size (read it from the model card, e.g.
   518x518), normalize per the model card (ImageNet mean/std).
   Document the exact preprocessing in the docstring AGAINST the
   model card, including the output convention (DA-V2 emits relative
   INVERSE depth — larger = closer). The pipeline needs relative
   DEPTH (larger = farther), so run() converts: relative_depth =
   1 / max(output, eps) with eps a named constant, and DOCUMENTS the
   conversion. run(frame) -> DepthResult(relative_depth float32 at
   frame resolution, depth_ms measured around session.run only).
   close() releases the session. No path logic, no artifact writing
   here — core/debugio.py owns that.
4. scripts/gate_a_bench.py — on 5 real frames from the sample: 3
   warmup + 20 timed runs per model (v2-small AND v3-small); print
   mean/p95 ms/frame per model; capture peak GPU memory (read Task
   Manager during the run and type the number into the printed table;
   also print onnxruntime arena info if exposed). Write depth PNGs
   (magma colormap, min-max normalized per frame) for the 5 frames
   and LOOK at them: sanity = far/sky light, ground dark (or the
   model card's convention), building edges crisp, no banding
   stripes. If T13 will run, bench YOLOv8n ONNX at 640 px the same
   way (the export happens ONCE offline with the ultralytics CLI;
   record the exact export command; no torch ends up imported at
   runtime).
5. Append one entry to docs/DECISIONS.md: "backend = directml,
   depth = DA-V2-small, ms/frame = X (p95 Y), peak VRAM = Z GB,
   evidence = bench table + PNGs", and the v3-small verdict
   (adopt / reject, with the measured reason).

GATE A (the team decides; you produce the evidence — never mark the
gate passed yourself)
PASS = depth ONNX on the RX 5700 XT at <= 200 ms/frame, <= 2 GB VRAM,
visually sane output on real frames.
FAIL pivot (same day, team decision): ncnn port path for depth, OR
cpu_reference with reduced image size. If measurements fail the gate,
STOP and report the numbers — do not silently switch backends or
lower thresholds.

TESTS
- tests/test_depth_shapes.py (no GPU): preprocess on a synthetic
  array → expected shape/normalization; postprocess/decode identity;
  eps conversion keeps finiteness (no zero division).
- One [gpu]-marked smoke test: real ONNX on one frame, asserts finite
  output with min < median < max. Skips when no GPU.
DONE WHEN
[ ] bench table with real numbers (both models) printed + saved
[ ] depth PNGs written and eyeballed; verdict noted
[ ] DECISIONS.md entry appended with model SHA256s + versions
[ ] models/depth.py grep-clean for torch; ruff/mypy/pytest green
[ ] <= 5-line summary of what changed
FAILURE PROTOCOL
If the GPU or DirectML is unavailable where you run, report the exact
error and stop — do not benchmark CPU and present it as GPU. Any
deviation becomes a DECISIONS.md entry first.
```

---

## T2 — Ingest + telemetry + keyframe

```text
==================================================================
V3R CODING-AGENT PROMPT — T2 · Ingest + telemetry + keyframe
Day 1, ~5 h. After T0 (consumes its contracts).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T2 of V3R: read video + telemetry, decide
keyframes, and record EVERY decision. You own ONLY the files listed
below.

PROJECT FACTS
- V3R turns drone video + per-frame telemetry (GPS lat/lon/alt +
  roll/pitch/yaw) into metre-scaled 3D objects; inference later runs
  via ONNX Runtime DirectML on an AMD RX 5700 XT (torch banned).
- Keyframe rule (plan §4.4.3): footprint width F = 2·h_agl·tan(fov_x/2)
  with h_agl = alt − ground_ref_altitude; take a keyframe when the
  plane has flown d_step = (1 − overlap_ratio)·F since the last one
  (defaults: overlap 0.65 → d_step = 0.35·F; at 100 m and 90 deg FOV:
  F = 200 m, d_step = 70 m ≈ every 5.6 s at 45 km/h). Plus rejects:
  blur (Laplacian variance below an adaptive threshold) and tilt
  beyond max_tilt_deg. EVERY decision becomes a row — rejects carry a
  reason. Nothing is silently dropped.
- Determinism: no RNG, no wall-clock in the decision math.
- Debug output is a deliverable: telemetry.csv row per frame decided,
  skipped.csv row per reject, keyframe JPGs for kept frames.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
ingest/source.py, ingest/telemetry.py, ingest/keyframe.py,
tests/test_ingest.py, tests/data/ (hand-checked expectations).

INPUTS ALREADY BUILT (trust these interfaces, do not change them)
- core/dataclasses: Frame(index, t_s, image_bgr, flight),
  FlightSample(t_s, lat_deg, lon_deg, alt_m, roll_deg, pitch_deg,
  yaw_deg), KeyframeResult(frame, is_keyframe, reason,
  d_since_last_m, footprint_w_m).
- core/settings: fov_x_deg, mount_tilt_deg, roll_offset_deg,
  yaw_offset_deg, image_long_side, overlap_ratio, max_time_gap_s,
  blur_margin_ratio, max_tilt_deg, ground_ref_settle_keyframes.
- core/debugio: create_run_folder, CsvAppender, TELEMETRY_COLUMNS,
  png_gate, keyframe_jpg(i), skipped_csv, telemetry_csv.
- The telemetry format is LOCKED in docs/DECISIONS.md by T0, from the
  real sample file. Parse exactly that; if the file does not match
  the documented format, raise TelemetryError naming the column.

YOUR TASK (do exactly this, in order)
1. ingest/source.py — FileSource(path): decode as fast as asked, not
   paced (cv2.VideoCapture; frame_at(index) -> Frame with
   t_s = index / fps; SourceError past EOF; properties frame_count,
   fps). StreamSource(url): probe(timeout_s) -> ProbeResult(ok,
   width, height, fps, reason) — bounded open + first-frame read
   within probe_timeout_s (5 s); frames() generator keeping only the
   NEWEST frame (drop-oldest policy; the dropped count is exposed —
   it lands in status later); StreamLostError if no frame arrives
   for lost_timeout_s (10 s). The stream code exists NOW; card T11
   only wires the UI icon and the probe command.
2. ingest/telemetry.py — parse the sample CSV exactly as locked in
   DECISIONS.md -> list[FlightSample]; at(t_s) -> FlightSample by
   linear interpolation; ANGLE WRAP: interpolate across 360 deg
   correctly (359 -> 1 goes +2, never −358); times outside the sample
   span raise TelemetryError (never extrapolate); missing columns or
   unparseable rows raise TelemetryError naming the column/row.
3. ingest/keyframe.py — the decision as a PURE function (R14):
   decide(candidate: Frame, state: KeyframeState, settings) ->
   KeyframeResult, where KeyframeState holds the last kept frame's
   position, time, and the running blur-score list. Rule exactly per
   the facts above; blur threshold = blur_margin_ratio × running
   median of accepted keyframes' blur scores, floored at an absolute
   minimum (new setting blur_floor, default 30.0, provisional,
   reason "Laplacian variance floor; measured in Task 12"). Tilt
   test on pitch/roll magnitude vs max_tilt_deg. reason strings:
   "" (kept), "blur", "tilt". A separate thin driver class feeds
   frames + telemetry and writes rows via debugio (kept frames also
   save keyframe_jpg when the PNG gate says so).
4. Every frame decided gets a telemetry.csv row (is_keyframe,
   skip_reason, d_since_last_m, footprint_w_m, blur_score, plus the
   telemetry fields); every reject ALSO gets a skipped.csv row
   (frame_index, timestamp, reason).
5. tests/data/: hand-check 5 frames from the real sample — for each,
   write the expected decision (keep/skip + reason) into a small
   committed file, produced by a human looking at the frames.

TESTS (green without GPU)
- Angle-wrap interpolation (359 -> 1 -> 3 stays monotone).
- d_step math on a synthetic straight flight (100 m AGL, 90 deg FOV
  → d_step 70 m; keyframes land at the analytically expected frames).
- Blur rejection counted with reason "blur"; tilt rejection with
  "tilt"; both rows present in telemetry.csv AND skipped.csv.
- Every reject has exactly one reason row (no silent drops).
- TelemetryError on missing column and on out-of-range time.
- 3-minute driver over the real sample: yield table printed; the 5
  hand-checked frames match expectations.
DONE WHEN
[ ] 3 min of the sample drive through ingest; decisions recorded
[ ] keyframe decisions match the hand-check on 5 frames
[ ] pytest + ruff + mypy green; torch grep clean
[ ] <= 5-line summary of what changed
FAILURE PROTOCOL
If the sample or its documented format is missing, STOP. If a rule
blocks you, name it. Do not pace the decoder "to be nice" — as fast
as asked is the contract.
```

---

## T3 — Depth stage + debug artifacts

```text
==================================================================
V3R CODING-AGENT PROMPT — T3 · Depth stage + debug artifacts
Day 2, ~4 h. After T1 + T2. Parallel-safe with T4/T5 IF you obey the
append-only rule on core/debugio.py.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T3 of V3R: the depth stage and its §5 debug
artifacts (raw npz, magma PNG, stats row, frame diff). You own ONLY
the files listed below.

PROJECT FACTS
- Depth = Depth Anything V2 Small ONNX via ONNX Runtime DirectML on
  the RX 5700 XT (torch banned). It outputs RELATIVE depth; the
  metric scale arrives later from the ground-plane fit (T5) — your
  stage only guarantees a consistent relative-depth field + artifacts.
- §5 artifact rule: every stage emits a machine-readable row AND a
  human-viewable artifact. Depth: stats.csv row always; raw npz
  always; PNG only when the PNG gate says so. The PNG gate exists so
  debug output never becomes the bottleneck.
- Determinism: two runs on the same input → same artifacts.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
models/depth.py (extend: postprocess + eps conversion — T1 built the
wrapper), ingest/diff.py (new), core/debugio.py (EXTEND APPEND-ONLY:
save_depth, save_diff, a small magma-colorize helper — do not modify
any existing function), tests/test_depth_artifacts.py, tests/test_diff.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T1: DepthModel(onnx_path, compute_backend) with preprocess documented
  against the model card and run(frame) -> DepthResult(relative_depth,
  depth_ms) where relative_depth is larger = farther.
- T2: FileSource/telemetry/keyframe driver producing kept keyframes.
- T0: debugio (RunFolder paths, CsvAppender, TELEMETRY_COLUMNS,
  png_gate), settings registry, DepthResult dataclass.

YOUR TASK (do exactly this, in order)
1. Verify/complete the model-card-documented postprocess in
   models/depth.py: letterbox undo (back to frame resolution),
   inverse-depth → depth conversion relative_depth = 1/max(out, eps),
   eps a named constant. Confirm the preprocessing docstring matches
   the HF model card (input size, normalization) — if T1 left a gap,
   fix it here and note it.
2. debugio.save_depth (append-only): writes depth/raw/frame_XXXXXX.npz
   (float32 relative depth, ALWAYS); depth/stats.csv row via
   CsvAppender (columns: frame, min, max, median, p90, fit_scale,
   fit_mad) — the row is written by the pipeline AFTER the scale fit
   in the same keyframe cycle so fit_scale/fit_mad are filled (T5
   supplies ScaleFit; until T5 is merged the pipeline passes
   fit_scale="", fit_mad=""); depth/png/frame_XXXXXX.png ONLY when
   png_wanted (magma colormap via cv2.applyColorMap on a min-max
   normalized uint8 conversion).
3. ingest/diff.py — pure function (R14): frame_diff(current_bgr,
   previous_bgr) -> (mean_diff, changed_pixel_pct) on grayscale |Δ|;
   changed = |Δ| > diff_pixel_threshold (setting, default 8,
   provisional, reason "below this is JPEG noise; measured in Task
   12"). debugio.save_diff (append-only): diff/metrics.csv row
   (frame, mean_diff, changed_pixel_pct) ALWAYS + diff png (magma on
   |Δ| normalized) when png_wanted.
4. Wire a pipeline fragment: kept keyframe → depth → diff →
   save_depth/save_diff + telemetry row fields depth_ms (real
   measured ms — never a constant). The fragment runs inside the CLI
   loop later (T8); here it is exercised by tests + a driver script
   over the real sample.
DONE WHEN
[ ] depth PNGs for 20 keyframes of the real sample look structurally
    correct (eyeball; note "20/20 sane" or the failures in DECISIONS.md)
[ ] stats.csv populated with real numbers for those keyframes
[ ] depth_ms column contains real measurements (no constants)
[ ] diff metrics plausible: low between consecutive flight frames,
    high across hard cuts
[ ] pytest + ruff + mypy green; torch grep clean; <= 5-line summary
TESTS (green without GPU — inject a mock depth provider; the mock
lives only in tests/, which is the one allowed place)
- preprocess shape + normalization on a synthetic array; decode
  identity; eps conversion finiteness.
- save_depth: npz loads back allclose; PNG gate honored (off → no
  png, npz still present); stats row has all 7 columns.
- save_diff: half-white/half-black synthetic pair → analytic mean and
  changed percentage.
- append-only check: existing debugio functions unchanged (compare
  signatures against the T0 list in the test).
- One [gpu]-marked smoke test: real ONNX on one frame; asserts finite
  stats (min < median < max).
FAILURE PROTOCOL
If depth PNGs look wrong (banding, inverted brightness vs the model
card's convention, edge soup), STOP and report with the PNGs — do not
"fix" it by scaling or histogram tricks. A model-output convention
discovery is a DECISIONS.md entry.
```

---

## T4 — Back-projection + PLY writers

```text
==================================================================
V3R CODING-AGENT PROMPT — T4 · Back-projection + PLY writers
Day 2, ~5 h. After T0 (uses core/geo). Parallel-safe with T3/T5.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T4 of V3R: turn a metric depth map into an
ENU point cloud, and write PLY + text-sample artifacts. You own ONLY
the files listed below.

PROJECT FACTS
- Geometry (plan §4.4.5): X_world = C + D(p) · R · d_cam(p), where C =
  camera ENU position, R = orientation matrix (cam->ENU), d_cam =
  normalized camera-ray direction, D(p) = scale · relative_depth(p).
- The cloud lives for one keyframe: down-accumulated into the height
  map by T6, written to PLY for debugging, then freed. Memory budget:
  transient cloud 2 M pts × 24 B ≈ 50 MB — full-frame back-projection
  at 1280x720 is fine (≈ 0.9 M pts).
- All geometry is PURE (R14): numpy in, numpy out, no files, no GPU,
  no clock — writers are thin glue outside the pure functions.
- Determinism: row-major pixel order, no RNG.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
geometry/backproject.py, io_formats/ply_writer.py, core/debugio.py
(EXTEND APPEND-ONLY: save_cloud), tests/test_backproject.py,
tests/test_ply_writer.py.

INPUTS ALREADY BUILT (trust these, do not change)
- core/geo: build_rays(width, height, fov_x_deg) -> (3,H,W);
  camera_rotation(sample, mount_tilt_deg, roll_offset_deg,
  yaw_offset_deg) -> (3,3) cam->ENU; enu_from_llh(...); 
  ray_ground_distance(...).
- core/dataclasses: DepthResult(relative_depth, depth_ms),
  ScaleFit(scale, mad_m, ground_frac, ok, reason), Frame.
- core/debugio: RunFolder.cloud_ply(i), RunFolder.cloud_text(i),
  png_gate.

YOUR TASK (do exactly this, in order)
1. geometry/backproject.py — pure functions:
   backproject(relative_depth, scale, rotation, cam_pos_enu, rays,
   colors_bgr) -> (xyz (N,3) float32, rgb (N,3) uint8): metric depth
   D = scale · r per pixel; X = C + D · R · d_cam; skip pixels with
   non-finite depth (return the skipped count too); colors converted
   BGR→RGB; deterministic row-major order.
   If T5 needs ray-plane distances, they come from core/geo — never
   duplicate the math here.
2. io_formats/ply_writer.py — binary_little_endian PLY: documented
   header (vertex count, properties x/y/z float32, red/green/blue
   uchar), write(path, xyz, rgb) plus read(path) -> (xyz, rgb) for
   tests. Keep it <= 100 lines.
3. debugio.save_cloud (append-only): PLY ALWAYS written (§5.2: PLY is
   the "always" column; openable in MeshLab/CloudCompare); text
   sample cloud/text/frame_XXXXXX.txt (first 500 points, one "x y z
   r g b" per line) only when png_wanted (the debug-only set).
DONE WHEN (the analytic acceptance test)
[ ] Synthetic scene: camera at a known ENU pose over a plane with one
    10 m box → construct synthetic relative depth as r = D_true / s
    with a known s (no model involved), back-project → the cloud
    reproduces the ground plane at z ≈ 0 ± 0.5 m and the box top at
    z ≈ 10 ± 0.5 m, with the box pixel count consistent.
[ ] Non-finite depth pixels are excluded AND counted.
[ ] PLY round-trip: write → read → allclose, colors equal.
[ ] Text sample is at most 500 lines.
[ ] pytest + ruff + mypy green; <= 5-line summary of what changed.
TESTS
- Ray construction: centre ray direction matches the documented cam
  convention; corner rays symmetric.
- Rotation composition: level camera + 15 deg mount tilt → known
  forward vector in ENU.
- Backproject: the analytic box test above (this is R14's "known
  analytic answer").
- Excluded-pixel counting; PLY round-trip; text-sample line cap.
FAILURE PROTOCOL
If the box test fails, debug the math against the analytic answer —
do not loosen the tolerance beyond ±0.5 m without a DECISIONS.md
entry. Never fudge the synthetic depth to make the test pass.
```

---

## T5 — Scale fit (the heart)

```text
==================================================================
V3R CODING-AGENT PROMPT — T5 · Scale fit (the heart of the procedure)
Day 2, ~5 h. After T0 (uses core/geo). Parallel-safe with T3/T4.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T5 of V3R: convert relative depth to metric
depth via the ground-plane scale fit against GPS height. This is the
one place the metric claim is earned — pair-review your math. You
own ONLY the files listed below.

PROJECT FACTS (plan §4.4.4 — the procedure, exactly)
- The depth model outputs relative depth r(p) (arbitrary units,
  larger = farther after T3's conversion). One unknown, the scale s,
  converts to metres: D(p) = s · r(p).
- Ground reference height z_g is known from h_agl = alt −
  ground_ref_altitude.
- Ground-pixel selection: pixels in the lower frame region where the
  relative-depth map is smooth (local variance below
  smoothness_max_var, window smoothness_window) — over the nadir-ish
  mount this is the open ground around structures. Water/smoke/shadow
  fail naturally: their depth field is smooth BUT inconsistent with
  the ground plane, so the fit residuals blow up and the MAD rejects
  them (step 5 below).
- Per ground pixel, its ray (intrinsics + orientation) intersects the
  plane z = z_g at a known metric slant distance D_geo(p) — pure
  geometry, from core/geo.ray_ground_distance.
- Robust one-parameter least squares over all ground pixels:
  s = Σ w·r·D_geo / Σ w·r² (weights = smoothness confidence), then
  ONE median-based outlier-rejection pass and a re-fit.
- Rejection: too few ground pixels (< min_ground_fraction of frame)
  or MAD above max_fit_mad_m → REJECT THE KEYFRAME with the reason —
  counted, logged, never a silent best-guess.
- ground_ref_altitude is refined over time (T6's feedback): the
  launch-point value is used for the first ground_ref_settle_keyframes
  (10) accepted keyframes. Your function takes ground_ref_alt as a
  parameter; the loop owns the refinement.
- Sanity anchor: at 100 m AGL, 90 deg FOV, ground pixels at frame
  bottom sit at ≈ 100–160 m slant range — a strong lever arm. The 15
  deg forward tilt INCREASES it.
- visual_only mode: without telemetry the scale is undefined — scale
  = 1.0, artifacts stamped UNSCALED upstream. A labelled mode, never
  a fallback that pretends to be metric.
- Consumer GPS honesty (§4.4.8): 2–5 m horizontal CEP, worse vertical;
  IMU 1–2 deg → 1.7–3.5 m ground shift at 100 m AGL. Relative
  structure is strong, absolute placement carries GPS/IMU error. Fit
  quality is recorded per keyframe — nobody writes "accurate" without
  a number.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
geometry/scale_fit.py, core/debugio.py (EXTEND APPEND-ONLY:
save_ground_overlay), tests/test_scale_fit.py.

INPUTS ALREADY BUILT (trust these, do not change)
- core/geo: build_rays, camera_rotation, ray_ground_distance.
- core/dataclasses: DepthResult, ScaleFit(scale, mad_m, ground_frac,
  ok, reason), Frame, FlightSample.
- core/settings: min_ground_fraction 0.20, max_fit_mad_m 3.0,
  smoothness_window 9, smoothness_max_var, outlier_mad_multiplier 3.0,
  ground_region_row_fraction 0.5, fov_x_deg, mount_tilt_deg.

YOUR TASK (do exactly this, in order)
1. geometry/scale_fit.py — PURE function (R14, no files, no clock):
   fit_scale(relative_depth, image_shape, flight_sample, rotation,
   rays, ground_ref_alt, settings) -> ScaleFit.
   a. Ground-pixel selection: rows below
      ground_region_row_fraction · H AND local variance of relative
      depth < smoothness_max_var (local variance via box-filter
      E[x²] − E[x]² with window smoothness_window; cv2.blur is
      allowed here as the box filter).
   b. Per ground pixel: D_geo(p) = ray_ground_distance(...) on the
      plane z = ground_ref_alt-relative ground level.
   c. Weighted fit s = Σ w·r·D_geo / Σ w·r² with
      w = 1 / (1 + var / smoothness_max_var).
   d. ONE outlier pass: residuals |s·r − D_geo|; drop pixels with
      residual > outlier_mad_multiplier × MAD(residuals); re-fit
      once. Report the final MAD in metres: MAD(s·r) vs D_geo.
   e. Rejections (R9 — counted, never silent): ground_frac <
      min_ground_fraction → ScaleFit(ok=False, reason="no_ground");
      MAD > max_fit_mad_m → reason="fit_mad". Success →
      reason="ok".
   f. visual_only: flight_sample is None → scale = 1.0, ok=True,
      reason="visual_only".
2. debugio.save_ground_overlay (append-only): green overlay of the
   selected ground mask on the keyframe → ground/frame_XXXXXX.png,
   ONLY when png_wanted (§5.2: "the one place the metric claim is
   earned — show the mask").
DONE WHEN
[ ] Synthetic frame with known geometry recovers the scale within
    5 %: construct plane + box at known depth, r = D_true / s with
    known s, run fit, assert |s_est − s| / s < 0.05.
[ ] A doctored "water" frame (a smooth low-texture region whose depth
    field is inconsistent with the ground plane) is rejected with
    reason="fit_mad".
[ ] A frame with too little ground (< 20 %) is rejected with
    reason="no_ground".
[ ] Outlier robustness: 10 % wildly-wrong pixels → still within 5 %
    after the rejection pass.
[ ] Overlay PNG written when the gate is on; not written when off.
[ ] pytest + ruff + mypy green; <= 5-line summary of what changed.
TESTS (analytic, R14)
- Perfect plane → exact recovery (near-zero MAD).
- Plane + outliers → robustness test above.
- Rejection paths return the EXACT reason strings.
- The function is pure: assert no file writes / no time calls (grep
  or design inspection noted in the test comment).
FAILURE PROTOCOL
If the synthetic test fails, debug against the analytic answer. Do
NOT loosen min_ground_fraction or max_fit_mad_m to make a test pass —
those thresholds are settings with reasons; changing them is a
DECISIONS.md entry with a measured justification.
```

---

## T6 — Height map fusion

```text
==================================================================
V3R CODING-AGENT PROMPT — T6 · Height map fusion
Day 3, ~4 h. After T0 + T4 (consumes clouds). Parallel-safe with T7
if T7 needs your ground filter — freeze its signature early.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T6 of V3R: the persistent ENU height map —
running weighted-mean fusion, hit counts, confidence, the unknown-cell
honesty rule, and the ground-reference feedback loop. You own ONLY
the files listed below.

PROJECT FACTS (plan §4.4.6 + §4.4.4 step 6, exactly)
- Grid in ENU. Cell size cell_m = gsd · cell_factor, where
  gsd = F / W (F = footprint width in metres from the keyframe rule,
  W = image width in pixels). Defaults: cell_factor 4 → 0.42 m at
  100 m / 90 deg / 1920 px.
- Per cell: running weighted mean of Z (Σw·z / Σw), hit count,
  colour mean, confidence = f(hits, fit quality). The plan leaves f
  open; the chosen f (document it in DECISIONS.md as provisional):
  confidence = min(hits, 2·min_hits) / (2·min_hits), multiplied by
  1.0 when the last ScaleFit.ok else 0.5.
- Cells with hits < min_hits (3) are UNKNOWN — never interpolated,
  never guessed. The mesh shows a hole there; status reports
  coverage %. Adding observations only tightens estimates (the
  "gets more accurate over time" behaviour is a weighted mean —
  nothing exotic).
- The grid grows to cover observations but is bounded by
  area_limit_km2 (4 km²) — exceeding it raises AreaLimitError (honest
  failure, R9), never silent clipping.
- Ground-reference feedback (the one documented circular dependency):
  for the first ground_ref_settle_keyframes (10) accepted keyframes,
  h_agl uses the launch-point ground_ref_altitude; after settle, the
  median of fused GROUND-cell heights feeds back (ground = robust
  local minimum filter, §4.5). Implemented as an explicit method the
  pipeline calls — no hidden side effects.
- latest.npz is the resume artifact (TODO-112): written atomically
  (tmp + os.replace) so kill -9 mid-run never leaves a half-written
  state.
- Debug artifacts (§5.2): latest.npz ALWAYS; relief PNG only when the
  PNG gate says so.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
geometry/heightmap.py, core/debugio.py (EXTEND APPEND-ONLY:
save_heightmap), tests/test_heightmap.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T4: backproject(relative_depth, scale, rotation, cam_pos_enu, rays,
  colors_bgr) -> (xyz, rgb, skipped_count).
- T5: ScaleFit(scale, mad_m, ground_frac, ok, reason).
- T0: settings (cell_factor, min_hits, area_limit_km2,
  ground_ref_settle_keyframes, ground_filter_kernel), debugio
  (RunFolder.heightmap_npz, heightmap_png(step), png_gate).

YOUR TASK (do exactly this, in order)
1. geometry/heightmap.py — pure class (R14: numpy in/out, no clock;
   file writing stays in debugio):
   HeightMapFusion(cell_m, area_limit_km2, min_hits, ...).
   accumulate(xyz, rgb, weight_per_point, fit_ok) → merges points
   into cells: weighted mean z (Σw·z/Σw), hit count, colour mean,
   confidence per the formula above. The grid origin/extent grows to
   the bounding box of observations; exceeding area_limit_km2 raises
   AreaLimitError.
   z_at(cell) / colour_at(cell) / confidence_at(cell): raise
   UnknownCellError when hits < min_hits — unknown cells RAISE on
   access rather than return 0.
   ground_ref_update(...) — after settle: ground_ref_altitude =
   median of ground-cell heights (ground = robust local minimum
   filter: grey erosion with ground_filter_kernel on the z-grid,
   unknown cells excluded). Returns the new ground_ref_altitude;
   the pipeline decides to adopt it (first 10 keyframes ignore it).
   coverage_pct() = known cells / total cells in the occupied
   bounding box.
   snapshot() -> dict of arrays {z (float32, NaN where unknown),
   hits (int32), confidence (float32), colour (uint8)}.
2. debugio.save_heightmap (append-only): latest.npz ALWAYS (the four
   arrays from snapshot(), atomic tmp + os.replace); relief PNG when
   png_wanted: shaded relief (normalize z, hillshade via gradient ·
   light direction) with unknown cells black.
DONE WHEN
[ ] Convergence test: 30 synthetic observations of a 12 m box from
    moving camera poses converge monotonically toward 12 m — the P90
    of the box's cell heights after k observations is non-decreasing
    within tolerance and lands at 12 ± 0.5 m.
[ ] Ground observed 10× sits at z ≈ 0 ± 0.2 m.
[ ] Unknown access raises UnknownCellError (2-hit cell raises, 3-hit
    cell returns).
[ ] latest.npz round-trip: keys z/hits/confidence/colour present,
    NaN exactly where hits == 0.
[ ] AreaLimitError raised past the limit.
[ ] Ground-ref feedback: after settle, update returns the fused
    median on a synthetic map with known ground.
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS (analytic, R14)
- Single-cell accumulation math with known weights (Σw·z/Σw by hand).
- min_hits boundary (2 raises, 3 returns); NaN where hits == 0.
- The convergence test above (pure: feeds synthetic clouds directly,
  no model, no GPU).
- Coverage math on a half-known grid.
- The confidence formula: fit_ok=False halves confidence.
FAILURE PROTOCOL
If the convergence test is noisy, check the weight scheme — do not
"smooth" unknown cells or interpolate. Unknown means unknown; a hole
is the honest output.
```

---

## T7 — Regions + meshes

```text
==================================================================
V3R CODING-AGENT PROMPT — T7 · Regions + meshes
Day 3, ~6 h. After T6 (uses the height map + its ground filter).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T7 of V3R: from the fused height map to
stable-ID object regions and closed prism/terrain meshes. You own
ONLY the files listed below.

PROJECT FACTS (plan §4.5, exactly)
- Ground height per cell: robust local minimum filter (reuse T6's
  ground function — do not duplicate it). normalized height = cell −
  ground.
- Object mask: nh > min_object_height_m (default 2.0) AND confidence
  ≥ min_region_confidence (setting, default 0.5, provisional — plan
  §4.5 names the gate; the value is measured in Task 12).
- Connected components (8-neighbour); filter: min_cells (25), fill
  holes (binary_fill_holes on the mask).
- Track: footprint overlap keeps the ID; a new region gets a new ID;
  missing for remove_after_updates (5) updates means removed.
  Overlap metric: intersection-over-union of cell sets ≥ keep_overlap
  (0.3) → same ID. Deterministic: regions sorted by (min_row,
  min_col) so IDs are reproducible (NFR-7: same input → same meshes
  and IDs).
- Buildings: prism-with-roof mesh — footprint polygon simplified with
  cv2.approxPolyDP (epsilon from a setting), roof at height = P90 of
  the region's cell heights, vertical walls down to ground, outward
  faces. This IS the "accurate position and overall structure"
  product.
- Trees/humans/vehicles/complex objects: the classifier is a declared
  extension point (ObjectClassifier returning unknown_structure;
  TODO-102/103 own real classification) — you do NOT build it, and
  no UI/settings/messages exist for it (R10).
- Terrain mesh: known ground cells only; holes under objects and
  unknown cells — visible honesty beats pretty guesses.
- Mesh files: vertices float32 (N,3), faces int32 (M,3), axes already
  Blender's (X east, Y north, Z up — ENU matches directly). OBJ is
  the human-inspection format (CLI harness); the npz that Blender
  consumes is written by scene/publisher in T10, not here.
- Debug artifacts (§5.2): objects appear in messages (always);
  regions.png (instance-coloured mask overlay on the latest keyframe)
  is the debug-only artifact; per-object OBJ files ship when the PNG
  gate is on.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
geometry/regions.py, geometry/meshing.py, io_formats/obj_writer.py,
core/debugio.py (EXTEND APPEND-ONLY: save_regions_overlay,
save_object_obj), tests/test_regions.py, tests/test_meshing.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T6: HeightMapFusion (z_at raises UnknownCellError on unknown,
  confidence_at, coverage_pct) and its robust-local-minimum ground
  function (ask for its exact name/signature; if T6 has not landed,
  STOP and coordinate — do not re-implement it).
- T0: settings (min_object_height_m, min_cells, keep_overlap,
  remove_after_updates, min_region_confidence, approx_poly_epsilon),
  debugio (RunFolder.objects_dir(step), regions png path, png_gate),
  Region dataclass.

YOUR TASK (do exactly this, in order)
1. geometry/regions.py — pure:
   extract_regions(fusion, settings) -> list[Region]: ground filter
   → normalized height → mask (nh > min_object_height_m AND
   confidence ≥ min_region_confidence) → 8-neighbour connected
   components → min_cells filter → hole fill → deterministic order.
   update_regions(prev_state, current_regions, step) -> (new_state,
   upserts, removes): IoU-based ID tracking per the facts above;
   removes collected after remove_after_updates consecutive misses.
2. geometry/meshing.py — pure:
   object_mesh(region, fusion, settings) -> (vertices, faces):
   approxPolyDP-simplified footprint, roof at P90 height, walls to
   ground, OUTWARD faces, closed manifold — the builder ASSERTS
   closure (every edge shared by exactly two faces) and raises
   MeshError otherwise. height_m = P90 of the region's cell heights.
   terrain_mesh(fusion, regions, settings) -> (vertices, faces):
   quads over KNOWN ground cells only; holes under object footprints
   and unknown cells.
3. io_formats/obj_writer.py — ASCII OBJ (v + f lines, 1-indexed),
   write(path, vertices, faces) + a read() for tests. <= 80 lines.
4. debugio append-only: save_regions_overlay (instance-coloured mask
   over the latest keyframe, png gated), save_object_obj (per-object
   OBJ into objects/step_XXXX/, gated with the PNG set).
DONE WHEN
[ ] Two separate synthetic boxes → exactly two regions with STABLE
    IDs across 5 successive observation steps, and two closed meshes.
[ ] A scripted merge sequence (two boxes growing into contact) tracks
    deterministically: the same observation sequence → the same ID
    assignment on two runs.
[ ] Terrain mesh has holes exactly at object footprints + unknown
    cells (test counts the skipped cells).
[ ] Mesh closure verified by an edge-count assertion in the test
    (every directed edge appears exactly once).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS (analytic, R14)
- Mask + components on a synthetic height map with two known blobs →
  two labels; min_cells filter drops the speck.
- IoU tracking: overlap ≥ 0.3 keeps the ID; drift below creates a
  new ID; 5 misses → remove.
- Roof height = P90 within tolerance; approxPolyDP reduces vertex
  count while preserving footprint area ± 10 %.
- Terrain hole count matches the expected skip set.
FAILURE PROTOCOL
If regions flicker between steps on the real clip, that is a tracking
bug — fix the IoU logic; do not add temporal smoothing hacks. Unknown
cells never become ground; objects never float on interpolated cells.
```

---

## T8 — CLI assembly (Gate B owner)

```text
==================================================================
V3R CODING-AGENT PROMPT — T8 · CLI assembly (Gate B owner)
Day 3, ~4 h, after T6 + T7. This is the first end-to-end of the core
pipeline: file in, artifacts out, no Blender.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T8 of V3R: v3r_cli.py wires T2–T7 into the
pipeline with no Blender, plus the auto-generated report and graphs.
Gate B is judged on your output — you produce the evidence, the team
marks the gate. You own ONLY the files listed below.

PROJECT FACTS
- The CLI exists to prove the §4.4 procedure on real footage before
  any UI depends on it. The core modules are written once and used by
  both the CLI and the worker (the plugin is the CLI plus a different
  publisher).
- Interface (§6.3, keep it small):
  python v3r_cli.py --video FILE [--telemetry FILE] [--out DIR]
                    [--max-seconds N] [--debug full|stats|off]
                    [--export-obj] [--backend directml|cpu_reference]
- --telemetry omitted → visual_only mode: relative depth, no metric
  scale, clouds in pixel units, EVERY artifact stamped UNSCALED. A
  labelled mode that unblocks Day-1 testing before real telemetry
  arrives — never a fallback that pretends to be metric.
- cpu_reference backend = same ONNX on CPU for debugging on any
  laptop; clearly slower; never the default.
- The loop per keyframe: source → keyframe decision → depth (GPU) →
  scale fit → back-project → cloud → height-map fusion → regions →
  meshes → artifacts + telemetry rows. OBJ export is CLI-only
  (§4.9).
- report.md (§5.4) is auto-generated, zero hand-typing; graphs come
  from telemetry.csv at run end. The worker (T10) reuses these
  modules on Stop — build them importable, not script-only.
- Debug/telemetry: the CLI writes the FULL §5.1 artifact tree; PNG
  gate per §5.5 (debug full|stats|off maps to the debug_artifacts
  modes; CSVs + npz/PLY always written).

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
v3r_cli.py, core/report.py, core/graphs.py, tests/test_report.py,
tests/test_cli.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T2: FileSource, telemetry parser, keyframe decide().
- T3: DepthModel + save_depth/save_diff.
- T4: backproject + PLY writers + save_cloud.
- T5: fit_scale + save_ground_overlay; ScaleFit.
- T6: HeightMapFusion + save_heightmap.
- T7: extract/update_regions + meshing + save_regions_overlay.
- T0: debugio (create_run_folder, write_meta, TELEMETRY_COLUMNS,
  CsvAppender, png_gate), settings registry, message dataclasses.

YOUR TASK (do exactly this, in order)
1. v3r_cli.py — argparse exactly per the interface above. Creates the
   run folder via debugio; writes meta.json (settings snapshot incl.
   CLI overrides, model hashes from models/weights/, package
   versions, machine info, input fingerprint); wires the keyframe
   loop with the real components; per-keyframe telemetry row (the
   §5.3 columns, real measured ms per stage); run ends with
   report.md + graphs. No bpy import reachable from this file (R1).
   --max-seconds bounds the run for quick tests. If a stage raises
   (TelemetryError, AreaLimitError, MeshError, SourceError), the CLI
   prints the counted failure and exits nonzero — no silent partial
   trees without a report.
2. core/report.py — generates report.md from telemetry.csv +
   skipped.csv + the run's message log (pure python over CSVs, csv
   module, so the worker reuses it): run summary (settings + source +
   model hashes); keyframe yield table (accepted/rejected by reason);
   scale-fit quality distribution (MAD table + bins); height-map
   coverage; object table (id, height, area, first_seen, last_updated,
   updates); latency table (per stage mean/p95 from the telemetry
   columns); anomalies (every degraded interval, every error code,
   with timestamps).
3. core/graphs.py — matplotlib (Agg backend, constrained_layout=True,
   English labels, legends outside the plot area):
   latency_per_stage.png (stacked bars per keyframe),
   height_convergence.png (P90 height of the 3 largest objects vs
   observations), scale_fit_quality.png (fit MAD over time),
   object_growth.png (objects + coverage vs time), frame_budget.png
   (keyframe cadence vs target cadence). Generated from telemetry.csv.
4. --export-obj: per-object OBJ files for human inspection (CLI-only).
DONE WHEN (Gate B evidence — run the checklist on the real clip and
paste the results; the team marks the gate)
[ ] 1. A real ≥ 3-min aerial clip runs end-to-end on the RX 5700 XT;
      the §5.1 artifact tree is fully populated.
[ ] 2. report.md numbers exist for: keyframe yield, fit MAD
      distribution, coverage, object count, per-stage latency.
[ ] 3. Synthetic unit tests pass: flat ground + two boxes of known
      height → two closed prism meshes within tolerance (tests run
      without GPU via a mock depth provider — the only mock, confined
      to tests/).
[ ] 4. No keyframe is silently dropped: every reject has a reason row.
[ ] 5. A 10-second sample runs in visual_only mode (artifacts stamped
      UNSCALED).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS
- --help works; unknown flag fails loudly.
- visual_only stamps UNSCALED (check meta.json + artifact names).
- report generation from a fixture CSV produces every §5.4 section
  with real numbers (no empty cells; no "TODO" strings in output).
- Graphs: five files exist, each > 0 bytes, after a fixture run.
- Failure surfacing: a fixture run with a corrupt telemetry row exits
  nonzero with the counted error.
FAILURE PROTOCOL
If the end-to-end run fails on real footage, bisect by stage using
the artifacts (the debug tree exists exactly for this) and report the
failing stage with its last artifact. Do not add stage-level try/except
fallbacks to "keep going" — a counted failure is the honest output.
```

---

## T9 — Blender process side

```text
==================================================================
V3R CODING-AGENT PROMPT — T9 · Blender process side
Day 4, ~6 h. Consumes only T0's message contract (runs in parallel
with T10 — use a fake worker in tests, never T10's code).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T9 of V3R: the Blender side of the two-process
design — spawn the worker, read its JSON lines without ever blocking
the UI, apply meshes on a timer with bounded work, stop cleanly. You
own ONLY the files listed below.

PROJECT FACTS (plan §4.2/§4.7/§4.9, exactly)
- Blender's Python is the same interpreter as the UI: anything heavy
  there freezes the viewport. Therefore the worker is a separate OS
  process with its own Python env; Blender only (a) writes a config
  file and spawns it, (b) reads its stdout without waiting, (c)
  applies mesh files on a timer. This also survives a worker crash —
  Blender shows the error and stays alive.
- The timer applies at most max_updates_per_tick (6) messages and
  returns before tick_budget_ms (30 ms) elapses — that is the ENTIRE
  non-blocking mechanism; no locks, no waiting, newest-version-wins
  queue.
- Blender -> worker traffic is ONLY the line "stop" on stdin.
- Message contract (§4.9, one JSON object per line, worker → Blender):
  status {state, frames_read, keyframes_used, keyframes_rejected,
    objects, coverage_pct, text} — every 1 s
  origin {lat, lon, alt_m} — once, at anchor
  terrain {version, file} — file = npz path
  object_upsert {object_id, version, file, height_m,
    footprint_area_m2, class} — class = "unknown_structure"
  object_remove {object_id}
  error {code, message, fatal} — never swallowed
- Mesh npz: vertices float32 (N,3), faces int32 (M,3), axes X east /
  Y north / Z up, written to temp then atomically renamed by the
  worker — the applier can assume a complete file.
- Anchoring is the honest "we need ~10 accepted keyframes before the
  first publish" state; the status text says exactly that (§4.8).
- Degraded state banner text (§4.4.7): "telemetry inconsistent —
  positions unreliable".
- Hard rules (§8 R1/R2): only blender/ may import bpy; only the main
  thread touches bpy; the Blender side never blocks.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
blender/process_manager.py, blender/applier.py, blender/operators.py,
blender/panel.py (extend the existing repo files where they exist),
tests/test_applier.py, tests/fake_worker.py (a scripted fake worker
that plays the worker role in tests).

INPUTS ALREADY BUILT (trust these, do not change)
- core/dataclasses message classes with to_line()/from_line() (T0).
- core/settings (max_updates_per_tick, tick_budget_ms, stop_timeout_s,
  queue_len, worker_python_path, probe_timeout_s).

YOUR TASK (do exactly this, in order)
1. blender/process_manager.py: collect settings → config.json (source
   kind, source, telemetry path, run folder, settings dict, model
   paths + SHA256); spawn the worker via subprocess.Popen(
   worker_python_path, worker/app.py, "--config", path); a stdout
   reader THREAD parses JSON lines and pushes them into a
   newest-version-wins queue (length queue_len 8; overflow drops the
   oldest and counts it) — the main thread NEVER reads stdout
   directly; stdin handle kept for "stop"; process exit watched from
   the timer. If the process dies, the timer notices and reports the
   exit code + the last log lines (§4.2).
2. blender/applier.py — the bpy-writing module:
   origin → creates/updates the V3R collection with custom props
   (lat, lon, alt_m);
   terrain {version, file} → V3R_TERRAIN object: in-place mesh replace
   (update the existing object's mesh data, never duplicate);
   object_upsert → V3R_OBJ_<id>: create if new, else replace mesh
   data in place; version check skips stale versions (newest wins);
   object_remove → delete the object;
   error → surfaced as a UI message, never swallowed;
   apply(queue) processes ≤ max_updates_per_tick messages and returns
   within tick_budget_ms (the test measures this).
3. blender/operators.py (extend the existing Start/Stop): Start
   validates settings, writes config.json, spawns, registers the
   bpy.app.timers timer (poll every 250 ms → apply → return);
   Stop sends the "stop" line, waits ≤ stop_timeout_s (10) ACROSS
   TIMER TICKS (never a blocking wait), then reports the exit code;
   exceeded → terminate + honest error message. Add an Open Run
   Folder operator (reveals runs/<run_id> in the file browser —
   supports §1.1 item 9: every demo run is a measurement run).
4. blender/panel.py: state banner from status messages (state,
   frames, objects, coverage, text) including the Anchoring honest
   text and the degraded warning string exactly as specified above.
5. Blender 5.2 registration: follow the repo's existing add-on
   registration (legacy bl_info or extension manifest — match what
   is there; if you add blender_manifest.toml, it must not break the
   legacy path). No bpy outside blender/ — enforced by a grep test.
DONE WHEN
[ ] A scripted fake worker (tests/fake_worker.py, the mock lives only
    in tests/) drives the applier with 200 object_upserts at 60/s:
    per-tick apply time stays ≤ tick_budget_ms and object versions
    deduplicate to newest (final state = last version per object, no
    duplicate objects).
[ ] origin → collection custom props test; object_remove test;
    stale-version skip test.
[ ] Stop path: stdin "stop" sent, exit code reported, terminate
    fallback after stop_timeout_s (tested with the fake worker).
[ ] State banner strings match the Anchoring/degraded texts exactly.
[ ] No "import bpy" outside blender/ (grep test in tests/).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS
- Fake-bpy harness in tests/ (R1: tests may mock).
- Queue: overflow drops oldest + counts; newest version wins.
- config.json round-trip contains all required §4.9 fields.
- Timer bounded-apply measured with the fake clock in tests/.
FAILURE PROTOCOL
Never "wait a bit" in the main thread — poll, apply, return is the
whole design. If the fake worker cannot reproduce a timing issue,
report it; do not add sleeps or locks to mask it.
```

---

## T10 — Worker app

```text
==================================================================
V3R CODING-AGENT PROMPT — T10 · Worker app
Day 4, ~5 h, after T8 (reuses its report/graphs modules). Runs in
parallel with T9 (which consumes only your MESSAGE CONTRACT — the
fake worker in T9's tests plays your role).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T10 of V3R: the worker process that wires the
whole pipeline behind a stdin/stdout contract — config file in, JSON
status lines out, clean stop, honest fatal errors, atomic publishes.
You own ONLY the files listed below.

PROJECT FACTS (plan §4.7/§4.8/§4.9, exactly)
- The worker never imports bpy and never opens a window (R2). It is
  spawned by Blender with a config file; Blender -> worker traffic is
  ONLY the line "stop" on stdin.
- Messages: one JSON object per line on stdout, using T0's message
  dataclasses (status every 1 s; origin once at anchor; terrain;
  object_upsert; object_remove; error {code, message, fatal}).
- State machine (§4.8): Idle → Starting (Start pressed) → Anchoring
  (worker up, reading frames) → Building (scale fit OK, origin
  published) → Degraded (telemetry consistency check fails 3×) →
  Building (check passes again) → Stopping (Stop pressed) → Idle
  (worker exited, code 0 or error). Building → Failed on fatal
  error / stream lost; Failed → Idle acknowledged.
- Flight-data consistency check (§4.4.7): after each accepted
  keyframe, the measured ground height at the footprint centre should
  match the fused map's ground height there; if |Δ| >
  gps_consistency_tolerance_m for 3 consecutive keyframes → the run
  enters degraded: processing CONTINUES, but the status banner shows
  "telemetry inconsistent — positions unreliable" and the run is
  flagged in report.md. No auto-correction magic (TODO-118 owns it);
  the check exists so bad telemetry is VISIBLE.
- Mesh publish: npz (vertices float32 (N,3), faces int32 (M,3), X
  east / Y north / Z up) written to temp then ATOMICALLY renamed — a
  kill -9 mid-run must leave no half-written mesh (§4.9).
- On Stop: finish the current keyframe, finalize report.md + graphs
  (§5.4 — reuse T8's core/report.py + core/graphs.py), send the final
  status, exit 0. Fatal → error JSON {code, message, fatal:true} +
  nonzero exit. The GPU session is released on EVERY exit path.
- Debug artifacts are written on the worker thread BETWEEN keyframes
  (§5.5) — the Blender process pays zero debug cost; nothing
  debug-related crosses the JSON channel.
- A plugin run writes the SAME §5.1 tree as the CLI (§1.1 item 9) —
  the worker calls the identical core modules + debugio; it is not a
  second pipeline.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
worker/app.py, scene/publisher.py, geometry/consistency.py,
tests/test_worker.py, tests/test_consistency.py.

INPUTS ALREADY BUILT (trust these, do not change)
- The full T2–T7 stack (source, telemetry, keyframe, depth, scale
  fit, backproject, height map, regions, meshing) and T3's debugio
  artifact writers.
- T8: core/report.py + core/graphs.py (finalize on Stop).
- T0: message dataclasses (to_line/from_line), debugio, settings
  (stop_timeout_s, gps_consistency_tolerance_m, debug_artifacts).

YOUR TASK (do exactly this, in order)
1. scene/publisher.py: messages → stdout (one JSON line, flush
   immediately, unbuffered) via to_line(); mesh npz writer (vertices/
   faces arrays, axes as specified) written to a tmp file then
   os.replace — atomic; publishes object_upsert (file = npz path
   under objects/step_XXXX/), terrain (npz path), origin (once at
   anchor), object_remove. The Blender-side applier (T9) consumes
   exactly this.
2. geometry/consistency.py — pure (R14): given the measured ground
   height at the footprint centre and the fused map's ground height
   there, maintain a streak: |Δ| > gps_consistency_tolerance_m for 3
   consecutive keyframes → degraded; a passing check recovers it.
   Every check emits a counted row. Returns
   ConsistencyState(degraded: bool, bad_streak: int, last_delta_m).
3. worker/app.py: parse --config; construct the components (T2
   source, T3 model, T5 fit, T4 backproject, T6 fusion, T7
   regions/meshing, publisher); run the keyframe loop; implement the
   §4.8 state machine with the exact state names above; status line
   every 1 s (mock clock in tests only); stdin line "stop" → finish
   current keyframe → finalize report.md + graphs → final status →
   exit 0; fatal → error JSON + nonzero exit; GPU session close() on
   every exit path; artifacts written between keyframes per §5.5.
DONE WHEN
[ ] Headless run on the real clip produces messages + files.
[ ] Stop ends within stop_timeout_s (measured).
[ ] kill -9 mid-run leaves no half-written mesh: the test publishes
    in a loop, kills the process, then verifies no tmp files remain
    and every published npz loads cleanly.
[ ] Degraded transition: inject 3 bad keyframes → Degraded; a good
    keyframe → back to Building (banner text per §4.4.7).
[ ] Final report.md + graphs exist and are complete after Stop (§5.4
    sections; the run folder matches the CLI tree — §1.1 item 9).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS
- Consistency math analytic (known deltas → streak → degraded →
  recovery).
- State machine transitions table-driven (every §4.8 edge).
- Atomic publish: simulate os.replace failure → tmp cleaned up, old
  file intact.
- Status cadence with a mock clock (mocks only in tests/).
- Config parsing: missing field → error JSON, nonzero exit.
FAILURE PROTOCOL
If the headless run diverges from the CLI on the same clip (different
meshes/numbers), that is a bug — the two front ends share core modules
and must be byte-equivalent on artifacts. Report the first divergent
stage; do not patch around it.
```

---

## T11 — Stream mode in Blender

```text
==================================================================
V3R CODING-AGENT PROMPT — T11 · Stream mode in Blender
Day 5, ~4 h. After T9 + T10. Cut order note: if behind, the probe
ICON polish is cut first, but raw-URL stream mode stays.
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T11 of V3R: live stream ingest with a truthful
connectivity probe and the UI icon. Same pipeline; only the ingest
differs. You own ONLY the files listed below.

PROJECT FACTS (plan §4.6, exactly)
- The worker opens the URL with a bounded probe (probe_timeout_s, 5 s):
  success = {ok, width, height, fps} JSON → UI shows green; failure =
  {ok:false, reason} → red. Probe is a separate small process — it
  never blocks the UI.
- During the run, the ingest keeps only the NEWEST frame (drop-oldest,
  counted), and if no frame arrives for lost_timeout_s (10 s) the run
  ends with StreamLostError — status shows it; nothing hangs.
- Digital HD sources only (§3.6): the 5.8 GHz analog VTX path is out
  of scope (risk R1); if frame statistics look analog-grade the UI
  reports "video quality below minimum" (blur/low-res heuristics,
  thresholds measured in Task 12). Analog input is never silently
  accepted.
- OBS test: OBS → (rtmp or udp) → worker, one live 3-min session.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
ingest/source.py (stream path completion), blender/panel.py (URL
field + probe icon), worker/app.py (probe subcommand only — coordinate
append-only with T10's owner if not merged), tests/test_stream.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T2: StreamSource skeleton (probe(), frames() with drop-oldest,
  StreamLostError).
- T9: panel + timer poll pattern; T10: worker app structure.
- T0: settings (probe_timeout_s, lost_timeout_s, queue_len).

YOUR TASK (do exactly this, in order)
1. worker probe subcommand: worker/app.py --probe URL →
   StreamSource.probe with probe_timeout_s → prints ONE JSON line
   {ok, width, height, fps} or {ok:false, reason} → exits. No
   pipeline started.
2. blender/panel.py: stream URL field + Probe operator → spawns the
   probe as a background process (non-blocking); icon states:
   checked/green (probe ok), red (probe failed — reason string shown
   next to the icon), unknown (not probed yet); result picked up by
   the existing 250 ms timer poll. Start on a stream URL starts the
   same pipeline with ingest=stream (config source kind "stream").
3. StreamSource runtime completion: drop-oldest while processing
   (dropped count → status text + telemetry), no frame for
   lost_timeout_s → StreamLostError → worker publishes error
   {code:"stream_lost", fatal:true} → state Failed → Idle
   acknowledged.
4. OBS end-to-end test (manual, recorded in DECISIONS.md): OBS →
   (rtmp or udp) → worker, one live 3-min session producing objects;
   note the OBS settings used (encoder, bitrate, keyframe interval).
DONE WHEN
[ ] The icon truthfully reflects a deliberately-broken URL vs a
    working one (manual check, recorded with screenshots).
[ ] A live 3-min OBS session produces objects in the viewport.
[ ] The drop-oldest counter is visible in status/telemetry.
[ ] StreamLostError path ends the run honestly (no hang).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS (no real network in CI)
- Probe timeout on an unroutable URL (e.g. localhost:1) fails FAST,
  not hang (assert elapsed < probe_timeout_s).
- StreamLostError after injecting silence into the frame source.
- Drop-oldest counter math (feed N frames slowly, read M newest).
- Probe JSON contract round-trip.
FAILURE PROTOCOL
Never "wait and hope" around streams — every wait has a timeout with
a name. If OBS gives a stream the probe accepts but the pipeline
cannot sustain, report the measured fps/latency; do not widen
timeouts silently.
```

---

## T12 — Measurements + docs

```text
==================================================================
V3R CODING-AGENT PROMPT — T12 · Measurements + docs
Day 5–6, ~6 h (Day 6 owner). The ODM baseline must be started Day 4
in the background (ODM is CPU-slow — hours).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T12 of V3R: turn the runs into MEASURED
numbers and honest documentation. This card produces no pipeline code
except settings-default updates — its deliverables are RESULTS.md,
USER_GUIDE.md, and DECISIONS.md entries backed by data.

PROJECT FACTS
- Accuracy honesty (§4.4.8): consumer GPS 2–5 m horizontal CEP, worse
  vertical; IMU ~1–2 deg → 1.7–3.5 m ground shift at 100 m AGL.
  Relative structure (building shapes, heights, spacing) is strong —
  it comes from visual depth. Absolute placement carries the GPS/IMU
  error. RESULTS.md reports absolute error against known references.
  Nobody writes "accurate" without a number next to it.
- §7 rule: every claim in RESULTS.md carries its measurement method;
  estimates are marked "estimate"; nothing is copied from a datasheet
  into a results table.
- Baselines: ODM (offline DSM + ortho + reference mesh on the same
  frames — pseudo-ground-truth), known-size references (3+ objects:
  road width, car, court — tape measure / map), OpenREALM literature
  numbers for context (cited, not measured by you).
- Metrics (§7, auto-generated into report.md where possible):
  time-to-first-object; footprint IoU vs ODM ortho; height RMSE vs
  ODM DSM at same cells; position error (centroid offset vs ODM,
  table + CDF); height convergence; latency per stage mean/p95; yield;
  rejection transparency.
- Flight-check calibration (§4.4.7): Task 12 turns the observed
  consistency-check residuals into camera-tilt/time-offset
  corrections (a measured DEFAULT update; runtime auto-correction is
  TODO-118 — you do not build that).
- The finish line is §1.1 (9 items) — all ticked or explicitly failed
  with reasons.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
docs/RESULTS.md, docs/USER_GUIDE.md, core/settings.py (defaults
update ONLY: measured values replace provisional defaults; flip the
provisional flags), docs/DECISIONS.md entries, scripts/odm_baseline.sh
(the ODM invocation, for reproducibility).

YOUR TASK (do exactly this, in order)
1. Confirm the ODM baseline run started Day 4 is progressing; run it
   to completion on the same clip (scripts/odm_baseline.sh records
   the exact command + settings so it is reproducible).
2. Fill the §7 tables with MEASURED numbers, computed from
   telemetry.csv / report.md / the ODM outputs (write the small
   analysis script under scripts/ so the numbers are recomputable;
   do all arithmetic in code, never by hand): the nine metrics above,
   each in its table/graph form, zero empty cells.
3. Flight-check calibration: from the consistency-check residuals
   (T10's counted rows), estimate the camera-tilt correction and the
   telemetry time offset (first cut); update the settings defaults
   (fov_x_deg, mount_tilt_deg) with the measured values, flip their
   provisional flags, and write DECISIONS.md entries with the
   numbers + method.
4. RESULTS.md also carries the absolute-accuracy honesty note (the
   §4.4.8 text above, with YOUR measured numbers in place of the
   generic bounds where available) and the known-size-reference
   checks.
5. docs/USER_GUIDE.md: tested install steps for the worker venv (or
   explicitly marked untested), start-to-objects walkthrough for file
   and stream mode, the run-folder explanation (debug artifacts =
   the research instrument; PNG gate rules §5.5), Stop / Open Run
   Folder usage, and what each report.md table means.
DONE WHEN
[ ] RESULTS.md has zero empty cells and zero unlabelled estimates.
[ ] Finish-line §1.1 items all ticked or explicitly failed with
      reasons: 1 file-mode live meshes; 2 stream probe + live
      processing; 3 Blender interactive during runs; 4 stop within
      timeout + GPU released; 5 metres/ENU + measured error in
      RESULTS.md; 6 CLI identical core + full tree + report.md;
      7 RX 5700 XT via DirectML, no torch; 8 TextureProjector stub
      raising NotImplementedError(TODO-101); 9 plugin run writes the
      §5.1 tree + finalizes report.md on Stop.
[ ] Settings defaults updated with measured values; provisional flags
    flipped where measured; DECISIONS.md entries written.
[ ] USER_GUIDE steps actually followed once on a clean venv (note the
    date + result).
[ ] All arithmetic done by script; numbers recomputed and matched.
FAILURE PROTOCOL
If ODM fails or a reference measurement is missing, report it and
mark that cell explicitly as failed-with-reason — an honest gap beats
an invented number. Never copy a datasheet figure into a results
table.
```

---

## T13 — YOLO proxies (stretch; first cut)

```text
==================================================================
V3R CODING-AGENT PROMPT — T13 · YOLO proxies (stretch)
Day 6, ~5 h. FIRST TO CUT if behind (plan §9.0 cut order).
Paste this ENTIRE block into the coding agent as its task.
==================================================================

ROLE
You implement task card T13 of V3R (stretch): YOLO person/vehicle
detection on keyframes, projected to ground positions and heights,
published as proxy markers. You own ONLY the files listed below.

PROJECT FACTS
- Detector: YOLOv8n ONNX (exported + benched in T1; the export
  command + SHA256 are in DECISIONS.md). Runs via ONNX Runtime
  DirectML on the RX 5700 XT (torch banned at runtime; the offline
  export step is recorded, not a repo dependency).
- Projection (§4.4.1–2 math, reused — no new geometry): box
  bottom-centre pixel → camera ray → intersect the ground plane
  z = z_g → ground_x, ground_y; box top-mid ray → distance →
  est_height.
- Publish: object_upsert with class = person|vehicle; mesh =
  capsule (person) / box (vehicle) at the measured size, through
  scene/publisher. Classification beyond person/vehicle is
  TODO-102/103 — do not build it (R10).
- Debug artifacts (§5.2 detection row): detections.csv ALWAYS
  (frame, class, conf, bbox, ground_x, ground_y, est_height); boxed
  JPGs when the PNG gate is on.
- Acceptance bar: ≥ 60 % of a hand-made detection list present; the
  hand list is committed in the test.
- If the cut order dropped this card: ship detector.py +
  detect_projection.py + tests, and leave the publish wiring as
  "# TODO(V3R-109): publish proxy markers" — the interface-stub
  pattern (R11). No fake button, no fake message.

CODE RULES
V3R CODE RULES — obey all. If a rule blocks you, STOP and say which
one; do not route around it.

ZONES (import direction is one-way, downward)
 R1. blender/ — may import bpy. core/ — NEVER imports bpy, never
     imports blender/. models/ — only onnxruntime + numpy + cv2
     preprocess. io_formats/ — file writers/readers. tests/ — may
     import everything; may use mocks. Nothing else may use mocks.
 R2. Only the Blender main thread touches bpy. The Blender side never
     blocks: poll, apply, return. The worker never imports bpy and
     never opens a window.

MODULE SHAPE
 R3. One file = one job. <= 200 lines/file, <= 30 lines/function,
     <= 4 params (else a frozen dataclass). Plain English names (no
     abbreviations except: id, url, fov, gpu, gpu_vram).
 R4. Every file starts with 2–4 comment lines: what it does, what it
     does NOT do. Every public function has a one-sentence docstring
     stating what it returns.
 R5. Type hints on every function signature. Data crossing module
     boundaries is a frozen dataclass, never a bare dict.
 R6. No globals, no singletons, no side effects at import time.
     Dependencies are passed in constructors.
 R7. No magic numbers outside core/settings.py. If a value could ever
     be tuned, it is a setting with a one-line reason. Physics
     constants (2, 360, pi) may stay local.

HONESTY (the anti-sugarcoating rules)
 R8. No fake data, no placeholder meshes, no random outputs, no
     "return something plausible on failure". Mocks exist only inside
     tests/.
 R9. Raise specific error classes; catch only where you can act.
     "except: pass" fails review. Every failure surfaces as a counted
     row (telemetry/status), not silence.
 R10. Unfinished feature = "# TODO(V3R-1xx): <what's missing>" with
      the id registered in docs/TODO.md. Do not create UI, settings,
      or messages for TODO features. Banned in code, comments, and
      commit messages: "MVP", "prototype", "temporary", "for now",
      "hack", "dummy", "quick and dirty".
 R11. Extension points are interfaces that raise
      NotImplementedError(TODO_id) — declared, tested at the interface
      level, never imitated.

TECHNOLOGY (Rev 3.0)
 R12. Inference runs in the worker process only. PyTorch (CUDA/ROCm)
      and ONNX Runtime are both allowed there. Blender-side add-on code
      must not import torch or onnxruntime. The backend (cuda | rocm |
      cpu) is a startup setting; an unavailable backend is a verbose
      error, never a silent fallback.
 R13. Anything CPU-side that grows past ~250 ms/keyframe must be
      reported in the task notes — it is an architecture violation,
      not a tuning problem.

GEOMETRY & TESTS
 R14. All geometry is pure: numpy in, numpy out, no files, no GPU, no
      clock. Each function has a test with a known analytic answer.
 R15. Every task ends with: ruff format + ruff check + mypy (basic) +
      pytest green, and a <= 5-line summary of what changed. Tests may
      not require a GPU (inject a mock depth provider); one marked
      [gpu] smoke test runs only where the GPU exists.

READABILITY BUDGET
 R16. Any module you cannot explain to a second-year student in <= 5
      sentences gets split before it merges. If a diagram in the plan
      no longer matches the code, fixing the mismatch is part of the
      task, never a follow-up.

REVIEW CHECKLIST (every PR): zones respected · sizes respected ·
settings not constants · no banned words · no torch · errors surfaced
not swallowed · diagram updated · tests green.

FILES YOU OWN
models/detector.py, geometry/detect_projection.py, core/debugio.py
(EXTEND APPEND-ONLY: save_detection), tests/test_detector.py,
tests/test_detect_projection.py.

INPUTS ALREADY BUILT (trust these, do not change)
- T1: YOLOv8n ONNX + bench numbers; DA-V2 depth stack.
- core/geo: build_rays, camera_rotation, ray_ground_distance.
- scene/publisher (T10), debugio, settings, Frame dataclass.

YOUR TASK (do exactly this, in order)
1. models/detector.py: ONNX wrapper like models/depth.py — preprocess
   640 px letterbox + normalization per the Ultralytics ONNX doc
   (documented in the docstring against it); postprocess: boxes +
   scores + NMS (cv2.dnn.NMSBoxes); class mapping: COCO ids
   {0: person, 2: car, 5: bus, 7: truck} → {"person", "vehicle"}
   (mapping table documented); run(frame) -> list of
   Detection(class, conf, bbox_xyxy) + detector_ms.
2. geometry/detect_projection.py — pure (R14): project_detection(d,
   rotation, cam_pos_enu, rays, ground_z) -> (ground_x, ground_y,
   est_height) per the facts above; points where the ray never hits
   the plane are dropped and COUNTED (R9).
3. debugio.save_detection (append-only): detections.csv row ALWAYS;
   detect/frame_XXXXXX.jpg with boxes drawn (cv2.rectangle, class +
   conf labelled) when png_wanted.
4. Publish wiring through scene/publisher: object_upsert with
   class set; capsule mesh for person / box mesh for vehicle at the
   measured size (meshing helpers exist in T7 — reuse the box path;
   the capsule is a small pure builder you add under geometry/).
DONE WHEN
[ ] detections.csv + boxed JPGs for a clip with visible people.
[ ] ≥ 60 % of the hand-made detection list present (the hand list is
    committed in tests/data/ with frame, class, bbox).
[ ] Projected positions land on the ground plane in the viewport
    (manual check, recorded with a screenshot).
[ ] pytest + ruff + mypy green; <= 5-line summary.
TESTS (analytic, R14)
- Letterbox round-trip on a synthetic image (box coords map back
  exactly).
- NMS on known overlapping boxes (survivor set = expected).
- Projection: known ground plane + pixel → known ENU point ±
  tolerance; box-top ray → known height ± tolerance.
- Class mapping table test (all four COCO ids).
FAILURE PROTOCOL
If detections are poor on the real clip (low recall), report the
measured recall + example frames; do not lower the conf threshold
silently — conf_min is a setting with a reason (add it if missing,
default 0.35, provisional). The 60 % bar is the acceptance bar; a
miss is measured, not hidden.
```

---

# Appendices

## A. Gate checklists (copy into docs/DECISIONS.md on Day 1)

Gates are go/no-go on the real machine, decided by the team in a
15-minute meeting, written in DECISIONS.md — never marked by an agent.
If a gate fails, the listed pivot happens THAT DAY, and only then does
work continue. No silent plan-drift, no runtime fallbacks invented later.

- **Gate A (Day 1)** — evidence from T1: depth ONNX runs on the RX
  5700 XT at ≤ 200 ms/frame, ≤ 2 GB VRAM, visually sane output.
  Fail → same-day pivot decision (ncnn depth path / cpu_reference
  with size cut). Nothing else proceeds on assumptions.
- **Gate B (Day 3)** — evidence from T8 (§6.4 checklist): (1) a real
  ≥ 3-min aerial clip runs end-to-end, §5.1 tree fully populated;
  (2) report.md numbers exist for keyframe yield, fit MAD
  distribution, coverage, object count, per-stage latency; (3)
  synthetic unit tests pass (flat ground + two boxes of known height
  → two closed prism meshes within tolerance; no GPU, mock confined
  to tests/); (4) no keyframe silently dropped — every reject has a
  reason row; (5) a 10-second sample runs in visual_only mode.
  Fail → Day 4 becomes fix-day; stream mode (T11) drops to Day 6 and
  T13 is cut.
- **Gate C (Day 5)** — evidence from T9/T10/T11: §1.1 items 1–4 + 7
  demonstrated live, and item 9 verified once on a real plugin run
  (run folder populated + report.md finalized on Stop). Fail →
  Day 6–7 fix only; the demo shows the CLI artifacts + partial plugin,
  honestly labelled.

## B. Parallel-safety table (who may run at the same time)

| Day | Run in parallel | Constraint |
|---|---|---|
| 1 | T0 first, alone | Everyone consumes its contracts — do not start T1/T2 before T0's dataclasses/settings/geo/debugio exist. |
| 1 | then T1 ∥ T2 | Disjoint files. Both append to docs/DECISIONS.md — append-only, no edits to others' entries. |
| 2 | T3 ∥ T4 ∥ T5 | All three EXTEND core/debugio.py — strictly append-only (new functions at the end). T5 consumes core/geo rays (T0), not T4's output. |
| 3 | T6 ∥ T7, then T8 | T7 calls T6's ground-filter function — freeze its signature early, or run T6 first. T8 last (needs both). |
| 4 | T9 ∥ T10 | T9 consumes only the §4.9 message contract (fake worker in tests). Merge both end of day → first end-to-end. |
| 5 | T11 ∥ T12 (start) | T11 = stream; T12 = ODM background job + docs. |
| 6 | T12 (finish) ∥ T13 | T13 is cut FIRST if behind (see C). |

Shared-file rules that make this safe:
- `core/debugio.py` is **append-only** for T3–T7 and T13 (new writer
  functions at the end; existing functions never change).
- `docs/DECISIONS.md` and `docs/TODO.md` are append-only for everyone.
- Nobody else may touch `core/settings.py` except T0 (create) and T12
  (default updates from measurements).

## C. Cut order if behind (drop from the bottom, never from the middle)

T13 (YOLO) → T11 (stream probe icon polish; keep raw URL mode) → OBJ
export → PNG re-render of past runs (the current run still writes CSVs
always, and full PNGs on the first run of any dataset per §5.5) →
**never** T2–T9, and **never** the §5 instrumentation itself.

## D. Acceptance protocol for the operator (every card)

1. The agent's PR contains: the ticked DONE WHEN checklist, pasted
   test output (green), and a ≤ 5-line summary.
2. You verify on the machine: pytest green; the artifacts it claims
   exist on disk (spot-check the run folder: meta.json, telemetry.csv,
   at least one PNG family); ruff + mypy clean.
3. Run the §8 review checklist: zones respected · sizes respected ·
   settings not constants · no banned words · no torch · errors
   surfaced not swallowed · diagram updated · tests green.
4. Any threshold the agent had to move is a DECISIONS.md entry with a
   measured reason — no exceptions.
5. Only then does the next card (or the parallel partner) merge.
