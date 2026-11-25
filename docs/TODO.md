# V3R TODO Register

Unfinished features live here, not in the code. Code marks them with
`# TODO(V3R-1xx): <what's missing>` (rule R10). Do not build UI, settings, or messages for a
TODO feature before it is implemented.

**Standing rule:** every TODO ships its own debug artifacts and telemetry columns in the same
commit as the feature. A TODO without its instrumentation is not done. The run folder and
`telemetry.csv` grow; the §7 analysis pipeline keeps working unchanged. Any new GPU stage adds
its latency and `gpu_mem_mb` columns from day one.

**Status values:** `not started` · `in progress` · `blocked` · `done`.

| ID | Feature | Status | Notes |
|---|---|---|---|
| TODO-101 | Texture projection | not started | `TextureProjector` interface raises `NotImplementedError(TODO-101)`; picks K frames per object, ray-casts colours onto UVs. |
| TODO-102 | Object classification | not started | Context classifier on footprint, height map and colour: building / tree / vehicle / other. |
| TODO-103 | Proxy models | not started | Swap prism for parametric proxy (cone-sphere, capsule, box) sized from height map and detections. |
| TODO-104 | Water bodies | not started | Smooth low-gradient, low-normalized-height, colour region to a flat translucent surface. |
| TODO-105 | Rubble | not started | Roughness metric on the height map over building footprints to a rubble proxy. |
| TODO-106 | Damage comparison | not started | Register two runs on the same origin; per-object delta height and footprint to a damage class. |
| TODO-107 | ODM background refinement | not started | Finished area to an offline ODM run; a refined tile swaps the live mesh. |
| TODO-108 | Absolute georeferencing export | not started | Export meshes and the height map in a standard projected CRS with the GPS-bounded error. |
| TODO-109 | Human markers and dwell tracking | not started | Ground-projected tracks over the height map plus a dwell-time table. |
| TODO-110 | Multi-strip flights | not started | Fuse overlapping flight strips with per-strip scale-fit quality comparison. |
| TODO-111 | Height-map compute-shader fusion | not started | Move height-map accumulation to a GPU compute path if CPU fusion exceeds budget. |
| TODO-112 | Resume interrupted runs | not started | Persist `resume_state.json` and continue from the last accepted keyframe. |
| TODO-113 | Linux ground station | not started | NCNN-Vulkan backend path and bench table for a Linux ground station. |
| TODO-114 | Automatic camera calibration | not started | Recover intrinsics and mount tilt from flight-check residuals; writes `calibration.json`. |
| TODO-115 | Offline batch quality pass | not started | Re-run finished areas at higher quality settings for a final report. |
| TODO-116 | Analog FPV intake (research-only) | not started | Explicitly out of scope for the product; research note only, never a quiet degraded mode. |
| TODO-117 | Visual pose refinement of telemetry | not started | Refine telemetry poses with feature matching only if the flight check shows IMU orientation is too coarse. |
| TODO-118 | Automatic telemetry offset correction | not started | Turn observed flight-check errors into camera-tilt and time-offset corrections. |

## Debug stages owned by full-plugin features

Each feature below adds these artifacts through `core/debugio.py` and these columns to
`telemetry.csv` in the same commit (plan §10.1).

| TODO | Extra debug artifacts | New telemetry columns |
|---|---|---|
| TODO-101 | `texture/qa_obj.png`; reprojection-error histogram in `report.md` | `tex_views_k`, `tex_reproj_px`, `tex_ms` |
| TODO-102 | `objects/classification.png`; class distribution and confusion table | `class_label`, `class_score` |
| TODO-103 | `objects/obj_XXXX_proxy.glb`; proxy-vs-prism volume error row | `proxy_fit_err_m3` |
| TODO-104 | `ground/water_mask_frame_XXXX.png`; water level time series | `water_frac`, `water_level_m` |
| TODO-105 | `objects/rubble_score.png`; flagged-object table | `rubble_score` |
| TODO-106 | `compare/delta_heightmap.png`; per-object delta table | n/a (offline run) |
| TODO-107 | `refine/tile_k_before_after.png`; live-vs-refined delta stats | `refine_delta_p95_m` |
| TODO-109 | `detect/track_XXXX_path.png`; dwell-time table | `active_tracks` |
| TODO-110 | `heightmap/strip_overlap.png`; per-strip MAD table | `strip_id` |
| TODO-113 | Backend bench table appended to `report.md` | `backend` |

All other TODOs keep the plan §5 artifacts as-is; calibration and resume write small JSONs
(`calibration.json`, `resume_state.json`) into the run folder so runs stay reproducible.