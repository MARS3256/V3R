# AGENTS.md — V3R agent rules

V3R is a Blender 5.2 add-on (GPL-3.0-or-later) that turns drone video plus per-frame flight
telemetry into metre-scaled 3D objects. Read `plans/architecture/V3R_System_Design_and_1_Week_Plan_v2.md`
for the design and `plans/architecture/V3R_Coding_Agent_Prompts.md` for the task cards.

## Progress log (required)

Maintain `plans/progress.md` for every task card or meaningful work chunk.

1. **Append one entry** to `plans/progress.md`, newest entry at the top.
2. Each entry records: date, task id, status (`in progress` / `blocked` / `done`), files
   changed, the exact verification commands and their result, blockers, and the next step.
3. **Update it before you finish** the task, not in a separate summary message. If a task is
   blocked, the entry says what is missing and who/what unblocks it.
4. Keep entries factual and short. Do not put secrets, tokens, or large file dumps in them.
5. If the task changes a decision or threshold, add the entry to `docs/DECISIONS.md` first and
   link it from the progress entry.

## Where things live

- `docs/DECISIONS.md` — append-only decision log; every deviation from a task card goes here.
- `docs/TODO.md` — the unfinished-feature register (`TODO-1xx`), never faked in code.
- `plans/progress.md` — the chronological work log described above.

## Non-negotiables

- Inference runs in the **worker process only**, inside its own Python environment. PyTorch
  (CUDA/ROCm) and ONNX Runtime are allowed there. Blender-side add-on code must not import
  `torch` or `onnxruntime`; it talks to the worker through JSON lines and files.
- The compute backend (`cuda` / `rocm` / `cpu`) is a startup setting. An unavailable backend
  is a verbose error, never a silent fallback.
- Core modules never import `bpy`; only `blender/` code touches Blender.
- Never invent missing input data. If the video or stream is absent, stop and record the
  blocker; do not fabricate frames, depth or poses.
- Run `ruff format`, `ruff check`, `mypy`, and `pytest` before calling a task done.
- After any change to `V3R/`, rebuild `V3R_addon.zip` so the installable add-on stays current.