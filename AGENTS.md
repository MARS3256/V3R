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

- No `import torch` / `from torch` anywhere; inference is ONNX Runtime only.
- Core modules never import `bpy`; only `blender/` code touches Blender.
- Never invent missing input data. If the telemetry sample or video is absent, stop and record
  the blocker; do not fabricate column names or values.
- Run `ruff format`, `ruff check`, `mypy`, and `pytest` before calling a task done.