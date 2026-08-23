---
name: skill-lifecycle
description: Witnessed lifecycle for the project skill pack — usage-telemetry sidecar, value/idle-driven auto-archive (never delete, restorable), pin-exemption, journaled reversible transitions. Use to record skill usage, review archive verdicts, archive or restore a skill, or pin one exempt. Max action is archive; nothing is ever deleted. Use when this named workflow matches the task.
---

# /skill-lifecycle — witnessed skill lifecycle (never delete)

Keeps `.claude/skills/` from rotting without ever destroying work: per-skill
usage telemetry, auto-archive of stale *low-witnessed-value* skills, and a
journaled, reversible archive/restore path. Hermes-curator parity (#2941), with
fak's twist: transitions are driven by witnessed value, not idle time alone.

All state lives beside the skills, inside the `claude` lane:

| Path | Role |
|---|---|
| `.claude/skills/.usage.json` | telemetry sidecar (use/view/patch counts, last activity, state, pin, origin) |
| `.claude/skills/.lifecycle-journal.jsonl` | append-only archive/restore transition log |
| `.claude/skills/.archive/<name>/` | where an archived skill lives — moved, never deleted |
| `.claude/skills/.archive/.snapshots/` | pre-archive tar.gz backups |

## Hard rules

- **Never delete.** Archive is a move; a collision is a structured refusal.
- **Pinned is exempt** — sidecar `pinned: true` or `pin: true` in SKILL.md
  front-matter blocks both auto- and manual archive.
- **Bundled/core skills are never auto-archived.** Origin comes from the
  sidecar, else git (tracked SKILL.md = bundled); unwitnessable = bundled.
- **Value guards idle.** Auto-archive needs idle > `--idle-days` **and**
  witnessed value (`use_count + patch_count`) < `--min-value`. An
  emergency-only skill with proven uses stays live.

## Commands

```
python .claude/skills/skill-lifecycle/skill_lifecycle.py record <skill> [--kind use|view|patch]
python .claude/skills/skill-lifecycle/skill_lifecycle.py status
python .claude/skills/skill-lifecycle/skill_lifecycle.py sweep [--idle-days 30] [--min-value 1] [--apply]
python .claude/skills/skill-lifecycle/skill_lifecycle.py archive <skill> --reason "..."
python .claude/skills/skill-lifecycle/skill_lifecycle.py restore <skill>
python .claude/skills/skill-lifecycle/skill_lifecycle.py pin <skill> | unpin <skill>
```

`sweep` is dry-run by default and prints one verdict per live skill with the
reason; `--apply` executes only the `archive` verdicts (snapshot → move →
sidecar state → journal entry, in that order).

## When invoked as a skill

1. Run `status` and `sweep` (dry-run) and show the operator the verdicts.
2. Only `sweep --apply` or `archive` after the operator confirms; `restore`
   is always safe to run on request.
3. After any transition, show the tail of `.lifecycle-journal.jsonl` so the
   operator sees the journaled, reversible record.

## Witness

`python .claude/skills/skill-lifecycle/skill_lifecycle_test.py` — 12 tests,
including the acceptance witness: an idle unpinned agent skill is archived
(bytes preserved, snapshot taken, journaled) while a pinned one is untouched,
and restore reverses the move exactly.
