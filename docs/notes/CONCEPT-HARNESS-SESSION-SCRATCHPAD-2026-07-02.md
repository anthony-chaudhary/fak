# The harness session scratchpad: known as a leak, not as a place (2026-07-02)

Every Claude Code session on this fleet is handed a private temp directory and told to
put all its temporary files there — intermediate results, one-off scripts, analysis
output, anything that doesn't belong in the repo:

    %LOCALAPPDATA%\Temp\claude\<project-slug>\<session-uuid>\scratchpad   (Windows)
    /tmp/claude/<...>/<session-uuid>/scratchpad                           (POSIX shape)

Writes there are permission-free and isolated from the working tree. That makes the
scratchpad the single busiest artifact surface the fleet has that fak does not model.
This note is the audit of what fak currently knows about it, the evidence for what it
misses, and the filed follow-ons. It also serves as the definition the tree previously
lacked — no doc in the repo said what the scratchpad *is*.

One disambiguation up front, because the word is overloaded three ways in this tree:
`compute.MemoryScratchpad` (internal/compute/capacity.go) is a GPU memory class, and the
README/FAQ "scratchpad" is a prose metaphor for the KV cache. Neither is this. This note
is about the harness's per-session temp directory only.

## What fak already understands — three edges, all defensive

The audit method was a repo-wide sweep for the concept and its path shapes. Every hit
that isn't GPU memory or the KV metaphor falls into one of three touchpoints, and all
three treat the scratchpad purely as a hazard that leaks *into* the tree:

1. **`fak sweep` knows its failure shape.** `isSweepJunk` (cmd/fak/sweep_plan.go:220)
   classifies a repo-root file whose name contains both `scratchpad` and `temp` as a
   misdirected harness-scratchpad write — the path separators got flattened into one long
   filename. Witnessed by cmd/fak/sweep_test.go ("flattened temp scratchpad").
2. **`.gitignore` carries a backstop glob.** `*scratchpad*` (with the comment naming the
   same flattened-write class), and the FILE_ADMISSION gate is the commit-time backstop
   for anything that slips past.
3. **The worktree janitor knows it as a disposable marker.** `tools/worktree_doctor.py`
   treats a `scratchpad` path segment as disposable (`DISPOSABLE_MARKERS`), and
   `tools/worker_worktree.py` deliberately spins worker worktrees there so the doctor can
   reap them. AGENTS.md's trunk-guard section points at this.

So: fak knows the scratchpad as a *source of junk*. Nothing in the tree knows it as a
*place where session work-product lives*. That asymmetry is the finding.

## What fak does not understand — two gaps, both witnessed today

**Gap 1: artifacts strand on session-id change.** The path is keyed by session-uuid, so
a resume, re-home, or throttle-cut successor gets a fresh empty scratchpad and no pointer
back. Witnessed this morning: session `af90ee4c` wrote seven host-cleanup triage notes
(t1–t7) plus an `elevated_cleanup.ps1` into its scratchpad; the successor session could
only find them because the operator hand-pasted the absolute path into the next prompt.
The content survived only because the predecessor promoted its findings to issues
[#2337](https://github.com/anthony-chaudhary/fak/issues/2337)–[#2343](https://github.com/anthony-chaudhary/fak/issues/2343)
before dying — a manual escape, not a mechanism. The resume plane is blind to all of it:
`internal/resume` has zero scratchpad awareness, the verified resume packet
([VERIFIED-RESUME-PACKET-CHECKPOINT-2026-06-26](VERIFIED-RESUME-PACKET-CHECKPOINT-2026-06-26.md))
doesn't carry the path, and the durable-session-state kill-safe ladder
([#2217](https://github.com/anthony-chaudhary/fak/issues/2217)) covers turn journals and
context residuals but not scratchpad artifacts. Filed as
[#2345](https://github.com/anthony-chaudhary/fak/issues/2345).

**Gap 2: dead-session dirs accrue with no reaper.** Measured on the agent-host today:
`Temp\claude\C--work-fak` holds 155 session dirs, ~49,000 files, 2.18 GB — every one
created since 2026-07-01 08:12, i.e. roughly 2 GB/day on one project slug, with 84 more
dirs under `C--work-job`. Each existing cleanup lane misses the class: worktree_doctor
reaps only git worktrees *inside* scratchpads, the runaway reaper is process-shaped, the
GCP janitor is GCP-only ([#2341](https://github.com/anthony-chaudhary/fak/issues/2341)),
and the proc guard observes without enacting
([#2337](https://github.com/anthony-chaudhary/fak/issues/2337)). Filed as
[#2344](https://github.com/anthony-chaudhary/fak/issues/2344) — with the fence that a
reaper must honor #2345's notion of "referenced by a resumable session" before deleting,
or it turns the stranding gap into a loss gap.

## Discipline until the planes learn it

Until #2344/#2345 land, the scratchpad contract for agents working this repo is:

- Write temp files there by absolute path. The flattened-junk class `fak sweep` catches
  exists precisely because relative or mangled writes land in the repo root instead.
- Treat the scratchpad as lossy. Anything worth keeping gets promoted before the session
  ends: a dated note in docs/notes/ for the record, a GitHub issue for follow-ons,
  ../fak-private for raw status text.
- When resuming another session's work, reconstruct the predecessor's scratchpad path
  from project-slug + session-uuid and read what it parked there — today that
  reconstruction is manual.

## Status

The definition and audit are this note (shipped). The resume-carry and reap-lane wiring
are `not yet`: the missing witnesses are a resume packet that names a predecessor
scratchpad path (#2345) and a janitor run that deletes an aged, unreferenced session dir
while sparing a live one (#2344). Next checkable step for each is on its issue.
