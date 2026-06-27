---
title: "DOS effective-usage audit (2026-06-22): is the fleet's own trust substrate actually doing work?"
description: "Audit of whether DOS (the dogfooded fleet governance substrate) is used effectively in this repo. Finds the arbiter's collision-safety silently defeated by a stale fak/ path prefix in dos.toml, reconciles the lane taxonomy to the real tree, and records the remaining gaps."
---

# DOS effective-usage audit — 2026-06-22

**Question:** the repo dogfoods DOS (`dos.toml`, the plugin's PreToolUse/PostToolUse/Stop
hooks, the `dos verify` ship referee, the lane arbiter). Is it *effectively used*, or is it
ceremony? **Verdict:** DOS is genuinely live and catches real things — but its dogfood
policy had drifted from the repo layout, and the drift **silently defeated the arbiter's
single most-exercised guarantee.** Fixed in this change; the remaining gaps are recorded
below as operator decisions / upstream debt.

Findings were each adversarially re-derived by an independent verifier before shipping.

## DOS is live (the part that works)

Measured from the per-call hook observation log + the lane journal (window ending
2026-06-22):

| signal | value |
|---|---|
| tool calls adjudicated by the hooks | ~18.5k (98.9% passed untouched) |
| Stop-hook "you're not actually done" blocks caught | 12 sessions (false-done refusals) |
| CI hard-gates on `dos review` | yes — `.github/workflows/ci.yml` exit-code gate |
| `dos verify` ship-referee grammar | recognizes `(fak <leaf>)` trailer + `vX.Y.Z:` release |

So the kernel runs on every turn, the Stop-gate has genuinely caught workers claiming a
task was done when it wasn't, and CI blocks an un-witnessed ship. That is real value.

## The critical bug: the arbiter was running on phantom trees

`dos arbitrate` is the "may two workers edit the repo at once?" gate — it admits a lane iff
its file tree is **disjoint** from every live lease. It was the **most-exercised** DOS
mechanism in the window. The lane trees in `dos.toml` were:

```toml
gateway = ["fak/internal/gateway/**"]   # ← every internal leaf, plus cmd / experiments / abi
```

But **the Go module is the repository root** (AGENTS.md) — the real tree is
`internal/gateway/**`, with **no `fak/` segment**. `fak/internal/` is not a directory; it is
only a doc-link convention. So **all 39 leaf/dir globs matched zero files**, and the
disjointness rung was reasoning about trees that do not exist.

This is not cosmetic. Reproduced live, before the fix:

```
# A peer holds the gateway lane (its canonical tree resolves to fak/internal/gateway/**).
# Worker B edits the REAL path internal/gateway/server.go on another lane:
dos arbitrate --lane gateway --tree internal/gateway/server.go \
  --leases '[{"lane":"x","mode":"shared","tree":["fak/internal/gateway/**"]}]'
→ ACQUIRE        # collision MISSED — two workers would edit the same files

# Same call, lease tree corrected to the real layout:
  --leases '[{"lane":"x","mode":"shared","tree":["internal/gateway/**"]}]'
→ REFUSE (lock-mode conflict)   # collision correctly caught
```

A second, worse shape (an undeclared leaf): `dos arbitrate --lane swebench` did **not**
fail — it silently **auto-picked an unrelated lane (`adjudicator`) and returned GO over
*that* tree**, so a worker editing `internal/swebench/**` held a lease on the wrong tree and
the swebench files stayed unleased. A false-GO over the wrong tree, not a clean error.

The root cause is a historical `fak/` → repo-root flattening that several tooling files
never tracked (the same class as the known-stale `tools/new_leaf.py` paths).

## What this change ships

1. **`dos.toml` — strip the stale `fak/` prefix** from every `[lanes.trees]` glob
   (`fak/internal/X/** → internal/X/**`, `fak/cmd → cmd`, `fak/experiments → experiments`,
   `fak/internal/abi → internal/abi`). The arbiter now detects real same-tree collisions.
2. **`dos.toml` — add the 19 real `internal/` leaves that had no lane at all**
   (appversion, boundarylint, deletioncert, demoui, enginecache, gpulease, leakcheck,
   metalgemm, modelengine, pathlint, pathutil, radixkv, ratelimit, rsiloop, swebench,
   toollint, tracesink, urllint, webbench). Each is a real Go package; several
   (e.g. swebench) are actively shipped. Now each leases its real tree.
3. **`tools/new_leaf.py` — emit `internal/<name>/**`** for a new leaf's lane (was
   `fak/internal/<name>/**`), so the golden path stops re-introducing dead globs.
4. **`tools/issue_lane_router.py` — normalize the `fak/` doc-link prefix** in
   `path_matches_lane` (lockstep: issue text names files as `fak/internal/...`; without the
   strip, path-confirmed routing would have gone dark against the corrected trees). Tests
   updated to the real layout + a both-conventions regression test.

**Measured effect:** `dos doctor` verifiability rose **27 → 33 of 50** commits; bindable
ship-stamp coverage (`commit_stamp_doctor.py`) **56% → 66%** (of non-bookkeeping commits); `lanes.trees` 47 → 66; and
`dos arbitrate` now refuses the real-path collision it previously admitted. No glob starts
with `fak/` anymore. Build/vet green; the four affected Python test suites pass.

One leaf is intentionally **not** added here: it is private (excluded from the public copy),
and `dos.toml` ships publicly, so adding it would surface a private token. Tracked separately.

## Remaining gaps (operator decisions / upstream debt)

- **`dos verify` is lane-agnostic by design.** It greps commit *subjects* for the ship
  grammar; it does **not** consult `lanes.trees`. So `dos verify fak <leaf>` returns SHIPPED
  for any matching trailer regardless of whether the leaf is a lane. The lane taxonomy
  matters for **arbitrate** (collision protection) and for `dos doctor`'s recognized-unit
  count, not for verify. (`commit_stamp_doctor`'s "off-lane" warning is a typo-catch
  heuristic, not a verify failure.)
- **Off-lane demo stamps.** A handful of shipped *demo/feature* commits stamp
  `(fak turntax)` / `(fak ctxdemo)` / `(fak guarddemo)` / `(fak turntaxdemo)` — names that
  live under `cmd/`, not `internal/`, so they are not lanes. **Decision for the operator:**
  either stamp demos with the owning lane (`(fak cmd)`), or declare per-demo `cmd/<demo>/**`
  lanes. (Not auto-gated here — it would fight an established fleet habit.)
- **The plan portfolio is empty.** `paths.plans_glob = "PLAN-*.md"` matches no file at the
  repo root, so every plan-facing DOS surface (`dos next-up`, `dos replan`, the
  class-cycle/lifecycle skills, which also have no `[lifecycle]` table) is inert.
- **The supervisor/dispatch watchdog is not running.** `supervise.target = 4` presumes a
  standing population loop, but the scheduled watchdog tasks are disabled — the loop the
  target presumes is not alive.
- **The MCP `dos_*` tools are barely used in practice** — invoked mostly in setup/audit
  bursts, not as part of ongoing work — and the admission rung mis-classifies those calls
  (and pure-meta tools like `TaskUpdate`, `ToolSearch`, `Read`) as empty-tree work-lanes,
  producing low-signal advisory cautions (DOS itself labels them "low signal"). The
  matcher-scoping / footprint-derivation fix is **upstream** in the dos-kernel plugin (see
  `docs/perf-dos-hook-cost.md`); this repo's lever is measurement (`docs/dos-kernel-transfer-playbook.md`).
- **`commit_stamp_doctor.py` is orphaned automation** — it has a test but is wired into no
  CI/cron path. Candidate: run it as an advisory `--min-coverage` check alongside the
  existing `dos review` CI gate so the trailer-coverage number is watched, not guessed.
- **The custom `[reasons.*]` vocabulary never fires through DOS itself** (0 occurrences in
  the observation log) — by design: each reason's enforcement is delegated to a named floor
  (a git hook, the architest gate), so the vocabulary is citable documentation, not an
  active DOS adjudication. Worth knowing when reasoning about "what DOS actually blocks."

## How to re-run this audit

```bash
python tools/commit_stamp_doctor.py -n 50      # bindable ship-stamp coverage + off-lane stamps
dos doctor --workspace .                        # verifiability, lane roster, .dos surface
dos helped --workspace .                        # what DOS actually refused/advised, by reason+tool
# arbiter collision check (should REFUSE a real-path collision against a real-tree lease):
dos arbitrate --lane docs --tree internal/gateway/x.go \
  --leases '[{"lane":"gateway","mode":"shared","tree":["internal/gateway/**"]}]'
```
