

---

<!-- Verification-evidence overlay appended 2026-07-02 by the overnight lab-machine
worker. Recorded here because the fleet worker's capability floor refuses outward
`gh issue comment` (preview-confirm gate; the confirm-token echo is stripped by the
harness) — an operator or labeled-token pass may copy the evidence block below onto
the issue. Scrubbed to generic GPU-server language; no private hostnames, channel
ids, tokens, or raw private run logs. -->

## Verification evidence (2026-07-02, overnight unattended pass)

Ask 1 of this issue (fail fast on wedged readback) is **shipped and witnessed** on
the private side; asks 2 and 3 remain open, non-blocking follow-ups.

**Commit:** private companion repo, `main`, SHA
`f65858a73854cc733fc9251d4e5a83eaacbd08bc`
(`fix(dgxbridge): fail fast on wedged readback (#2372)`). Confirmed present on the
private trunk with only memory-sync commits after it. It adds
`tools/dgxbridge/internal/framed.go` (`ReadbackPreflight`: one bounded sentinel-echo
round-trip, default ~20s budget, typed `READBACK_WEDGED` error), wires
`Preflight: true` for every CLI verb except `selftest`, and adds pure tests for the
OK and typed-wedged preflight paths.

**Witnesses re-run 2026-07-02 from current private `main` HEAD:**

- `GO111MODULE=off go test ./internal` from the private dgxbridge tree — `ok … 0.279s`, exit 0.
- Temp-module staging (throwaway dir outside any repo): `go test ./internal/dgxbridge` — `ok … 1.058s`, exit 0.
- Temp-module `go build ./cmd/dgxbridge` — exit 0.
- Live probe (cheap, non-disruptive, no lab jobs touched): bridge `doctor` with
  `-probe -probe-wait 90s -settle 12s -timeout 5m` returned **READY** — token and
  control channel resolved, and *a running control session answers* — exit 0.

**Public-repo cleanliness:** public `cmd/dgxbridge/` still holds only the scrub stub
`main.go` (unchanged since `7ede91ce`); no `internal/dgxbridge/` exists in the
public tree; `git status` shows no dgxbridge paths dirty. The buildable staging copy
lived only in a temp directory and was never committed.

**Residual (keeps this issue open):**

- Ask 2 — sentinel-robust base64/spill-file framing: `execHubFramed` still reads
  the scoped tail window; its `framed` flag is a deliberate no-op in `f65858a`.
- Ask 3 — readback failover across hub sessions on `READBACK_WEDGED`: not
  implemented.

The "done when" contract is therefore half-met: a wedged session now yields a typed
`READBACK_WEDGED` within the ~20s preflight budget instead of burning `-timeout`
per attempt (witnessed by the pure wedge test), but the chatty-long-output base64
round-trip rung is not yet built. The 14x/37-min burn shape is prevented at the
fail-fast rung; structural impossibility of sentinel loss awaits ask 2.

<!-- dispatch-contract sections appended 2026-07-02; derived from the issue prose
above + the verified repo state. No intent change. Scoped to the RESIDUAL work
(asks 2 and 3) since ask 1 is shipped and witnessed per the evidence above. -->

## Generation stream

gen/now. Current-product operator-loop defect with a live forensic witness (the
14x sentinel-loss burn across two audited sessions, 2026-07-01..02) and acceptance
criteria checkable today; every lab GPU-server workflow depends on bridge readback.
Labels already on the issue: `bug`, `dev-ex`.

## Parent context

Provenance chain on the issue: #2365 behavioral-lens dogfood, forensic run
wf_8f4794e7. Predecessor surface: the bridge readback hardening line in the private
companion repo (`tools/dgxbridge`); the public `cmd/dgxbridge/main.go` has been a
stub since `7ede91ce`.

## Current state

Ask 1 shipped: private commit `f65858a73854cc733fc9251d4e5a83eaacbd08bc` adds
`ReadbackPreflight` (typed `READBACK_WEDGED`, ~20s default budget) wired as
`Preflight: true` for every CLI verb except `selftest`, with pure OK/wedged tests —
all witnessed green from current private `main` (see evidence above). Asks 2 and 3
are NOT built: `execHubFramed` still reads the scoped `!tail` window (its `framed`
parameter is a no-op) and no readback failover across hub sessions exists.

## Why now

(gen/now) Until ask 2 lands, a chatty/truncated tail window can still lose the
completion sentinel mid-command — the preflight only guarantees the session was
readable at command start. Each recurrence burns operator time on the top
lab-machine dependency (all GPU-server work reads results back through this path).

## Working spine

In the private `tools/dgxbridge/internal` package: (ask 2) make `execHubFramed`
honor its `framed` flag — wrap the remote command so the result returns as a single
base64 line (spill-file + fresh probe fetch), immune to tail-window truncation;
(ask 3) on `READBACK_WEDGED`, retry the READBACK (never the command) through a
different live hub session, since scratch state is shared and session rotation is
the empirically working recovery.

## In scope

Private-repo edits under `tools/dgxbridge/` only: `internal/framed.go` (base64
frame + failover), `internal/sessions.go` (pick an alternate live session), pure
tests in `internal/dgxbridge_test.go` for the framed round-trip and the failover
path, and the agent guide's failure-mode table.

## Out of scope

Any private bridge source in the public repo (the public stub stays a stub); the
legacy Python driver (`tools/dgxsh.py` — frozen); hub-side/remote shell changes;
retrying the remote COMMAND itself on wedge (only the readback fails over);
lab-job-disturbing live tests.

## Done condition

The issue's own "done when": a deliberately wedged hub session yields a typed
`READBACK_WEDGED` in <=20s (shipped, witnessed), AND a chatty long-output command
round-trips its result through the base64 frame, AND a wedged readback recovers
through an alternate live session — so the 14x/37-min shape is structurally
impossible.

## Witness

Pure tests in the private `internal` package that fail before and pass after: a
fake-transport test where the tail window truncates the sentinel but the base64
frame still round-trips; a failover test where session A never replies and session
B serves the readback. Optional live rung (cheap, non-disruptive): one bridge
`selftest` after deploy, recorded in scrubbed language.

## Acceptance gate

`GO111MODULE=off go test ./internal` green from the private dgxbridge tree AND
temp-module `go test ./internal/dgxbridge` + `go build ./cmd/dgxbridge` green from
a throwaway staging dir (the same three witnesses re-run for ask 1 above).

## Closure binding

Resolving private commit cites #2372 in the subject (the fail-fast half is already
bound by `f65858a`'s subject). The issue closes only when asks 2 and 3 land with
their witnesses.

## Lane

dgxbridge — the edit surface is the private companion repo's `tools/dgxbridge/**`;
no public lane lease is needed beyond this overlay (the public tree carries only
the stub and scrubbed docs).

## Work unit

leaf

## Expected steps

6 — implement the base64 frame in `execHubFramed`, add the truncated-tail framed
test, implement readback failover, add the failover test, run the three acceptance
witnesses, commit citing #2372.

## Assumptions

- Scratch state on the box is shared across hub sessions, so a readback (not the
  command) can be served by any live session — the empirical recovery seen in the
  49db5c19 forensics.
- The remote shell can base64 a spill file in one line small enough for the
  transcript path (chunking already exists for file pull if not).

## Confusion risks

- Do not re-implement ask 1 — the preflight is shipped and witnessed; a new commit
  duplicating `f65858a` is the failure mode the dispatch explicitly forbids.
- The `framed` no-op flag in `execHubFramed` looks like dead code but is the
  deliberate seam for ask 2 — extend it, don't strip it.
- Failover applies to the READBACK only; re-sending the command through a new
  session would double-execute remote side effects.

## Coordination

- Private working tree currently has unrelated dirty files (`tools/dgxsh.py`,
  `tools/dgx_witness_fetch.py`, nightrun logs) — commit by explicit path only.
- The bridge is shared with live overnight lab jobs; live testing stays at the
  cheap selftest rung and must not restart sessions that host running jobs.

## Trigger

2026-07-01..02 behavioral audit (14x sentinel loss, sessions 49db5c19/9fa4b6b2)
filed as #2372; this overlay recorded by the 2026-07-02 overnight verification
pass after witnessing the fail-fast half shipped.

## Batch policy

One issue for the whole readback-hardening contract; asks 2 and 3 stay on #2372
rather than splitting, since they share the done-when. Re-grooms update this
overlay in place instead of filing duplicates.

## Likely files

- private `tools/dgxbridge/internal/framed.go`
- private `tools/dgxbridge/internal/sessions.go`
- private `tools/dgxbridge/internal/dgxbridge_test.go`
- private `tools/dgxbridge/BRIDGE-AGENT-GUIDE.md`
