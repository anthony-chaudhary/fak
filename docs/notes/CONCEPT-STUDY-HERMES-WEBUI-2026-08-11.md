# Concept study — nesquena/hermes-webui (2026-08-11)

**Source:** `https://github.com/nesquena/hermes-webui`
**Pinned:** `e9f1e2a4594ee000d65b6f31e0c66cabd7eb59a1` — every `path:line@sha` below resolves
against this commit, not against their tree.
**Licence verdict (repo-wide):** **MIT — DIRECT-PORT and ADAPT both permitted**, with notice
and attribution preserved. See [Licence gate](#licence-gate).

**Reading lens (operator-supplied):** study this for a *separately buildable UI/UX for fak* —
what a real end user needs from the terminal experience, not what fak-as-substrate needs. So
the axes of interest are the **separability of a presentation layer from the engine** and the
**end-user-facing surfaces**: transcript, activity disclosure, approvals, onboarding, and what
a client is told when a run dies.

hermes-webui is a browser chat UI over a local agent CLI. That makes it a useful mirror rather
than a competitor: they had to *extract* a client from an engine that never expected one, and
the seams they cut are exactly the seams fak would cut. Six of their eight interesting
techniques are about the boundary, not the pixels.

## What was read

Acquisition: shallow clone into the session scratchpad, `--filter=blob:none`, read-only, never
into the fak tree. SHA pinned before the first read.

**Fan-out:** six parallel readers, one per subsystem the README map exposed — the runtime-adapter
protocol (`api/runtime_adapter.py`, `docs/rfcs/hermes-run-adapter-contract.md`), the transcript
and interruption model (`api/models.py`), the replay journal (`api/run_journal.py`), agent
health (`api/agent_health.py`), the lint/CI/test discipline (`scripts/ruff_lint.py`,
`.github/workflows/tests.yml`, `TESTING.md`, `eslint.runtime-guard.config.mjs`), and the
frontend/no-build-step posture (`static/`, `DESIGN.md`).

**Completeness critic.** <!-- CRITIC -->

**Their worldview** (reconstructed from the RFC's stated motivation, their config defaults, and
their non-goals): one person, one machine, one agent CLI they did not write and cannot change.
The user wants to drive that CLI from a browser or a phone instead of a terminal. Two facts
follow from that and explain nearly every design choice worth borrowing:

1. **They do not own the engine.** The RFC opens by calling the adapter a *"protocol translator,
   not a runtime surrogate"* (`docs/rfcs/hermes-run-adapter-contract.md:15`) and spends its
   longest section on a four-bucket classifier for *who owns which piece of state* (`:233-262`).
   When you cannot change the engine, the contract is the only place discipline can live.
2. **Their client is a browser tab, so it disconnects constantly.** A refresh, a phone locking,
   a tunnel dropping. Their forcing function (`:36-49`) was concrete: approvals lived in live
   in-process callbacks, so a refresh mid-approval lost the prompt entirely. Everything about
   replay addressing falls out of that one bug.

fak's user is the opposite on both counts — fak *owns* its loop, and its client is a terminal in
the same process. That difference is real and it is why several of their techniques are
DIVERGENT for fak. But it cuts the other way too: the moment fak has a client that is *not* the
process that started the run — which the fleet UI plan already anticipates at G3 — fak inherits
their exact problem without having done their exact thinking.

## Licence gate

`LICENSE:1@e9f1e2a` — MIT, "Copyright (c) 2025 Hermes Web UI Contributors", confirmed by
`pyproject.toml`. No NOTICE file, no submodules, vendored JS confined to `static/vendor`.

DIRECT-PORT is therefore permitted. It is nonetheless **not used** for any borrow below: the
source is Python and every surviving candidate is a *contract discipline* (a wire field, a
taxonomy, an env allowlist) rather than an algorithm, so INSPIRE is the honest route. The
licence gate matters here for what it *rules in* — none of these were dropped for provenance.

## Candidate table

Witnessed at the **axis**, not the capability name. "PRESENT-on-axis" below always means fak's
own code was read on that specific seam, not that fak has something in the neighbourhood.

| Borrow | Source `path:line@sha` | The AXIS | Their-worldview reason | Witness on-axis | Route | Filed |
|---|---|---|---|---|---|---|
| Monotonic `seq` + `event_id` as SSE `id:`, reconnect by cursor, at-least-once with client dedupe | `docs/rfcs/hermes-run-adapter-contract.md:158-180` | Zero-loss resumption of one run's event stream across a reconnect | A browser tab refreshing mid-approval lost the prompt; they could not fix the engine, only the wire | **PARTIAL** — `agent.ProgressEvent` (`internal/agent/loop_observe.go:47-55`) has no `Seq`/`EventID`; the SSE frame (`internal/gateway/native_serve.go:156-171`) has no `id:`; observer sheds on backpressure with no gap signal. fak's own `session_changes.go:139-158` already does this correctly | INSPIRE | **#6486** |
| Interruption-cause taxonomy + evidence sentence, rendered where the turn died | `api/models.py:2148-2290`, grace window `:3573-3590` | A terminated run tells the surface which *class* of thing killed it | Their engine dies in ways they cannot instrument, so the UI must classify from outside and admit `unknown` | **ABSENT** — every loop failure becomes the fixed string `"upstream model error"` (`internal/gateway/native_serve.go:174-182`); real `err` goes only to `s.logf`. Local twin: `cmd/fak/chat.go:92` | INSPIRE | **#6487** |
| Replay reader returns a typed failure status rather than partial data | `api/run_journal.py` (8 statuses incl. `replay_noncontiguous`, `cursor_invalid`) | A reader that cannot serve the whole range says so instead of serving a prefix | Same rotated-log hazard behind a resumable reader | **PARTIAL** — capability present (`ReadAllSegments`, `VerifySegments`), axis absent: `rotate.go:196-215` has **zero** non-test callers vs 12 for tail-only `ReadRows` (`journal.go:1105`), with rotation armed at 64 MB (`cmd/fak/guard_support.go:704`) | INSPIRE | **#6488** |
| Git env allowlist + `GIT_TERMINAL_PROMPT=0`, with `GIT_INDEX_FILE` the one intentional exception | `docs/workspace-git.md:1-80` | A git subprocess cannot be re-aimed or wedged by inherited ambient config | Their agent runs against the user's own machine's git config, so ambient env was a live hazard | **PARTIAL** — `internal/gitbroker/runner.go:396-398` inherits `os.Environ()` whole, no scrub, no `GIT_TERMINAL_PROMPT=0`. fak already knows the technique at `internal/patchcommit/patchcommit.go:53` | INSPIRE | **#6489** |
| Diff-scoped lint + unconditional whole-tree compile to plug the hole scoping opens | `scripts/ruff_lint.py:88-106,151-176`; `.github/workflows/tests.yml:125-135` | Attributing a gate failure to the change under test rather than the tree | A shared long-lived branch made whole-tree lint unattributable | **PARTIAL, small** — fak has *both halves* already (`cmd/fak/validate.go:83,262-290` scoped gofmt; `:90-97` whole-tree build+vet). Residual gap is only that `make ci`'s `gofmt-check` (`Makefile:349-358`) reports whole-tree with no ownership split | INSPIRE | **#6490** |
| An onboarding checklist addressed to the *user's* AI assistant, with a stated blast radius | `docs/onboarding-agent-checklist.md:8-9,11-27,29-40` | The assistant a **user** brings to a broken install, as a distinct reader from a contributor | Their user installs a local app and asks their own assistant to fix it | **WORLDVIEW-FINDING** — fak's `CLAUDE.md`/`AGENTS.md` address contributors only, by their own headers; nothing addresses a user's assistant, and nothing states what it must not delete or print | INSPIRE | **#6491** |
| One turn = one compact `Used N tools` disclosure, not N content cards | `DESIGN.md:126` | Transcript legibility when a turn used many tools | A chat UI where tool spam drowns the answer | **PRESENT-on-axis** — `cmd/fak/chat.go:96` already prints one bracketed metrics line per turn (`[turn N: … engine calls, … denied, … served]`) | — | not filed |
| Typed loop-event vocabulary | `api/runtime_adapter.py:89` (8-method `RuntimeAdapter` Protocol) | A closed set of lifecycle transitions a client can switch on | Needed to translate an opaque CLI into UI state | **PRESENT-on-axis, fak richer** — `internal/agent/loop_observe.go` has a closed `ProgressEventKind` set carrying kernel `Verdict` and `Taint`, for which hermes has **no analogue** | — | not filed |
| Closed control-verb vocabulary normalised to 5 outcomes | `api/runtime_adapter.py:158-189` (`accepted/not-active/unsupported/conflict/expired`) | A control verb's outcome is drawn from a closed set, not free text | Their engine's failure modes were undocumented, so they closed the set at the boundary | **PRESENT-on-axis** — `internal/gateway/http.go:1245`, `POST /v1/fak/session/{trace}/{run\|budget\|pace\|priority\|wall\|throughput}` | — | not filed |
| Throwaway index so a git op cannot see a half-built tree | `docs/workspace-git.md` (`GIT_INDEX_FILE` exception) | Isolating a git operation's index from concurrent operations | Concurrent agent workspaces on one clone | **PRESENT-on-axis** — `internal/patchcommit/patchcommit.go:53` | — | not filed |
| Tri-state liveness (`True`/`False`/`None`) with a named reason | `api/agent_health.py:717-831` | Distinguishing "not alive" from "cannot tell" | A local process they did not spawn and cannot always see | **PRESENT-on-axis** — `internal/fleetpane/health.go:52,87,92` carries `Reason` with "no monitor JSON" / "monitor JSON was not parseable"; `docs/fleet-ui-generation-plan.md:97-98` already *requires* distinguishing no-config / unavailable / stale / healthy-zero | — | not filed |
| Mutation testing on gates (`LIFECYCLE_TEST_BITE=drop-anchor-persistence`) | `TESTING.md:118-137` | Proving a green gate is load-bearing rather than vacuous | A large single-file backend where a gate silently rotting was survivable | **DIVERGENT** — fak has negative-control tests at unit grain but no gate-level mutation surface. Recorded, not filed: the technique needs a named gate-mutation vocabulary fak does not have, so it is not ship-alone today | — | not filed |
| Zero-build-step product UI (stdlib Python + vanilla JS; ESLint as a runtime-error guard, not a build step, at `eslint.runtime-guard.config.mjs:29-32` — exactly 2 rules with a stated zero-false-positive admission policy) | `static/`, `eslint.runtime-guard.config.mjs` | No toolchain between source and running artifact | Solo maintainer; a build step they'd have to babysit is a tax with no payer | **DIVERGENT, same worldview** — fak's hand-rolled Go TUI holds the *same* property by a different route. Their target is a browser; `docs/fleet-ui-generation-plan.md:151` explicitly non-goals a web application. Their user wants a chat GUI over a local CLI from a phone; fak's is an operator supervising loops from a terminal | — | not filed |
| 27k-line `routes.py` with no module-splitting policy; navigability via issue-numbered tests + a contract index | `api/routes.py` | Keeping a large surface navigable | Solo maintainer optimising for grep-ability over architecture | **DIVERGENT — fak stronger.** fak's tiered packages + architest enforce what they navigate by convention; their docs had drifted ~3× against this file | — | not filed |

## The one finding worth restating

fak is **not** behind on structured loop progress — it is ahead. `ProgressEvent` carries the
kernel's `Verdict` and `Taint`, which no part of hermes has an analogue for, because hermes has
no kernel to report. The coarse read ("do we have structured progress? yes, #5148 shipped it")
would have closed this study with nothing filed.

The gap only appears one level down, at the **addressing**: fak's richer events are *unaddressed*
(no `seq`, no `id:`, shed-on-backpressure with no gap signal), so they cannot survive the one
thing an end-user surface does constantly — disconnect and come back. That is #6486, and it is
the single highest-value item here for the operator's stated lens.

The mirror-image finding is #6491: hermes writes documentation for a reader fak does not
acknowledge exists — the assistant a *user* brings to a broken install. fak's agent docs say, in
their own headers, that they are for contributors. That is not a gap in fak's code; it is a gap
in who fak thinks its readers are, which is exactly what the operator's "not just as the base
thing" lens was asking about.

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the witness discipline each row above ran through.
- Epic [#2388](https://github.com/anthony-chaudhary/fak/issues/2388) — the owned-turn track; **#6486** and **#6487** are filed as children.
- [`docs/fleet-ui-generation-plan.md`](../fleet-ui-generation-plan.md) — the existing G0→G3 plan for fak's terminal fleet surface; this study's lens overlaps its G3 ("additional clients") and its read-model discipline.
