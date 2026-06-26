# CHARTER — the ten principles fak is built to satisfy

This is fak's constitution: the small set of commitments every surface is meant to
advance. It is the *why* above the *what* — read [`README.md`](README.md) for what
fak is, [`AGENTS.md`](AGENTS.md) for how to work in the repo, and
[`CLAIMS.md`](CLAIMS.md) for what is actually shipped. This page names what all of
that is *for*.

The charter is a north star, not a new gate. It does not block a commit. Instead it
binds to the machinery the repo already runs: each principle points at a **surface
that embodies it** and a **deterministic scorecard that keeps it honest**, and where
no measuring stick exists yet, the charter says so in plain `not yet` language rather
than claiming alignment it can't witness. The repo already folds **18 scorecards**
into one debt number through [`tools/scorecard_control_pane.py`](tools/scorecard_control_pane.py);
most of this charter is already measured there. The job of this document is to make
the goal explicit, map it to that machinery, and grade the gap honestly.

## The charter

1. **Agentic built by default.**
2. **Industry leading value.**
3. **Low ego, flexible, Buddhist, works with anyone.**
4. **DOS verified.**
5. **Self improving, growth mindset.**
6. **Up to date** — zero-day or pre-release support for popular concepts, papers, and models.
7. **Great by default** — all optimizations and concepts "just work" out of the box.
8. **Agentic first. Built by agents, for agents.**
9. **Win-win-win** — e.g. a security win that also improves performance.
10. **Coherent and remaining human-steerable, good for humanity.**

## How the charter stays honest

A charter that only inspires is decoration. This one is wired to evidence:

- **Each principle has a surface and a stick.** The surface is the code or doc that
  makes the principle real; the stick is the scorecard whose debt number falls when
  the principle is better served. The score family lives in `tools/*_scorecard.py`,
  is folded by the control pane, and is re-derived from the git tree on every run —
  so the number can't be gamed by editing prose.
- **The grades below were grounded, not asserted.** Each row's alignment was mapped
  against the actual tree by an agent and then adversarially verified by a second
  agent told to *refuse* an unwitnessed claim and default to lowering the grade. That
  pass corrected two first drafts — it knocked "agentic-first" from A to B (the
  agent-readiness stick measures whether an agent *can* build on fak, not whether
  agents *are* the primary builders) and split "industry value" into an honesty grade
  and a realized-value grade. Four rows could not finish the second agent pass before
  a session limit; they are marked `verify pending` and graded conservatively.
- **`not yet`, not failure.** A principle with no dedicated measuring stick is capped
  below A and named as a gap to build, not scored as if the absence were success.
  This is the same incomplete-state discipline [`AGENTS.md`](AGENTS.md) and
  [`CLAIMS.md`](CLAIMS.md) already enforce.

## Alignment scorecard (2026-06-26)

Grades are current alignment, A (structurally embodied **and** measured **and**
witnessed) through F. The control-pane key is the debt metric that already tracks the
principle, or `— none yet` where the stick is missing.

| # | Principle | Embodied by | Control-pane stick | Grade | Worst-first next action |
|---|---|---|---|---|---|
| 1 | Agentic by default | `AGENTS.md`, `.mcp.json`, `docs/integrations/`, `examples/mcp/` | `agent` (friction_debt **0**) | **A** | Add a real-world *adoption* witness; the affordances are perfect, usage in the wild is unmeasured. |
| 2 | Industry leading value | `docs/industry-scorecard/`, `BENCHMARK-AUTHORITY.md` | `parity` (parity_debt **0**) | **B+** | Convert the 67 honest-gap rows to measured or out-of-scope; refresh the ~47 stale SOTA bars. |
| 3 | Low ego, works with anyone | `docs/integrations/interoperability.md`, compatibility matrix (44 harnesses) | — none yet | **B** | Build an `interop` stick: extract + run each integration recipe's fenced command in CI. |
| 4 | DOS verified | `dos.toml` `[reasons.*]`, the `(fak <leaf>)` trailer, `dos_verify` / `dos_commit_audit` | `ship_integrity` KPI (inside `code`) | **B** | Promote `dos-verified` to a portfolio-level stick (stamp-adoption rate, closure-audit pass rate). |
| 5 | Self improving | `tools/scorecard_control_pane.py` (folds 18), `/score-2x`, `internal/rsiloop`, `guard-rsi` | `guard_rsi` (**1**) + the whole fold | **A−** | Close the remaining guard-RSI debt; promote the plan-mode RSI loops to journal-closing loops. |
| 6 | Up to date | `tools/idea_scout.py` (daily arXiv+GitHub), `docs/notes/RESEARCH-*-triage-*` | — none yet *(verify pending)* | **B** | Build a `currency` stick: idea-scout cadence, triage latency, model-adoption lag. |
| 7 | Great by default | `cmd/fak/serve.go` defaults, `tools/token_defaults_scorecard.py` | token-defaults (debt **0**, 4/6 savers on) *(verify pending)* | **A−** | Witness the `elide-result` bounded-loss saver on real traffic, then default it on (4/6 → 5/6). |
| 8 | Agentic first | `AGENTS.md`, `tools/new_leaf.py`, `docs/dispatch-loop.md`, this charter (agent-authored) | `agent` measures *usability*, not *primacy* | **B** | Add an agent-*primacy* witness (agent-authored commit / issue / PR share); usability is already A. |
| 9 | Win-win-win | the unified `kernel.Syscall()` seam (one decision, many budgets), `compounding-benefits-of-a-saved-call` | `conflation` (live drift **+2**) *(verify pending)* | **B** | Re-label the 2 unlabeled `OBSERVED` metric help strings; re-pin conflation to 0. |
| 10 | Human-steerable | `tools/steerability_scorecard.py`, `tools/stability_scorecard.py`, `POLICY.md`, `docs/ROLLBACK.md` | `steer` (live drift **+2**), `stability` (**0**) *(verify pending)* | **B** | Split the 2 dispatch god-files along their verb seams (`/modularize`); steer 2 → 0. |

### Row notes (the honesty behind the grades)

- **#1 Agentic by default — A.** Verified: `agent_readiness_scorecard.py` scores
  100/100, grade A, `friction_debt = 0` across 23 KPIs, with a live witness
  (`experiments/agent-live/claude-code-fak-guard-live-pilot-2026-06-25.json`) of
  Claude Code running through `fak guard` with a dangerous call denied and useful work
  continued in the same session. The only gap is *adoption in the wild*, which is not
  the same thing as *readiness for adoption*.
- **#2 Industry value — B+ (honesty A, realized value B+).** The industry scorecard
  itself grades A (`parity_debt = 0`, 100% of 89 dimensions positioned) — fak is
  *honest and complete* about where it stands. But only 5 of ~90 rows are measured
  *leads* (e.g. 4.1× fleet serving vs a tuned warm-cache stack) and 67 are explicit
  no-claim gaps. The principle's literal claim — value *at or above* SOTA — is proven
  narrowly, so the realized grade is B+ even though the measuring discipline is A.
- **#3 / #4 / #6 — B, capped by a missing stick.** These three principles are
  structurally strong and witnessed in working artifacts, but none has a dedicated
  deterministic scorecard folded into the control pane, so each is capped below A by
  rule. Building those three sticks (interop, dos-verified, currency) is the single
  highest-leverage charter move — it turns three B's into *measurable* B's that can
  then be driven to A. New sticks ship as a `fak` subcommand in Go (the control pane
  already runs `guard-rsi` and `dogfood` that way), never a new `tools/*.py` — the
  `pythongate` ratchet reds the trunk on a new Python tool.
- **#5 Self improving — A−.** This is the engine that runs every other row: the
  control pane folds 18 scorecards into one ratcheted number
  (`tools/scorecard_baseline.json`, pinned `total_debt = 366` @ `ba46040`), `/score-2x`
  drives debt down while a surface is dirty and *hardens the metric* when it
  saturates, and `internal/rsiloop` keeps-or-reverts on a witnessed signal. The −
  is `guard_rsi_debt = 1` plus the honest note that some guard-RSI loops are still
  plan-mode scaffolds, not journal-closing loops. (Grounded directly; the agent pass
  for this row hit the session limit.)
- **#8 Agentic first — B (corrected from A).** The adversarial pass earned its keep
  here: agent-readiness being A proves the repo is *built for* agents, but "built *by*
  agents" as a measured fact is unwitnessed — there is no agent-contribution or
  feature-ownership metric, and git authorship is human (with `Co-Authored-By`). The
  practice is real (this charter was drafted by an agent workflow); the *measurement
  of primacy* is the gap.

### Portfolio anchor

The committed ratchet (`tools/scorecard_baseline.json` @ `ba46040`) pins
`total_debt = 366`. The charter-aligned sticks are mostly already at the floor:
`parity 0 · agent 0 · stability 0 · steer 0 · conflation 0 · demo 0 · robustness 0 ·
readme 0`. The heaviest debt sits in `slop 261` (the long tail of "great by default"),
then `seo 49 · appeal 19 · hygiene 15 · product 10 · code 8 · persona 2 · doc 1 ·
guard_rsi 1`. At HEAD the control pane's early-warning lens flags two charter-relevant
regressions vs the pinned floor — **steerability +2** (a dispatch god-file) and
**conflation +2** (two unlabeled metric help strings) — which are the two cheapest
worst-first wins on the board.

## Actioning the charter

Worst-first, by leverage rather than by principle number:

1. **Give three principles a measuring stick** (low-ego, dos-verified, up-to-date).
   Each is a B *only* because nothing measures it. Ship `fak interop-score`,
   `fak dos-verified-score`, and `fak currency-score` as Go subcommands, fold each
   into `tools/scorecard_control_pane.py` via the existing `cmd:` entry pattern, and
   re-pin. This is the move that makes the charter self-policing.
2. **Retire the two live early-warnings** — the cheapest real wins on the board.
   Re-label the two `OBSERVED` metric help strings (`conflation` +2 → 0) and split the
   two dispatch god-files along their verb seams (`steer` +2 → 0, via `/modularize`),
   then `--pin` the control pane back down.
3. **Close the realized-value gaps.** Convert industry's 67 no-claim rows into measured
   head-to-heads or explicit out-of-scope declarations, and add the agent-primacy
   witness that would let "agentic first" earn its A honestly.
4. **Drain the tail.** `slop 261` is the single largest debt and the biggest drag on
   "great by default"; run `/slop-score` worst-first.

Every item above is already a named scorecard or skill. The charter does not invent
new process — it *orders* the process that exists around a single set of goals.

## Governance

- **Amending the charter.** The ten principles change only by an explicit edit to this
  file with a DCO sign-off — never silently. Treat a change here the way you would a
  change to the frozen ABI: deliberate, reviewed, and rare.
- **Keeping it honest.** The scorecards keep the charter honest; `/score-2x` keeps the
  scorecards honest (debt down while a surface is dirty, the bar up when it saturates,
  so a frozen A is never mistaken for a finished job).
- **Companions.** [`AGENTS.md`](AGENTS.md) is the agent's working contract,
  [`CLAIMS.md`](CLAIMS.md) is the per-capability claim ledger, [`STATUS.md`](STATUS.md)
  is the critical path, and [`INDEX.md`](INDEX.md) is the full map. This charter is the
  layer above all of them: the goals they exist to serve.
