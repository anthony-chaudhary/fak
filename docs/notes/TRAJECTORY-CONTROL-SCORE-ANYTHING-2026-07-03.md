# Trajectory control — score anything, steer by curve (2026-07-03)

The design note for the **trajectory control** concept family: a first-class,
harness-native way to keep long-horizon sessions, sub-agent fleets, and
error-detour recoveries pointed at their declared objective. One primitive
carries the whole family: **anything you want to progress gets a named score**,
and every conversation about progress becomes one of two moves — *improve the
score* or *improve the scoring method*.

This is the epic spine note. The epic is filed as **#2533** with 42 children
(**#2534–#2575**: spine #2534–#2542, then scorers, hierarchy/sub-agents,
steering, integrations, meta, and production hygiene — the epic carries the
grouped checklist). All children were validated contract-dispatchable through
`fak issue cohort --from-issues` before filing and carry `fak-trajctl-key`
dedupe markers. This note is the durable statement of the concept, the
positioning, and the seams.

## The problem (four concrete failure shapes)

1. **Long-horizon sessions** drift: the goal stated in turn 1 is diluted by turn
   200; context fills with operational detail and the session optimizes for the
   most recent error message instead of the objective.
2. **Sub-agent fleets get distracted**: a spawned worker adopts a nearby, more
   tractable objective and reports success on the wrong thing; the parent
   accepts self-reported completion.
3. **Error side-quests**: a failing tool call opens an unbounded repair detour;
   three hours later the session is an expert on a linker flag and the feature
   is untouched.
4. **Unrelated infra fixes**: the session (correctly!) fixes a broken harness or
   flaky CI first — but nothing enforces the *return* to the main objective, and
   nothing accounts for the paused time.

fak already detects fragments of this per-subsystem (`loopmgr` witness-collapse
admission, `resume` watchdog retry gates, `dispatch_status.silent_workers`,
`guardrsi` refusal recovery) — but there is **no general forward-progress score
against a declared objective**, and no controller that reads one. `loopdrive`'s
GOAL.md witness is the closest thing and it is deliberately binary:
witnessed-done or not. Trajectory control generalizes that bit into a curve.

## Doctrine

1. **Score anything.** Anything you want to progress gets a named objective and
   a score. A goal without a score is a wish; a score without a curve is a
   snapshot. Steering reads curves, never points.
2. **Improve the score or improve the scorer.** Scoring methods are themselves
   scored — a scorer's calibration against witnessed outcomes is an objective
   like any other. This is the same shape as the scorecard family's
   intent-vs-literal honesty stick, applied to the scorers.
3. **Scores are witnessed, not self-reported.** Every score row carries a
   witness rung, same doctrine as `dos verify`:
   - **W3** deterministic evidence (witnessed commit, green suite, benchmark harness)
   - **W2** transcript-derived heuristic (sessionaudit signals, tool-stream shape)
   - **W1** judge verdict (structured, pinned-schema LLM judgment)
   - **W0** self-report (never gates anything alone)
4. **Steer by regime.** Intervening in a high-scoring session is harm, not help
   — [arXiv:2602.03338](https://arxiv.org/abs/2602.03338) shows mid-trajectory
   intervention consistently *degrades* performance in high-success regimes and
   helps only in low-success ones. The controller's default action is nothing;
   every rung above "annotate" sits behind a regime gate.
5. **Detours are objectives too.** A side-quest is not noise to suppress — it is
   a child objective with its own score and a budget. Trajectory control means
   the detour *returns*. (Corollary from
   [DataPRM](https://arxiv.org/abs/2604.24198): scorers that punish recoverable
   errors prune self-correcting trajectories — score the recovery, don't kill
   the detour.)

## Positioning (the disambiguation contract)

**Trajectory control (`trajctl`) is the live, forward-progress control plane
over declared objectives.**

- **Not `fak traj`** — that is the *retrospective* trajectory-corpus toolkit
  (`internal/trajectory` recorder + `internal/trajhook` corpus scorers +
  similar/cluster/score/gc/export) used for gardening past turns. trajhook's
  `Score` is *notability of a past turn*; trajctl's score is *progress of a
  live objective*.
- **Not the scorecard family** — a scorecard is one deterministic measurement of
  a repo surface folding into a `*_debt` integer with a CI ratchet. trajctl
  scores *runs*, not repo surfaces.
- **Not `score-signal`** — that is the auto-arming feeder that files issues on
  CI scorecard regressions.
- **Not `fak signal steer`** — that is the operator's free-text input channel to
  a running session; trajctl *uses* that channel as one actuator rung.
- **Not `loopdrive`'s GOAL.md witness** — that is the binary termination policy
  (WITNESSED_DONE / BUDGET_SPENT); trajctl adds the continuous curve between 0
  and done, and imports the GOAL.md `Objective`/`Plan`/`Budget` spec as its
  canonical session objective.

## Core model

```
Objective {
  id, parent_id      // hierarchy: epic > session goal > sub-agent assignment > detour
  statement          // the goal text (GOAL.md Objective, /goal condition, dispatch issue)
  scorers[]          // attached scoring methods (registry names)
  budget             // turns / tokens / wall-clock before escalation (esp. detours)
  status             // active | paused (detour open) | met | abandoned (explicit)
}

ScoreRow {
  objective_id
  value              // normalized [0,1] progress-rate semantics where possible
  method, version    // registered scorer
  witness            // W3 | W2 | W1 | W0
  evidence           // pointer: commit SHA, ledger path, transcript span, verdict blob
  ts, session_id, run_id
}

Curve = time-ordered ScoreRows per (objective, method) → derived closed-vocabulary
signals: HEALTHY | STALL (flat curve × high activity) | DRIFT (declining
alignment) | DETOUR_OVERRUN (detour past budget while parent paused).
```

The ledger is append-only JSONL with a pinned schema id, same discipline as
`docs/nightrun/*.jsonl` and the guard decision journals. Objectives and scores
survive process death; the memq/notes read-back discipline (re-verify evidence
pointers at read time, stale rows surface as stale) applies.

## The scorer seam ("score anything" made modular)

A scorer is a registered method `(Objective, Evidence window) → ScoreRow`.
Planned built-ins, in witness-rung order:

- **W3 witnessed-commit progress**: fraction of the declared plan's phases with
  a `dos verify`-witnessed commit. The spine's first scorer — zero model calls.
- **W3 suite/benchmark progress**: test-witness pass fraction; AgentBoard-style
  progress rate `f(state, goal) ∈ [0,1]` for benchmark harnesses
  ([AgentBoard](https://arxiv.org/abs/2401.13178)).
- **W2 activity/progress divergence**: sessionaudit behavioral signals (edit
  churn, repeat failure signatures, sleep-polls) folded against the progress
  curve — high activity × flat progress = STALL.
- **W2 detour detection**: tool-stream shape opens/closes detour objectives.
- **W1 judge scorer**: structured verdict against the objective statement
  (rubric-based, per [SWE-TRACE](https://arxiv.org/abs/2604.14820): rubric
  criteria include trajectory discipline and budget awareness; verify to steer
  early, not to rerank finished runs).
- **W0 self-report**: the session's own claimed progress — recorded, never
  load-bearing.

The registry is the extension point: any leaf can register a scorer the way
`trajhook.Registry` registers corpus scorers today. Scoring methods carry
versions so a method change is visible in the curve's provenance.

## The steering ladder (weakest sufficient rung wins)

```
observe → annotate (ledger only)
        → nudge   (re-anchor: re-inject objective + curve through the session steer channel)
        → warn    (advisory guard rung / exit summary)
        → suspend (structured refusal reason from the closed vocabulary, e.g. TRAJECTORY_STALL)
        → escalate (operator / Slack)
```

Every rung above *annotate* sits behind the **regime gate**: recent curve
healthy ⇒ do nothing. Nudge outcomes are themselves ledgered and scored (did
the curve recover after the nudge?) so the intervention policy is calibrated
from evidence, not vibes — this is the
[failure-prediction ≠ failure-prevention](https://arxiv.org/abs/2602.03338)
lesson made structural. Re-anchor content follows checkpoint-and-re-read:
serialize objective + curve + plan state, read it back fresh, rather than
trusting context continuity (the goal-drift literature's most robust
mitigation: [2505.02709](https://arxiv.org/abs/2505.02709),
[2603.03258](https://arxiv.org/abs/2603.03258),
[2510.07777](https://arxiv.org/abs/2510.07777)).

## Use cases → mechanisms

1. **Long-horizon sessions**: score-at-turn-end (stop hook) writes the curve;
   drift/stall trigger a regime-gated re-anchor nudge; compaction boundaries
   (PreCompact) re-anchor unconditionally (cheapest safe point).
2. **Sub-agent distraction**: the dispatch/Task prompt carries an objective
   contract (objective + scorer); the parent accepts work on the *witnessed*
   child score, never the child's self-report — the DOS "kernel doesn't believe
   the agents" doctrine applied to progress. Handoffs between models get a
   fresh anchor (cascade drift: a strong model conditioned on a weak agent's
   trajectory inherits its drift, [2603.03258](https://arxiv.org/abs/2603.03258)).
3. **Error side-quests**: an error burst opens a detour objective with a turn/
   token budget; DETOUR_OVERRUN fires the return-to-main nudge; the detour's
   own score records whether the repair actually landed.
4. **Unrelated infra fixes**: same detour machinery — the infra fix is scored on
   *its* objective, the main objective's paused time is visible in the curve,
   and the controller enforces the return.

## Anti-gaming fences

- **Intent-vs-literal honesty**: every trajctl metric gets an
  `intent_literal_scorecard` row; a score whose name invites a meaning its
  denominator doesn't support is debt (the cache-hit vanity-metric lesson).
- **No length-gameable steering signals**: any score that rises with session
  length by construction is banned as a steering input.
- **Bounded scoring**: every scorer declares its termination condition; scoring
  work itself consumes a detour budget (the #2364 anti-pattern registry's
  "unbounded scoring loop" is a named enemy of this epic, not a risk we accept).
- **Scores gate harness actions only** — they never feed a model reward loop
  directly.

## SOTA anchors

- [AgentPRM (arXiv:2511.08325)](https://arxiv.org/abs/2511.08325) — agent steps
  scored by *promise and progress toward the goal*, not step correctness.
- [AgentBoard (arXiv:2401.13178)](https://arxiv.org/abs/2401.13178) — progress
  rate as a matching score `f(state, goal) ∈ [0,1]` over sub-goals; grounding
  accuracy as a second axis.
- [SWE-TRACE (arXiv:2604.14820)](https://arxiv.org/abs/2604.14820) — rubric
  process-reward models for long-horizon SWE agents; steer early with
  verification instead of reranking finished trajectories.
- [QLASS (arXiv:2502.02584)](https://arxiv.org/abs/2502.02584) — per-step value
  guidance without a proprietary judge.
- [Process Reward Agents (arXiv:2604.09482)](https://arxiv.org/html/2604.09482v1)
  — online step-wise rewards steering a frozen policy.
- [DeepVerifier (arXiv:2601.15808)](https://arxiv.org/html/2601.15808) — rubric
  verifier built on a failure taxonomy beats vanilla LLM-judge by 12–48% F1.
- Goal drift: [evaluation (2505.02709)](https://arxiv.org/html/2505.02709v1),
  [inherited/cascade drift (2603.03258)](https://arxiv.org/pdf/2603.03258),
  [drift as KL with restoring forces (2510.07777)](https://arxiv.org/html/2510.07777).
- [Failure prediction ≠ prevention (arXiv:2602.03338)](https://arxiv.org/pdf/2602.03338)
  — regime-aware intervention gating is load-bearing.
- [DataPRM (arXiv:2604.24198)](https://arxiv.org/html/2604.24198v1) — don't
  prune self-correcting trajectories; score the recovery.
- Industry: the "agent control plane" category (Forrester, 2025-12), Microsoft
  [Agent Control Specification](https://commandline.microsoft.com/agent-control-specification-runtime-governance/)
  (canonical verdict shape, fail-closed),
  [MI9 runtime governance (arXiv:2508.03858)](https://arxiv.org/pdf/2508.03858),
  AgentScope 1.0 interrupt-and-re-anchor.

## The epic spine (what ships first)

The end-to-end spin, in dependency order — each leaf is one issue:

1. `internal/trajctl` leaf: Objective + ScoreRow model, witness rungs, JSONL ledger.
2. `fak trajctl declare`: declare/close objectives; GOAL.md spec import.
3. Scorer registry + the W3 witnessed-commit progress scorer.
4. W2 activity/progress divergence (stall) scorer from sessionaudit signals.
5. `fak trajctl curve`: the fold + closed signal vocabulary.
6. Stop-hook score-at-turn-end (the curve gets a point every turn).
7. First steering rung: regime-gated re-anchor nudge via the session steer channel.
8. Dogfood on a real long-horizon session; witness report.
9. Docs: glossary positioning + disambiguation rows + observability doc.

Follow-on production waves (filed as children of the epic): the remaining
scorers (judge, rubric, benchmark, detour, token-efficiency), the objective
hierarchy and sub-agent contracts, the upper steering rungs and regime
calibration, integrations (dispatch tick, watchdog, loopdrive, /metrics,
session_audit, superloop, Slack), the meta loop (scorer calibration, provenance
audit, anti-gaming fences), and production hygiene (retention, fleet
aggregation, dos.toml config, replay backtest, score-signal feeder, operator
skill).
