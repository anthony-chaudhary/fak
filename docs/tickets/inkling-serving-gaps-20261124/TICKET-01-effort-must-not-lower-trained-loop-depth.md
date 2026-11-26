<!-- fak-sync-key: inkling-serving-gaps-20261124-tk01 -->
# fix(agent): reasoning effort must not lower a looped model below its trained pass count

```routing
lane: internal/agent
paths: ["internal/agent/effort_budget.go", "internal/agent/effort_budget_test.go", "internal/agent/effort_trained_loops_test.go", "internal/capabilitymatrix/matrix.go", "internal/capabilitymatrix/checkpoint_trained_loops_test.go", "internal/gateway/loop_depth.go", "internal/gateway/loop_depth_test.go", "internal/gateway/loop_depth_irreversible_test.go", "internal/ggufload/inkling2_trained_loops_test.go", "cmd/fak/serve_inkling2.go", "docs/tickets/inkling-serving-gaps-20261124/TICKET-01-effort-must-not-lower-trained-loop-depth.md"]
expected_steps: 3
repo: fak
priority: P1
class: correctness
dependencies: []
```

Process cause: verification-gap

## Why this ticket exists

`agent.ResolveLoopDepth` (public fak `internal/agent/effort_budget.go`) computes
`ceil(maxLoops*step/4)`, where step is none=0, low=1, medium=2, high=3, ultra=4.
The gateway falls back to `maxLoops` when no depth is set
(`internal/gateway/loop_depth.go`).

Inkling-2-Mini is trained at `num_passes: 3` and fixed there. With
`maxLoops=3`, effort maps to depth like this:

| Effort | Passes |
|---|---|
| none | 1 |
| low | 1 |
| medium (the default) | 2 |
| high | 3 |

So every tier below high runs the model below its trained depth.

The TML varloop study (`docs/research/STUDY-tml-varloop-destructive-actions-2026-11-24.md`)
found that one pass proposes destructive actions about 3.4x as often as three.
Quality knees at three passes.

This is latent today: no capability row sets `max_loops`, and Inkling is not
served. It turns live the day either changes.

## Scope

- Effort and loop depth are separate levers. Use the trained pass count as the
  default and the floor.
- Effort may raise depth above the trained count only for checkpoints that
  declare variable depth (varloop). It never lowers depth below that count.
- Any request whose tool scope includes irreversible tools gets a floor of the
  trained count (3).

## Acceptance witness

- Table tests in `internal/agent`: for a fixed-depth 3 model, every effort tier
  resolves to 3. For varloop with max 5, effort none resolves to at least 3.
- A gateway test: with no `loops` set, depth resolves to the trained count, not
  `maxLoops`.

## Trained-loop capability and floor

- `capmatrix.Capabilities.TrainedLoops` (`trained_loops`, env key `trainedloops`)
  is the checkpoint's trained pass count, sourced from GGUF `loop.count`.
  `max_loops` stays the ceiling. Unknown (0) means fixed-depth at `max_loops`.
- Effort resolves as `max(effort-derived, trained_loops)` via
  `agent.ResolveTrainedLoopDepth`, capped at `max_loops`.
- `agent.ResolveIrreversibleLoopFloor` floors a granted depth at the trained
  count when the request's tool scope includes an irreversible tool. The gateway
  resolves this per request from `ChatRequest.Tools` (after the policy cap, so
  a cap cannot drop irreversible work below the trained count).
- Source of `trained_loops`: the serve load path registers the loaded
  checkpoint's `loop.count` under the served model slug with
  `capmatrix.RegisterCheckpointTrainedLoops`. The Metal route does not register
  it yet, so there it stays unknown (fixed-depth at `max_loops`).
- Scope rule: a tool is irreversible-scope when its name classifies irreversible
  by name, or it is a shell tool (`bash`, `shell`, `exec`, ...). A shell can run
  any command, and its destructive reach is only known per call.

## Cost of the floor

For a fixed-depth checkpoint trained at 3 passes (Inkling-2-Mini):

- `none` and `low` go from 1 pass to 3 passes per token.
- `medium` goes from 2 to 3, and `high` stays at 3.
- Decode cost scales with passes, so `none` and `low` run roughly 3x the core
  forward work they did before. Operators who need the cheap tier must accept
  the trained depth or use a checkpoint that declares variable depth.
- The shell-scope rule adds the same floor to every request that offers a shell
  tool, at any effort tier. Tool-using Inkling turns at `none` and `low` pay the
  3x core cost even when the turn is routine. This is accepted for safety; a
  narrower rule would need per-call tool arguments at depth-selection time.
