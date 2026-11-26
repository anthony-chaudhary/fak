# fak agent — live A/B on your own model

Compare how a model handles the same tool task with and without fak’s policy
checks. The example uses a test booking system. Each run writes its own
transcript hash and measurements.

## Run the comparison

With Ollama running locally:

```sh
./run.sh --local
```

To select another OpenAI-compatible server, pass `--base-url` and `--model`.
To use a cloud provider, choose the provider and the environment variable for
your API key. The provider bills both arms of the comparison.

```sh
./run.sh --provider gemini --key-env GEMINI_API_KEY
```

Read `agent-report.json` for the result and `transcript_sha` for the run identity.

## Reading the A/B report

Compare `model turns` and `tool errors` for retries. Inspect `injection in
context` and `destructive op executed` for the policy outcome. Check `task
completed` on both arms before interpreting a difference in turn counts.
Results depend on the model.

The HEADLINE also prices the spared turns into the currency a user feels —
wall-clock. On a **live** run it reads `time saved by fak : Ns  (K turns × Mms
observed mean E2E/turn)`, where the report carries a measured `time_saved_seconds`
= `turns_saved × mean-per-turn-latency` (the fak arm's observed mean end-to-end
per-turn latency, also emitted as `mean_turn_latency_ms`, with each arm's raw
wall-clock in `elapsed_ms`). This is the same pricing the live info panel and guard
exit summary use.

Each result applies to the model, task and settings in its report.

## What this does not claim

- The offline A/B uses a deterministic mock planner. It shows the mechanics of
  the policy checks, not how a real model behaves.
- A live run measures one model, one task and one set of settings. It does not
  claim a general improvement in tool-use quality or safety.
- `time saved by fak` is a projection from turns spared times observed per-turn
  latency. It is not a measured end-to-end saving for a user workflow.
- One test booking task is not a safety benchmark. The adversarial coverage is
  in the red-team battery linked below.

## See also

- [`CLAIMS.md`](../../CLAIMS.md) #104 — the live-seam claim this example backs.
- [`experiments/agent-live/`](../../experiments/agent-live/) — the real live-run
  witnesses (`turntax-injection-live.json` and peers; each carries a distinct
  per-trial `transcript_sha`, `live: true`).
- [`examples/agentdojo-redteam/`](../agentdojo-redteam/) — the deterministic,
  Go-only red-team battery (the static-attacker counterpart to this live A/B).
