# Issue #8624 — observed small-model Ultracode cache frontier

**Verdict:** on `qwen2.5:0.5b` through the sanctioned `fak-realmodel` node, every bounded agentic cell at widths 1, 2, 4, and 8 retained the frozen accepted outcome while reducing scoped context and reusing a shared prefix. The evaluator therefore hill-climbed both multi-agent modes through width 8.

## Replay

```powershell
fak ultracode bench --scenario access-frontier `
  --scenario-input docs/_witnesses/issue-8624-ultracode-smallmodel/access-frontier.json `
  --widths 1,2,4,8 --json
```

Files:

- `source-artifact.json` — source-linked observed campaign: node/runtime/model, frozen task digest, bounded scout/writer/reviewer roles, accepted responses, Ollama `cached n_tokens` excerpts, and request usage telemetry. Its `raw_capture_sha256` binds the compact checked-in receipt to the complete raw capture retained during the run.
- `access-frontier.json` — evaluator input (`evidence_kind: observed_run`).
- `report.json` — replayed `fak-ultracode-access-frontier/1` result.

## Result

| Mode | Width | Full-context counterfactual | Scoped child context | Shared-prefix reads | Total avoided | Verdict |
|---|---:|---:|---:|---:|---:|---|
| scout_writer | 1 | 4,095 | 1,639 | 1,024 | 3,480 | GAIN |
| scout_writer | 2 | 8,190 | 3,280 | 2,628 | 7,538 | GAIN |
| scout_writer | 4 | 16,380 | 6,562 | 5,908 | 15,726 | GAIN |
| scout_writer | 8 | 32,760 | 13,126 | 12,468 | 32,102 | GAIN |
| multi_writer | 1 | 4,095 | 1,641 | 1,024 | 3,478 | GAIN |
| multi_writer | 2 | 8,190 | 3,280 | 2,628 | 7,538 | GAIN |
| multi_writer | 4 | 16,380 | 6,560 | 5,836 | 15,656 | GAIN |
| multi_writer | 8 | 32,760 | 13,120 | 12,252 | 31,892 | GAIN |

The single-agent width-1 control also gained, but it is not a throughput baseline. It establishes that the same harness semantics benefit from scoping and prefix reuse before concurrency is increased.

## Measurement boundary

- `full_context_input_tokens` is an observed full-context prompt count multiplied by equal agent width; it is a context-access counterfactual, not a second latency run.
- `scoped_context_input_tokens` is Ollama's per-request `prompt_eval_count` summed across bounded role contexts.
- `shared_prefix_read_tokens` is the largest Ollama llama-server `cached n_tokens` value for each task in the source log. This is runtime-authored KV/prompt-cache telemetry, not a value inferred from wall time.
- All responses normalized to the same `ACCEPTED` outcome digest before savings were credited.
- The claim excludes raw single-request throughput, traditional batching, billed tokens, and spend. It applies only to this observed model/runtime/task envelope.
