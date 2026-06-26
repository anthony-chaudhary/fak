# vCache Scorecard Epic #788 Closure

Issue #788 asked to move the vCache scorecard from an offline proof to a live
dogfood and release-gate surface. This note is the closure map for the epic: it
names the shipped artifact or gate that satisfies each definition-of-done item
without adding a new performance claim.

## Definition of done

| DoD item | Evidence |
| --- | --- |
| Child tickets under the epic are closed | #790, #791, and #792 were already closed. #789 is satisfied by the live telemetry dogfood artifact below; #793 is satisfied by the debug-stats surfaces below. |
| Reproducible live telemetry artifact exercises `fak vcache score --telemetry ...` | `experiments/agent-live/VCACHE-SCORECARD-TELEMETRY-DOGFOOD-2026-06-25.md` documents the replay command, and `experiments/agent-live/vcache-score-codex-telemetry-2026-06-25.json` is the frozen `fak.vcache.score.v1` artifact. It reports `active_source: telemetry`, `two_x_better: true`, `economics.multiplier: 7.1330210550495075`, `economics.hit_rate: 0.9553410520385182`, `economics.cache_read_tokens: 10163712`, `economics.rebate_token_equiv: 9147340.8`, and `economics.cost_token_equiv: 1491490.2`. |
| Docs and benchmark authority describe the threshold and failure modes | `docs/serving/vcache-scorecard-playbook.md` is the operator playbook, `BENCHMARK-AUTHORITY.md` carries the authority boundary for vCache claims, and `internal/benchcatalog/catalog.go` registers the `vcache` benchmark entry as the offline scorecard surface. |
| CI/dogfood can re-run the scorecard artifact | `Makefile` target `vcache-gate` runs `tools/vcache_scorecard_gate_test.py` and `tools/vcache_scorecard_gate.py`; `tools/recent_feature_dogfood.py` also has vCache score and benchmark dogfood steps. |
| Operator debug output reports cache/compaction health | `fak guard --debug-stats` and `fak serve --debug-stats` are wired in `cmd/fak/guard.go` and `cmd/fak/serve.go`. `internal/gateway/debug_stats.go` renders one payload-free per-turn line with prompt/completion/cache-read/cache-create tokens, compaction state, and resetScore shadow health. `internal/gateway/debug_stats_test.go` covers the five reset health states, compaction/cache-hit rendering, flattening, and log-independence. |

## Validation

Validate this closure with:

```sh
python tools/vcache_scorecard_gate_test.py
python tools/vcache_scorecard_gate.py --json
go test ./cmd/fak ./internal/vcachescore ./internal/gateway
.\scripts\ci.ps1
```

`.\scripts\ci.ps1` is the repo's full Windows CI gate for this host. No pass is
inferred from the artifact alone; the scorecard gate still has to execute and
fail closed if the default 2x floor regresses.
