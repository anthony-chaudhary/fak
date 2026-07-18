# Mac serving-path cache-value witness — contract half landed, Mac-hardware half `not yet` (#2727)

**Verdict: `not yet` for #2727's full done condition.** The host-independent half —
exact-span KV eviction + the WITNESSED cache-value P&L wired end-to-end on the in-kernel
serving path — is now contract-tested green. The remaining half (a real Mac session's
`cachevalue report` output with tok/s numbers) is gated on Mac hardware this fleet's
Windows workers do not have, and on #2723 (the Mac bench spine, still open).

## What is now witnessed (host-independent, runs in CI on any host)

`internal/gateway/cachevalue_evict_witness_test.go`
(`TestInKernelServeWitnessesCacheValueAcrossExactSpanEviction`) drives one multi-turn
coding-agent-style session over HTTP `/v1/chat/completions` into the gateway with a REAL
`agent.InKernelPlanner` (real `model.Session`, real tokenizer, RadixAttention reuse ON,
the #579 KV-MMU span bridge ON — the production posture of a Mac `fak serve --gguf`
session) and asserts, on one session:

1. **Realized fak-authored reuse** — turn 2 serves prompt tokens from the cached KV
   prefix (`cacheobs` tap, the same tap `internal/gateway/serving_metrics.go` renders).
2. **Exact-span eviction on the served path** — a poisoned tool result admitted through
   the live serve entrypoint mechanically drives `model.KVCache.Evict` of that result's
   span (`freed > 0`), bit-exact to a session that never saw it (`exact = true`).
3. **Reuse survives the eviction** — the turn AFTER the eviction still realizes reuse.
   This is the differentiator over whole-cache-reset engines: exact-span means the
   surviving prefix keeps earning.
4. **The WITNESSED Track-1 ledger row is non-zero** — the same
   `cachevalueledger.Append` call `cmd/fak/serve.go` makes at serve exit produces a
   `provider="fak"`, `mechanism="kv_prefix_reuse"` row with `reused_tokens > 0`, and
   `ScoreLedger` folds it to `realized_reuse_ratio > 0` under the #1066 honesty fence —
   the WITNESSED line of `fak cachevalue report`, never provider-OBSERVED savings.
5. **`fak_serving_*` fires natively** — the in-kernel worker renders goodput and a
   non-zero `fak_serving_prefix_cache_hit_rate` onto the same normalized schema the
   vLLM/SGLang scrape emitters feed (#2727 scope item 2).

Witnessed run (synthetic 2-layer model, contract numbers — NOT performance numbers, NOT
a Mac session): `turns=4 reused=503/1074 (ratio 0.468), exact-span evict freed=132
exact=true, serving hit rate=0.468`. Gate: `go test ./internal/gateway -count=1` +
`go vet ./internal/gateway`, green 2026-07-18.

Everything this test exercises sits ABOVE the compute backend: on macOS/Metal the same
planner tap, eviction bridge, ledger fold, and metrics render run unchanged — Metal only
moves where the GEMM executes. That is the argument for calling this the Mac serving
path's wiring witness without Mac hardware; the assumption it rests on is named below.

## What remains for the full #2727 done condition (operator-gated)

A real multi-turn agent session on a Mac (`fak claude-mac-fak`, or the fastest path the
epic #2722 children landed), with a workload that revisits/evicts KV spans, then:

```bash
fak cachevalue report --since <session date>   # WITNESSED line must be non-zero
```

captured into a `docs/notes/` artifact alongside the session's tok/s (blocked on #2723,
the fak-vs-llama.cpp-vs-MLX Mac bench spine, still open — `docs/bench-plan.md`'s
`node-macos-a` still records zero runs). The serve-exit ledger row lands automatically
(`cmd/fak/serve.go` appends to `docs/nightrun/cache-value.jsonl` on exit); the operator
only has to run the session and capture the report.

## Generation frame (gen/next)

- **Promotion evidence (what moves this toward `now`):** one captured Mac session
  artifact per the runbook above — a non-zero WITNESSED line from real Mac traffic plus
  tok/s from #2723 promotes the epic's "fast AND caches for you" claim to demonstrable.
- **Demotion/retirement evidence:** if #2723's bench lands and the Mac in-kernel path is
  retired in favor of the MLX ride-adapter (#2724) as the flagship Mac path, this witness
  demotes to the adapter's scrape-emitter equivalent and the in-kernel Mac claim retires.
- **Invalidating assumption:** "the wiring above the compute backend is identical on
  Metal" — if the Metal residency path ever bypasses the host `cacheobs` tap or the #579
  bridge (e.g. a future device-resident KV that never reports host reuse), this contract
  test stays green while the Mac claim silently breaks. A Mac-host run of this same test
  (`go test ./internal/gateway -run TestInKernelServeWitnessesCacheValue -count=1` on
  the Mac node) is the cheap check.

## Links

- Issue: #2727 (epic #2722; depends on #2723 for the tok/s half).
- Test: `internal/gateway/cachevalue_evict_witness_test.go`.
- Serve-exit seam: `cmd/fak/serve.go` (`cachevalueledger.Append`), tap:
  `internal/agent/inkernel_planner.go` (`cacheobs.Default.ObserveLabeled`).
- Showcase path: `docs/qwen36-claude-dogfood-playbook.md` (the `claude-mac-fak`
  runbook this witness backs).
