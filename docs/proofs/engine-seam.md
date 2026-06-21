# I1 · engine-seam

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/enginecache/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

The engine seam is fak's `abi.EngineDriver` boundary — the point where the kernel,
after folding adjudication at Submit, dispatches an ALLOWED tool call to a concrete
inference backend at Reap. Three drivers live in scope: the **in-kernel model**
(`internal/modelengine`, id `inkernel`, the default — a real greedy Prefill+Step decode
over a kernel-owned KV cache), and two offline-deterministic drivers in
`internal/engine` — the **Mock** (synthetic echo, the offline fallback) and the
**Cassette** (content-addressed record/replay). Alongside sits `internal/enginecache`,
the client that binds `cachemeta`'s remote-invalidation directives to a serving engine's
documented cache-reset endpoint (SGLang `/flush_cache`, vLLM `/reset_prefix_cache`).

This is **regime A (algebraic/structural)**: "correct" here is not a numeric tensor claim
(those are proven one layer down in `internal/model`) but two structural invariants of the
seam itself — (1) **determinism**: a fixed engine maps the same request to the same
response shape and content, with the input genuinely driving the result; and (2)
**invalidation binding / fail-closed staleness**: an invalidation directive is routed to
the correct engine reset call, and the seam never *pretends* a precise span was evicted —
when the engine cannot do exact-span eviction and the caller requires it, the call fails
**closed** rather than silently leaving a stale entry under a "success."

---

## THEOREM 1 — EngineDriver determinism (same request → same response shape)

**REGIME** A — structural (determinism / input-drivenness of the seam).

**STATEMENT** For a fixed registered engine, `Complete(ctx, c)` is deterministic in
`(c.Tool, c.Args)`: identical requests yield identical response shape (`StatusOK`, engine
meta, integer-parseable token accounting) and identical payload content; distinct requests
are distinguishable (the input drives the result, it is not a constant).

**PROOF** The strongest seam witness is the in-kernel driver. `Engine.Complete`
(`fak/internal/modelengine/modelengine.go:139`) materializes the args, deterministically
byte-tokenizes `(tool,args)` into a bounded prompt (`tokenize`,
`fak/internal/modelengine/modelengine.go:184` — a pure function, no RNG/clock), then runs a
**greedy** decode (`sess.Generate`, line 151) with `EOSTokenID=-1`
(`SyntheticConfig`, `fak/internal/modelengine/modelengine.go:80`) so the length is fixed at
`genTokens=16`. Greedy argmax over a fixed forward pass has no stochastic seam, so the
result is a pure function of `(tool,args)`. The two offline drivers share the property:
`Mock.Complete` (`fak/internal/engine/engine.go:48`) builds its body as a pure
`fmt.Sprintf` of the tool + truncated args; `CassetteEngine.Complete`
(`fak/internal/engine/engine.go:126`) is content-addressed by
`sha256(tool ‖ 0 ‖ args)[:16]` (`callKey`, `fak/internal/engine/engine.go:88`), so identical
requests replay the same recorded response and a miss is a typed `StatusError`. The lone
mutable field `Mock.calls` is a counter not surfaced in the `Result`, so it cannot perturb
shape or content.

**WITNESS**
```
go test ./internal/modelengine/ -run 'TestDecodeIsDeterministicAndInputDriven|TestCompleteRunsRealDecode' -count=1 -timeout 120s -v
go test ./internal/engine/ -run 'TestMockCompleteStatusAndUsage|TestCassetteReplayHitAndMiss' -count=1 -timeout 120s -v
```
`TestDecodeIsDeterministicAndInputDriven` (`modelengine_test.go:72`) asserts the load-bearing
property directly: two identical calls decode `equalInts` token sequences (line 78), and a
*different* `(tool,args)` decodes a *different* sequence (line 83). `TestCompleteRunsRealDecode`
pins the response shape (StatusOK, `engine==inkernel`, exactly `genTokens` in-vocab tokens).
`TestMockCompleteStatusAndUsage` and `TestCassetteReplayHitAndMiss` carry the offline drivers.

**VERDICT** PROVEN — 2026-06-20, all four green on this macOS node (`ok .../internal/modelengine 0.282s`,
`ok .../internal/engine 0.275s`).

**DOS** bound at ship.

---

## THEOREM 2 — enginecache binds invalidation correctly (fail-closed, no silent stale)

**REGIME** A — structural (directive→endpoint binding + fail-closed staleness gate).

**STATEMENT** A non-empty invalidation directive set is bound to the correct documented
engine reset endpoint (SGLang `/flush_cache`, vLLM `/reset_prefix_cache`); when exact-span
eviction is *required* but the engine supports only whole-prefix reset the call fails
**closed** before any reset (so no stale span is served as if evicted); an empty set is a
no-op; a non-2xx control response surfaces as an error carrying the status code.

**PROOF** `Client.Invalidate` (`fak/internal/enginecache/enginecache.go:57`) is the binding
seam. Empty → zero `Result`, no call (lines 59-61). Otherwise it resolves the engine
(`inferEngine`, `fak/internal/enginecache/enginecache.go:186`) and routes to the documented
endpoint (`endpoint`, `fak/internal/enginecache/enginecache.go:139`). The staleness core is the
fail-**closed** gate: `SupportsExactSpan` is hard-false for both public engines
(`fak/internal/enginecache/enginecache.go:130`), so `checkRequiredScope`
(`fak/internal/enginecache/enginecache.go:113`) returns an error **without issuing any reset**
when `RequiredScope==exact_span`. The system refuses to pretend a precise span was evicted,
rather than silently leaving it stale under a "success." When exact-span is not required, N
span directives collapse to **one** whole-prefix reset — an over-invalidation that is a
*superset* of the named span, so the invalidated entry cannot be served stale after a
successful reset. A non-2xx reset is surfaced as an error with the status code (lines 107-109).

**WITNESS**
```
go test ./internal/enginecache/ -count=1 -timeout 120s -v
```
Witnessed by `TestInvalidateSGLangFlushesRadixCache` (POST `/flush_cache?timeout=30` + bearer),
`TestInvalidateVLLMResetsPrefixCache`, `TestInvalidateExactSpanRequiredFailsBeforeWholeCacheReset`
(server never called; error names `exact-span eviction required`),
`TestInvalidateExactSpanUnsupportedUsesWholePrefixReset` (2 directives → 1 reset),
`TestSupportsExactSpanIsFalseForCurrentPublicEngines`, `TestInvalidateNoopsWithoutDirectives`,
`TestInvalidateReportsHTTPFailure`.

**VERDICT** PROVEN — 2026-06-20, all seven green (`ok .../internal/enginecache 0.270s`). This
proves the directive is correctly *bound* to the engine reset call and that the failure modes
are fail-closed/typed. It does **not** drive a live serving engine to observe a post-reset miss
(Theorem 3).

**DOS** bound at ship.

---

## THEOREM 3 — end-to-end "served fresh after invalidate" (the engine-side effect)

**REGIME** A — structural (the post-reset cache-miss effect on a real serving engine).

**STATEMENT** After `Invalidate` succeeds against a live serving engine, a subsequent request
for the invalidated prefix/span observes a cache **miss** (recompute), not the
pre-invalidation cached value.

**PROOF / GAP** `enginecache` is a *client* seam: it stops at the HTTP control boundary.
Proving the engine then actually evicts requires a real SGLang/vLLM process or a high-fidelity
serving fake with an observable prefix cache — neither is in-tree. The current suite uses
`httptest` stubs that assert the reset endpoint was POSTed (Theorem 2), not the engine-side
eviction.

**WITNESS** None today. Closed by a fake serving engine that (a) caches a keyed prefix
response, (b) accepts `/flush_cache`|`/reset_prefix_cache`, (c) returns a recomputed (different)
value on the next request after a reset — the test asserting response 2 differs from response 1.

**VERDICT** OPEN — 2026-06-20. The binding and fail-closed behavior are PROVEN (Theorem 2);
the engine-side eviction effect is honestly un-witnessed.

**DOS** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **enginecache-end-to-end-not-served-stale** → ✅ PROVEN by `TestEndToEndNotServedStaleSGLang`. enginecache hosts no cache itself; it translates cachemeta directives into the documented engine control-plane resets (SGLang POST /flush_cache, vLLM POST /reset_prefix_cache). The test stands up a stateful fakeServingEngine whose prefix cache is cleared iff its documented reset endpoint is POSTed, then drives the REAL Client.Invalidate against it. Property asserted end-to-end: (a) cold serve MISSes and warms a value; (b) re-serve WITHOUT invalidation is a HIT returning the identical value (non-vacuity guard); (c) Invalidate succeeds (status 200) and drives exactly one whole-cache reset; (d) the post-invalidation serve for the same prefix is a MISS and returns a strictly-newer recomputed value, never the pre-invalidation cached value. PROVEN for both SGLang (TestEndToEndNotServedStaleSGLang, with idle-timeout query) and vLLM (TestEndToEndNotServedStaleVLLM). The metamorphic CONTRAST TestNoInvalidateLeavesCacheStale proves non-vacuity: with NO Invalidate the warmed value is still served (HIT, 0 resets), so the anti-stale effect is a direct consequence of Invalidate, not of re-serving. All 3 new tests pass and the full enginecache package stays green.
