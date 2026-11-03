# pkg/logprobparity

Scores a candidate endpoint's greedy continuations against a reference oracle
using vLLM `check_logprobs_close` semantics: token-id match first, top-k
membership on divergence, text-piece segmentation fallback for text-only
candidates. API: [`client.go`](client.go) (`Client.Generate`, `Score`,
`FoldPrompts`, `OraclePrompts`) and [`logprobparity.go`](logprobparity.go)
(`Check`, `SegmentText`, verdicts). Endpoint wire formats live in their own
profile files dispatched through the `apiProfiles` registry in `client.go`:
`profile_openai.go` (POST `/v1/completions`) and `profile_probs.go`
(POST `/completion`).

## Debt lane receipt (2026-11-03, flow ticket debtwave2-pkg-logprobparity)

Baseline census 2026-11-02 (`fak debt-lanes --lane logprobparity`): maturity
6.5/8, total_debt 3.69, health critical 0.65; issues `unproven_runtime`,
`unbenchmarked`, `modularity_deficit`, `model_hardcoding`.

Retired by this lane's pass:

- **Modularity split.** `Client.Generate` carried both endpoint wire formats
  inline behind one `switch`; each now lives in its own profile file
  (`profile_openai.go`, `profile_probs.go`) and dispatches through the
  `apiProfiles` registry. Behavior is byte-identical to HEAD (cross-validated
  diff of request bodies, URLs, error strings, retry logic, decode order).
- **Unbenchmarked.** `benchmark_test.go` measures the two production hot
  paths: `Check` (agreeing prefix, divergent-with-top-k) and `SegmentText`
  (full-match recovery); `go test -bench .` runs all three.

Residual flags, recorded with reason (census skips non-Go files, so this
receipt moves no scored field):

- **`model_hardcoding` (client.go: llama x3)** — keyword hits on the
  wire-format *family* name, not a model checkpoint. `"llama"` is the
  documented CLI value (`fak compute-trace logprobs --api openai|llama`,
  `cmd/fak/computetrace_logprobs.go:52`) and the exported `APILlama` constant's
  value; renaming or re-keying it changes the public surface and breaks the
  sole consumer, which sits outside this ticket's `pkg/logprobparity/`-only
  boundary. The structural hardcoding (per-API inline bodies inside one
  function) is gone; the family name is intrinsic to the `/completion` wire
  contract.
- **`unproven_runtime`** — the dogfood witness belongs in
  `internal/maturity/runtime-proofs.json`, also outside this boundary. Next
  checkable step: run `fak compute-trace logprobs --endpoint <live-url>
  --ref <oracle.json|llama-server-url>` end-to-end and register a
  `logprobparity` proof entry in a widened-boundary follow-up.

Witness: `go test ./pkg/logprobparity/... -count=1` ok; `go vet` clean;
`gofmt -l` empty; `fak debt-lanes --workspace . --target-repo fak --lane
logprobparity --json` -> maturity 6.5->7.5, total_debt 3.69->1.19,
`unbenchmarked` retired from health issues.
