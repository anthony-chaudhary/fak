# DeepSWE Adapter Smoke - 2026-06-26

Status: `ADAPTER_FIXTURE_COMPLETE`

This packet proves the `RunnerDeepSWE` adapter path no longer stops at the old
placeholder. A configured adapter executable was invoked through
`fak swebench run --agent deepswe`, received the
`fak.swebench.deepswe-request.v1` request, and emitted SWE-bench prediction JSON.

This is fixture evidence only. It does not claim a real DeepSWE/R2E-Gym model
score, pass@1, or resolve rate.

## Command

```powershell
$env:FAK_DEEPSWE_RUNNER='go'
$env:FAK_DEEPSWE_RUNNER_ARGS='run ./cmd/fak-deepswe-runner --fixture'
$env:FAK_SANDBOX_ENV_ALLOW='LocalAppData,GOCACHE'
go run ./cmd/fak swebench run `
  --agent deepswe `
  --difficulty testdata/swebench_smoke.json `
  --filter full `
  --limit 2 `
  --model DeepSWE-Preview-fixture `
  --timeout 30s `
  --max-steps 7 `
  --preds-only `
  --output experiments/agent-live/deepswe-adapter-smoke-20260626
```

The `FAK_SANDBOX_ENV_ALLOW` line is only for the local `go run` fixture
adapter on Windows: the adapter subprocess otherwise cannot see Go's build
cache location after env masking. A prebuilt DeepSWE/R2E-Gym adapter should not
need those Go-specific variables.

Official-grader readback:

```powershell
go run ./cmd/fak swebench eval `
  --predictions experiments/agent-live/deepswe-adapter-smoke-20260626/predictions.json `
  --run-id deepswe-adapter-smoke-20260626 `
  --out experiments/agent-live/deepswe-adapter-smoke-20260626/eval.json
```

## Artifacts

- `summary.json`: packet summary and no-claim fence.
- `predictions.json`: two canonical SWE-bench prediction rows.
- `meta.json`: run metadata, `2` total instances, `2` done, `0` failed.
- `eval.json`: official grader command plus the local gate reason.

The local official grader is gated because this host does not have the
`swebench` Python harness importable. Run the command in `eval.json` on a
Docker/SWE-bench host for benchmark-native scoring.
