# Test-execution routing alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6353](https://github.com/anthony-chaudhary/fak/issues/6353) tracks real execution witnesses.

The same native-allowed, Windows-with-WSL, CI-only, and unavailable probes are routed to exact executable choices. Arms: fak native; GOOS-only baseline; fak + GitHub Actions; Go native; WSL wrapper; GitHub Actions routing; Bazel platform constraints. Completion requires route accuracy, latency/throughput, CPU/RSS/network, setup/operator time, total cost, pinned versions/configuration, commands, and independent read-back. Three Windows/amd64 local samples were 31.07, 30.70, and 27.19 ns/op (median 30.70 ns/op; zero allocations). Unavailable arms remain measurement-zero; no cross-system ranking exists yet.

## GitHub Actions CI-only route contract

The executable fallback is the `workflow_dispatch` contract in `.github/workflows/ci.yml`. `internal/testroute` renders:

```text
gh workflow run ci.yml --ref main -f go_test_args_json=<JSON array of go-test arguments>
```

The dedicated `testroute-ci-only` job runs on `ubuntu-24.04`, takes the Go toolchain from `go.mod` through `actions/setup-go`, validates the input as an array of strings, and passes those strings verbatim to `go test`. Push and pull-request CI keep their existing jobs; a manual route dispatch runs only this dedicated test job. Evidence for a real dispatch must pin the source/head SHA and workflow revision, preserve the exact JSON input and command, and independently read the Actions run/job/timing APIs. CPU, RSS, network, rankings, and dollar cost remain unreported unless a hosted API actually supplies them; a missing billing endpoint or unavailable field is recorded as such rather than estimated.
