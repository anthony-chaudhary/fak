# TICKET-05a: Pi launcher qwen-prefix window row overrides the router-reported window for cloud Qwen routes

GitHub mirror: #13778. Parent: [TICKET-05](TICKET-05-pi-launcher-garden-launchd.md) (#13762).

## Current state

A CLI end-to-end run on 2026-10-09 against the live local router, with a temp HOME, found that the two Pi paths disagree on the window for a cloud Qwen route:

- `fak pi --dry-run --model qwen3.8-max` (launcher) resolves `contextWindow` 131072.
- `fak pi config --from-router --write` writes 163840 for the same model. The router reports 1000000, and `harnesskit.DeriveContextEnvelope` quality-caps that to 163840.

Cause: `internal/projectassets/pi_model_window.go` has a catch-all registry row, `{match: piHasModelPrefix(id, "qwen"), window: DefaultPiServedWindow}`, that matches every `qwen*` id. `PiModelServedWindow` checks the registry before the router-reported or caller window, so the launcher pins 131072 for cloud Qwen routes. Halo Qwen (`Qwen3.8-27B-UD-Q2_K_XL`, router 131072) is correct in both paths.

Related commits: 51185bc3243 (selected-window propagation), 0c191ba16bc (shared router compaction across model windows).

## Working spine

1. Record the precedence decision. Recommendation: when the router reports a served window, it wins. The registry is the fallback for routes the router does not report.
2. Narrow the `qwen` prefix row to the Halo Qwen ids, or remove it in favour of the exact Halo row that already exists.
3. Make `PiModelServedWindow` and its launcher caller prefer a router-reported window over the registry.
4. Add a fail-before/pass-after test: a cloud Qwen id with a router-reported 1000000 window resolves to the same capped value in the launcher and in the `--from-router` config path, and the Halo Qwen id still resolves to 131072.

## Witness

- `go test -run 'PiModelServedWindow|PiWindow' ./internal/projectassets/ ./cmd/fak/` exits 0, and the new test fails on the parent commit.
- Captured output of `fak pi --dry-run --model qwen3.8-max` and `fak pi config --from-router --write` in a temp HOME, both showing the same window.

## Done condition

- [ ] The precedence decision is recorded in the `pi_model_window.go` doc comment.
- [ ] For a cloud Qwen route, the launcher and the router-derived config report the same window.
- [ ] Halo Qwen still resolves 131072 in both paths.
- [ ] A fail-before/pass-after test citing 51185bc3243 is on trunk.
