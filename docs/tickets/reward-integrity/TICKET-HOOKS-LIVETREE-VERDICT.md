<!-- fak-hooks-key: live-tree-findings-must-affect-verdict -->
# fix(hooks): stop live-tree governance tests passing on nonempty findings

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12378

## Current state

Four `internal/hooks` tests are named or documented as live-tree clean gates, but their nonempty-finding branches only call `t.Logf`:

- `gate_godfile_test.go:233-255`
- `gate_hardware_test.go:99-116`
- `gate_pythongate_test.go:13-32`
- `gate_tierdeclared_test.go:11-30`

A direct run reported 18 god-file/function offenses, numerous hardware-tell findings, and two undeclared leaves while every selected test returned PASS. Comments claim zero findings and, for god-file growth, claim the test reds CI.

## Parent context

Parent: #3831. Historical early-admission parent: #2653.

## Why now

The live package currently demonstrates that multiple blocking-looking findings can coexist with green tests. That makes the clean-tree claim unreliable precisely when autonomous workers use it as a completion witness.

## Problem frame

- Priority: P0 false-positive verification.
- Centrality: Stewardship (the shared-trunk governance obligation).
- P1: advanced - maintainers cannot cite a green clean-tree test while its detector reports blocking violations.
- P2: advanced - findings become machine-visible verdicts rather than hidden verbose logs.
- P3: preserved - align tests with existing gate policy without inventing new detectors.
- P4: advanced - deterministic parity proves nonempty findings affect the claimed verdict.

## Core through-line

Run existing live-tree detector -> classify finding as blocking or explicitly advisory -> make the Go test verdict and name reflect that classification. A test named `LiveTreeClean` cannot pass merely by logging a blocking finding.

## Working spine

The existing detector result flows directly into the named live-tree test verdict and the authoritative hygiene classification.

## Blast radius and affected lanes

- Primary lane: `hooks`.
- Current findings may require separately routed paydown or explicit, justified nonblocking classification.
- No production gate detector changes are required unless test/gate verdict parity exposes drift.

## Quarantined fallback mechanism

Do not land a permanently red trunk. Existing legitimate exceptions must be classified at the detector/baseline authority with a linked debt owner. If a check is intentionally advisory, rename it and remove all clean/gate claims; silent `t.Logf` is not an enforcement fallback.

## Scoped acceptance criteria

- [ ] Every `LiveTreeClean` test fails on a blocking nonempty finding.
- [ ] Intentionally advisory audits are named and documented as advisory, not clean gates.
- [ ] Current blocking findings are resolved or explicitly routed; they are not absorbed by an upward baseline rewrite.
- [ ] A focused policy test prevents regression to a log-only verdict.
- [ ] The package test and authoritative hygiene command agree on blocking versus advisory status.

## Gold-plating boundary

No redesign of the hook registry, no blanket baseline regeneration, no mass god-file refactor in this issue, and no weakening of hardware/tier/python/god-file detectors to keep tests green.

## Done condition

The selected tests can no longer print blocking findings and return PASS under a name or comment that asserts a clean tree.

## Witness

The deterministic gate witness is the selected live-tree test suite below.

### Non-Forgeable Witness

```text
go test ./internal/hooks -run 'Test(GodfileGate|HardwareTell|PythonToolGate|TierDeclared)_LiveTreeClean$' -count=1 -v
```

## Acceptance gate

`go test ./internal/hooks -run 'Test(GodfileGate|HardwareTell|PythonToolGate|TierDeclared)_LiveTreeClean$' -count=1 -v`

## Closure binding

The resolving commit cites this issue in its subject and carries a `(fak hooks)` trailer.

## Likely files

- `internal/hooks/gate_godfile_test.go`
- `internal/hooks/gate_hardware_test.go`
- `internal/hooks/gate_pythongate_test.go`
- `internal/hooks/gate_tierdeclared_test.go`

## Lane

`hooks`

## Dependencies and dedupe

Closed #5531 and #5849 repaired earlier individual god-file offenses. Closed #2653 shifted several detectors earlier. None owns the shared log-only false verdict, and no exact open match was found.

## Expected steps

6

```routing
lane: hooks
paths: internal/hooks/gate_godfile_test.go, internal/hooks/gate_hardware_test.go, internal/hooks/gate_pythongate_test.go, internal/hooks/gate_tierdeclared_test.go
expected_steps: 6
```
