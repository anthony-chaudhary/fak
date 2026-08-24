# Work delivery: recording is not readiness

FAK tracks delivery as five independent facts: **recorded**, **compile-admitted**, **verified**, **integrated**, and **release-ready**. A commit proves only that work was recorded. It does not silently add source to the compile stream, claim tests passed, prove a push, or authorize a release.

## Captured walkthrough

The self-checking witness is [`internal/workdelivery/e2e_test.go`](../internal/workdelivery/e2e_test.go). Its stable artifacts are:

- [`happy-path.json`](../internal/workdelivery/testdata/e2e/happy-path.json): recording → explicit compile admission → verification → integration → explicit witnessed release readiness.
- [`failure-path.json`](../internal/workdelivery/testdata/e2e/failure-path.json): an aggregate `full-tests` red recursively splits by tree and test group, then ends at `unit-bad-test` with its exact gate, evidence, and retry command.

Run it from the repository root:

```bash
go test ./internal/workdelivery -run '^TestCaptured' -count=1
```

The happy-path fixture deliberately includes `fixture/recorded_broken.go` as **recorded but excluded**. The test fails if that path enters the admitted compile set. It then applies one receipt per axis and checks that every downstream axis remains unchanged until its own receipt arrives. Release admission additionally requires the matching witnessed readiness receipt; a `ready` state alone fails closed.

## Operator commands

Inspect or advance one declared axis:

```bash
fak work-delivery status --file UNIT.json
fak work-delivery transition --file UNIT.json --axis compile_admission --to admitted --out UNIT.next.json
```

Localize a failure and repeat on the returned child scope:

```bash
fak work-delivery diagnose --file OBSERVATION.json --json
```

Resolve local vocabulary such as `CI red` to the canonical stage and bottleneck registry:

```bash
fak work-delivery stages --local 'CI red'
```

The rule is mechanical: consume the receipt for the stage you are deciding. Never infer release readiness from commit, build, test, or push success.

## Final-mile uptake: release is not activation

Operator-facing work has two additional independent facts:

- **activated**: a witnessed runtime receipt names the exact `module@rev`, build digest, and configuration digest currently running.
- **operator-accepted**: a captured operator journey (for example, a post-hook result appearing in the TUI or a click changing views) is witnessed against that same runtime identity.

`release-ready` never implies either fact. Activation does not imply acceptance. A stale build, changed configuration, missing journey, or self-reported evidence fails closed. `fak work-delivery inspect --file UNIT.json` prints the missing final-mile receipt as the next action.
