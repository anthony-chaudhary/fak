---
name: verify
description: "Bind a done-claim to a GREEN test run of the changed package, not just diff shape. Use after a commit claims a package/feature is done and you want to run that commit's affected tests and report CLAIM_TEST_GREEN / CLAIM_TEST_RED / CLAIM_TEST_UNRUN before folding the claim as true."
---

# verify - bind "done" to a green test run, not just diff shape

`dos commit-audit` (and the dispatch-tick witness it feeds) grades whether a
commit's DIFF did the KIND of thing its subject claimed — `diff-witnessed` vs a
forgeable `subject-only`. That is commit SHAPE, explicitly not correctness: a
diff-witnessed commit can still fail its own tests. This skill runs the missing
rung — the resolving commit's affected-package tests — and binds the pass to
"done". It is additive to the shape witness, never a replacement.

## Load-Bearing Rule

The only path to a GREEN done-binding is a test that actually RAN and PASSED.
A runner that never fired, found no test-bearing changed package, or faulted is
`CLAIM_TEST_UNRUN` — a valid, surfaced state, never a pass. Do not fold a
done-claim as true on shape alone, and never let UNRUN masquerade as GREEN.

## Step 0 - Identify The Resolving Commit

Take the commit whose done-claim you are checking (the resolving commit for the
issue, usually `HEAD` after a ship). Its changed `.go` files determine which
packages carry the test binding:

```bash
git show --name-only --pretty=format: <sha>
```

Map each changed file to its package dir; a changed dir with **no** `*_test.go`
carries nothing to bind a pass to — it contributes to UNRUN, not a vacuous pass.

## Step 1 - Run The Affected-Package Tests (WSL-aware)

Run the changed packages' tests through the repo's own affected-test seam, which
is WSL-aware — on native Windows it drives the tests through `./test.ps1`, else
`go test`:

```bash
fak affected            # runs the affected packages' tests through the WSL-aware seam
```

If you are checking one known package directly, the explicit equivalent is:

```bash
go test ./<changed-pkg>/... -count=1
```

Do not substitute a build (`go build`) or a vet for a test run — a green build
is shape, not the test binding.

## Step 1b - Execute Real-World Smoke Testing Earlier in the Process

Mocks hide integration bugs (Hermes' rule). Whenever touching CLI, runtime, or security-critical paths, prove real execution early in development:

- **Isolated Validation Smoke**: Run `fak validate --mine <changed-files> --smoke` in your inner loop to compile and execute preflight, version, and offline agent checks in an isolated overlay.
- **Fast Local Smoke Tier**: Run `make smoke` (`smoke-exec` + `dogfood-test`) or `make test-fast` before staging.
- **Dogfood Launcher Witness**: Run `bash scripts/dogfood-claude_test.sh` to verify process supervision and fail-fast deadlines.
- **Attestation Trailer**: Add `Smoke-verified: <command>` or `E2E-verified: <command>` in commit or test files to satisfy the `E2E_OVER_MOCKS` boundary hook.

## Step 2 - Grade The Binding

Fold the run into exactly one claim, the same grader the dispatch-tick witness
records into the `.witness` sidecar's `test_claim` field:

| Ran? | Passed? | Claim | Meaning |
|---|---|---|---|
| yes | yes | `CLAIM_TEST_GREEN` | affected tests ran and passed — done-binding holds |
| yes | no | `CLAIM_TEST_RED` | affected tests ran and FAILED — a diff-witnessed commit can still be RED; this is the gap the rung closes |
| no | — | `CLAIM_TEST_UNRUN` | no resolving commit, no test-bearing changed package, or the runner was off/faulted — valid, surfaced, NOT a pass |

`ran=false` always outranks a stale `passed` bit: a run that never happened is
UNRUN, never GREEN.

## Step 3 - Read The Recorded Witness (Optional)

On a live dispatch tick with the in-tick runner enabled, the shell records this
same binding for you. The default runner is opt-in via `FAK_WITNESS_TEST_RUN`
(unset -> the rung records UNRUN so the hot loop never eats an in-tick
`go test`); this `verify` skill is the always-on manual consumer of the same
rung. When a sidecar exists, read it rather than re-running:

```bash
# the per-worker resolve log's sibling: resolve-<N>-<stamp>.witness
# its `test_claim` key carries CLAIM_TEST_GREEN / _RED / _UNRUN
```

Preserve the rung alongside the shape verdict — a `diff-witnessed` +
`CLAIM_TEST_GREEN` is a strictly stronger done-binding than `diff-witnessed`
alone, and `diff-witnessed` + `CLAIM_TEST_RED` is a REFUTATION.

## Step 4 - Report, Do Not Fold Blindly

- `CLAIM_TEST_GREEN`: the done-claim is test-bound; safe to fold as done.
- `CLAIM_TEST_RED`: do not fold as done; the change fails its own tests —
  re-open or fix, and say so.
- `CLAIM_TEST_UNRUN`: do not fold as GREEN; report the missing test run and why
  (no changed test-bearing package, runner off/faulted). UNRUN is honest, not a
  pass.

## Anti-Patterns

- Reporting a diff-shape `diff-witnessed` as if it proved the tests pass.
- Letting `CLAIM_TEST_UNRUN` read as "probably green".
- Running a build or a vet and calling it the test binding.
- Grading a package that has no `*_test.go` as a pass (that is a vacuous run —
  it is UNRUN).
- Requiring every commit to carry a test run — UNRUN is a valid state, not a
  failure.
