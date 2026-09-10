# fix(hooks): pass the public repository root to companion boundary checks

<!-- fak-hooks-key: companion-boundary-public-root -->

```routing
lane: developer-tooling
paths: [".githooks/pre-commit", ".githooks/pre-commit.ps1"]
expected_steps: 3
```

## Working spine

Public staged files -> companion-aware pre-commit hook -> companion boundary checker with the resolved public root -> valid public commit admitted.

## Current state

Both public pre-commit hooks change the Go command's working directory to the companion repository and then pass `--fak-dir .`. The relative dot consequently resolves to the companion repository rather than the public checkout whose index the hook is validating. Verified public changes are then inspected under the wrong repository role and can be rejected as public/private placement violations.

The defect is visible at `.githooks/pre-commit:102` and `.githooks/pre-commit.ps1:118`. Both hooks already resolve the public checkout as `REPO_ROOT` or `$repoRoot`; they do not pass that value to `--fak-dir`.

## Why this is next

The hook is the mandatory commit gate. A false refusal blocks otherwise verified public fixes, including commits whose direct boundary check succeeds when given the correct absolute public root.

## Parent context

Standalone public developer-tooling correctness leaf; no parent issue.

## Core through-line

Keep executing the boundary tool from the companion repository, but pass the hook's pre-resolved absolute public repository root to `--fak-dir`. Apply the same correction to Bash and PowerShell. Preserve every boundary and leak check; this is a path-resolution repair, not a bypass.

## Gold-plating boundary

Do not redesign companion discovery, alter boundary policy, weaken hook failures, add environment overrides, or change unrelated hygiene and formatting checks.

## Verifiable Witness

With these two public files staged:

```text
internal/metalgemm/icb_replay_spec_test.go
internal/metalgemm/icb_types.go
```

the current companion-aware hook has twice ended in `HOOK_REFUSED` because its boundary command uses the companion working directory as `--fak-dir`. Running the same boundary command with the absolute resolved public checkout passed to `--fak-dir` accepts those two staged public files.

After the fix, exercise the actual hook rather than a bypass:

```bash
.githooks/pre-commit
```

```powershell
& .githooks/pre-commit.ps1
```

## Witness

Both real hook entry points must reach the successful companion boundary result for the staged public files. The boundary invocation must visibly contain the resolved public root for `--fak-dir` and must retain `--staged` and the companion `--private-dir`.

## Definition of done

Completion requires every scoped acceptance criterion and the real staged-hook witness.

## Scoped Acceptance Criteria

- [ ] Bash passes absolute `REPO_ROOT` to `--fak-dir`.
- [ ] PowerShell passes absolute `$repoRoot` to `--fak-dir`.
- [ ] Both hooks keep the companion as the Go command working directory.
- [ ] Both hooks retain staged boundary enforcement and leak auditing.
- [ ] The actual Bash and PowerShell hooks accept the witnessed valid public staged files.
- [ ] A real boundary violation remains rejecting; no bypass is introduced.

## Acceptance gate

Both hook entry points pass on the witnessed valid staged public change, and source review confirms that all existing checks remain enabled.

## Closure binding

The resolving commit cites this issue and carries the `developer-tooling` lane.

## Lane

developer-tooling

## Likely files

- `.githooks/pre-commit:18-20,101-106` — resolved Bash public root and companion boundary invocation.
- `.githooks/pre-commit.ps1:19-27,117-123` — resolved PowerShell public root and companion boundary invocation.

## Expected steps

3

## Blast radius and affected lanes

Only public commit-hook path resolution changes. Runtime, inference, serving, and boundary-policy packages are unaffected.

## Quarantined fallback mechanism

No fallback or bypass is permitted. If companion validation cannot run against the correct public root, the hook continues to reject the commit.

- Centrality: Core
- P1 Context: advanced — restores the mandatory public commit gate.
- P2 Net value: preserved — policy remains unchanged.
- P3 Adaptation: N/A — no adaptive runtime surface.
- P4 Operations: advanced — removes a deterministic false refusal.

## Work estimate

Estimate: 1 point

## Overall completion contribution

Contribution: 1/4 points for companion-aware hook reliability.

## Completion standard

demo
