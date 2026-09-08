---
name: cross-validate
description: "Cross-validate code changes with independent subagents, execute on-device test witnesses, and generate GitHub issues for discovered follow-ons by default (DOS style). Use after implementing any fix or feature, before reporting completion."
---

# cross-validate — adversarial subagent audit, on-device proof & auto-ticketing

DOS discipline: **never believe an authoring agent's self-report.** A return string
claiming "implemented, all tests pass" is narrative, not proof. This skill coordinates
independent on-device verification, spawns a dedicated subagent to adversarial-audit
the work, and durable-tickets discovered follow-on issues by default.

## Load-Bearing Rule

Do not report completion without:
1. **On-device test execution** (`CLAIM_TEST_GREEN`)
2. **DOS diff audit** (`dos commit-audit HEAD` -> `diff-witnessed`)
3. **Independent subagent cross-validation** (`CONFIRMED_VALID`)
4. **Follow-on issue generation** (tickets filed by default for edge cases or soak needs)

---

## Step 0 — Identify Scope & Staged Changes

Inspect the exact changeset to verify what was touched:

```bash
git status -s
git diff --stat HEAD~1..HEAD
```

Ensure no extraneous files are included. All changes must fall within the lane's declared tree.

For native tuning, apply the `AGENTS.md` default before expensive execution: independently
review validity and planned cost against existing authorization and budget, binding the review
to exact source/configuration/workload identity, objective, load, and SLO. Check the full
configuration delta and justification for inseparable bundles; attribute results to the bundle
only. Require fresh reference measurements where changed objective or load makes history
incompatible; SLO-only changes may use reassessed compatible history. Candidate changes
invalidate affected review and witnesses only. This review grants neither spend nor security
authority and preserves existing user authorization, budget limits, and security controls.

---

## Step 1 — On-Device Build & Test Verification

Compile without modifying in-tree binaries:

```bash
fak-dev buildcheck --vet
```

Execute the affected package tests on-device (routed through WSL on native Windows):

```bash
# Whole affected test suite via repo seam:
fak affected

# Or explicit package:
.\test.ps1 ./internal/<changed-pkg>/...
```

Only proceed when the status is `CLAIM_TEST_GREEN`. An unrun or faulted test is `CLAIM_TEST_UNRUN`, never a pass.

---

## Step 2 — DOS Commit Audit & Stamp Witness

Verify that the commit subject follows Conventional Commits with a recognized `(fak <leaf>)` trailer and DCO sign-off:

```bash
fak commit --preview
dos commit-audit HEAD
```

Confirm the witness level is `diff-witnessed`. If the audit returns `CLAIM_UNWITNESSED` or `subject-only`, the commit makes a claim that the diff does not corroborate — fix the diff or the message before proceeding.

---

## Step 3 — Spawn Concurrent Verification Triad

Spawn a concurrent Verification Triad in a single message turn using multiple `task` tool calls to independently audit the landed work: `cross-validator` (adversarial audit) + `issue-auditor` (edge cases & ticket generation) + `tester` (regression & matrix execution).

```json
[
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "cross-validator",
      "description": "Adversarial cross-validation of HEAD",
      "prompt": "You are an adversarial verifier auditing recent code changes in commit HEAD.\nDo NOT trust the author's narrative or claims.\n\nYour task:\n1. Run `git show --stat HEAD` and inspect the raw diff with `git show HEAD`.\n2. Verify that edits did not leak outside declared package boundaries.\n3. Check for regressions, concurrency hazards, race conditions, memory leaks, boundary conditions, and nil-pointer risks.\n4. Verify whether the tests actually test the fix/feature or merely pass vacuously.\n\nEmit a structured verdict:\n- VERDICT: CONFIRMED_VALID | DEFECT_DETECTED\n- EVIDENCE: diff inspection + invariant audit\n- DIFF_AUDIT: analysis of edge cases and invariants"
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "issue-auditor",
      "description": "Audit edge cases & file tickets",
      "prompt": "You are an issue auditor inspecting the changes in commit HEAD.\nYour task:\n1. Uncover unhandled edge cases, boundary conditions, missing platform support (Windows/Linux/Darwin), or soak test needs in the newly added code.\n2. If real follow-ons or gaps exist, draft structured GitHub issue tickets with clear reproduction steps and acceptance criteria.\n\nEmit a structured report:\n- AUDIT_SUMMARY: unhandled edge cases and risk surfaces\n- TICKETS: list of drafted/filed GitHub issues (#N) with Problem/Today/Better because/Witness"
    }
  },
  {
    "tool": "task",
    "parameters": {
      "subagent_type": "tester",
      "description": "Regression & matrix test execution",
      "prompt": "You are a deterministic tester executing tests for commit HEAD on-device.\nYour task:\n1. Execute on-device package tests: `.\\test.ps1 ./internal/<pkg>/...` (or `go test -v ./internal/<pkg>` and `go vet ./internal/<pkg>`).\n2. Execute regression suites and matrix verification across affected packages.\n3. Observe actual test output and confirm deterministic execution without flaky passes.\n\nEmit a structured verdict:\n- VERDICT: CLAIM_TEST_GREEN | CLAIM_TEST_FAILED\n- EVIDENCE: on-device test command + exact exit code\n- TEST_SUMMARY: passed/failed count and execution duration"
    }
  }
]
```

Inspect the returned subagent verdicts. If `DEFECT_DETECTED` or tests fail, address the identified defects and re-verify.

---

## Step 4 — Auto-Ticket Discovered Issues by Default

Do not leave discovered edge cases, platform gaps, or follow-ons in chat text.
By default, file durable GitHub tickets for all identified follow-on work:

```bash
# Fan out 3..50+ follow-on QA/dogfood/edge-case issues from the spine commit:
fak-dev issue fanout --title "<Feature/Fix Title>" --leaf <leaf> --spine <commit-sha> --json

# Or create targeted follow-on issue via GitHub CLI:
gh issue create --title "<type>(<scope>): <problem>" --body "..."
```

Every ticket must state:
- **Problem**: the specific unhandled case or gap discovered.
- **Today**: current behavior or limitation.
- **Better because**: concrete benefit of addressing it.
- **Witness**: the command, test, or artifact that will prove the fix.

---

## Step 5 — Final Landing & Receipt

Check that no residual remains before push:

```bash
dos review origin/main..HEAD
```

Confirm `has_residual: false`.
Emit the final non-forgeable receipt:
- Commit SHA: `<sha>`
- Diff Witness: `diff-witnessed`
- Test Status: `CLAIM_TEST_GREEN`
- Cross-Validation: `CONFIRMED_VALID`
- Discovered Issues: `#<issue_numbers>`
