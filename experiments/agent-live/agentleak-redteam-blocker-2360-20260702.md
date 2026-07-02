# Issue #2360 Agent Leak Red-Team Witness Blocker

Date: 2026-07-02
Ticket: https://github.com/anthony-chaudhary/fak/issues/2360
Outcome: BLOCKED. The final synthetic canary witness suite was not produced.

## Authoritative dependency state

Fetched with:

```text
gh issue view 2360 --json number,title,body,state,labels,url
gh issue view 2356 --json number,title,state,closedAt,url
gh issue view 2357 --json number,title,state,closedAt,url
gh issue view 2358 --json number,title,state,closedAt,url
gh issue view 2359 --json number,title,state,closedAt,url
gh issue view 2361 --json number,title,state,closedAt,url
gh issue view 2362 --json number,title,state,closedAt,url
gh issue view 2363 --json number,title,state,closedAt,url
```

All required dependency tickets were still OPEN:

| Issue | State | Title |
|---|---|---|
| #2356 | OPEN | feat(toolprocgate): add AgentRun spawn envelope and broker core |
| #2357 | OPEN | feat(runtime): contain descendant process trees under the agent run boundary |
| #2358 | OPEN | feat(policy): default-deny inherited env, secret, filesystem, and network scope for child tools |
| #2359 | OPEN | feat(resultgate): quarantine child and subagent output before context or logs |
| #2361 | OPEN | feat(obs): surface agent leak-prevention events and descendant tree state |
| #2362 | OPEN | feat(dispatch): broker dispatch, account, and resume worker launches under AgentRun |
| #2363 | OPEN | feat(gateway): broker MCP and ToolExec child launches under AgentRun |

## Local evidence

- Current branch: `main`.
- HEAD at inspection: `40f73c83 docs(agentleak): define AgentRun boundary (#2355) (fak docs)`.
- `git log --oneline --grep "#2356\|#2357\|#2358\|#2359\|#2361\|#2362\|#2363\|#2360"` found no local resolving commits for the dependency set.
- The working tree contains many concurrent dirty and untracked files in the likely dependency areas. Those edits may be prerequisite work in progress, but they are not landed evidence and cannot be swept into a #2360 final witness commit.

## Acceptance command result

Command:

```text
go test ./internal/toolprocgate ./internal/policy ./internal/leakcheck ./internal/gateway ./cmd/fak -run "Test.*AgentLeak|Test.*Canary|Test.*Subagent|Test.*ToolProc"
```

Observed after an import-only compile repair in `cmd/fak/guard_test.go`:

```text
ok   github.com/anthony-chaudhary/fak/internal/toolprocgate
FAIL github.com/anthony-chaudhary/fak/internal/policy [build failed]
ok   github.com/anthony-chaudhary/fak/internal/leakcheck [no tests to run]
ok   github.com/anthony-chaudhary/fak/internal/gateway [no tests to run]
ok   github.com/anthony-chaudhary/fak/cmd/fak

internal/policy/inherited_test.go: ResolveLaunch undefined on *InheritedTable
```

Earlier in the same inspection, `internal/policy/inherited.go` was momentarily absent while `policy.go` referenced `InheritedRule`, `InheritedTable`, and `compileInheritedCapabilities`; the tree changed concurrently and the file reappeared. The remaining `ResolveLaunch` mismatch is still a hard #2358 blocker.

## Missing prerequisites blocking #2360

- #2358 is red in the acceptance package: inherited capability tests and implementation disagree on the public method name, so the package does not build.
- #2356 is not landed as a resolving commit; spawn broker code and guard adapter tests are present only as dirty local work.
- #2357 has shipped earlier process-tree binding pieces, but the issue remains open and no final AgentRun descendant containment closure is present in history.
- #2359 output-admission code is present only as dirty local work, not a landed prerequisite.
- #2361 leak-prevention observability/reporting code is present only as dirty local work, not a landed prerequisite.
- #2362 dispatch/account/resume broker adapter code is present only as dirty local work, not a landed prerequisite.
- #2363 gateway/MCP tool-process adapter code is present only as dirty local work, not a landed prerequisite.

## Safe preparatory work

Performed:

- Added missing `os/exec` and `internal/toolprocgate` imports to `cmd/fak/guard_test.go` so the in-progress guard spawn-broker tests can compile once the policy blocker is resolved.

Not added:

- No final synthetic canary suite was added. A valid #2360 suite should wait until #2356, #2357, #2358, #2359, #2361, #2362, and #2363 are landed and green, then exercise the registered child/subagent/tool-process channels instead of local in-progress stubs.
