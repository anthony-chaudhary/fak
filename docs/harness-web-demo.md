---
title: "Local native harness web UI"
description: "harnesswebdemo is the local operator home over fak's public pkg/harnesskit contract. Its first screen shows current agent/session totals, recent runs,"
---
# Local native harness web UI

`harnesswebdemo` is the local operator home over fak's public `pkg/harnesskit` contract. Its first screen shows current agent/session totals, recent runs, the canonical goal registry, and direct links to the Web gateway, health, sessions, loops, fleet, tasks, metrics, and diagnostics. The run composer follows that operational overview instead of replacing it.

The same loopback product submits deterministic offline turns, renders semantic message start/delta/completion, tools, artifacts, approvals, and typed failures, and reconnects from an exclusive sequence cursor. It does not parse terminal output or expose a remote listener.

```text
fak harness web --selfcheck
fak harness web
# open http://127.0.0.1:8787

# source-checkout compatibility wrapper:
go run ./cmd/harnesswebdemo --selfcheck
```

The Run agent section includes three offline operating scenarios:

- **Tool run:** message → tool progress → artifact → completion.
- **Approval run:** a scoped approval request; approve or deny exactly once before the run continues.
- **Failure run:** a typed retryable error and terminal failed run.

The Theme control proves that branding/layout tokens can change without rebuilding the kernel. The selfcheck captures the HTML, verifies the overview's run/goal/dashboard projections, drives all three scenarios over HTTP, validates `fak.harness.run/v1` envelopes, resumes after a cursor, and reports a SHA-256 receipt.

This is a practical local product-development surface, not yet a complete coding-harness replacement: a live model/tool adapter, durable sessions across process restart, authenticated non-loopback deployment, full screen-reader/browser automation, and the independent second implementation remain follow-ons under #6790.

Captured normal, approval, and failure renders are indexed in [the operating-envelope witness](_witnesses/harness-web-demo/README.md).
## Native coding workspace capability

Launch the gateway with its existing bounded coding catalog, then point the browser at it:

```text
fak serve --native --native-code-workspace <workspace> [...model flags]
fak harness web -fak-url http://127.0.0.1:8080 -workspace <workspace>
```

`GET /healthz` exposes `native_code_workspace.armed` and the six tool names without
returning the private workspace path. The browser probes that contract at startup and
projects it through `GET /api/status`. That status read also refreshes the gateway's
bounded `/debug/vars` session/fleet projection and reads the local goal registry (the
same path as `fak goal list`; override it with `FAK_GOAL_REGISTRY`). `-workspace`
resolves the same operator-declared root once and exposes only a stable `ws-…` identity
in browser status, never the private path.

The gateway's native catalog remains the sole execution path: Read/Write/Edit/Grep/Glob
are root-confined (including symlink checks), while Bash admits only focused `go test`,
`git diff`, and `git status --short` commands. Shell composition, destructive commands,
and ambient-credential enumeration are typed `COMMAND_DENY` refusals. The browser never
executes a browser-supplied command.

Rollback is exact: stop `fak harness web` and restart without `-fak-url`/`-workspace` for
the deterministic offline product, or stop the workspace-armed `fak serve` process. Use
`fak harness gallery show --id coding-workspace` for the corresponding user-needs pack.


The real edit/test/diff browser receipt and restart capture are archived in [the live coding witness](_witnesses/harness-web-demo/LIVE-CODING-GPT-5.6-SOL-2026-08-15.md). The shipped-binary extraction, temp-built launch, and rollback evidence are archived in [the shipped fak launch witness](_witnesses/harness-web-demo/SHIPPED-FAK-LAUNCH-2026-08-15.md).
