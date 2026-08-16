# Local native harness web UI

`harnesswebdemo` is a separately built browser product over fak's public `pkg/harnesskit` contract. It serves embedded assets on loopback, submits deterministic offline turns, renders semantic message start/delta/completion, tools, artifacts, approvals, and typed failures, and reconnects from an exclusive sequence cursor. It does not import `internal/`, parse terminal output, or expose a remote listener.

```text
go run ./cmd/harnesswebdemo -selfcheck
go run ./cmd/harnesswebdemo
# open http://127.0.0.1:8787
```

The page includes three offline operating scenarios:

- **Tool run:** message → tool progress → artifact → completion.
- **Approval run:** a scoped approval request; approve or deny exactly once before the run continues.
- **Failure run:** a typed retryable error and terminal failed run.

“Switch skin” proves that branding/layout tokens can change without rebuilding the kernel. The selfcheck captures the HTML, drives all three scenarios over HTTP, validates `fak.harness.run/v1` envelopes, resumes after a cursor, and reports a SHA-256 receipt.

This is a practical local product-development surface, not yet a complete coding-harness replacement: a live model/tool adapter, durable sessions across process restart, authenticated non-loopback deployment, full screen-reader/browser automation, and the independent second implementation remain follow-ons under #6790.

Captured normal, approval, and failure renders are indexed in [the operating-envelope witness](_witnesses/harness-web-demo/README.md).
## Native coding workspace capability

Launch the gateway with its existing bounded coding catalog, then point the browser at it:

```text
fak serve --native --native-code-workspace <workspace> [...model flags]
go run ./cmd/harnesswebdemo -fak-url http://127.0.0.1:8080
```

`GET /healthz` exposes `native_code_workspace.armed` and the six tool names without
returning the private workspace path. The browser probes that contract at startup and
projects it through `GET /api/status`. This proves that the connected session is capable
of bounded coding; the full browser-driven read/patch/diff/test witness remains #6962.
