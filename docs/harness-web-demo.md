# Local native harness web UI spine

`harnesswebdemo` is the smallest separately built browser surface over fak's public `pkg/harnesskit` contract. It serves embedded assets on loopback, submits a deterministic offline turn, renders semantic message/tool events, and reconnects from an exclusive sequence cursor. It does not import `internal/`, parse terminal output, or expose a remote listener.

```text
go run ./cmd/harnesswebdemo -selfcheck
go run ./cmd/harnesswebdemo
# open http://127.0.0.1:8787
```

The selfcheck captures the HTML render, posts a real HTTP run request, validates five `fak.harness.run/v1` envelopes, resumes after cursor 3, and reports a render SHA-256. This is the runnable local spine for #6882, not completion of parent #6790: approvals, failures, accessibility checks, a second skin, and captured live-browser screenshots remain in that parent.
