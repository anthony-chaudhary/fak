# MCP tool-result shape (the `SyscallResponse` wire)

The fak gateway is an MCP server (JSON-RPC 2.0, stdio or `POST /mcp`). Six tools
route a proposed call or result through the kernel:
`fak_adjudicate`, `fak_syscall`, `fak_admit`, `fak_changes`, `fak_revoke`,
`fak_context_change`. Every one returns its payload through the **same MCP
tool-result envelope** (`mcpToolResult` in `internal/gateway/mcp.go`):

```json
{
  "content": [{ "type": "text", "text": "<the JSON document, stringified>" }],
  "isError": false
}
```

`isError` is **always `false`**. A deny/quarantine is a successful adjudication —
the outcome lives in the verdict *inside* the `text`, never in `isError`. A
JSON-RPC `error` object is reserved for protocol/build faults (bad params,
unknown tool), not for a refusal.

The `text` field is a JSON-encoded document. For `fak_adjudicate` and
`fak_syscall` (and `fak_admit`) that document is a **`SyscallResponse`**. This
doc specifies that shape; `fak_changes` / `fak_revoke` / `fak_context_change`
return their own response structs (`ChangesResponse`, `RevokeResponse`,
`ContextChangeResponse`) through the identical envelope.

## `SyscallResponse` fields

Defined in `internal/gateway/wire.go`:

| field                | JSON key              | type             | when present                                   |
| -------------------- | --------------------- | ---------------- | ---------------------------------------------- |
| `Verdict`            | `verdict`             | `WireVerdict`    | always                                          |
| `Result`             | `result`              | `ResultEnvelope` | execute path (`fak_syscall` / `fak_admit`) only |
| `RepairedArguments`  | `repaired_arguments`  | raw JSON         | only on a `TRANSFORM` verdict                  |
| `TraceID`            | `trace_id`            | string           | echoed when a trace id is in play              |

### `WireVerdict` (the `verdict` object)

| field         | JSON key       | type                | meaning                                                       |
| ------------- | -------------- | ------------------- | ------------------------------------------------------------- |
| `Kind`        | `kind`         | string              | `ALLOW` \| `DENY` \| `TRANSFORM` \| `QUARANTINE` \| `REQUIRE_WITNESS` \| `DEFER` \| `KIND_<n>` |
| `Reason`      | `reason`       | string (omitempty)  | closed refusal vocabulary, e.g. `POLICY_BLOCK`, `SELF_MODIFY` |
| `By`          | `by`           | string (omitempty)  | which adjudicator decided (forensics)                         |
| `Disposition` | `disposition`  | string (omitempty)  | deny-loopback class: `RETRYABLE` \| `WAIT` \| `ESCALATE` \| `TERMINAL` |
| `Detail`      | `detail`       | `map[string]string` | bounded disclosure (e.g. the offending self-modify `claim`)   |

`reason` is one of the closed core vocabulary (`internal/abi/reasons.go`):
`DEFAULT_DENY`, `POLICY_BLOCK`, `SELF_MODIFY`, `LEASE_HELD`, `TRUST_VIOLATION`,
`MALFORMED`, `MISROUTE`, `RATE_LIMITED`, `SECRET_EXFIL`, `UNWITNESSED`,
`OVERSIZE`, `UNKNOWN_TOOL` (plus out-of-tree `REASON_<n>` codes). `disposition`
is derived from that reason by `kernel.Disposition`: `MISROUTE`/`MALFORMED` →
`RETRYABLE`; `RATE_LIMITED`/`LEASE_HELD` → `WAIT`; `SELF_MODIFY`/`TRUST_VIOLATION`
→ `ESCALATE`; everything else → `TERMINAL`. A `REQUIRE_WITNESS` verdict carries
`ESCALATE` so the client can route it to a witness/human-approval queue.

### `ResultEnvelope` (the `result` object, execute path only)

| field     | JSON key  | type                | meaning                                          |
| --------- | --------- | ------------------- | ------------------------------------------------ |
| `Status`  | `status`  | string              | `OK` \| `ERROR` \| `PENDING` \| `UNKNOWN`        |
| `Content` | `content` | string              | the tool result bytes, resolved (never a `Ref`)  |
| `Meta`    | `meta`    | `map[string]string` | side-band, e.g. `{"admit":"quarantined"}`        |

## One concrete example per verdict class

Each block below is the value of the envelope's `text` field — i.e. the
`SyscallResponse` document a client gets after `JSON.parse`-ing `content[0].text`.

### ALLOW (`fak_syscall` — adjudicated and executed)

```json
{
  "verdict": { "kind": "ALLOW", "by": "tool" },
  "result": {
    "status": "OK",
    "content": "{\"rows\":3}"
  },
  "trace_id": "t-7f3a9c"
}
```

On the adjudicate-only path (`fak_adjudicate`) an ALLOW carries no `result`:

```json
{ "verdict": { "kind": "ALLOW", "by": "tool" }, "trace_id": "t-7f3a9c" }
```

### DENY (refusal as a value — `isError` is still `false`)

```json
{
  "verdict": {
    "kind": "DENY",
    "reason": "SELF_MODIFY",
    "by": "selfmod",
    "disposition": "ESCALATE",
    "detail": { "claim": "fak/internal/kernel/kernel.go" }
  },
  "trace_id": "t-7f3a9c"
}
```

A model-fixable refusal instead loops back as `RETRYABLE`:

```json
{
  "verdict": { "kind": "DENY", "reason": "MISROUTE", "disposition": "RETRYABLE" },
  "trace_id": "t-7f3a9c"
}
```

### TRANSFORM (the call is admitted with repaired canonical arguments)

`repaired_arguments` is raw JSON the client should run **instead of** what it
proposed. It is present only for this verdict kind.

```json
{
  "verdict": { "kind": "TRANSFORM", "by": "canon" },
  "repaired_arguments": { "path": "/srv/data/report.csv", "mode": "r" },
  "trace_id": "t-7f3a9c"
}
```

### QUARANTINE (a poisoned/secret-shaped result was paged out at admit-time)

On the execute path, a result the context-MMU quarantines overrides the submit
verdict: `verdict.kind` becomes `QUARANTINE` and the offending bytes do not
reach context. The `result.meta` carries the admit marker.

```json
{
  "verdict": { "kind": "QUARANTINE", "reason": "SECRET_EXFIL", "disposition": "TERMINAL" },
  "result": {
    "status": "OK",
    "content": "",
    "meta": { "admit": "quarantined" }
  },
  "trace_id": "t-7f3a9c"
}
```

## Notes

- The MCP `arguments` object for `fak_adjudicate`/`fak_syscall` **is** a
  `SyscallRequest` (`{tool, arguments, read_only, trace_id, witness}`); `fak_admit`
  takes a `{tool, result, trace_id, witness}` `AdmitRequest`.
- `REQUIRE_WITNESS` and `DEFER` are valid `verdict.kind` values too; an unknown
  registered restrictive kind renders as `KIND_<n>` and fails closed
  (`disposition: ESCALATE`).
- Live `--stdio` transport capture against a real Claude Code client is not
  included here — that needs an interactive session and is out of scope for this
  doc. The shapes above are taken directly from `wire.go` / `mcp.go`.
