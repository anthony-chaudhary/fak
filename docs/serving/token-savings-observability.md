# Token-saving observability

`fak serve` and `fak guard` expose a bounded `token_savings` receipt on the existing
read-scoped `GET /debug/vars` endpoint. Loopback callers need no bearer; remote callers
need the read bearer. The receipt contains aggregate counts and bounded reason enums only:
no prompt text, tool arguments, paths, traces, or tool results.

```powershell
(Invoke-RestMethod http://127.0.0.1:8080/debug/vars).token_savings |
  ConvertTo-Json -Depth 6
```

Each default-on lever reports:

- `configured`: whether the resolved gateway posture armed it;
- `state`: `active`, `ready` (armed but not yet exercised), `bypassed`, or `off`;
- `reason` and `bail_reasons`: why it fired or safely returned identity;
- `fired`, `units`, and measured `saved_bytes` or `saved_tokens` where that
  transform has a truthful effect measure;
- `rollback`: the exact flag or environment switch that restores the baseline.

The native MCP filter additionally emits the same receipt in every `tools/list` response
at `_meta["fak/tool_filter"]`. Its active/control proof is executable as
`go test ./internal/gateway -run TestNativeMCPFilterABProof`: the active arm must reduce
exact descriptor bytes, the `FAK_ABLATE_MCP_TOOL_FILTER=1` arm must restore the full list,
held intent queries must retain recall, and an `--expose`-hidden tool must remain both
undiscoverable and uncallable.

Interpret `ready/not_observed` as **not yet proven to have saved anything in this process**,
not as a win. A `bypassed` state is an intentional safe bailout: fak kept the original
request/tool surface and paid the token cost rather than risking a missing capability or
an ambiguous rewrite.
