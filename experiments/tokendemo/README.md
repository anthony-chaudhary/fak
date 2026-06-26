# tokendemo served-read cache witness (#877)

Archived gateway-visible evidence that a repeated same-file read sent through fak's
**served boundary** — the HTTP and MCP surfaces a non-Go agent would use — is answered
by fak's cache after the first read, not by rerunning the real read engine.

This closes the integration gap left by `cmd/tokendemo -timing`, which drives
`kernel.Syscall` in-process. The witness here crosses a real `httptest` gateway (the
same `Handler()` as `fak serve`) over `POST /v1/fak/syscall` and `POST /mcp`, against
the real confined-filesystem read engine (`fakread`).

## Regenerate

```bash
go run ./cmd/tokendemo -served        # human table
go run ./cmd/tokendemo -served-json   # the archived schema (this file's source)
go test ./cmd/tokendemo/ -run Served  # the gate
```

`served-read-cache-877.json` is one captured `-served-json` run
(`schema: fak.tokendemo.served-read-cache.v1`). Timing fields are point-in-time; the
counts are invariant.

## What the witness shows

8 served calls (4 over HTTP, 4 over MCP) of one same-file `Read`:

| evidence | value | meaning |
| --- | --- | --- |
| `raw_engine_calls` | 8 | the raw baseline reruns `os.ReadFile` every call |
| `fak_engine_calls` | 1 | only the first served call reaches `fakread` |
| `vdso_hits` | 7 | every repeat is served from the cache |
| call #1 `fak_source` | `engine` (`fakread`) | first read goes to the engine |
| calls #2-8 `served_by` / `tier` | `vdso` / `2` | repeats return tier-2 cache metadata |

Gateway-visible `/metrics` rows (not just local counters):

- `fak_kernel_submits_total 8`, `fak_kernel_engine_calls_total 1`, `fak_kernel_vdso_hits_total 7`
- `fak_gateway_http_requests_total{route="/v1/fak/syscall",method="POST",status="200"} 4`
- `fak_gateway_http_requests_total{route="/mcp",method="POST",status="200"} 4`
- `fak_gateway_operations_total{operation="syscall",verdict="ALLOW",...,by="vdso"} 7`
- `fak_gateway_vdso_hit_ratio 0.875`

The MCP rows prove the warmed tier-2 entry filled over HTTP is reused over MCP on the
same gateway.

## Honesty bounds

- **Not a wall-clock win.** For a trivial local read the served path (HTTP/MCP
  round-trip) is *slower* than a raw `os.ReadFile`; `raw_minus_fak_ns` is negative and
  is reported as-is. The win is engine-call collapse (7 of 8 reads never touch the
  engine), not latency.
- **Tool-side only.** Cached bytes are still returned to the caller; this does not claim
  model-context token savings (no KV/context mechanism removes them) or that native
  Claude Code tools under `fak guard -- claude .` are routed through this cache.
