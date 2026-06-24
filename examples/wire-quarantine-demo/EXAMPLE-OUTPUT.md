# Example output

A captured run of `./run.sh` — `fak serve` with the offline mock planner (no model, key,
or GPU), `demo.py` POSTing three client-produced tool results to `/v1/fak/admit`. A clean
result is admitted unchanged; a secret-shaped and an injection-shaped result are each
quarantined and paged out, server-side. A `✓` means the verdict matched expectation.
Reproduce: `./examples/wire-quarantine-demo/run.sh`.

```
fak — wire-side result quarantine demo  POST /v1/fak/admit · kernel=http://127.0.0.1:8080 · no model, key, or GPU
  an UNTRUSTED client POSTs a tool RESULT it produced · the kernel SCREENS it server-side · a ✓ means the verdict matched expectation

  ✓ CLEAN result admitted        read_file → DEFER  content passed through  ifc_taint=trusted
  ✓ SECRET result quarantined     fetch_url → QUARANTINE (SECRET_EXFIL)  paged_out=ng-q1  ifc_taint=quarantined (rose)
  ✓ INJECTION result quarantined  fetch_url → QUARANTINE (TRUST_VIOLATION)  paged_out=ng-q2

summary: wire quarantine test passed  ·  clean result admitted · secret + injection quarantined and paged out · IFC ledger raised
  the load-bearing result: an UNTRUSTED client's own tool result was screened SERVER-SIDE.
  The secret and the injection never reach the model's context — the client never had to be trusted.
```

Exit code: `0` (CI-usable).

## The raw wire — what the kernel actually returns

The demo asserts against the JSON body of `POST /v1/fak/admit`. The two responses below
are captured verbatim from a fresh server (`curl … | python -m json.tool`).

**A clean result is admitted unchanged** — verdict `DEFER` (no admitter objected), the
content passes through, and the trace's IFC stamp stays `trusted`:

```json
{
    "verdict": { "kind": "DEFER", "by": "normgate" },
    "result": {
        "status": "OK",
        "content": "{\"text\":\"hello world, the build is green\"}",
        "meta": { "ifc_taint": "trusted" }
    },
    "trace_id": "wire-quarantine-demo"
}
```

**A secret-shaped result is quarantined and paged out** — verdict `QUARANTINE`
(`SECRET_EXFIL`); the in-context `content` is replaced with an opaque stub (the
`sk-live…` key is **gone** — it never reaches the model); `meta.quarantine_id` is the
paged-out pointer; and `meta.ifc_taint` rose to `quarantined` (the trace's IFC ledger was
raised):

```json
{
    "verdict": { "kind": "QUARANTINE", "reason": "SECRET_EXFIL", "by": "normgate" },
    "result": {
        "status": "OK",
        "content": "{\"_note\":\"obfuscated threat caught on normalized view\",\"_quarantined\":true,\"by\":\"normgate\",\"id\":\"ng-q1\",\"len\":82,\"reason\":\"SECRET_EXFIL\"}",
        "meta": {
            "admit": "quarantined",
            "ifc_taint": "quarantined",
            "normgate": "quarantined",
            "quarantine_id": "ng-q1"
        }
    },
    "trace_id": "wire-quarantine-demo"
}
```

The injection-shaped result is identical in shape, with `reason` `TRUST_VIOLATION`.

When the client omits `trace_id`, the gateway mints one and returns it both in the
`X-Trace-Id` response header and the body's `trace_id` — so a stateless client can still
correlate a session's taint without managing ids itself.
