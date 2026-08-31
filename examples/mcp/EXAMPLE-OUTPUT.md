# Example output

A real run of [`verify.py`](verify.py) on a clean checkout — **no model, key, GPU, or
network**; the MCP stdio transport needs no listener and no auth. The capability floor is
[`../dev-agent-policy.json`](../dev-agent-policy.json). A `✓` means the check matched
expectation. Reproduce: `python examples/mcp/verify.py`.

```
fak — MCP stdio adjudication proof  newline-delimited JSON-RPC over stdin/stdout · no model, key, or GPU
  floor: examples\dev-agent-policy.json

  ✓ A  initialize handshake  serverInfo=fak-gateway · protocol 2024-11-05
  ✓ B  tools/list exposes schema-light bootstrap tools  fak_adjudicate, fak_syscall, fak_tools_search; fak_admit deferred
  ✓ C  fak_tools_search discovers deferred fak_admit  routable without eager schema load
  ✓ D  fak_admit screens a benign result  DEFER · result OK · IFC taint recorded
  ✓ E  fak_adjudicate refuses git_push  DENY (POLICY_BLOCK/RETRYABLE)
  ✓ F  fak_adjudicate allows git_status  ALLOW

summary: PASS  ·  the kernel admitted a client result and adjudicated proposed calls over the MCP stdio transport, with no model, key, or GPU.
  this is the path your editor's MCP client uses (.mcp.json wires `fak serve --stdio`).
$ echo $?
0
```

## The frames behind it (abridged)

The transport is **newline-delimited JSON-RPC 2.0** over stdin/stdout — one compact JSON
object per line, no `Content-Length` headers. stdout carries only protocol frames (the
server's logs go to stderr). These are the actual request/response pairs:

```jsonc
// A. handshake — negotiate a protocol version, name the server
>> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"py-verifier","version":"0"}}}
<< {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"tools":{}},"protocolVersion":"2024-11-05","serverInfo":{"name":"fak-gateway","version":"0.31.0"}}}
>> {"jsonrpc":"2.0","method":"notifications/initialized"}        // a notification — no reply

// B. bootstrap discovery — heavy fak_admit schema is intentionally deferred (#3231)
>> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
<< {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"fak_adjudicate",...},{"name":"fak_syscall",...},{"name":"fak_tools_search",...},{"name":"fak_read",...}]}}

// C. on-demand discovery still finds the deferred tool
>> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fak_tools_search","arguments":{"query":"fak_admit","detail_level":"name"}}}
<< {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"tools\":[{\"name\":\"fak_admit\"},...]}"}],"isError":false}}

// D. a benign client-run result traverses result admission
>> {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fak_admit","arguments":{"tool":"git_status","result":{"status":"ok","source":"mcp-stdio-verifier"},"trace_id":"mcp-stdio-verifier"}}}
<< {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"{\"verdict\":{\"kind\":\"DEFER\",\"by\":\"toolprocgate\"},\"result\":{\"status\":\"OK\",\"content\":\"{...}\",\"meta\":{\"ifc_taint\":\"tainted\"}},\"trace_id\":\"mcp-stdio-verifier\"}"}],"isError":false}}

// E. a shared-history mutation is refused — DENY as a VALUE, not a JSON-RPC error
>> {"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"fak_adjudicate","arguments":{"tool":"git_push","arguments":{}}}}
<< {"jsonrpc":"2.0","id":5,"result":{"content":[{"type":"text","text":"{\"verdict\":{\"kind\":\"DENY\",\"reason\":\"POLICY_BLOCK\",\"by\":\"monitor\",\"disposition\":\"RETRYABLE\"},\"trace_id\":\"gw-1\"}"}],"isError":false}}

// F. a read is permitted — the floor is not a blanket deny
>> {"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"fak_adjudicate","arguments":{"tool":"git_status","arguments":{}}}}
<< {"jsonrpc":"2.0","id":6,"result":{"content":[{"type":"text","text":"{\"verdict\":{\"kind\":\"ALLOW\",\"by\":\"monitor\"},\"trace_id\":\"gw-2\"}"}],"isError":false}}
```

The server version and generated adjudication `trace_id` (`gw-N`) can vary by build/run; the checks accept the supported protocol set and ignore generated trace IDs. A `DENY`
is **deny-as-value** — a normal, successful tool result (`isError:false`) whose embedded
`verdict.kind` is `DENY`; the JSON-RPC `error` channel is reserved for protocol faults
(bad method, malformed frame), never a policy refusal.
