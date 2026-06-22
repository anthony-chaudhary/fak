# Example output

A real run of [`verify.py`](verify.py) on a clean checkout — **no model, key, GPU, or
network**; the MCP stdio transport needs no listener and no auth. The capability floor is
[`../dev-agent-policy.json`](../dev-agent-policy.json). A `✓` means the check matched
expectation. Reproduce: `python examples/mcp/verify.py`.

```
fak — MCP stdio adjudication proof  newline-delimited JSON-RPC over stdin/stdout · no model, key, or GPU
  floor: examples/dev-agent-policy.json

  ✓ A  initialize handshake  serverInfo=fak-gateway · protocol 2024-11-05
  ✓ B  tools/list exposes the adjudication verbs  fak_adjudicate, fak_admit, fak_syscall (+3 more)
  ✓ C  fak_adjudicate refuses git_push  DENY (POLICY_BLOCK/TERMINAL)
  ✓ D  fak_adjudicate allows git_status  ALLOW

summary: PASS  ·  the kernel adjudicated every proposed call over the MCP stdio transport, with no model, key, or GPU.
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

// B. discover the tools your agent will call
>> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
<< {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"fak_adjudicate",...},{"name":"fak_syscall",...},{"name":"fak_admit",...},{"name":"fak_changes",...},{"name":"fak_revoke",...},{"name":"fak_context_change",...}]}}

// C. a shared-history mutation is refused — DENY as a VALUE, not a JSON-RPC error
>> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fak_adjudicate","arguments":{"tool":"git_push","arguments":{}}}}
<< {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"verdict\":{\"kind\":\"DENY\",\"reason\":\"POLICY_BLOCK\",\"by\":\"monitor\",\"disposition\":\"TERMINAL\"},\"trace_id\":\"gw-1\"}"}],"isError":false}}

// D. a read is permitted — the floor is not a blanket deny
>> {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fak_adjudicate","arguments":{"tool":"git_status","arguments":{}}}}
<< {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"{\"verdict\":{\"kind\":\"ALLOW\",\"by\":\"monitor\"},\"trace_id\":\"gw-2\"}"}],"isError":false}}
```

The only non-deterministic field is `trace_id` (`gw-N`), which the checks ignore. A `DENY`
is **deny-as-value** — a normal, successful tool result (`isError:false`) whose embedded
`verdict.kind` is `DENY`; the JSON-RPC `error` channel is reserved for protocol faults
(bad method, malformed frame), never a policy refusal.
