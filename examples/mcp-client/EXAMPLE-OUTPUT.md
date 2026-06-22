# Example output

A real run of [`client.py`](client.py) on a clean checkout — **no model, key,
GPU, or network**. The MCP stdio transport needs no listener and no auth; the
capability floor is [`../dev-agent-policy.json`](../dev-agent-policy.json).
Reproduce: `python3 examples/mcp-client/client.py` (or `examples/mcp-client/run.sh`).

```
fak — third-party MCP client walkthrough  stdio · fak serve --stdio · JSON-RPC 2.0 · stdlib only
  floor: examples/dev-agent-policy.json

initialize  server=fak-gateway v0.31.0 · protocol negotiated → 2024-11-05
tools/list  6 tools: fak_adjudicate, fak_syscall, fak_admit, fak_changes, fak_revoke, fak_context_change

calling each tool with a small payload (showing the response shape)

  fak_adjudicate  verdict only, no execution (the production path for a client that runs its own tools)
    → args   {"tool":"git_status","arguments":{}}
    → verdict{"kind":"ALLOW","by":"monitor"}
    → result {"verdict":{"kind":"ALLOW","by":"monitor"},"trace_id":"gw-1"}

  fak_syscall  adjudicate AND execute through the kernel (returns verdict + admitted result)
    → args   {"tool":"git_status","arguments":{},"read_only":true}
    → verdict{"kind":"ALLOW","by":"monitor"}
    → result {"verdict":{"kind":"ALLOW","by":"monitor"},"result":{"status":"OK","content":"{\"tool\":\"git_status\",\"engi…

  fak_admit  screen a result YOU executed through the result-side stack before it enters context
    → args   {"tool":"web_fetch","result":{"text":"hello from an external tool"}}
    → verdict{"kind":"DEFER","by":"normgate"}
    → result {"verdict":{"kind":"DEFER","by":"normgate"},"result":{"status":"OK","content":"{\"text\": \"hello from an ext…

  fak_changes  drain the cross-agent change feed (typed mutations + revocations since a cursor)
    → args   {"since":0}
    → result {"events":[],"cursor":0}

  fak_revoke  refute a world-state witness; entries admitted under it are evicted fleet-wide
    → args   {"witness":"sha256:deadbeefcafef00d"}
    → result {"witness":"sha256:deadbeefcafef00d","evicted":0,"trust_epoch":1}

  fak_context_change  negative-only recall mutation (needs a real recall image — see note below)
    → args   {"image_dir":"examples/mcp-client/no-such-image","step":1,"reason":"walkthrough probe"}
    → error  JSON-RPC error code=-32602 "load core image: open examples/mcp-client/no-such-image/manifest.json: no such file or directory…"

done  ·  all six tools answered over stdio. Five returned a SyscallResponse (deny-as-value lives in verdict.kind); fak_context_change used the JSON-RPC error channel because no recall image was supplied — exactly the result-vs-error split in docs/mcp-tool-result.md.
$ echo $?
0
```

## Reading the output

- **`initialize`** — the handshake negotiated `2024-11-05` (the version the client
  offered, which the server supports) and named the server `fak-gateway`.
- **`tools/list`** — all six tools are discoverable.
- **`fak_adjudicate` / `fak_syscall`** — `git_status` is a read; the floor returns
  `ALLOW`. `fak_syscall` also executes it and returns the admitted `result`.
- **`fak_admit`** — a result *you* executed is screened through the result-side
  stack; here it returns through the normal tool-result channel.
- **`fak_changes` / `fak_revoke`** — the change feed is empty (cursor `0`) and the
  unknown witness evicts nothing (`evicted:0`) on a fresh server — the honest
  shape of each.
- **`fak_context_change`** — given a path with **no** recall image, this returns a
  JSON-RPC **error** (`-32602`), not a deny-as-value result. That is the contract:
  the error channel is for protocol/build faults; a policy refusal would instead
  be a successful result with `verdict.kind = DENY`. Point `image_dir` at a real
  persisted recall image to see it succeed.

The only non-deterministic field is `trace_id` (`gw-N`). The transport path is
identical over HTTP — run `fak serve --addr 127.0.0.1:8080` and add
`--http http://127.0.0.1:8080/mcp` to drive the same six tools over `POST /mcp`.
The full wire shape of every field is [`../../docs/mcp-tool-result.md`](../../docs/mcp-tool-result.md).

> Captured on a POSIX host; on Windows the paths print with `\` separators and
> the `fak_context_change` error reads "The system cannot find the path
> specified" — the frames are otherwise identical.
