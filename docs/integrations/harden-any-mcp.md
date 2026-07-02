---
title: "Harden any MCP server against tool poisoning with fak"
description: "fak hardens any MCP server against tool poisoning: a context-MMU quarantines poisoned results out of context and a capability allow-list blocks unwired tools."
---

# Harden any MCP server against tool poisoning

MCP is exploding, and **tool poisoning is its named, unsolved problem** (the MCP
Top-10: Tool Poisoning, Memory Poisoning). Today an agent implicitly *trusts the tool
descriptions and tool results* a server hands it — that trust is the attack surface.
`fak` closes it **by structure**: route the server's proposed calls and results through
the kernel's two independent gates and neither a poisoned description nor a poisoned
result can reach the model by arguing past a classifier.

This is the 60-second "harden any MCP server" path. The mechanism is shipped and
verified; this page is the packaging. For the breadth of the MCP surface see
[`README.md`](README.md) and [`../../examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md).

## The two gates

| Vector | What the server does | fak's structural answer | The tool your agent calls |
|---|---|---|---|
| **Poisoned RESULT** | returns bytes carrying a prompt-injection or a leaked secret | the **context-MMU** ([`../../internal/ctxmmu`](https://github.com/anthony-chaudhary/fak/tree/main/internal/ctxmmu)) holds the bytes out of context entirely and swaps in a stub pointer | `fak_admit` (screen a result you ran) or `fak_syscall` (run it through the kernel) |
| **Poisoned DESCRIPTION** | advertises a tool whose description is an injection | the **capability allow-list** — a tool that was never wired into the policy cannot be invoked no matter what its description says | `fak_adjudicate` (verdict before you run) |

An attacker has to beat **two independent gates** rather than fool one classifier. The
floor is the lock (the allow-list) and the containment (the bytes never reach the
model), not detection.

## Drop fak in front of your server (one block)

Add `fak` as an MCP server alongside the one you want to harden. `fak serve --stdio`
speaks newline-delimited JSON-RPC over stdin/stdout (no listener, no auth surface) and
exposes the `fak_*` tools above.

**Claude Code** — copy [`../../examples/mcp/.mcp.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/.mcp.json) to
your project root (it wires `fak serve --stdio`), or:

```bash
claude mcp add fak -- fak serve --stdio --policy examples/dev-agent-policy.json
```

**Cursor** — add the same `mcpServers` block to `.cursor/mcp.json` (project) or
`~/.cursor/mcp.json` (global).

**Any MCP client (HTTP)** — `fak serve --addr 127.0.0.1:8080`, then `POST /mcp`:

```bash
go build -o fak ./cmd/fak
./fak serve --addr 127.0.0.1:8080 --policy examples/dev-agent-policy.json
# then POST JSON-RPC frames to http://127.0.0.1:8080/mcp
```

Then route the server you are hardening through the gates: `fak_adjudicate` a proposed
call before you run it, and `fak_admit` the result before it enters context (or do both
at once with `fak_syscall`). The full input schemas are in `tools/list`; the result
envelope is specified in [`../mcp-tool-result.md`](../mcp-tool-result.md). Adjust the
policy to your own floor — see [`../../POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).

## Prove it: the poisoned-MCP A/B (no model, no key, no network)

```bash
go run ./cmd/poisonedmcpdemo            # -> the before/after table
go run ./cmd/poisonedmcpdemo -json      # -> the same, CI-usable
```

A mock tool-poisoning MCP server returns a tool whose description is a prompt-injection
and results that carry an injection marker and a leaked secret. The "with fak" arm
drives the **real** `internal/ctxmmu` gate (the exact code path `fak_admit` /
`fak_syscall` fold every result through) and the call-side allow-list. The before/after:

```
vector                                     without fak          with fak
result: summarize_doc (injection)          IN CONTEXT           QUARANTINED · TRUST_VIOLATION · trap held out of context
result: lookup_config (leaked secret)      IN CONTEXT           QUARANTINED · SECRET_EXFIL     · trap held out of context
result: search_kb (benign policy)          IN CONTEXT           ALLOWED · (not a blanket block)
description: exfiltrate_creds (poisoned)   may coerce the model DENY · tool not allow-listed (effects gated)
```

The poisoned bytes never reach the model; the benign result is still admitted, so fak is
not a blanket block. The demo's test asserts the trap is quarantined and the benign
control is allowed.

## Honest fences

- The load-bearing guarantee is **containment + the lock**, not detection. The result
  *detector* (the regex floor in `internal/ctxmmu`) is ~100% evadable by design — it is
  a helpful bonus, not the floor. The win for the poisoned-*result* case is the
  deterministic quarantine floor for the payloads it does catch; the durable guarantee
  for *effects* is the allow-list (a never-wired tool can't be invoked no matter what a
  description says).
- This page is the **security / governance wedge**. The KV-reuse savings are self-host
  only and are deliberately not the pitch here — see
  [`../../README.md`](https://github.com/anthony-chaudhary/fak/blob/main/README.md) for that axis.
- fak does not today *transparently* proxy an arbitrary MCP server's wire; it provides
  the gate tools (`fak_adjudicate` / `fak_admit` / `fak_syscall`) the agent calls to put
  the kernel's two gates in front of every call and result. Add it alongside the server
  you want to harden and route through it.

## Related

- [`../../examples/mcp/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp/README.md) — the one-paste MCP
  setup (`.mcp.json` + the `fak_*` tools) this hardens.
- [`../../examples/mcp-client/client.py`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mcp-client/client.py) — a
  dependency-free reference client that drives all six `fak_*` tools over stdio and HTTP.
- [`../../README.md`](https://github.com/anthony-chaudhary/fak/blob/main/README.md) §"Security: the lock, not the screener" — the
  two-gate framing and the flight-booking A/B this mirrors.
