---
title: "Put fak in front of an agent-memory system (mem0 / OpenMemory / MCP)"
description: "fak is the reference monitor beneath an agent-memory store, not a competitor to it. Wire mem0's OpenMemory MCP server behind fak guard and every add/search/delete crosses a default-deny capability floor: oversized writes refused, secret-shaped writes refused, a prompt-injected delete_all refused, and every recalled memory trust-gated before it re-enters context. Zero mem0 code."
---

# Run your memory store through fak

[mem0](https://github.com/mem0ai/mem0), [Letta](https://docs.letta.com), and
[Zep/Graphiti](https://github.com/getzep/graphiti) own the part agents actually buy:
fact-extraction, embeddings, semantic recall, per-user scoping. They do **not** own a
*write barrier*. Every fact an LLM extracts is promoted to the durable store by default,
a `delete_all` fires on whoever asks, and a recalled memory flows back into context with
no trust check. mem0's own docs concede there is no request middleware and no pre-write
hook.

`fak` is the layer beneath: a **reference monitor at the agent's syscall boundary**. It
keeps the store's retrieval engine intact and adds a default-deny floor in front of every
memory operation. This guide wires mem0's local **OpenMemory MCP server** behind
`fak guard` so the gate is invisible to your agent and you change no mem0 code.

> **fak does not replace your memory store.** It has no embedder, no vector search, no
> fact-extraction pass. The play is composition: mem0 keeps recall; fak adds the gate, the
> audit trail, and the trust screen on the way back in. The durability-classification
> *thesis* behind this lives in
> [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md); this page is the runnable
> gate.

---

## What the gate gives you

Three things mem0 has no mechanism for, with the policy in
[`examples/mem0-openmemory-policy.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mem0-openmemory-policy.json):

1. **Fail-closed writes.** `add_memories` is allowed, but an oversized payload (a context
   dump masquerading as a "fact") is refused `OVERSIZE`, and a secret-shaped payload (a
   private-key header, an `AKIA…` / `sk-…` / `xoxb-…` token) is refused `SECRET_EXFIL`
   before it can be laundered into durable memory.
2. **Capability-gated destruction.** `delete_all_memories`, `delete_memory`, and
   `reset_memory` are denied by structure. A prompt-injected agent that is *talked into*
   wiping the store still can't — the refusal is on the tool **name**, independent of what
   the model was convinced to do.
3. **Trust-gated read-back.** This one is automatic and needs no policy. Under
   `fak guard`, a `search_memory` **result** is a tool result, so it routes through the
   kernel's result-side floor (`/v1/fak/admit`) and is quarantined if it carries an
   injection or a secret — *before* the recalled bytes reach the model. mem0 returns stored
   text with no screen; this closes indirect prompt-injection through the memory store
   (OWASP **Memory Poisoning T1**), and it does not depend on any classifier judgment.

---

## Prove the floor before you wire anything (no model, no mem0, no key)

The floor is the same code whether a model is in the loop or not, so you can verify every
verdict offline with `fak preflight`. With [Go 1.26+](https://go.dev/dl/) and a clone:

```bash
go build -o fak ./cmd/fak
POL=examples/mem0-openmemory-policy.json

fak policy --check "$POL"                                                              # manifest valid

fak preflight --policy "$POL" --tool search_memory       --args '{"query":"prefs"}'   # ALLOW
fak preflight --policy "$POL" --tool add_memories        --args '{"text":"the user prefers afternoon meetings"}'  # ALLOW
fak preflight --policy "$POL" --tool delete_all_memories --args '{}'                   # DENY  POLICY_BLOCK
fak preflight --policy "$POL" --tool delete_memory       --args '{"memory_id":"x"}'   # DENY  POLICY_BLOCK
fak preflight --policy "$POL" --tool add_memories        --args '{"text":"my key AKIAIOSFODNN7EXAMPLE"}'          # DENY  SECRET_EXFIL
fak preflight --policy "$POL" --tool not_a_listed_tool   --args '{}'                   # DENY  DEFAULT_DENY
```

Captured verbatim from the binary (the `add_memories` oversize case uses a 9 KB `text`):

```
verdict=ALLOW reason=NONE          by=monitor    # search_memory
verdict=ALLOW reason=NONE          by=monitor    # add_memories (normal fact)
verdict=DENY  reason=POLICY_BLOCK  by=monitor    # delete_all_memories
verdict=DENY  reason=POLICY_BLOCK  by=monitor    # delete_memory
verdict=DENY  reason=OVERSIZE      by=monitor    # add_memories (9 KB text)
verdict=DENY  reason=SECRET_EXFIL  by=monitor    # add_memories (AKIA… token)
verdict=DENY  reason=DEFAULT_DENY  by=monitor    # an unlisted tool
```

A refusal cites a **named, closed-vocabulary reason**, not a model judgment — so the gate
is a property you can diff and test, not a prompt you hope holds.

---

## Wire it live, under `fak guard`

1. Run mem0's OpenMemory MCP server locally (FastAPI on `:8765`; Docker compose, local
   Qdrant — see [mem0's OpenMemory docs](https://docs.mem0.ai/openmemory/overview)). It
   exposes the tools `add_memories`, `search_memory`, `list_memories`,
   `delete_all_memories`.

2. Add it to your agent as an MCP server (the agent's own tool), exactly as you would
   without fak.

3. Launch the agent through the gate, with this policy as the floor:

   ```bash
   fak guard --policy examples/mem0-openmemory-policy.json -- claude
   ```

Now every memory tool call the model proposes surfaces on the proxied stream, crosses
`k.Decide`, and is dropped / repaired / allowed before your agent dispatches it to
OpenMemory. On exit, `fak guard` prints the tally:

```
fak guard: 47 kernel decision(s) — 44 allowed, 3 denied, 0 repaired, 1 quarantined
  blocked: POLICY_BLOCK   x2     # a delete_all the model proposed
  blocked: SECRET_EXFIL   x1     # a token it tried to memorize
```

For a durable, hash-chained record of every memory decision, add an audit journal:

```bash
FAK_AUDIT_JOURNAL=~/mem-audit.jsonl fak guard --policy examples/mem0-openmemory-policy.json -- claude
```

---

## Honest limits — read before you rely on it

- **The store must be the agent's own MCP tool on the proxied stream.** fak gates the call
  because it appears as a model-proposed `tool_use` that the proxy sees. If your agent
  calls mem0 **out-of-band** (the Python/TS SDK directly, not as a model tool call), that
  write never crosses `fak guard` — it is invisible to the gate. fak is not an outbound MCP
  client today; a transparent outbound-MCP proxy would be net-new.
- **Match the tool names your harness reports.** This policy uses the bare OpenMemory tool
  names. Some harnesses namespace MCP tools (Claude Code surfaces them as
  `mcp__<server>__<tool>`). Run once with `fak guard --log -` and read the `tool=` field in
  the `gateway_operation` lines, then set `allow` / `deny` to those exact strings (verify
  any name with `fak preflight`). The floor matches names exactly; a name it doesn't know
  falls to `DEFAULT_DENY`, which is the safe direction.
- **The floor is replace-not-merge.** This manifest *is* the whole capability floor, so it
  includes the standard coding-agent tool set alongside the gated memory tools. Start from
  it (or `fak policy --dump`) and edit; don't expect it to layer on top of another policy.
- **This gates the write; it does not classify durability.** Refusing `delete_all` and
  oversized/secret writes is the cheap, high-value half. The deeper move — an
  *expire-by-default durability barrier* that refuses to promote a transient fact ("it's
  3pm") to durable memory at all — is a separate integration that leans on a write-time
  classifier (today a lexical prior, not an oracle). It belongs in audit/warn mode first.
  The thesis and the buildable ladder are in
  [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md).

---

## Where this sits among the integration paths

This page is the cheapest of three ways fak composes with a memory store:

| Path | Effort | What it adds | This page |
|---|---|---|---|
| **Proxy / MCP interception** | **S** | The gate above, zero store-side code | ✅ |
| Durability write-barrier in front of `add()` | M | Expire-by-default promotion gate (warn-first; leans on the classifier) | [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md) |
| Store as a `memq` backend | L | fak's deterministic, caps-gated, no-hard-delete algebra over the store's recall | `internal/memq` |

The honest framing across all three: fak is **security and governance over an unchanged
memory store**, not better recall. For a benign-drift threat model it is overhead; for a
memory-poisoning, compromised-agent, or `delete_all` threat model it is the refusal an
attacker can't talk the agent out of.

### Discover fak-native memory tools

If the agent needs to discover fak's own memory surface instead of reading this page,
query the self-feature catalog:

```bash
fak feature query memory --detail fak_memory_run --json
```

Over MCP, call `fak_feature_query` with `{"query":"memory"}`. It returns lightweight
cards for `fak_memory_drivers`, `fak_memory_explain`, `fak_memory_run`, and the registered
memory drivers. Discovery is read-only; `fak_memory_run` still defaults to `apply=false`,
so effectful memory changes remain proposals unless the caller explicitly supplies the
apply capability. The planning spine is
[`SELF-FEATURE-QUERY-SPINE-2026-06-30.md`](../notes/SELF-FEATURE-QUERY-SPINE-2026-06-30.md).

---

## Cross-references

- [`examples/mem0-openmemory-policy.json`](https://github.com/anthony-chaudhary/fak/blob/main/examples/mem0-openmemory-policy.json) — the floor this page deploys.
- [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md) — why context ≠ memory, and the expire-by-default durability gate.
- [`claude.md`](claude.md) — the `fak guard` integration this builds on.
- [`debugging.md`](debugging.md) — why was a memory call denied/transformed? Reproduce it offline with `fak preflight --explain`.
- [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the manifest schema (allow / deny / arg_rules / self_modify_globs).
