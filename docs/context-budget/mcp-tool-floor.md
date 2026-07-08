# MCP tool-schema floor — committed baseline (#3230)

Part of epic **#3229** (shrink the always-sent context budget). This is the
measured baseline the reduction levers ratchet down.

## What this number is

Every tool fak's MCP server advertises ships its full JSON schema in every
`tools/list`, and that set is re-sent on **every** turn — the model pays for it
whether or not the tool is ever called. That fixed per-turn tax is the *tool
floor*. `fak footprint` prices it offline and deterministically, reusing the
same estimator the agent request footprint uses (`agent.RequestFootprint`, via
`internal/mcpfootprint`), so the number can never drift from
`EstimateAnthropicTokens`.

Regenerate at any time:

```
fak footprint            # human table, largest-first
fak footprint --json     # schema fak-mcp-footprint/1
fak footprint --top 8    # just the heaviest N
```

## Baseline (measured)

```
mcp-footprint: 24 tools · floor 5460 est. tokens (21843 bytes, ESTIMATED)
```

Heaviest contributors (the cold-schema deferral targets for #3231/#3232):

| rank | est. tokens | bytes | tool |
|-----:|------------:|------:|------|
| 1 | 558 | 2234 | fak_memory_run |
| 2 | 508 | 2033 | fak_memory_explain |
| 3 | 305 | 1223 | fak_context_restore |
| 4 | 305 | 1222 | fak_context_change |
| 5 | 290 | 1161 | fak_context_spans |
| 6 | 257 | 1028 | fak_admit |
| 7 | 254 | 1017 | fak_context_value |
| 8 | 249 |  999 | fak_adjudicate |

The full 24-tool breakdown is what `fak footprint` prints; only the head is
pinned here so a drift is legible in review.

## Witness

- `internal/mcpfootprint.TestRealFakMCPFloor` prices the **real** registry and
  asserts the floor is a faithful partition (floor bytes == sum of per-tool
  bytes) and non-trivial — the number above is reproducible, not hand-typed.
- `internal/mcpfootprint.TestPricePartitionsExactly` /
  `TestPerToolSortedDescending` lock the estimator invariants.
- `cmd/fak.TestMCPFootprintVerbJSON` witnesses the `fak footprint --json` shape.

## Cross-links

- **#3229** — epic: shrink the always-sent context budget (this baseline is its
  measurement floor).
- **#3233** — the *live* gateway footprint (same estimator, measured on a real
  request rather than the static registry).
- **#3231** — defer cold `fak_*` MCP schemas (drives the number above down).
- **#3232** — gateway tool-floor deferral (the 10× lever on this same floor).
- **#3234** — the userland analogue: `fak skill footprint` for the resident
  `.claude/skills` description floor.
