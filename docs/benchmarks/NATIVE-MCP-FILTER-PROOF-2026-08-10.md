# Native MCP tool-filter A/B proof — 2026-08-10

Command:

```bash
fak mcp-filter-proof --json
```

Captured artifact: [`experiments/mcp-filter-proof-2026-08-10.json`](../../experiments/mcp-filter-proof-2026-08-10.json).

The offline held corpus exercises five cold capabilities by natural intent. A task passes
only when ranked search returns the required tool first **and** the same exposed-registry
route exists for first call. The tuned control is the complete `tools/list` descriptor
array; the active arm is the default bootstrap plus `fak_tools_search`.

Captured result: **PASS** — 5/5 task success, 5/5 search recall, 5/5 first-call route
existence, and exposed-registry security parity. The active arm advertised 4 of 19 tools
and removed 16,253 exact JSON descriptor bytes (20,164 → 3,911); the full-list control
removed zero. These are bytes, not estimated provider tokens. This proves the native
registry and held intents in this artifact; it does not claim universal model task
accuracy. Live operators should read `/debug/vars.token_savings.native_mcp_filter`; any
`bypassed` state means fak restored the full list rather than risking capability loss.
