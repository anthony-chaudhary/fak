# Native MCP tool-filter live model A/B — 2026-08-10

Command shape:

```bash
OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
OPENAI_API_KEY="$FAK_GROQ_API_KEY" \
go run ./cmd/fak mcp-filter-proof --live --model llama-3.1-8b-instant --json
```

Captured artifact: [`experiments/mcp-filter-live-proof-2026-08-10.json`](../../experiments/mcp-filter-live-proof-2026-08-10.json).

## Verdict: NOT_YET — keep the native default, but do not claim live accuracy parity

This is the first model-driven run of the same held tool-selection intents against the
native default-on `tools/list` view and the tuned full-list control
(`FAK_ABLATE_MCP_TOOL_FILTER=1`). The active arm still removed **16,253 exact descriptor
bytes** (20,164 → 3,911), and the independently exercised exposure-floor parity remained
true. It did **not**, however, complete a held task: 0/3 task success, 1/3 search recall,
and 2/3 valid first recovery calls. The full-list control was also 0/3 because the provider
rejected all three oversized requests during this account-limited run. That makes this a
useful failure/bailout witness, not a net-true accuracy comparison.

The active failures were privacy-safe reason classes only:

- one search recalled the right tool but the model emitted no post-search call;
- one model-generated call was rejected by provider tool validation;
- one search call had invalid arguments.

The control failures were provider request failures. No prompt text, arguments, API key,
or provider response body is stored in the artifact.

## Operator action

The immediate rollback remains:

```bash
FAK_ABLATE_MCP_TOOL_FILTER=1
```

Live processes expose the current state and reason at
`/debug/vars.token_savings.native_mcp_filter`. A native safety uncertainty still fails open
to all exposed tools. This run does **not** justify reverting the lossless registry/search
mechanism by itself—the control did worse under its descriptor/TPM cost—but issue #6404
stays open until a provider/account capable of completing both arms produces a comparable
held-corpus result. The first-class `--live` harness makes that rerun durable rather than a
throwaway script.
