# Managed-cache message-prefix ablation — 2026-08-05

Issue: #2186 (duplicate contract: #2176).

## Decision

Enable the ordered message-prefix 1h upgrade by default whenever managed-cache is ACTIVE.
Keep `FAK_ABLATE_TTL_1H_HEAD_ONLY=1` as the distinct head-only control arm.

## Break-even

Anthropic's 1h cache write costs 2x the 5m write, while a 1h cache hit costs the same
0.1x input rate. For a stable prefix of `P` tokens:

- head-only / 5m after a 5–60 minute idle: rewrite `1.25P`;
- message-prefix / 1h after that idle: one initial `2P` write, then each resume reads `0.1P`;
- incremental first-write premium: `0.75P`;
- avoided expired-tail rewrite on one qualifying resume: `1.15P`.

Therefore one observed 5–60 minute resume repays the premium (`0.75 / 1.15 = 0.652`
qualifying resumes). The default remains conditional on ACTIVE posture; OFF is unchanged.

## Ablation row and witness

The pre-instrumentation provider ledger gives the prize-size lower bound but cannot split
head from history: `docs/nightrun/cache-savings.jsonl` records the 2026-07-11 Anthropic
row with **8,773,566 cache-creation tokens**. That missing split is why #2186 adds the
following current-state witness rather than relabeling old totals:

| arm | switch | request result | creation attribution |
|---|---|---|---|
| managed cache OFF | `cacheTTL1H=false` | caller's 5m layout | `head_5m` |
| head-only control | `FAK_ABLATE_TTL_1H_HEAD_ONLY=1` | system/tools become 1h; messages unchanged | `cache_creation_tokens_head_only` / `head_1h` |
| message-prefix default | managed cache ACTIVE | eligible system/tools and message prefixes become 1h in provider order | `cache_creation_tokens_message_prefix` / `message_prefix_1h` |

The cumulative split is emitted in `/debug/vars.cache_attribution`; each
`gateway_inference_turn` JSON line carries `cache_creation_span`. The provider's usage
response reports only one aggregate creation count, so attribution is deliberately by the
admitted request layout for that turn; it does not pretend to recover a per-breakpoint token
split the provider did not send.

## Refusal and ordering contract

The rewrite is identity on malformed/ambiguous cache controls, caller-selected TTLs, and a
message prefix containing the same closed UUID/ISO-8601 volatility shapes used for head
refusal. Every earlier breakpoint must be eligible for 1h before a later message breakpoint
is upgraded, preserving Anthropic's longer-before-shorter TTL rule.
