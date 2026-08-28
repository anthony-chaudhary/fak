# GPT-5.6 Sol official API terms checked 2026-08-27

Issue: [#9575](https://github.com/anthony-chaudhary/fak/issues/9575)

**Verdict:** as checked on **August 27, 2026**, OpenAI's official model and API pricing pages list GPT-5.6 Sol standard pricing at **$4.00/MTok input, $0.40/MTok cached input, $5.00/MTok cache writes, and $20.00/MTok output**. Prompts above 272K input tokens use 2x input and 1.5x output pricing for the full request. The model page lists a 1,050,000-token context window and 128,000 maximum output tokens. Promotional pricing is stated to remain available at least through **November 21, 2026**.

## Official commercial terms

Only these official OpenAI sources support this section:

- OpenAI model page: <https://developers.openai.com/api/docs/models/gpt-5.6-sol>
- OpenAI API pricing page: <https://developers.openai.com/api/docs/pricing>

| Term | Official value checked 2026-08-27 |
|---|---:|
| Standard input | $4.00 per 1M tokens |
| Standard cached input | $0.40 per 1M tokens |
| Standard cache writes | $5.00 per 1M tokens |
| Standard output | $20.00 per 1M tokens |
| Long-context threshold | More than 272,000 input tokens |
| Long-context application | 2x input and 1.5x output, applied to the full request |
| Long-context input / cached / write / output | $8.00 / $0.80 / $10.00 / $30.00 per 1M tokens |
| Context window | 1,050,000 tokens |
| Maximum output | 128,000 tokens |
| Promotional period | Available at least through 2026-11-21 |

The pricing page also lists tools separately from model tokens. On the checked date it lists web search at $10.00 per 1,000 calls plus search-content tokens at model rates; 1 GB Hosted Shell/Code Interpreter containers at $0.03 per 20-minute session per container, with separate prices for larger containers; and file-search tool calls at $2.50 per 1,000 calls plus separately metered storage. These fees must not be hidden inside a token-only estimate.

The four requested comparison buckets (35K, 64K, 128K, and 200K) remain below the >272K long-context threshold. A job can still cross the threshold through its actual request input, so receipts preserve raw counters and select the rate from observed input rather than from the bucket label alone.

## Evidence boundary

This note deliberately separates three evidence classes:

1. **Official terms:** the table above is a dated transcription of OpenAI's official pages. It can change after 2026-08-27 and must be snapshotted again before future accounting.
2. **Observed telemetry:** provider usage counters, invoices, local harness events, quality artifacts, retries, setup, compaction, intervention, and wall time belong in job receipts. This note contains no observed GPT-5.6 Sol job telemetry.
3. **Estimates and derived accounting:** costs calculated from terms and counters, local hardware allocation, operator labor, bounds, and completed-job rollups are derived. They must cite their inputs and carry uncertainty. They are not official prices or raw counters.

The normative provider-neutral ledger contract is [`../standards/provider-job-accounting.md`](../standards/provider-job-accounting.md). Its scrubbed fixtures are synthetic validation examples, not benchmark claims. The same schema is intended to accept the Claude and Gemini records planned in #9588 without provider-specific field changes.
## Counter-only provider evidence

The GPT fixture also appends `provider_counter_observation` record `counters-openai-gpt-5.6-sol-2026-08-18-two-turn`, sourced from the scrubbed live calibration witness. It preserves only the observed counters: input `28,473` then `39,739`; cached input `3,456` on both turns; cache-write input `0` on both turns. Output, reasoning, outcome, quality, elapsed time, and cost remain `null`; task revision and realized ratio are explicitly unavailable.

This row is not a quality-qualified job receipt. It cannot establish a completed job, a quality result, total wall time, attempted-job cost, quality-qualified completed-job cost, or a realized 200:1/300:1 ratio. #9552 next needs a matched run bound to an explicit task revision and complete attempted-job envelope. #9578 next needs the quality artifact, elapsed time, complete cost inputs, and any defensible realized ratio.
