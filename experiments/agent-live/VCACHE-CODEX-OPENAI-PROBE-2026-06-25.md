# vCache Codex/OpenAI Probe - 2026-06-25

## Question

Can this host prove or refute vCache savings for Codex/OpenAI from real
`cached_tokens` telemetry?

## Current status

**Proven for this Codex CLI thread from Codex-authored session telemetry.** The
active process exposes Codex management metadata only:

```text
CODEX_MANAGED_BY_NPM
CODEX_MANAGED_PACKAGE_ROOT
CODEX_THREAD_ID
```

There is no `OPENAI_API_KEY` in this process, and the Codex CLI session does
not expose raw OpenAI Responses or Chat Completions usage JSONL for this run.
However, the local Codex session JSONL records `event_msg` / `token_count`
events with `last_token_usage.input_tokens` and
`last_token_usage.cached_input_tokens`. Those counters are enough to prove the
Codex side without an API key.

Captured Codex CLI session witness:

```powershell
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json
```

Result:

```json
{
  "status": "PROVEN",
  "requests": 68,
  "baseline_token_equiv": 10638831,
  "actual_token_equiv": 1491490.2,
  "saved_token_equiv": 9147340.8,
  "saved_pct": 85.98069468346664,
  "cache_read_tokens": 10163712,
  "first_positive_request": 1,
  "correctness_depends_on_hit": false
}
```

Frozen summary artifact:
`experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.json`.
Replayable telemetry artifact:
`experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl`.

## Why the Codex savings number is so large

The Codex proof is large because this workload is almost the ideal prompt-cache
shape: one long thread repeatedly sends a very large, mostly identical prefix,
then appends a small fresh suffix for the next turn. Current OpenAI prompt-cache
documentation says cache hits require exact prompt-prefix matches, caching is
automatic for prompts of at least 1024 tokens, and cached input can cost up to
90% less than uncached input:
<https://developers.openai.com/api/docs/guides/prompt-caching>. Codex pricing
also treats input tokens, cached input tokens, and output tokens as separate
usage fields:
<https://developers.openai.com/codex/pricing#credits-overview>.

The captured Codex session has this aggregate shape:

```text
requests:       68
input tokens:   10,638,831
cached tokens:  10,163,712
uncached input: 475,119
cached share:   95.53%
```

The verifier's OpenAI/Codex accounting model is:

```text
baseline = total_input
actual   = uncached_input + read_mult * cached_input
saved    = cached_input * (1 - read_mult)
```

With `read_mult = 0.1`, the proof becomes:

```text
saved = 10,163,712 * 0.9 = 9,147,340.8 token-equiv
saved_pct = 85.98%
```

That is not magic and it is not a compression claim. The full context is still
being supplied to Codex. The provider is charging/serving the already-seen
prefix on the cheaper cached-input path instead of redoing full uncached prefill
for every request.

The per-row shape explains the effect. The first ten recorded rows had cached
shares of roughly `9.8%, 96.2%, 73.4%, 6.8%, 54.8%, 86.2%, 88.6%, 92.2%,
99.6%, 99.0%`. By the end of the same live thread, rows were mostly around
`99.0%` to `99.9%` cached. That means most later turns were paying the uncached
rate only for the newest delta: the latest prompt, recent tool output, and any
changed metadata.

Why Codex is especially favorable:

- The stable prefix is huge: system/developer instructions, tool schemas, repo
  guidance, conversation history, and accumulated tool results.
- The thread is append-heavy. If earlier bytes remain stable, the longest common
  prefix keeps growing.
- Tool definitions and instructions are near the beginning of the prompt. If
  they do not change, they are cache-friendly.
- Codex session telemetry reports cached reads but not a separate cache-write
  charge. So the proof is read-discount arithmetic over observed
  `cached_input_tokens`, unlike the Claude artifact where cache creation tokens
  materially affect break-even.

Important boundary: this proves observed Codex/OpenAI prompt-cache economics for
this live Codex session. It does **not** prove that fak's vCache layer caused
those savings. It supports the vCache design premise: if fak can keep prefixes
stable, route related turns together, and avoid cache-breaking edits near the
front of the prompt, Codex/OpenAI traffic can spend most repeated context on the
cached-input path. Correctness still depends on sending the full context; a cache
hit is a rebate, not a trust dependency.

## Verifier path now supported

`fak vcache prove-telemetry` accepts OpenAI-compatible usage rows in either raw
Responses shape:

```json
{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}
```

or raw Chat Completions shape:

```json
{"usage":{"prompt_tokens":2006,"prompt_tokens_details":{"cached_tokens":1920}}}
```

The verifier splits total input tokens into uncached and cached portions, then
applies `--read-mult` to only the cached portion. With the default `--read-mult
0.1`, the first example proves:

```text
baseline token-equiv: 2006.0
actual token-equiv: 278.0
saved token-equiv: 1728.0
cache read/write tokens: 1920 / 0
```

It also accepts Codex CLI session `token_count` events:

```json
{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":2006,"cached_input_tokens":1920}}}}
```

and `codex exec --json` `turn.completed` rows:

```json
{"type":"turn.completed","usage":{"input_tokens":24763,"cached_input_tokens":24448,"output_tokens":122}}
```

## How to test live

### 1. Replay the frozen Codex witness

This validates the arithmetic and parser without using a key or sending a new
request:

```powershell
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json
```

Expected: `status: PROVEN`, `requests: 68`, `saved_pct: 85.98`.

### 2. Test the active Codex CLI thread

Extract only token counters from the local Codex session, then replay them:

```powershell
$out = "experiments/agent-live/vcache-codex-live-$($env:CODEX_THREAD_ID).jsonl"
python tools/vcache_codex_session_extract.py --thread-id $env:CODEX_THREAD_ID --out $out
go run ./cmd/fak vcache prove-telemetry --file $out --json
```

If automatic discovery misses the session file, pass it explicitly:

```powershell
$session = Get-ChildItem "$env:USERPROFILE\.codex\sessions" -Recurse -File -Filter "*$env:CODEX_THREAD_ID*.jsonl" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
python tools/vcache_codex_session_extract.py --session $session --out experiments/agent-live/vcache-codex-live.jsonl
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-live.jsonl --json
```

The extractor intentionally writes only `input_tokens` and
`cached_input_tokens`. It should not preserve prompts, tool outputs, diffs, or
model text.

### 3. Test `codex exec --json`

For a noninteractive run, capture Codex's JSONL stream and sanitize it:

```powershell
codex exec --json "Say ok after inspecting no files." | Tee-Object -FilePath experiments/agent-live/codex-exec-raw.jsonl
python tools/vcache_codex_session_extract.py --session experiments/agent-live/codex-exec-raw.jsonl --out experiments/agent-live/codex-exec-usage.jsonl
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/codex-exec-usage.jsonl --json
```

One tiny `codex exec` may not prove much by itself. The useful test is a sequence
of similar turns in the same thread or session, where the first turn warms the
prefix and later turns should report higher `cached_input_tokens`.

### 4. Run controlled A/B tests

Run at least five requests per condition:

```text
A: stable prefix, small changing suffix
B: same size, but randomize early prefix bytes each request
```

Expected result:

```text
A: cached_input_tokens / input_tokens climbs after warm-up
B: cache share stays low or collapses
```

This distinguishes real prefix caching from a verifier bug. If both A and B show
high cache shares, the test is not isolating the cache key. If both show low
shares, the prefix is too short, the prefix is changing, the model/account path
does not expose prompt caching, or the requests are routed in a way that misses
the cache.

### 5. Probe retention and routing

After a warmed high-hit request, wait fixed intervals and send the same-shaped
turn:

```text
30s, 2m, 5m, 10m, 30m, 60m
```

Record when `cached_input_tokens` drops. OpenAI documents in-memory retention as
generally 5-10 minutes of inactivity, up to one hour, with extended retention
available on some models. That gives fak a concrete target for warming cadence
and affinity routing.

Also compare same-thread versus new-thread behavior:

```text
same thread: should retain high prefix reuse
new thread: may lose part or all of the hit depending on provider routing/keying
```

### 6. Run the raw OpenAI API probe

For raw OpenAI API traffic, issue #727 still has the optional provider probe:
run repeated Codex/OpenAI requests with a stable 1024+ token prefix, capture
provider JSONL containing `cached_tokens`, and feed it to:

```powershell
python tools/vcache_openai_probe.py --out experiments/agent-live/vcache-openai-probe.jsonl
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-openai-probe.jsonl --read-mult 0.1
```

That command is the acceptable close condition for a raw OpenAI API-key run.
The Codex CLI thread is already proven from local `token_count` telemetry above.

## What to optimize next

The live implication for fak is not merely "long prompts are cheaper." It is:

```text
maximize stable prefix length
minimize early-prefix churn
keep related turns on the same provider/cache route when possible
budget every request at uncached price
record cached-token telemetry as a realized rebate
```

Concrete things to test before claiming product-level savings:

- Move volatile metadata, timestamps, request IDs, and per-turn scratch content
  late in the prompt.
- Keep tool schemas byte-stable and ordered deterministically.
- Keep durable repo guidance stable, and avoid injecting giant changing
  instruction blocks at the front of every turn.
- Measure cache share distribution, not just aggregate savings. A single long
  thread can hide cold misses behind many hot later turns.
- Separate provider-observed cache savings from fak-caused savings. The former is
  what this artifact proves; the latter requires an A/B where fak changes prefix
  stability or routing while the task workload stays fixed.
