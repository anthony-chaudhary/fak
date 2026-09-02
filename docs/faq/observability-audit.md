---
title: "fak FAQ — Observability, audit, and debugging"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Observability, audit, and debugging

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Your API gateway handles transport concerns (TLS, auth, routing, rate caps); `fak` sits on the agent's model path as the layer that understands tool calls and tool results, so the two stack rather than compete. A gateway sees opaque request bodies; `fak` decodes the turn, adjudicates each proposed tool call against the capability floor, screens inbound tool results for quarantine, and surfaces every verdict in a `fak` response extension plus an in-band note for clients that don't read it. It also ships intelligent tiered request routing as a library, but that router is explicitly not on the live serve request path today, so don't count on `fak` to replace your gateway's routing. Run your gateway at the edge and `fak` on the model path.

## Observability, audit, and debugging

The three correlated surfaces that tell you what the gate is doing, how to debug a denied call, and the consumer-side witnesses you can run over any answer.

## What observability does fak give me, and how are the three surfaces correlated?

`fak serve` exposes three correlated observability surfaces — a Prometheus `/metrics` endpoint, a live `/debug/vars` JSON snapshot, and a structured stdout access log — and a single `trace_id` threads all three together. The access log writes two JSON lines per request (`gateway_operation` carrying the verdict and `gateway_http_request` carrying transport details), `/debug/vars` gives you the same view as `/metrics` as one JSON object you can read right now, and every response carries an `X-Trace-Id` header that also appears in the access log and the per-operation verdict log. Point your scraper at `/metrics`, eyeball `/debug/vars` during an incident, and grep the access log by `trace_id` to follow one request across all three.

```bash
curl -s http://127.0.0.1:8080/metrics | grep fak_kernel
```

## What kernel counters does fak track, and what does the vDSO hit ratio tell me?

`fak` tracks per-kernel counters for submits, vDSO hits, engine calls, denies, transforms, quarantines, and admitted results, surfaced on `/metrics` as `fak_kernel_…_total` plus the derived gauge `fak_gateway_vdso_hit_ratio`. The vDSO hit ratio is `VDSOHits/Submits` — the fraction of tool calls answered from the in-process fast path with no adjudication and no engine call — so a high ratio means a cache-friendly workload and a low one means most calls fell through to a full decision. `denies`, `transforms`, and `quarantines` count how often the floor refused a call, rewrote its arguments, or held a tool result out of context. The vDSO cache also exports its own view (`fak_vdso_lookups_total`, `hits_total`, `hit_rate`) plus miss attribution under a closed vocabulary (`DESTRUCTIVE|MISSING_HINTS|RESOURCE_MISNAMED|WITNESS_REVOKED|NOT_CACHED`).

## How do I debug a tool call that fak denied?

Run `fak preflight` to replay that exact call through the policy and print the verdict, the reason code, and which rung decided it — no server, model, or network required. Pass the tool name and JSON args (and your policy file) and it prints `verdict=… reason=… by=monitor`; add `--explain` or `--json` to dump the full per-rung Decision trace so you can see whether the grammar rung, the preflight ladder, or the adjudicator monitor refused it. The reason comes from a closed 17-code vocabulary (`DEFAULT_DENY`, `POLICY_BLOCK`, `SELF_MODIFY`, `UNKNOWN_TOOL`, and so on), so the refusal is citable rather than free text. A `DEFAULT_DENY` usually means the tool was never allow-listed; a `POLICY_BLOCK` or `SELF_MODIFY` means an explicit deny or a write-shaped self-modify rule fired.

```bash
fak preflight --tool refund_payment --args '{}' --policy floor.json --explain
```

## What do fak's refusal reason codes mean?

Every refusal carries exactly one code from a closed 17-reason vocabulary, so you can route on it instead of parsing free text: `DEFAULT_DENY`, `POLICY_BLOCK`, `SELF_MODIFY`, `LEASE_HELD`, `TRUST_VIOLATION`, `MALFORMED`, `MISROUTE`, `RATE_LIMITED`, `SECRET_EXFIL`, `UNWITNESSED`, `OVERSIZE`, `UNKNOWN_TOOL`, `RESULT_SECRET_DISCOVERED`, `SECRET_REDACTED`, `SHELL_DIALECT`, `PII_REDACTED`, and `PII_EXFIL`. `DEFAULT_DENY` is the fail-closed floor — the tool was never allow-listed; `POLICY_BLOCK` is an explicit named deny; `SELF_MODIFY` fires on a write-shaped call that touches a guarded path or runs a mutating shell command; `MALFORMED` and `MISROUTE` flag broken or unrepairable call shapes. The vocabulary is forward-compatible: an unknown code renders as `REASON_<n>` and never panics. Each code also maps to a disposition (`RETRYABLE`, `WAIT`, `ESCALATE`, or `TERMINAL`) so the next agent turn knows whether retrying, waiting, or escalating is appropriate.

## Does fak's audit log record my tool arguments or result contents?

No — the stdout access log records tool names, verdicts, reason codes, dispositions, and timings, but never request bodies, tool arguments, or result content. Each request emits a `gateway_operation` line with the tool name and verdict fields and a `gateway_http_request` line with `duration_ms`, status, bytes, and route, both stamped with `trace_id`; neither carries a payload or even a digest of one. This is a deliberate privacy guarantee: you can ship the access log to a central collector without leaking what the agent was working on. If you opt into the separate durable decision journal (via `FAK_AUDIT_JOURNAL`), it adds content digests (`ArgsDigest`/`ResultDigest`) and a tamper-evident hash chain — still digests only, never the raw bytes.

## What is the durable decision journal and how is it different from the access log?

The decision journal is an opt-in, append-only, tamper-evident ledger that writes one hash-chained JSONL row per audit event (`DECIDE`, `DENY`, `QUARANTINE`, or `VDSO_HIT`), enabled by setting the `FAK_AUDIT_JOURNAL` environment variable; off by default, the package stays inert. Unlike the stdout access log, which stores no payload and no digest, the journal records the tool name, `trace_id`, verdict, reason, and content digests (never the blobs themselves), and each row's hash chains over the previous row so any post-hoc tampering breaks `Verify` at the first altered link. A vDSO fast-path hit is journaled like an engine call, so the audit trail is complete even for calls that never reached the model. Reopening the journal continues the chain rather than forking it, and each write is flushed to the OS file before returning so a crash loses no recorded row.

## How do I see what happened on a turn — was a tool call dropped or a result quarantined?

Read the `fak` extension object on the gateway response: it carries an `adjudications` array (one entry per proposed tool call, including dropped ones) and a `result_admissions` array (one entry per inbound tool result the kernel screened). Each adjudication shows `tool_call_id`, `tool`, whether it was `admitted`, the `verdict`, and `repaired_arguments` only when the verdict kind is `TRANSFORM`; a quarantined result shows up under `result_admissions` with `verdict.kind == "QUARANTINE"`, meaning its bytes were paged out and never reached the model. The object is omitted on turns with no tool activity. Because Claude Code reads content blocks but not the `fak` extension key, the same drops, repairs, and quarantines are also prepended to the message as a leading `[fak] …` text block so they remain visible on the Anthropic wire.

## How fast is fak's adjudication decision, and is the latency observable?

The adjudication decision itself is sub-millisecond — a captured access-log line shows a policy `DENY` at `duration_ms` ≈ 0.511 — because the decision is an in-process fold with no spawned hook and no engine round-trip. That number is the `adjudicate` operation duration from a real captured access log, observable per request via the `duration_ms` field on each `gateway_operation` line and correlatable by `trace_id`. The in-process fold is often faster than the OS clock granularity, which is why `fak bench` uses an inner calibration loop to measure it. The honest fence: this is the decide-path latency, not a serving-throughput figure; `fak bench`'s gate is a regression sentinel for the decide path that passes only if the in-process p50 beats the spawned-hook baseline.

## How can I check whether a candidate answer or tool result is degenerate before it reaches the model?

Pipe the text through `fak answer-shape`, the consumer-facing witness that grades how repetitive (looping or degenerate) and how long (verbose or runaway) a piece of text is against thresholds you choose. It reports a single `RepeatFraction` in `[0,1]` — the max of four sub-signals (n-gram repeat, repeated-line-block, short-period tiling, and a compression-redundancy signal) so it trips on whichever way the text actually degenerated — plus a rune-length count, and exits 0 in shape, 1 degenerate, and 2 on a usage error so it composes as a pipeline gate. It reads stdin on `-` (or no source), is pure and deterministic, and runs off the hot path with no model, session, or kernel dependency. Tune it with `--max-repeat`, `--max-chars`, and `--ngram`; repetition fractions below a 24-rune floor are reported but never trip the verdict.

```bash
some_model_output | fak answer-shape --max-repeat 0.5 --max-chars 8000
```

## What does fak doctor add over fak answer-shape?

