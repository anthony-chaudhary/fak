---
title: "fak FAQ — Sessions, recall, and persistence"
description: "How fak manages long-running sessions, context compaction, recall, resume state, memory trust, and safe restoration of dropped context."
---

# Sessions, recall, and persistence

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Front a real engine (vLLM, SGLang, llama.cpp, or a cloud provider) for anything where tokens per second matters, and reach for the in-kernel engine when you specifically want the kernel-owned KV cache and its provable span eviction on a self-hosted model. Point `fak serve --base-url <upstream/v1>` at your existing OpenAI-compatible server to keep its throughput while gaining the capability floor, result quarantine, and audit trail; that is where most deployments should start. Drop `--base-url` and pass `--gguf` only when you want the in-kernel path's correctness reference and reuse behavior, accepting that it is not a tuned production server.

## Sessions, recall, and persistence

What a session holds, what is process-local and resets on restart, and how `fak recall` persists a finished session as a durable core dump.

## What is a session in fak, and why is it called a core dump?

A session in `fak` is a small page table over a content-addressed swap device, not a flat transcript replayed token by token. As an agent runs, the context-MMU already pages every heavy or poisoned tool result out to a content-addressed store at write time, so the finished session is just roles plus digests plus descriptors plus quarantine state pointing into that store. That is structurally a core dump: answering a follow-up demand-pages only the working set the query touches, and never re-executes the whole history back into context. `recall.Session` is the reloaded core image, `recall.Recorder` is the live in-process recorder that holds the MMU and an in-memory CAS until it persists.

## Does quarantine and taint state survive a process restart?

The live quarantine and taint state is process-local and is gone when the process exits. The context-MMU keeps that state in plain in-memory maps under one mutex (`held`, `cleared`, an order list, and counters), allocated fresh on `New()` with no disk backing, so a restart starts clean. The quarantined bytes themselves live in a content-addressed store keyed by digest, so a page-in request for a dropped id just fails closed with "no quarantined result". This is exactly the gap `fak recall` closes by persisting the seal to disk; without recall, in-process held and cleared state and the in-memory CAS do not outlive the process.

## What does fak recall do?

`fak recall` records a finished agent session through the write-time quarantine gate, persists it as a durable core image, then reloads it in a fresh process to prove the quarantine survived the boundary. The recorder drives the shipped context-MMU over each tool result (plus a de-obfuscating scan as defense-in-depth, fail-closed to quarantine), then writes two files: `manifest.json` (the page table: roles, digests, descriptors, and quarantine state) and `cas.json` (the content-addressed swap device). The whole pass is offline and deterministic. The CLI default runs an airline-support session with two benign results, one injection, and one secret leak, then reloads it.

```bash
fak recall --dir recall-image --out recall-report.json
```

## What does a fak core image actually contain on disk?

A core image holds a manifest page table plus a content-addressed swap device, and nothing that re-injects poison. The `manifest.json` carries the version, session id, a world-version frozen at persist time, the list of pages, the cleared set, and any context-change tombstones. Each page records its step, role, descriptor, CAS digest, length, taint, quarantine flag and id, reason, durability class (turn, session, or durable), witness, and trust epoch. A quarantined page's descriptor carries only safe sealed metadata of the form `tool: [sealed: reason, N bytes]`, never the poisoned bytes and never their de-obfuscated text. The `cas.json` is a digest-to-bytes map that does hold a copy of every byte, including the sealed poison, the way a real core dump holds the whole process image.

## What survives a session boundary, and what is lost?

What survives is everything written into the on-disk core image; what is lost is the live in-process gate state. Surviving across the boundary: the page table, the frozen quarantine seals, the cleared clearance set, the tombstone context-changes, the witness and trust-epoch metadata, and the CAS bytes. Process-local and gone on restart: the live context-MMU maps (held, cleared, order, counters) and any recorder state you never persisted. The durability proof is that `Load(dir)` rebuilds a session with its own CAS loaded from disk plus a fresh MMU gate, so a resolve provably does not lean on the recording process being alive.

## Can a witness clearance alone un-quarantine a result after reload?

No. A clearance alone cannot launder still-poisoned bytes; a reloaded quarantined page pages back into a new context only if a witness `Clear()` ran AND the bytes pass a fresh content re-screen. This is the recall moat (rung 4): two independent gates, so clearing the id is necessary but not sufficient. The re-screen folds the de-obfuscating scan plus the whole registered result-admitter chain, most-restrictive-wins, so a session recorded under a weaker gate is re-caught by every detector the fleet ships now. In the committed demo, the injection page stays refused even after a clearance because the re-screen re-quarantines it, while a genuinely benign cleared page does release, which proves the gate discriminates on content rather than hard-denying.

## How is fak recall different from RAG over a chat transcript?

Naive RAG over history re-pastes transcript bytes ungated, while `fak recall` re-screens every page through the trust gate on the way back into context. A reloaded core image refuses to page a quarantined slice into a new window unless a witness clearance ran and the bytes pass a fresh content re-screen, so a poisoned result that an embedding ranker might happily surface is still walled off. The honest limit is that recall makes the gate's decision durable and re-screenable, it does not improve the decision itself: a crafted injection that never trips the detector's marker set at write time is never quarantined, and recall will resolve it. The re-screen is the lever that re-catches such a page once the patterns are tightened.

## What is the difference between the recall core dump and the audit journal?

They are two independent durable surfaces: the recall core dump is the reloadable session image, while the journal is an append-only, tamper-evident decision ledger. The journal (`internal/journal`, opt-in via `FAK_AUDIT_JOURNAL`, off by default) writes one hash-chained JSONL row per audit event with a monotonic sequence number, tool name, trace id, verdict, reason, and content digests, where each row's hash chains over the previous one. It stores digests only, never argument or result bodies, so it leaks no payload. The journal is the regulated-audit surface; the recall image is the durable session memory. Recall persistence and the journal do not depend on each other.

## How do deletion certificates relate to persistence?

A deletion certificate is a portable, re-checkable receipt that binds a bit-exact KV-cache eviction to the tamper-evident journal that recorded it, so a deletion claim survives as verifiable evidence. Under one ed25519 signature it carries the evicted count, the span, an equivalence record asserting `MaxAbsDelta == 0` (the byte-identical claim), and an anchor row from the journal pinned to the result digest. `Verify` fails closed on a signature mismatch, any non-zero delta, an absent or broken journal chain, or a subject relabel. Honest bounds: the v1 signature is self-attesting (it proves integrity, not issuer independence; third-party RFC-3161 or CT-log anchoring is an open stub), and it proves deletion from the inference working set and agent memory only, not from weights, backups, or replicas.

## If I want a memory to be absent from future context, do I delete it from the core image?

You file a tombstone, not a delete: the recall-side analogue of deletion is a negative-only, evidence-preserving tombstone. `Session.RequestContextChange` records a tombstone that suppresses future page-in for resolve, recall, and working-set ranking, but never deletes the CAS bytes or mutates the original page row, so the audit evidence stays intact. The tombstone is written into the manifest's context-changes and re-persisted, so it is durable across reloads. Operator and agent surfaces include `fak debug --cmd tombstone`, the HTTP route `POST /v1/fak/context/change`, and the MCP tool `fak_context_change`.

## What happens if the on-disk swap device is tampered with before reload?

A tampered core image fails closed at load: `recall.Load` verifies that every CAS blob hashes to its digest key, and if any blob does not match it refuses the whole image. Because the store is content-addressed, the digest is the identity, so flipping a byte inside a stored blob under its unchanged key is detected. The witness `TestCorruptCASFailsClosed` decodes the CAS, flips a byte inside a stored blob, and asserts the load is rejected. This is the same integrity discipline a deletion certificate uses when it re-derives its anchor row from the journal.

## Is the recall core image zero-copy, and what is the storage tradeoff?

