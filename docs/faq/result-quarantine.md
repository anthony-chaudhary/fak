---
title: "fak FAQ — The wall — how result quarantine works"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# The wall — how result quarantine works

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Running the check in the same address space as the agent loop is what makes fail-closed affordable: there is no per-call process spawn or socket round-trip to wedge on, so refusing by default never costs a hook launch. The decide path is a fold over registries read with a single atomic pointer load (no mutex, zero allocations on the hot path), and a witness proves no `os/exec` spawn happens on it. The measured in-process versus spawned-hook gap is roughly 2,400–2,849×, but that figure is a subsystem regression sentinel for the decide path, not a fleet-speed headline.

## The wall — how result quarantine works

The second, independent gate: how a suspicious tool result is held out of the model's context, why the wall holds even when the detector that flags it is fooled, and what that protects against.

## What is result quarantine in fak?

Result quarantine is the write-time gate that decides whether a tool result is allowed to enter the model's context, holding poisoned, secret-shaped, or polluted results out entirely. It is the call-side adjudicator's dual: where the adjudicator screens proposed tool *calls*, the context-MMU (`ctxmmu`) screens tool *results* at the moment they would be written into the conversation. A result either enters as-is (Allow), is paged out to a small pointer because it is benign but oversize (Transform), or is held out of context because it looks like a secret, an injection, or pollution (Quarantine).

## How does a quarantined result get held out of the model's context?

`fak` pages the offending bytes out to a content-addressed blob store and replaces the result payload in-place with a tiny stub like `{"_quarantined":true,"id":...,"reason":...,"len":...}`, so the dangerous bytes are physically absent from context. The kernel mints a quarantine id, pins the bytes in the content-addressed store so the bounded cache cannot reclaim them before a gated read, and stamps the result's metadata with the quarantine id. The model only ever sees the stub pointer; the poison never reaches attention. If even writing the stub fails, the path fails closed to an inline reference tagged as quarantined rather than letting the bytes through.

## What does the result detector actually screen for?

The screen, `ScreenBytes`, runs three first-match-wins checks over a result body: secret exfiltration, prompt injection, and byte-repeat pollution. Secret detection is an RE2 pattern matching shapes like `sk-...`, `AKIA...`, `ghp_...`, `xox[baprs]-...`, and PEM private-key blocks, returning `SECRET_EXFIL`. Injection detection is a lowercased substring scan over markers like "ignore previous instructions", "you are now", and "reveal your system prompt", returning `TRUST_VIOLATION`. Pollution detection is a byte-repeat predicate returning `OVERSIZE`. The same predicate backs both the post-tool admission gate and closed-API clients' pre-send transcript screening.

## How does the byte-repeat pollution predicate work?

The pollution predicate flags a result whose body is at least 512 bytes and contains a 16-byte chunk repeated back-to-back more than 50 times. It takes the first 16 bytes, steps through the body in 16-byte strides counting consecutive equal chunks, and resets the run to zero on any mismatch — so only a contiguous, blatant repeat trips it. A 16-byte chunk repeated 60 times (960 bytes) is quarantined as `OVERSIZE`. This is a deliberately conservative binary seal: it catches the most obvious context-flooding pollution without wrongly sealing a benign result.

## What is the taint ledger and where does it live?

The taint ledger is an in-process, process-local record of which results are held and which have been cleared, kept in memory under a single mutex. It holds maps of held ids to content-addressed references, a cleared set, a FIFO order list, and counters for total/quarantine/paged/evicted. It is in-memory only with no disk backing, so this live state is gone on process exit — the quarantined *bytes* live in the shared content-addressed store keyed by digest, but the live held/cleared maps reset on restart. The `fak recall` core-dump path is what persists quarantine state across the process boundary.

## Is the taint ledger bounded, or can it leak memory over a long-running process?

The ledger is bounded to a default of 8192 held ids (overridable via `FAK_CTXMMU_MAX_HELD`), closing a real process-lifetime leak where every quarantine once minted a permanent entry with no removal path. When the cap is reached, the oldest ids are evicted FIFO: the content-addressed handle is unpinned, the id is dropped from the held and cleared maps, and the order list's backing array is compacted. An evicted id's bytes were never in context, so a later page-in of that id is refused exactly like an unknown id — correct fail-closed degradation, never a leak. A bad env value fails safe to the default.

## How do quarantined bytes ever get back into context if they were a false positive?

Quarantined bytes page back in only on an explicit page-in request that comes *after* a witness clears the id, and both checks fail closed. Clearing records clearance only for an id that is currently held, keeping the cleared set a subset of the held set. Page-in refuses an id that was never held ("no quarantined result") and refuses an id that was held but never cleared ("no witness clear()"). So nothing re-enters context by accident; it takes a held id, an explicit clearance, and an explicit page-in, all three.

## How do I see quarantine decisions on the HTTP wire?

Quarantine decisions surface in the `fak` response extension under `result_admissions`, one entry per inbound tool result the kernel screened. Each entry carries the tool call id, the tool name, and a verdict whose `kind` is one of `ALLOW`, `DENY`, `TRANSFORM`, `QUARANTINE`, `REQUIRE_WITNESS`, or `DEFER`; a quarantined result shows up as `kind: "QUARANTINE"` with its reason. The extension is omitted entirely on a turn with no tool activity. Claude Code reads content blocks but not the `fak` key, so the gateway also prepends a leading `[fak] ...` text block describing the quarantine.

## What happens to a poisoned tool result in the gateway proxy path?

On the proxy path, the gateway screens every inbound tool-role message and, on a quarantine or transform, forwards the paged-out envelope so the poison never reaches the model. An un-admittable result is held out fail-closed with a stub carrying reason `ADMIT_ERROR` and a `QUARANTINE`/`TERMINAL` verdict. A quarantine also resets the relevant upstream KV span so a tuned engine's cache cannot keep serving the poisoned prefix. The counter `fak_gateway_context_pollutions_blocked_total` is the live "context saved" signal.

## How does result quarantine relate to the addressable KV cache?

They are one decision enforced in two media: the quarantine verdict bars the bytes from text context, and the KV side bars the corresponding K/V from attention state. The result detector's verdict drives a write-time eviction of the tool-result span from the kernel-owned KV cache, leaving it bit-identical to a session that never saw the poison — verified at `max|Δ| = 0` with a non-vacuity control showing the poison-vs-never delta is non-zero. This bridge is proven on a synthetic model in `internal/kvmmu` today and is not yet wired into the live `fak agent` HTTP loop, so treat the KV-eviction half as mechanism-proven, not production-served.

## Does quarantine survive a session boundary, or is it lost when the process exits?

The live quarantine maps are process-local and reset on restart, but `fak recall` persists a finished session as a durable core image whose quarantine seals survive the boundary. A reloaded image refuses to page a quarantined slice into a new context unless a witness clearance ran *and* the bytes pass a fresh content re-screen against the full registered admitter chain — clearance alone cannot launder still-poisoned bytes. The re-screen folds the current detectors, so a session recorded under a weaker gate is re-caught by every screen the fleet ships now. A sealed page persists with a safe descriptor only (`tool: [sealed: reason, N bytes]`), never the poisoned bytes.

## What is the difference between the kernel's binary quarantine and fak answer-shape?

The kernel's repeat predicate is a conservative *binary* seal — at least 512 bytes, a 16-byte chunk repeated more than 50 times — while `fak answer-shape` is a *graded*, tunable witness over the same concern. `answer-shape` emits a repeat fraction in `[0,1]` (the max of n-gram, repeated-line-block, short-period, and compression signals) judged against caller thresholds like `--max-repeat` and `--max-chars`, catching softer loops the kernel's binary gate deliberately admits. The two share the idea of degenerate repetition but not code: the kernel's is a fixed seal on the hot path, `answer-shape`'s is an off-hot-path consumer witness with no kernel dependency.

## Does the audit log of a quarantine leak the poisoned bytes?

No — the audit surfaces record names, verdicts, reasons, and content *digests*, never the poisoned bytes or result content. The stdout access log carries the tool name and verdict fields with no payload and no digest at all. The opt-in durable journal (enabled by `FAK_AUDIT_JOURNAL`) records the tool name, trace id, verdict, reason, and a result digest derived from the frozen reference — it never materializes a blob, so it leaks no payload into the log. A quarantine page's saved descriptor is safe sealed metadata only.

## What reason codes can a quarantine carry, and where do they come from?

A quarantine carries one code from the kernel's closed 17-reason refusal vocabulary: secret-shaped results return `SECRET_EXFIL`, injection-shaped results return `TRUST_VIOLATION`, and byte-repeat pollution returns `OVERSIZE`. These come from the same fixed vocabulary the call-side adjudicator uses, so a result refusal is as structured and citable as a call refusal — never free-text. An unknown forward-compatible code renders as `REASON_<n>` and never panics. (On the gateway proxy path, a result that cannot be admitted at all is held out fail-closed with the wire-level marker `ADMIT_ERROR`, which is a fail-closed signal rather than a vocabulary code.)

## Does quarantine guarantee you catch every injection, or only contain the ones it flags?

