---
title: "fak FAQ — Security and the threat model"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Security and the threat model

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

It has been witnessed live, not just modeled: a real `fak agent` run against gemini-2.5-flash showed 7 versus 6 turns, exactly 1.00 retry-turn per error, across 3 of 3 trials. This measures the clean-recovery floor where an injected error costs one extra model turn, recorded in a committed artifact. The sample is small (n=3, one model), so it is presented as a floor rather than a general distribution; the broader turn-tax decomposition around it remains a transparent cost model on the baseline side.

## Security and the threat model

What `fak` is built to stop, why two structural gates beat one classifier, and — stated up front — what it explicitly cannot protect against.

## What is fak's threat model: who is the attacker and what are they assumed to control?

`fak`'s threat model treats the language model itself as the untrusted program and assumes the attacker controls everything the model reads: the prompt, retrieved documents, and tool results. The model is ring-3 userspace; the harness is the kernel adjudicating each tool call (the syscall) from evidence the model did not author. So the question is never "did the model get fooled" but "can a fooled model still pull an irreversible lever or pull poison into its own context" — and the answer is gated by structure, not by trusting model output. A refusal does not depend on catching the attack: a tool you never allow-listed is refused regardless of how convincing the injected text is.

## Why are two structural gates better than one well-trained classifier?

Two independent structural gates raise the bar to a conjunctive one: an attacker must beat both, where a single classifier is one point of failure. `fak`'s two gates are the lock (a default-deny capability floor — an irreversible tool that was never allow-listed cannot run, so no injected context changes the verdict) and the wall (result quarantine — poisoned bytes are held out of the model's context entirely). Neither gate is a detector you can talk past. The evadable screener that flags suspicious results sits on top of the wall as a bonus; if it misses, the result is still quarantined by policy, and if it fires, that is extra signal — the floor never depends on it.

## Which OWASP Agentic Top-10 and MCP Top-10 risks does fak target structurally?

`fak` structurally targets Tool Poisoning (MCP03) and Memory Poisoning (T1) by containment and by a capability floor, not by per-attack recognition. For MCP03, untrusted tool results pass a write-time admission gate before they can enter the model's context; a result screened as secret-shaped, injection-shaped, or pollution is paged out to a tiny stub so the poisoned bytes never reach attention. For T1, recall's promotion gate refuses to fold a result into the durable session image unless it is classified durable, and a quarantined page stays sealed across the process boundary unless a witness clears it and a fresh content re-screen passes. The dangerous lever not existing and the poison never arriving are what carry the guarantee, not a model recognizing the attack.

## What does "fail-closed" actually mean inside fak's kernel?

Fail-closed means that when the policy is silent, ambiguous, or broken, the decision defaults to deny rather than allow. A zero policy is the empty floor where every call is refused with `DEFAULT_DENY`; an empty adjudicator chain folds to `DEFAULT_DENY`; and if every rung defers, the verdict is still a deny. The fold is a most-restrictive-wins lattice where an unknown verdict kind ranks as a deny, so a new or malformed rung can only tighten the floor, never loosen it. Config loading is fail-loud to match: a typo'd field name or an unknown refusal reason is a hard startup error, never a silent fallback to a more permissive default.

## Can fak stop a malicious argument to a tool that IS on the allow-list?

Not in the general case — `fak` bounds which tool NAMES can run, but it does not bound the resolved EFFECT of an allow-listed coarse tool's arguments, and the docs say so plainly. An allow-listed `send_email` with attacker-chosen recipients, or a coarse `Bash` running `rm -rf`, is the explicit gap. There are partial, restrict-only mitigations: arg-level predicates can deny by a path glob, a regex, or a max-byte bound on one decoded argument string, and the `SELF_MODIFY` floor refuses write-shaped calls that touch a guarded glob. But those inspect one decoded string, not the resolved effect, and the regex form is detection-shaped and evadable. The honest guidance is to keep exfil-shaped and destructive tools OFF the allow-list and reach for finer argument-scoped capabilities (path/host/amount as first-class constraints), which are roadmap, not shipped.

## If a tool call is admitted, does fak limit its blast radius?

No — once a call is allow-listed and admitted, `fak` does not contain what that call then does in the outside world. The kernel decides whether the call may run and whether its RESULT may re-enter context; it does not sandbox the call's side effects, so an admitted `delete_file` deletes the file. Blast-radius containment is a defense-in-depth job for a separate layer: run the actual tool execution inside a sandbox (for example E2B) so an admitted-but-overbroad action is bounded by the sandbox, while `fak` governs the gate and the result. `fak` governs the syscall boundary; the sandbox governs the effect.

## Does fak protect against request-volume abuse, denial-of-service, or rate-based attacks?

No — `fak` is not a rate limiter or a DoS shield, and request volume is outside what it structurally defends. The kernel's job is per-call adjudication and result admission, not traffic shaping; the closed refusal vocabulary even reserves a `RATE_LIMITED` reason code, but the floor is a permission decision, not a throughput governor. The gateway has operational hardening that is incidental, not a volume defense: a 4 MiB request-body cap, HTTP read/write/idle timeouts, and optional bearer-or-`x-api-key` auth gating every route except `/healthz`. For abuse by request volume, put `fak` behind your own rate limiter or reverse proxy, the same defense-in-depth posture you would use for any upstream.

## Why is the result detector deliberately built to be evadable?

`fak` treats its result detector as roughly 100% evadable by design because the security guarantee is structural, and a guarantee that leaned on pattern-matching would be only as strong as the patterns. The screener is a first-match scan for secret-shaped strings, a fixed set of injection marker phrases, and blatant byte-repeat pollution; any of those is trivially reworded or obfuscated to slip past. So the load-bearing protection is the quarantine POLICY and the capability lock — neither runs the detector. If the screener fires it is a helpful bonus; if it misses, an unlisted irreversible tool is still refused and a poisoned result is still walled by policy. Building it to be beatable is the point: it keeps the floor honest by never letting the detector become load-bearing.

## How does fak keep poison out of the model's context without trusting the detector to catch it?

`fak` quarantines a flagged tool result by physically replacing its bytes with a tiny stub before it can enter context, so the poison is absent from attention rather than merely "not shown." At the write-time admission gate, a quarantined result's payload is paged out to a content-addressed blob store and the in-context payload becomes a small `{"_quarantined":true,...}` pointer; the real bytes only page back in after an explicit witness clear AND a fresh re-screen, both fail-closed. Because `fak` owns the KV cache as a kernel object, the matching K/V span can also be evicted so the model is mechanically incapable of attending to it — verified byte-identical to a session that never saw the poison at max|Δ| = 0. The KV-eviction bridge is proven on a synthetic model in the kvmmu package and is not yet wired into the live agent HTTP loop; the context-side page-out is on the shipped serving path.

## Does the audit log record tool arguments, results, or request bodies?

No — `fak`'s audit surfaces record tool NAMES, verdicts, dispositions, and timings, never request bodies, tool arguments, or result content. The stdout access log emits two JSON lines per request carrying the tool name plus verdict, reason, disposition, duration, status, route, and a `trace_id`, with no payload field at all. The opt-in durable decision journal goes one half-step further: it stores content DIGESTS (the frozen Ref hash) rather than blobs, so it can prove WHICH bytes were seen without leaking them. This is deliberate — the audit trail is reviewable and correlatable by `trace_id` across the access log, the response header, and the per-operation verdict log, without becoming a secondary place secrets pile up.

## How does a memory-poisoning attack survive a session boundary, and how does fak block it?

`fak` blocks memory poisoning at the session boundary by sealing quarantined results into a durable core image and refusing to page them back into a new context without re-clearing them. When `fak recall` persists a finished session, a quarantined page is written with only a safe sealed descriptor (`tool: [sealed: reason, N bytes]`) — never the poisoned or obfuscated bytes — and on reload the rung-4 gate refuses to resolve that page unless a witness clear ran AND a fresh content re-screen passes, so clearance alone cannot launder still-poisoned bytes. The re-screen folds the whole registered admitter chain, so a session recorded under a weaker gate is re-caught by every detector the fleet ships now. The honest limit: recall makes the gate's decision durable and re-screenable, but it does not improve the original decision — an injection that never tripped the gate in the first place is never sealed.

## When a fak policy refuses a call, is that an error your agent has to handle?

No — a refusal is a successful response carried as a value, not an exception, so your agent never treats "the kernel said no" as a crash. On the served path a denied tool call returns HTTP 200 with the verdict in the response body; HTTP error statuses are reserved for malformed requests, auth failures, and upstream faults. The denied call is simply dropped from the model's tool-call list for that turn, with the structured verdict (reason from the closed 17-code vocabulary plus a disposition like `RETRYABLE`, `WAIT`, `ESCALATE`, or `TERMINAL`) available in the `fak` response extension and, for Claude Code, also prepended as a leading `[fak]` text block. Deny-as-value is what lets the agent loop read the refusal in-band and adapt on the next turn rather than erroring out.

## What should I pair fak with for a complete agent security posture?

