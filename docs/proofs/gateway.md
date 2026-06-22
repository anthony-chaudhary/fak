---
title: "fak proof: gateway wire verdict parity"
description: "Proof that fak's MCP and HTTP gateway returns the same kernel verdict as in-process and drops failed calls fail-closed, never a network bypass."
---

# D10 · gateway

The gateway is the kernel-adjudicated wire: it fronts the in-process fak kernel over MCP (newline-delimited JSON-RPC) and an OpenAI- and Anthropic-compatible HTTP surface so an agent written in any language can route its tool calls through the syscall boundary without writing Go (`gateway.go:1-27`). It computes nothing numerical; what it *computes* is a **routing of every wire request onto the one in-process kernel's decision** and a **fail-closed projection of that decision back onto the wire**. "Correct" here is regime **D — decision-procedure soundness**: (a) the network seam must not be a bypass — a verdict over the wire must equal the verdict the same kernel renders in-process, and the wire must never hand the kernel a pre-trusted handle that skips a rung; and (b) a call that fails adjudication must be *dropped fail-closed* — structurally removed from what the client receives, with the default branch being deny, not allow. Both are discharged below by deterministic Go witnesses run on this macOS native node (go1.26 darwin/arm64); no oracle fixture, network, or RNG is involved.

---

## THEOREM 1 — wire-fronted kernel returns the SAME verdict as the in-process kernel (no network bypass)

**REGIME** D — decision-procedure soundness (single-seam composition).

**PROOF.** There is exactly one decision seam. `s.adjudicate` calls `v := s.k.Decide(ctx, tc)` (`gateway.go:368`) and `s.syscall` calls `r, v := s.k.Syscall(ctx, tc)` (`gateway.go:397`), both over the *single* `*kernel.Kernel` held in `Server.k` (`gateway.go:157`), constructed once in `New` (`gateway.go:256`). Every wire entry point — the HTTP `/v1/fak/{syscall,adjudicate}` handlers, the Anthropic `/v1/messages` proxy, and the OpenAI `/v1/chat/completions` proxy — routes through these two methods; none contains an independent decision. Untrusted wire bytes are re-validated into an `abi.ToolCall` by `buildCall` (`gateway.go:413`), which **mints a tainted, agent-scoped Ref itself** (`gateway.go:421`) precisely because the wire never carries an `abi.Ref`, so a client cannot smuggle a pre-trusted CAS handle to skip the IFC / self-modify rungs. `renderVerdict` (`wire.go:31`) is a pure, total name projection of the folded `abi.Verdict` onto the closed wire vocabulary — it changes the *name*, never the *decision* — and the projection is exhaustively pinned by `TestRenderVerdict` (`gateway_test.go:138`).

The verdict parity is witnessed by a matched pair over **one shared kernel and one shared `toolAdj` adjudicator** (`gateway_test.go:53`): in-process `srv.adjudicate(ctx,"deny_thing",…)` yields `DENY` (`TestServedAdjudicateEmitsDecisionEvents`, `gateway_test.go:299`), and the wire `POST /v1/fak/syscall {deny_thing}` yields `DENY/POLICY_BLOCK/TERMINAL` (`TestHTTPSyscallDenyIsValueNot5xx`, `gateway_test.go:234`). An unknown tool `DEFAULT_DENY`s identically over the wire (`TestHTTPSyscallUnknownToolDefaultDeny`), and an unwitnessable require-witness gate fails **closed** to `DENY/UNWITNESSED` over the wire (`TestHTTPSyscallWitnessFailsClosed`, `gateway_test.go:314`). *Honest gap:* no single test runs one call both ways and `reflect.DeepEqual`s the two `WireVerdict`s; parity rests on the matched pair plus the structural single-seam argument. A literal A==B equality test driven by a shared table would be the tightest form and is the obvious next witness.

**WITNESS.** `go test -run 'Verdict|Adjud|HTTPSyscall|DefaultDeny|DenyIsValue|FailsClosed' ./internal/gateway/ -count=1 -timeout 180s -v` — all green (incl. the 9 `TestRenderVerdict` subcases).

**VERDICT.** PROVEN — 2026-06-20, on this macOS native node (16 PASS / 0 FAIL / 0 SKIP across the run).

**DOS.** bound at ship.

---

## THEOREM 2 — a tool-call that fails adjudication is DROPPED fail-closed, and the drop is solid

**REGIME** D — decision-procedure soundness (fail-closed fold).

**PROOF.** `adjudicateProposed` (`adjudicate_proposed.go:10`) folds each proposed call through `s.adjudicate` and appends to the surviving `kept` slice **only** on `case "ALLOW"` (`:25`) or `case "TRANSFORM"` (`:28`). The `default:` branch (`:35`) increments `dropped` and appends nothing — so `DENY`, `QUARANTINE`, `REQUIRE_WITNESS`, `DEFER`, and any unknown restrictive kind all fall through to the drop. A call the kernel cannot even build/adjudicate (`aerr != nil`, `:17`) is likewise dropped with a synthesized `DENY/MALFORMED` — fail-closed on error, never passed through raw. The drop is *applied* by replacing the assistant's tool calls with the survivor set: on the Anthropic wire, `completeAnthropicTurn` sets `asst.ToolCalls = kept` (`messages.go:203-204`), so a dropped call never becomes a `tool_use` content block the client could execute.

Solidity is witnessed by an A/B over **one fixed planner that always proposes the same egress call**: `TestAnthropicProxyResultTaintGatesProposedExfil` (`anthropic_exfil_floor_test.go:36`). Branch A injects an untrusted `tool_result`; `admitInboundResults` (`gateway.go:516`) routes it through `k.AdmitResult` and raises the IFC taint high-water mark on the trace; the already-wired sink-gate then `DENY`s the identical egress call, and the test asserts **zero** surviving `tool_use` blocks (`:80`) plus a recorded `DENY/TRUST_VIOLATION` (`:86`). Branch B sends the byte-identical call with no taint → `ALLOW` → exactly one surviving block (`:111`). Only the taint differs; only the survivor count differs — that is what makes the drop *causal and solid* rather than incidental. The OpenAI-wire twin (`TestChatProxyResultTaintGatesProposedExfil`) and the multi-verdict mix allow-kept / deny-dropped / transform-kept (`TestAnthropicMessagesPassthroughPreservesCacheAndAdjudicates`) corroborate.

**WITNESS.** `go test -run 'Floor|Exfil|Adjud' ./internal/gateway/ -count=1 -timeout 180s -v` — all green.

**VERDICT.** PROVEN — 2026-06-20, on this macOS native node (tainted session: 0 `tool_use` survive + `DENY/TRUST_VIOLATION` + IFC ledger raised; clean session: identical call kept, 1 `tool_use`, ledger `Trusted`).

**DOS.** bound at ship.
