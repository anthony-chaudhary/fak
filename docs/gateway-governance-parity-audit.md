---
title: "Gateway governance parity — gap audit"
description: "An honest audit of what `fak serve` already covers against the enterprise-gateway governance checklist, for the three named controls (PII/secret redaction, per-key/per-tenant rate limits, provider failover) — with follow-on issues filed for each true gap."
---

# Gateway governance parity — gap audit (#3280)

Part of epic **#3256** (all-in-one agent runtime — MLP), workstream C (corporate
adoption). This is the honest gap-audit the ticket asks for: what `fak serve`
(`internal/gateway`) **already covers** against the enterprise-gateway governance
checklist, scoped to the three commonly-checked controls — **PII/secret redaction**,
**per-key/per-tenant rate limits**, and **provider failover** — plus the surrounding
floor. It is a *gap-closing* audit, not a rebuild.

Verdict up front: **failover is present + test-covered; secret redaction and
provider-token rate limiting are present + test-covered; general-PII redaction and
per-key/per-tenant rate limits are the two true gaps** — each now has a scoped follow-on
(#5378, #5379).

## The surrounding floor (already shipped)

| Control | Status | Evidence |
|---|---|---|
| Request routing / tiered serving | ✅ | `routing.go` (`Router.Route`, tiers, size/cost/latency/hybrid strategies) |
| Policy adjudication (pre-exec verdict) | ✅ | `internal/adjudicator`; `fak_adjudicate` / `POST /v1/fak/adjudicate` |
| Result quarantine (poison/injection) | ✅ | `internal/normgate`; `QUARANTINE` verdict, page-in re-screen |
| Audit (tamper-evident rows) | ✅ | audit journal, `/metrics`, `/debug/vars`, `X-Trace-Id` joins |
| Auth — single key | ✅ | `RequireKey` (constant-time SHA-256 compare) |
| Auth — multi-key, per-tenant identity | ✅ | `keyset.go` (#5332): each api key → org/project **principal**, rides the request as `X-Fak-Principal`, attributed in audit |
| Revocation (witness → evict) | ✅ | `fak_revoke` / `POST /v1/fak/revoke` |

The vendor-neutral shape of this floor is written up in
[`docs/standards/agent-tool-governance-gateway.md`](standards/agent-tool-governance-gateway.md).

## The three named controls

### 1. PII / secret redaction — PARTIAL (secret ✅, PII ⚠️ gap)

**Secret / credential redaction — present + test-covered.**

- `internal/normgate` masks credential spans **in place** by default
  (`SECRET_REDACTED`, warn-first — the rest of the output stays in context) and **seals**
  the whole result under the opt-in fail-closed posture (`SECRET_EXFIL`). An *obfuscated*
  secret (only visible on a de-obfuscated view) seals even under the permissive default.
  Tests: `internal/normgate/normgate_test.go`
  (`TestPlaintextSecretRedactedByDefault`, `TestPlaintextSecretSealsUnderFailClosed`,
  `TestObfuscatedSecretSealsEvenByDefault`, `TestTrustedLocalSecretRedactedByDefault`).
  Surfaced to the model as a one-line warn via `secretRedactedWarn` (`messages.go`).
- Outbound (proposed-exfil) taint gate on the served proxy path:
  `proxy_exfil_floor_test.go` (`TestChatProxyResultTaintGatesProposedExfil`) and
  `anthropic_exfil_floor_test.go` (`TestAnthropicProxyResultTaintGatesProposedExfil`).
- 403-body scrub for the operator `/debug/vars` surface: `forbidden_detail.go`
  (`scrubForbiddenDetail`, `tokenShapedRe`); `forbidden_detail_test.go`
  (`TestScrubForbiddenDetail_RedactsSecretsKeepsReason`, `..._RedactsBearerAndEntropy`,
  `..._Bounded`).
- Egress-delivery redaction before send: `delivery_send.go` (`redactedBody`,
  `egressfloor`); wire surface `wireRedactionsFrom` (`wire.go`), `redaction_fakext_test.go`
  (`TestWireRedactionsFrom`).

**General-PII redaction — GAP.** All of the above match **credential shapes**
(`sk-…`, `Bearer …`, high-entropy runs) and prompt-injection. There is **no** detector for
general PII — email addresses, phone numbers, national IDs (SSN), payment card / IBAN
numbers, postal addresses. A tool result or model response carrying a customer's email or
card number passes through unmasked. → **follow-on #5378** (add a deterministic PII needle
set composing with the `normgate` mask-in-place / fail-closed seam, inbound + outbound).

### 2. Per-key / per-tenant rate limits — PARTIAL (provider-budget ✅, per-key ⚠️ gap)

**Provider-token rate limiting — present + test-covered.** `token_admission.go`
(`TokenRateGate`) enforces a rolling-window TPM + concurrency budget built on the pure caps
arithmetic in `internal/ratelimit`: reserve-on-estimate, settle-on-provider-truth, shed
locally (typed 429) instead of a 429 storm at the provider. Tests:
`token_admission_test.go` (`TestTokenRateGateReserveSettleAndWindowRoll`,
`TestTokenRateGateConcurrencyAndTargetUtilization`, `TestServedAdmissionTokenGateWiring`,
`TestEstimateServedTokenUsageSplitsInputOutput`).

**Per-key / per-tenant limits — GAP.** `TokenRateGate` is scoped to **one provider budget
(one account/seat)** — its own honest fence states the per-seat pool composes *above* it.
The tenant identity axis exists independently (`keyset.go` principal), but nothing keys the
rate gate by the authenticated principal, so a single noisy key/tenant can consume the whole
provider budget and there is no per-principal ceiling. → **follow-on #5379** (key a
`TokenRateGate` pool by the keyset principal; compose with budgets #C1 + tenancy #B5;
fail-closed; 429 names the principal + cap).

### 3. Provider failover — PRESENT + test-covered ✅

- **Health-aware tier fallback chain.** `Router.Route` returns an ordered
  `Decision.Fallbacks`; a hard-down tier (`SetHealth`) is routed around, `ErrNoTier` is a
  structured refusal (503/replan), never a silent mis-route. Tests: `routing_test.go`
  (`TestRouter_Health_FallsBackToNextTier`, `TestRouter_Health_AllDownIsErrNoTier`).
- **Persistent-403 auto-demote / auto-switch.** `Router.DemoteModel` routes around a model
  a credential is not entitled for and **self-recovers** after a cooldown (entitlement that
  returns is picked up with no operator un-demote). Tests: `routing_demote_test.go`
  (`TestRouter_DemoteModel_RoutesAroundDeniedTier`, `..._SelfRecoversAfterCooldown`,
  `..._ExtendsNeverShortens`, `..._NoOpGuards`).
- **Account-scoped credential failover.** `AccountFailoverFunc` / `onAccountFailover`
  (`gateway.go`) swap in a replacement upstream credential on account failure, observed on
  `/debug/vars` and fleet metrics. Tests: `upstream_error_visibility_test.go`
  (`TestDebugVarsExposeAccountFailover`), `fleet_membership_test.go`
  (`TestFleetMembershipDispatchFailoverAndTypedVerdict`),
  `fleet_membership_metrics_test.go` (`TestFleetMembershipMetricsFailoverAndLiveGauges`).

Fail-closed on residency is preserved throughout: `MaxCostPerMTok` refuses rather than
silently serving a pricier tier, and `ErrNoTier` refuses rather than mis-routing.

## Summary

| Named control | Verdict | Follow-on |
|---|---|---|
| Secret / credential redaction | present + test-covered | — |
| General-PII redaction | gap | **#5378** |
| Provider-token rate limiting | present + test-covered | — |
| Per-key / per-tenant rate limits | gap | **#5379** |
| Provider failover (tier + model-demote + account) | present + test-covered | — |

Both #3280 acceptance criteria are met: this note is the honest gap-audit, and each of the
three named controls is either present + test-covered or has a scoped follow-on for its
remaining gap.

## Evidence & assumptions (generation close-out)

- **Promotion evidence** (what earned "present + test-covered"): the named tests above run
  under the lane gate `go test ./internal/gateway` / `./internal/normgate`; each control is
  cited to a concrete impl file *and* a test that exercises it.
- **Demotion / retirement evidence:** nothing is retired here. Two capabilities the ticket
  assumed might be "partial/absent" are demoted from *unknown* to *confirmed-present* by the
  citations above (secret redaction, provider failover); the two genuine gaps are demoted
  from *this ticket's scope* to filed follow-ons (#5378, #5379) rather than left implicit.
- **Invalidating assumption:** this audit reads the redaction machinery as *credential/
  injection only* from the pattern sets (`tokenShapedRe`, normgate secret canon) and their
  tests. If a PII needle set is later found wired through a policy/config path not exercised
  by the gateway/normgate tests, the "PII gap" (#5378) would be invalid and should be closed
  as already-covered instead of implemented.
