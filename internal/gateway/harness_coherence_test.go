package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
)

// TestInboundProtectedPrefixDigestIsContentFree proves the digest is a HASH, never the bytes: it is
// a fixed-width hex string that does not contain the prompt content, is stable for the same prefix,
// and CHANGES when the protected prefix changes (the harness-rewrite signal). A body with no
// cache_control breakpoint has no stable cached head, so it digests to "".
func TestInboundProtectedPrefixDigestIsContentFree(t *testing.T) {
	secret := "SUPER-SECRET-SYSTEM-PROMPT-TEXT"
	body := `{"system":[{"type":"text","text":"` + secret + `","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`

	d := inboundProtectedPrefixDigest([]byte(body))
	if d == "" {
		t.Fatal("digest empty for a body that carries a cache_control breakpoint")
	}
	if len(d) != 64 { // sha256 hex
		t.Fatalf("digest %q is not a 64-char sha256 hex string", d)
	}
	if strings.Contains(d, secret) || strings.Contains(d, "text") || strings.Contains(d, "ephemeral") {
		t.Fatalf("digest leaks content: %q", d)
	}

	// Stable: same prefix bytes -> same digest (idempotent across turns).
	if d2 := inboundProtectedPrefixDigest([]byte(body)); d2 != d {
		t.Fatalf("digest not stable: %q != %q", d2, d)
	}

	// A changed protected prefix (the harness rewriting its own head) -> a different digest.
	rewritten := `{"system":[{"type":"text","text":"REWRITTEN-SUMMARY","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`
	if dr := inboundProtectedPrefixDigest([]byte(rewritten)); dr == d || dr == "" {
		t.Fatalf("rewritten prefix must yield a different non-empty digest; got %q (orig %q)", dr, d)
	}

	// No breakpoint -> no stable cached head -> "".
	noBP := `{"messages":[{"role":"user","content":"hi"}]}`
	if dn := inboundProtectedPrefixDigest([]byte(noBP)); dn != "" {
		t.Fatalf("no cache_control breakpoint must digest to \"\"; got %q", dn)
	}
	// Empty body -> "".
	if inboundProtectedPrefixDigest(nil) != "" {
		t.Fatal("nil body must digest to \"\"")
	}
}

// TestFakBailReasonForMapping pins the #1132 reason mapping: a clean fire ("") and a healthy
// under_budget no-op both map to "" (no bail); any other reason passes through as a real bail.
func TestFakBailReasonForMapping(t *testing.T) {
	if got := fakBailReasonFor(agent.CompactReasonNone); got != "" {
		t.Fatalf("clean fire must map to \"\"; got %q", got)
	}
	if got := fakBailReasonFor(agent.CompactReasonUnderBudget); got != "" {
		t.Fatalf("under_budget must map to \"\"; got %q", got)
	}
	for _, r := range []string{
		agent.CompactReasonPrefixMismatch, agent.CompactReasonCachedSpan,
		agent.CompactReasonWindowNoDrop, agent.CompactReasonSpliceFailed,
		agent.CompactReasonRedecodeFail, agent.CompactReasonNoBreakpoint,
	} {
		if got := fakBailReasonFor(r); got != r {
			t.Fatalf("real bail %q must pass through; got %q", r, got)
		}
	}
}

// TestObserveBuildsTurnObservationAndCountsStable proves a served turn is folded into a real
// compactcohere coordinator (the Decision comes back) and that the shared accumulators increment —
// the seam actually drives the shipped decision surface, not a stub.
func TestObserveBuildsTurnObservationAndCountsStable(t *testing.T) {
	h := newHarnessCoherenceMetrics(compactcohere.DefaultProviderCacheTTL)
	now := time.Unix(1_000_000, 0)

	// First turn on a fresh prefix: stable (nothing to compare against), block-by-default posture.
	d := h.observe("trace-A", now, "digestX", false, "", false, false, 100, 0)
	if d.Event != compactcohere.EventStable {
		t.Fatalf("first turn event = %q, want stable", d.Event)
	}
	if d.HarnessPosture != compactcohere.PostureBlock {
		t.Fatalf("default posture = %q, want block", d.HarnessPosture)
	}
	snap := h.snapshot()
	if snap.observedTurns != 1 {
		t.Fatalf("observedTurns = %d, want 1", snap.observedTurns)
	}
	if snap.events[compactcohere.EventStable] != 1 {
		t.Fatalf("stable count = %d, want 1", snap.events[compactcohere.EventStable])
	}
	if snap.harnessRewrites != 0 || snap.quarantineAtRisk != 0 {
		t.Fatalf("no rewrite/risk yet, got rewrites=%d risk=%d", snap.harnessRewrites, snap.quarantineAtRisk)
	}
}

// TestHarnessRewriteCountedOnPrefixDigestDelta is the core #1132 inference: fak forwards the inbound
// protected prefix verbatim, so a CHANGED inbound-prefix digest across turns can only be the harness
// rewriting its own history. The seam must attribute it to harness_rewrite and count the burst.
func TestHarnessRewriteCountedOnPrefixDigestDelta(t *testing.T) {
	h := newHarnessCoherenceMetrics(compactcohere.DefaultProviderCacheTTL)
	now := time.Unix(2_000_000, 0)

	h.observe("trace-B", now, "digest-1", false, "", false, false, 500, 0)                // turn 1: establishes the prefix
	now = now.Add(time.Second)                                                            // within TTL — not a cold-ttl
	d := h.observe("trace-B", now, "digest-2-DIFFERENT", false, "", false, false, 0, 800) // turn 2: prefix changed

	if d.Event != compactcohere.EventHarnessRewrite {
		t.Fatalf("event = %q, want harness_rewrite (digest changed)", d.Event)
	}
	if !d.BurstObserved {
		t.Fatal("a harness rewrite must mark BurstObserved (it bursts the provider cache)")
	}
	snap := h.snapshot()
	if snap.harnessRewrites != 1 {
		t.Fatalf("harnessRewrites = %d, want 1", snap.harnessRewrites)
	}
	if snap.events[compactcohere.EventHarnessRewrite] != 1 {
		t.Fatalf("harness_rewrite event count = %d, want 1", snap.events[compactcohere.EventHarnessRewrite])
	}
	if snap.burstsObserved != 1 {
		t.Fatalf("burstsObserved = %d, want 1", snap.burstsObserved)
	}
}

// TestQuarantineAtRiskCounted proves the trust-hole signal: a fak-sealed (quarantined) span that
// PRECEDES a harness rewrite may have been folded into the harness's summary, surviving the kernel's
// quarantine. The seam must raise and count quarantine_at_risk exactly on that ordering.
func TestQuarantineAtRiskCounted(t *testing.T) {
	h := newHarnessCoherenceMetrics(compactcohere.DefaultProviderCacheTTL)
	now := time.Unix(3_000_000, 0)

	// Turn 1: fak seals a span (sealed=true) on a stable prefix — the exposure begins.
	d1 := h.observe("trace-C", now, "digest-1", false, "", false, true /*sealed*/, 400, 0)
	if d1.QuarantineAtRisk {
		t.Fatal("no rewrite yet — quarantine_at_risk must not fire on the seal turn alone")
	}
	// Turn 2: the harness rewrites its history (prefix digest changed). The earlier seal may now be
	// inside the harness summary -> quarantine_at_risk.
	now = now.Add(time.Second)
	d2 := h.observe("trace-C", now, "digest-2-DIFFERENT", false, "", false, false, 0, 900)
	if d2.Event != compactcohere.EventHarnessRewrite {
		t.Fatalf("turn 2 event = %q, want harness_rewrite", d2.Event)
	}
	if !d2.QuarantineAtRisk {
		t.Fatal("a seal preceding a harness rewrite must raise QuarantineAtRisk")
	}
	snap := h.snapshot()
	if snap.quarantineAtRisk != 1 {
		t.Fatalf("quarantineAtRisk = %d, want 1", snap.quarantineAtRisk)
	}
}

// TestRenderMetricsEmitsHarnessCoherenceFamily proves the metric family reaches /metrics with the
// shared accumulators behind it: after folding a stable turn, a harness rewrite, and a
// quarantine-at-risk through a REAL Server's gatewayMetrics, the rendered scrape carries the family
// with the witnessed counts — and the operator-line summary folds the SAME numbers (so the two
// views can never disagree, the explicit #1132 requirement).
func TestRenderMetricsEmitsHarnessCoherenceFamily(t *testing.T) {
	srv := newTestServer(t)
	now := time.Unix(4_000_000, 0)

	// Turn 1: seal a span on a stable prefix.
	srv.metrics.observeHarnessCoherence("t", now, "dig-1", false, "", false, true, 300, 0)
	// Turn 2: the harness rewrites the prefix -> harness_rewrite + quarantine_at_risk + burst.
	now = now.Add(time.Second)
	srv.metrics.observeHarnessCoherence("t", now, "dig-2", false, "", false, false, 0, 700)

	text := srv.renderMetrics()
	for _, want := range []string{
		"fak_harness_coherence_turns_total 2",
		`fak_harness_coherence_events_total{event="harness_rewrite"} 1`,
		`fak_harness_coherence_events_total{event="stable"} 1`,
		`fak_harness_coherence_events_total{event="cold_ttl"} 0`, // emitted at 0 so the panel exists
		"fak_harness_coherence_harness_rewrites_total 1",
		"fak_harness_coherence_quarantine_at_risk_total 1",
		"fak_harness_coherence_bursts_total 1",
		"fak_harness_coherence_posture 1", // block-by-default (fak is coping)
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	// The operator-line summary folds the SAME accumulators -> agreement by construction.
	sum := srv.metrics.harnessCoherenceSummary()
	if sum.ObservedTurns != 2 || sum.HarnessRewrites != 1 || sum.QuarantineAtRisk != 1 || sum.BurstsObserved != 1 {
		t.Fatalf("summary disagrees with metrics: %+v", sum)
	}
	if sum.Posture != string(compactcohere.PostureBlock) {
		t.Fatalf("summary posture = %q, want block", sum.Posture)
	}
	if sum.Events["harness_rewrite"] != 1 || sum.Events["stable"] != 1 {
		t.Fatalf("summary events disagree: %+v", sum.Events)
	}
}

// TestPostureYieldsAfterFakBailStreak proves the standing posture is actually driven by the
// coordinator's yield logic: a sustained fak-compaction bail streak flips the posture to allow (hand
// the harness net back), and the rendered posture gauge follows.
func TestPostureYieldsAfterFakBailStreak(t *testing.T) {
	srv := newTestServer(t)
	now := time.Unix(5_000_000, 0)
	// DefaultBailStreakToYield (3) consecutive real bails -> allow.
	for i := 0; i < compactcohere.DefaultBailStreakToYield; i++ {
		now = now.Add(time.Second)
		srv.metrics.observeHarnessCoherence("y", now, "dig", false, agent.CompactReasonPrefixMismatch, false, false, 0, 0)
	}
	if got := srv.metrics.harnessCoherenceSummary().Posture; got != string(compactcohere.PostureAllow) {
		t.Fatalf("after a sustained fak-bail streak posture = %q, want allow", got)
	}
	if text := srv.renderMetrics(); !strings.Contains(text, "fak_harness_coherence_posture 0") {
		t.Fatalf("posture gauge should read 0 (allow) after the bail streak\n%s", text)
	}
}
