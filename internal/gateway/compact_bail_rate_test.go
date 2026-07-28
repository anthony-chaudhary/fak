package gateway

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// compactCell drives the REAL observeCompaction the serve path calls (internal/gateway
// messages.go), so these numbers are produced by production bookkeeping rather than by a
// hand-built map. fires counts fired attempts; bails is reason -> count.
func compactCell(t *testing.T, fires int, bails map[string]int) *gatewayMetrics {
	t.Helper()
	m := newGatewayMetrics(time.Now())
	for i := 0; i < fires; i++ {
		m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonNone, Dropped: 3, ShedTokens: 400}, false)
	}
	for reason, n := range bails {
		for i := 0; i < n; i++ {
			m.observeCompaction(agent.CompactOutcome{Reason: reason}, false)
		}
	}
	return m
}

// rawBailRate is the rate that shipped before #5443: every non-fire over every attempt.
// Reproduced here (rather than read from a helper) so the test pins the DEFECT as well as the
// fix — if the old reading were not really unseparable, the fixture would prove nothing.
func rawBailRate(m *gatewayMetrics) float64 {
	snap := m.compactionSnapshotData()
	fired, bailed := snap.attempts["fired"], snap.attempts["bailed"]
	if fired+bailed == 0 {
		return 0
	}
	return float64(bailed) / float64(fired+bailed)
}

func candidateBailRateOf(m *gatewayMetrics) float64 {
	snap := m.compactionSnapshotData()
	_, candidateBails := compactBailPartition(snap.bailReasons)
	return compactCandidateBailRate(snap.attempts["fired"], candidateBails)
}

func round4(v float64) float64 { return math.Round(v*1e4) / 1e4 }

// TestCandidateBailRateSeparatesHealthyFromBrokenWhereRawRateCannot is the #5443 witness.
//
// Two cells carry the SAME 900-request non-candidate flood — short auxiliary pings, non-JSON
// bodies, absent messages[], undecodable messages[] — which is what real mixed fleet traffic
// looks like once a compaction budget is set and every Anthropic passthrough is attempted:
//
//	healthy: 50 fires,  0 eligible bails, 900 non-candidate bails
//	broken:   1 fire,  49 eligible bails, 900 non-candidate bails
//
// The healthy cell compacts on nearly every attempt that COULD compact; the broken one bails
// 98% of them, mostly to prefix_mismatch — the single fak-fault cache signal. Under the raw
// rate they read 0.9474 and 0.9989 and no alert threshold usefully separates them. Under the
// candidate rate they read 0.0000 and 0.9800.
func TestCandidateBailRateSeparatesHealthyFromBrokenWhereRawRateCannot(t *testing.T) {
	nonCandidateFlood := map[string]int{
		agent.CompactReasonTooFewMsgs:   600,
		agent.CompactReasonNonJSON:      150,
		agent.CompactReasonNoMsgsKey:    100,
		agent.CompactReasonDecodeFailed: 50,
	}
	floodTotal := 0
	for _, n := range nonCandidateFlood {
		floodTotal += n
	}
	if floodTotal != 900 {
		t.Fatalf("flood = %d, want 900 — the fixture's arithmetic must be exact", floodTotal)
	}

	healthyBails := map[string]int{}
	for r, n := range nonCandidateFlood {
		healthyBails[r] = n
	}
	healthy := compactCell(t, 50, healthyBails)

	brokenBails := map[string]int{
		agent.CompactReasonPrefixMismatch: 30,
		agent.CompactReasonSpliceFailed:   12,
		agent.CompactReasonRedecodeFail:   7,
	}
	for r, n := range nonCandidateFlood {
		brokenBails[r] = n
	}
	broken := compactCell(t, 1, brokenBails)

	// --- the DEFECT, pinned. ---
	rawHealthy, rawBroken := round4(rawBailRate(healthy)), round4(rawBailRate(broken))
	if rawHealthy != 0.9474 || rawBroken != 0.9989 {
		t.Fatalf("raw bail rate = healthy %.4f / broken %.4f, want 0.9474 / 0.9989 — the fixture no longer reproduces the reported defect", rawHealthy, rawBroken)
	}
	// A conventional "most attempts are declining" alert sits somewhere at or below 0.95. At
	// any such threshold BOTH cells fire, including the one where the compactor is working
	// perfectly. The only thresholds that separate them live in the 0.0516-wide sliver
	// (0.9474, 0.9989] at the very top of the range — which is what "cannot be alerted on"
	// means concretely.
	for _, threshold := range []float64{0.5, 0.75, 0.9, 0.9474} {
		if !(rawHealthy >= threshold && rawBroken >= threshold) {
			t.Fatalf("raw rate at threshold %.4f: healthy %.4f / broken %.4f — the defect claim requires BOTH to alert", threshold, rawHealthy, rawBroken)
		}
	}
	if gap := rawBroken - rawHealthy; gap >= 0.06 {
		t.Fatalf("raw-rate gap = %.4f, want < 0.06 — a healthy and a broken compactor must be indistinguishable for this fixture to reproduce #5443", gap)
	}

	// --- the FIX. ---
	newHealthy, newBroken := round4(candidateBailRateOf(healthy)), round4(candidateBailRateOf(broken))
	if newHealthy != 0.0000 || newBroken != 0.9800 {
		t.Fatalf("candidate bail rate = healthy %.4f / broken %.4f, want 0.0000 / 0.9800", newHealthy, newBroken)
	}
	const alertAt = 0.9
	if newHealthy >= alertAt || newBroken < alertAt {
		t.Fatalf("candidate rate does not separate at the %.2f threshold: healthy %.4f / broken %.4f", alertAt, newHealthy, newBroken)
	}

	// --- the counters the fix must NOT redefine. ---
	for _, c := range []struct {
		name       string
		m          *gatewayMetrics
		wantFired  uint64
		wantBailed uint64
		wantNonCan uint64
	}{
		{"healthy", healthy, 50, 900, 900},
		{"broken", broken, 1, 949, 900},
	} {
		snap := c.m.compactionSnapshotData()
		if snap.attempts["fired"] != c.wantFired || snap.attempts["bailed"] != c.wantBailed {
			t.Errorf("%s attempts = fired %d / bailed %d, want %d / %d — attempts{} keeps its existing meaning for gauge continuity",
				c.name, snap.attempts["fired"], snap.attempts["bailed"], c.wantFired, c.wantBailed)
		}
		nonCandidate, candidate := compactBailPartition(snap.bailReasons)
		if nonCandidate != c.wantNonCan {
			t.Errorf("%s non-candidate bails = %d, want %d", c.name, nonCandidate, c.wantNonCan)
		}
		if nonCandidate+candidate != snap.attempts["bailed"] {
			t.Errorf("%s partition = %d + %d, want it to sum to bailed %d — no bail may be dropped by the split",
				c.name, nonCandidate, candidate, snap.attempts["bailed"])
		}
	}
}

// TestCandidateBailRateFailsOpenOnUnregisteredReason: a reason internal/agent has not
// registered must count as a CANDIDATE, so a vocabulary addition that skips registration
// leaves the rate conservatively high instead of silently shrinking the measured population.
func TestCandidateBailRateFailsOpenOnUnregisteredReason(t *testing.T) {
	m := compactCell(t, 1, map[string]int{
		agent.CompactReasonTooFewMsgs: 98, // never a candidate
		"reason_from_the_future":      1,  // unregistered
	})
	nonCandidate, candidate := compactBailPartition(m.compactionSnapshotData().bailReasons)
	if nonCandidate != 98 || candidate != 1 {
		t.Fatalf("partition = %d non-candidate / %d candidate, want 98 / 1 (the unregistered reason must land on the candidate side)", nonCandidate, candidate)
	}
	if got := round4(candidateBailRateOf(m)); got != 0.5 {
		t.Fatalf("candidate bail rate = %.4f, want 0.5000 — the unknown reason must raise the rate, never lower it", got)
	}
}

// TestCompactionBailRateSeriesRendered drives the real Prometheus rendering function so the
// fix is reachable from a scrape, not merely computable in a test. Reverting the
// metrics_render.go edit reds this.
func TestCompactionBailRateSeriesRendered(t *testing.T) {
	m := compactCell(t, 1, map[string]int{
		agent.CompactReasonTooFewMsgs:     900,
		agent.CompactReasonPrefixMismatch: 49,
	})

	var b strings.Builder
	m.writeCompactionMetrics(&b)
	out := b.String()

	for _, want := range []string{
		// unchanged by this fix
		`fak_gateway_compaction_attempts_total{outcome="fired"} 1`,
		`fak_gateway_compaction_attempts_total{outcome="bailed"} 949`,
		`fak_gateway_compaction_bail_reason_total{reason="too_few_msgs"} 900`,
		`fak_gateway_compaction_bail_reason_total{reason="prefix_mismatch"} 49`,
		// added alongside
		"fak_gateway_compaction_non_candidate_bails_total 900",
		"fak_gateway_compaction_candidate_bail_rate 0.98",
		"# TYPE fak_gateway_compaction_candidate_bail_rate gauge",
		"# TYPE fak_gateway_compaction_non_candidate_bails_total counter",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("/metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

// TestCompactionBailHelpEnumeratesEveryEmittedReason is the #5441 witness. The HELP for
// fak_gateway_compaction_bail_reason_total claims its label set is CLOSED; that claim is
// falsifiable, and it was false — 9 members declared against 13 emitted, so decode_failed
// (#5387), cached_span, malformed_body and burst_unprofitable were invisible to anyone who
// built an alert over the declared labels. The list is now derived from
// agent.CompactBailReasons(), so this asserts the derivation reaches the rendered bytes.
func TestCompactionBailHelpEnumeratesEveryEmittedReason(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var b strings.Builder
	m.writeCompactionMetrics(&b)

	help := ""
	for _, line := range strings.Split(b.String(), "\n") {
		if strings.HasPrefix(line, "# HELP fak_gateway_compaction_bail_reason_total ") {
			help = line
			break
		}
	}
	if help == "" {
		t.Fatalf("no HELP line for fak_gateway_compaction_bail_reason_total\n--- got ---\n%s", b.String())
	}

	reasons := agent.CompactBailReasons()
	if len(reasons) < 13 {
		t.Fatalf("agent.CompactBailReasons() = %v (%d) — the compaction path emitted 13 bail reasons at the time of #5441; a shrunk vocabulary needs re-checking, not a relaxed test", reasons, len(reasons))
	}
	for _, r := range reasons {
		if !strings.Contains(help, r) {
			t.Errorf("HELP omits emitted bail reason %q — a closed-set claim that drops a label is what #5441 reports\n--- HELP ---\n%s", r, help)
		}
	}
	// The four that were missing, named explicitly so a regression says WHICH.
	for _, r := range []string{
		agent.CompactReasonDecodeFailed,
		agent.CompactReasonCachedSpan,
		agent.CompactReasonMalformedBody,
		agent.CompactReasonBurstUnprofitable,
	} {
		if !strings.Contains(help, r) {
			t.Errorf("HELP omits %q — one of the four members the hand-typed closed set dropped", r)
		}
	}
	// A HELP line must stay one line, or the exposition is malformed.
	if strings.Contains(help[len("# HELP "):], "\n") {
		t.Errorf("HELP line contains a newline")
	}
	if !strings.Contains(help, strings.Join(reasons, "|")) {
		t.Errorf("HELP does not carry the vocabulary as a pipe-joined list; it must be derived from agent.CompactBailReasons(), not re-typed\n--- HELP ---\n%s", help)
	}
}
