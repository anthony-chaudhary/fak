package gatewayusageledger

import (
	"math"
	"strings"
	"testing"
	"time"
)

// compactionCell builds one exit row in a fixed (48k/0-20 by default) cell. Every fixture
// here is synthesized in memory — no ledger file, no .fak state — so the witness cannot
// drift with the operator's corpus.
func compactionCell(budget int, turns uint64, fired uint64, shed uint64, cached uint64, bailed uint64, reasons map[string]uint64) Row {
	return NewRow("exit", "guard", "claude", "", 0, &Provenance{CompactHistoryBudget: budget}, Counters{
		ObservedTurns:         turns,
		CompactionFired:       fired,
		CompactionShedTokens:  shed,
		CachedPromptTokens:    cached,
		CompactionBailed:      bailed,
		CompactionBailReasons: reasons,
	}, time.Unix(5000, 0))
}

// TestCandidateBailRateExcludesNonCandidates drives FoldCompaction — the entry point
// `fak cachevalue compaction` (cmd/fak/cachevalue_compaction.go) and the Prometheus
// exposition (cmd/fak/cachevalue_metrics.go) both call — over a mixed population of
// SERVED requests, GENUINE declines, and NON-CANDIDATES, and pins the thing that
// actually matters: the RATE.
//
// The population below is the shape #5388 reports on real traffic — a stream of sub-3
// message subagent pings and unparseable bodies riding the same Anthropic passthrough as
// a handful of real sessions. Under the old accounting every one of those counted as a
// declined compaction, so bail_rate read 0.99 and could not have read anything else.
func TestCandidateBailRateExcludesNonCandidates(t *testing.T) {
	rows := []Row{
		// SERVED: two compactions actually fired and shed.
		compactionCell(48000, 5, 2, 100, 900, 0, nil),
		// The attempt stream: 4 genuine declines about real candidates, 194 requests that
		// were never candidates at all (190 too_few_msgs + 1 non_json + 1 no_messages_key
		// + 2 decode_failed).
		compactionCell(48000, 6, 0, 0, 0, 198, map[string]uint64{
			"under_budget":       3,
			"burst_unprofitable": 1,
			"too_few_msgs":       190,
			"non_json":           1,
			"no_messages_key":    1,
			"decode_failed":      2,
		}),
	}

	rep := FoldCompaction(rows, "")
	seg := findSeg(t, rep, 48000, "0-20")

	// Logged so `-v` carries the before/after reading of the same population, not just a
	// pass: the old accounting's rate and the new one, side by side.
	t.Logf("population: %d fires, %d bails (%d never candidates) -> bail_rate=%.4f candidate_bail_rate=%.4f",
		seg.Fires, seg.Bails, seg.NonCandidateBails, seg.BailRate, seg.CandidateBailRate)

	// Continuity: Bails and BailRate are untouched, so the existing gauge and any consumer
	// reading them keeps its meaning — and the near-1.0 pinning stays literally visible.
	if seg.Bails != 198 || seg.Fires != 2 {
		t.Fatalf("fires/bails = %d/%d, want 2/198 (raw counters must be preserved for continuity)", seg.Fires, seg.Bails)
	}
	if seg.BailRate < 0.98 {
		t.Fatalf("BailRate = %.4f, want >= 0.98 — the fixture must reproduce the pinned-near-1.0 old accounting", seg.BailRate)
	}

	// The partition: 194 of the 198 bails were never candidates.
	if seg.NonCandidateBails != 194 {
		t.Fatalf("NonCandidateBails = %d, want 194 (190 too_few_msgs + 1 non_json + 1 no_messages_key + 2 decode_failed)", seg.NonCandidateBails)
	}

	// The fix: 4 declines over 6 eligible attempts (2 fired + 4 declined) = 0.6667, a rate
	// that can move. Under the old accounting the SAME population read 0.99.
	if math.Abs(seg.CandidateBailRate-4.0/6.0) > 1e-9 {
		t.Fatalf("CandidateBailRate = %.4f, want %.4f (4 candidate bails / (2 fires + 4 candidate bails)); BailRate on the same population was %.4f",
			seg.CandidateBailRate, 4.0/6.0, seg.BailRate)
	}
	if seg.CandidateBailRate >= seg.BailRate {
		t.Fatalf("CandidateBailRate %.4f did not fall below BailRate %.4f — non-candidates are still in the denominator", seg.CandidateBailRate, seg.BailRate)
	}

	// Nothing is hidden: the full per-reason mix still carries every non-candidate, so
	// decode_failed stays assertable as fak-fault on its own (#5387).
	if seg.BailReasons["too_few_msgs"] != 190 || seg.BailReasons["decode_failed"] != 2 {
		t.Fatalf("BailReasons lost the non-candidate detail (it must stay visible): %+v", seg.BailReasons)
	}

	// TopBailReason is restricted to candidates: too_few_msgs wins on raw volume 190-to-3
	// and would be the single cause an operator is pointed at — the one bucket with nothing
	// to do. under_budget must win instead, at 3/4 of the CANDIDATE bails.
	if seg.TopBailReason != "under_budget" {
		t.Fatalf("TopBailReason = %q, want under_budget (a non-candidate must not win on volume)", seg.TopBailReason)
	}
	if math.Abs(seg.TopBailShare-0.75) > 1e-9 {
		t.Fatalf("TopBailShare = %.4f, want 0.75 (3 under_budget / 4 candidate bails)", seg.TopBailShare)
	}

	// The rendered table prints the candidate rate and the held-out population, so the
	// operator reads both without re-folding the ledger.
	out := RenderCompaction(rep)
	if !strings.Contains(out, "candbail") || !strings.Contains(out, "noncand") {
		t.Fatalf("render missing the candbail/noncand columns:\n%s", out)
	}
	if !strings.Contains(out, "0.67") {
		t.Fatalf("render missing the candidate bail rate 0.67:\n%s", out)
	}
	if !strings.Contains(out, "under_budget·75%") {
		t.Fatalf("render top_bail did not restrict to candidates (want under_budget·75%%):\n%s", out)
	}
}

// TestCandidateBailRateSeparatesHealthyFromBroken is the informativeness witness: a metric
// that reads ~1.0 on a healthy compactor AND on a broken one signals nothing, no matter how
// correct each individual classification is. Two cells carry the SAME non-candidate flood;
// one serves every real candidate, the other declines every real candidate. bail_rate cannot
// tell them apart — both land above 0.94, five points from each other. candidate_bail_rate
// puts them at opposite ends of its range.
func TestCandidateBailRateSeparatesHealthyFromBroken(t *testing.T) {
	rows := []Row{
		// HEALTHY (interactive 48k): 50 fires, and the only bails are non-candidates.
		compactionCell(48000, 5, 50, 5000, 5000, 900, map[string]uint64{"too_few_msgs": 900}),
		// BROKEN (headless 96k): the same 900-request non-candidate flood, but of the 50
		// real candidates only 1 fired — 49 were declined under_budget.
		compactionCell(96000, 5, 1, 100, 900, 949, map[string]uint64{"too_few_msgs": 900, "under_budget": 49}),
	}

	rep := FoldCompaction(rows, "")
	healthy := findSeg(t, rep, 48000, "0-20")
	broken := findSeg(t, rep, 96000, "0-20")

	// The defect: under the old accounting both cells read near 1.0 and sit within a few
	// points of each other. No threshold separates them.
	if healthy.BailRate < 0.94 || broken.BailRate < 0.94 {
		t.Fatalf("fixture does not reproduce the defect: BailRate healthy=%.4f broken=%.4f, both must be >= 0.94", healthy.BailRate, broken.BailRate)
	}
	if gap := math.Abs(broken.BailRate - healthy.BailRate); gap > 0.06 {
		t.Fatalf("BailRate gap = %.4f, want <= 0.06 — the fixture must show bail_rate CANNOT separate the two states (healthy=%.4f broken=%.4f)",
			gap, healthy.BailRate, broken.BailRate)
	}

	// The fix: the healthy cell declined nothing eligible, the broken cell declined almost
	// everything eligible. Same corpus, same non-candidate flood, opposite readings.
	if healthy.CandidateBailRate != 0 {
		t.Fatalf("healthy CandidateBailRate = %.4f, want 0 (every eligible attempt fired)", healthy.CandidateBailRate)
	}
	if math.Abs(broken.CandidateBailRate-49.0/50.0) > 1e-9 {
		t.Fatalf("broken CandidateBailRate = %.4f, want %.4f (49 declines / 50 eligible)", broken.CandidateBailRate, 49.0/50.0)
	}
	if gap := broken.CandidateBailRate - healthy.CandidateBailRate; gap < 0.9 {
		t.Fatalf("CandidateBailRate gap = %.4f, want >= 0.9 — the rate is still not informative (healthy=%.4f broken=%.4f)",
			gap, healthy.CandidateBailRate, broken.CandidateBailRate)
	}

	// A cell whose bails were ALL non-candidates has no actionable top reason: the operator
	// must not be pointed at too_few_msgs, where there is nothing to do.
	if healthy.TopBailReason != "" {
		t.Fatalf("healthy TopBailReason = %q, want empty (no candidate bailed)", healthy.TopBailReason)
	}
	if broken.TopBailReason != "under_budget" {
		t.Fatalf("broken TopBailReason = %q, want under_budget", broken.TopBailReason)
	}
}

// TestNonCandidatePartitionPlacements pins the exact membership decision, reason by reason,
// so a later edit to the vocabulary cannot quietly move one across the line. The rule is
// WHERE the compactor decided, not whether the outcome was benign: the eligibility gate runs
// before a candidate span exists (non-candidate), everything after it aborts a real
// candidate (declined). decode_failed is a structural fault and still a non-candidate;
// prefix_mismatch and malformed_body are structural faults that ARE candidates.
func TestNonCandidatePartitionPlacements(t *testing.T) {
	for _, r := range []string{"too_few_msgs", "non_json", "no_messages_key", "decode_failed"} {
		if !compactionReasonIsNonCandidate(r) {
			t.Errorf("%q must be a NON-candidate: it is decided at the eligibility gate, before any compactible span exists", r)
		}
	}
	for _, r := range []string{
		"under_budget", "burst_unprofitable", "cached_span", "no_breakpoint", "window_no_drop",
		"splice_failed", "redecode_failed", "prefix_mismatch", "malformed_body",
	} {
		if compactionReasonIsNonCandidate(r) {
			t.Errorf("%q must stay a CANDIDATE bail: a real candidate existed and was declined or aborted", r)
		}
	}
	// Fail-open: an unrecognized reason stays in the eligible denominator, so a vocabulary
	// member added upstream leaves the rate conservatively high instead of silently
	// shrinking the measured population.
	if compactionReasonIsNonCandidate("some_reason_added_later") {
		t.Errorf("an unrecognized reason must default to CANDIDATE (fail-open)")
	}
}

// TestCandidateBailRateWithoutReasonMap — a row that recorded a bail COUNT but no reason map
// (the pre-#1407 schema, still present in the durable ledger) has no classification to lean
// on. Those bails must stay in the eligible denominator: an unclassified bail is never
// assumed benign, so the candidate rate degrades to the old rate rather than under-reporting.
func TestCandidateBailRateWithoutReasonMap(t *testing.T) {
	rep := FoldCompaction([]Row{compactionCell(48000, 5, 10, 100, 900, 90, nil)}, "")
	seg := findSeg(t, rep, 48000, "0-20")
	if seg.NonCandidateBails != 0 {
		t.Fatalf("NonCandidateBails = %d, want 0 (no reason map to classify from)", seg.NonCandidateBails)
	}
	if seg.CandidateBailRate != seg.BailRate {
		t.Fatalf("CandidateBailRate = %.4f, want it equal to BailRate %.4f when nothing can be classified out", seg.CandidateBailRate, seg.BailRate)
	}
	if math.Abs(seg.CandidateBailRate-0.9) > 1e-9 {
		t.Fatalf("CandidateBailRate = %.4f, want 0.9 (90 bails / 100 attempts)", seg.CandidateBailRate)
	}
}

// TestNonCandidateBailsClampedToBails guards the one way the two durable counters can
// disagree: a row whose reason map sums to MORE than its CompactionBailed total (a truncated
// or double-counted write). The eligible population must floor at zero rather than wrap the
// unsigned subtraction into a ~1.8e19 denominator.
func TestNonCandidateBailsClampedToBails(t *testing.T) {
	rep := FoldCompaction([]Row{
		compactionCell(48000, 5, 3, 100, 900, 2, map[string]uint64{"too_few_msgs": 50}),
	}, "")
	seg := findSeg(t, rep, 48000, "0-20")
	// 3 fires, 0 eligible bails → a rate of 0, not a wrapped one.
	if seg.CandidateBailRate != 0 {
		t.Fatalf("CandidateBailRate = %v, want 0 (non-candidates exceeded the bail total; the eligible count must floor at 0)", seg.CandidateBailRate)
	}
}
