package gateway

import (
	"testing"
	"time"
)

// governorQualityMetrics drives the LIVE loop the way a served turn does
// (observeVCacheTurn -> observeVCacheGovernorDecision) so the rows under test are the same
// rows a real `fak serve` session journals, not hand-built fixtures.
func governorQualityMetrics(t *testing.T) *gatewayMetrics {
	t.Helper()
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("head", 1, 40000, 0, 40000) // cold create
	m.observeVCacheTurn("head", 2, 50, 40000, 500)  // warm read
	recs := m.vcacheGovernorDecisionRecords()
	if len(recs) < 2 {
		t.Fatalf("live loop journaled %d governor rows, want >= 2", len(recs))
	}
	return m
}

// forgedDecision returns a valid verdict guaranteed to DIFFER from the recorded one, so a
// tamper test always mutates the row no matter what the live loop happened to classify.
func forgedDecision(current string) string {
	if current == "lazy_rebuild" {
		return "ride_natural"
	}
	return "lazy_rebuild"
}

// TestVCacheGovernorQualityScoresVerifiedChain is the "scorable" half of the #1492 QA bar:
// a live-journaled window verifies end to end and produces a score over [0,1].
func TestVCacheGovernorQualityScoresVerifiedChain(t *testing.T) {
	m := governorQualityMetrics(t)
	q := m.vcacheGovernorQualityVars()
	if q == nil {
		t.Fatal("journaled verdicts must produce a quality block")
	}
	if !q.ChainVerified {
		t.Fatalf("live journal must verify; broke at seq %d", q.ChainBreakSeq)
	}
	if q.ChainBreakSeq != 0 {
		t.Fatalf("verified chain must report no break seq, got %d", q.ChainBreakSeq)
	}
	if q.Records != len(m.vcacheGovernorDecisionRecords()) {
		t.Fatalf("audited %d rows, journal holds %d", q.Records, len(m.vcacheGovernorDecisionRecords()))
	}
	if q.Score < 0 || q.Score > 1 {
		t.Fatalf("score %v outside [0,1]", q.Score)
	}
	if q.Kept > q.Records {
		t.Fatalf("kept %d exceeds records %d", q.Kept, q.Records)
	}
	if want := float64(q.Kept) / float64(q.Records); q.Score != want {
		t.Fatalf("score %v != kept/records %v", q.Score, want)
	}
	if len(q.ByDecision) == 0 {
		t.Fatal("verified chain must break the score down by decision")
	}
	sum := 0
	for _, slice := range q.ByDecision {
		sum += slice.Records
	}
	if sum != q.Records {
		t.Fatalf("by_decision covers %d rows, want %d", sum, q.Records)
	}
	if q.Schema != vcacheGovernorQualitySchema {
		t.Fatalf("schema = %q", q.Schema)
	}
	if q.Provenance != "DECISION" {
		t.Fatalf("provenance = %q, want DECISION", q.Provenance)
	}
}

// TestVCacheGovernorQualityRefusesEditedRow is the "non-forgeable" half: editing a decision
// in place breaks that row's own hash, and the score fails closed to 0.0.
func TestVCacheGovernorQualityRefusesEditedRow(t *testing.T) {
	m := governorQualityMetrics(t)
	recs := m.vcacheGovernorDecisionRecords()
	recs[0].Decision = forgedDecision(recs[0].Decision) // edit without re-signing

	q := vcacheGovernorQuality(recs)
	if q == nil {
		t.Fatal("tampered window must still render a block reporting the break")
	}
	if q.ChainVerified {
		t.Fatal("an edited decision must NOT verify")
	}
	if q.ChainBreakSeq != recs[0].Seq {
		t.Fatalf("break seq = %d, want %d (the edited row)", q.ChainBreakSeq, recs[0].Seq)
	}
	if q.Score != 0 || q.Kept != 0 {
		t.Fatalf("unverified chain must fail closed to score 0/kept 0, got %v/%d", q.Score, q.Kept)
	}
	if q.ByDecision != nil {
		t.Fatal("unverified chain must not publish a per-decision slice")
	}
}

// TestVCacheGovernorQualityRefusesResignedRow closes the obvious attack on the previous
// test: an attacker who edits a row AND recomputes its hash still cannot succeed, because
// the successor's prev_hash no longer ties back. This is the property the chain buys.
func TestVCacheGovernorQualityRefusesResignedRow(t *testing.T) {
	m := governorQualityMetrics(t)
	recs := m.vcacheGovernorDecisionRecords()

	recs[0].Decision = forgedDecision(recs[0].Decision)
	recs[0].Hash = hashVCacheGovernorDecision(recs[0].PrevHash, recs[0]) // re-sign the edit

	q := vcacheGovernorQuality(recs)
	if q.ChainVerified {
		t.Fatal("re-signing an edited row must still break the successor link")
	}
	if q.ChainBreakSeq != recs[1].Seq {
		t.Fatalf("break seq = %d, want %d (the successor whose prev_hash no longer ties)",
			q.ChainBreakSeq, recs[1].Seq)
	}
	if q.Score != 0 {
		t.Fatalf("unverified chain must score 0, got %v", q.Score)
	}
}

// TestVCacheGovernorKeepBit pins the deterministic bit per verdict class, including the
// fail-closed default for a verdict this scorer has not opted into.
func TestVCacheGovernorKeepBit(t *testing.T) {
	for _, tc := range []struct {
		name     string
		decision string
		read     int64
		create   int64
		want     bool
	}{
		{"ride_natural vindicated by a read", "ride_natural", 40000, 0, true},
		{"ride_natural with no read is a false warm", "ride_natural", 0, 0, false},
		{"heartbeat_pin vindicated by a read", "heartbeat_pin", 128, 0, true},
		{"heartbeat_pin paying to hold a cold prefix", "heartbeat_pin", 0, 4096, false},
		{"lazy_rebuild correctly let it lapse", "lazy_rebuild", 0, 4096, true},
		{"lazy_rebuild gave up a still-hot prefix", "lazy_rebuild", 40000, 0, false},
		{"evict correctly dropped a cold prefix", "evict", 0, 0, true},
		{"evict dropped a prefix still being read", "evict", 1, 0, false},
		{"no_cache never warmed", "no_cache", 0, 0, true},
		{"no_cache leaked a create (D4 breach)", "no_cache", 0, 1, false},
		{"no_cache leaked a read (D4 breach)", "no_cache", 1, 0, false},
		{"explicit_cache never implicitly warmed", "explicit_cache", 0, 0, true},
		{"explicit_cache implicitly warmed", "explicit_cache", 0, 4096, false},
		{"unknown verdict fails closed", "teleport", 40000, 0, false},
		{"empty verdict fails closed", "", 0, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := vcacheGovernorKeepBit(vcacheGovernorDecisionRecord{
				Decision:            tc.decision,
				CacheReadTokens:     tc.read,
				CacheCreationTokens: tc.create,
			})
			if got != tc.want {
				t.Fatalf("keep bit = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestVCacheGovernorQualityDeterministic — the metric is a pure function of the rows, so
// scoring the same window twice cannot drift. Without this the score is unfalsifiable.
func TestVCacheGovernorQualityDeterministic(t *testing.T) {
	m := governorQualityMetrics(t)
	recs := m.vcacheGovernorDecisionRecords()
	first := vcacheGovernorQuality(recs)
	second := vcacheGovernorQuality(recs)
	if first.Score != second.Score || first.Kept != second.Kept || first.Records != second.Records {
		t.Fatalf("score drifted across identical inputs: %+v vs %+v", first, second)
	}
}

// TestVCacheGovernorQualityOmittedWhenEmpty keeps the no-phantom guard the sibling vcache
// blocks hold: no verdicts journaled => no block.
func TestVCacheGovernorQualityOmittedWhenEmpty(t *testing.T) {
	if q := vcacheGovernorQuality(nil); q != nil {
		t.Fatalf("empty journal must omit the block, got %+v", q)
	}
	m := newGatewayMetrics(time.Now())
	if q := m.vcacheGovernorQualityVars(); q != nil {
		t.Fatalf("a gateway that served no cache activity must omit the block, got %+v", q)
	}
	var nilMetrics *gatewayMetrics
	if q := nilMetrics.vcacheGovernorQualityVars(); q != nil {
		t.Fatal("nil metrics must omit the block")
	}
}

// TestVCacheGovernorChainVerifiesMidWindowAnchor — the journal is a bounded ring, so after
// drop-oldest the retained window legitimately starts mid-chain with a non-empty prev_hash.
// That must verify, not read as tampering.
func TestVCacheGovernorChainVerifiesMidWindowAnchor(t *testing.T) {
	m := governorQualityMetrics(t)
	recs := m.vcacheGovernorDecisionRecords()
	tail := recs[1:] // simulate drop-oldest having trimmed the head
	if len(tail) == 0 {
		t.Skip("need >= 2 rows to exercise a mid-window anchor")
	}
	if tail[0].PrevHash == "" {
		t.Fatal("mid-window anchor should carry its predecessor's hash")
	}
	if breakSeq, ok := verifyVCacheGovernorChain(tail); !ok {
		t.Fatalf("mid-window anchor must verify, broke at seq %d", breakSeq)
	}
}
