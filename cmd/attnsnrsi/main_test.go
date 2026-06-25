package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// snSpan mirrors the ctxplan fixture: Bytes=40 ⇒ TokenCost ceil(40/4)=10, so a Budget of
// 10 fits exactly one span and a mis-ranked forecast is observable as a single keep/elide
// flip.
func snSpan(id, desc string, step int) ctxplan.Span {
	return ctxplan.Span{ID: id, Descriptor: desc, Step: step, Bytes: 40, Durability: ctxplan.DurabilityTurn}
}

// controlledSession is the proven snfitness_test scenario: span:a is the one the model
// attends to (mass 0.9), span:b / span:c idle. The lexical baseline predicts the idle
// spans (fitness 0); the learned forecast (trained on the witnessed attention via
// LearnFromAttention, the #858 loop) promotes the attended span and scores ~0.9. The
// strict gain is exactly the keep-direction the rsiloop KEEP gate fires on.
func controlledSession() (baseline, learned ctxplan.Forecast, session []ctxplan.Turn) {
	spans := []ctxplan.Span{
		snSpan("span:c", "gamma grape", 1),
		snSpan("span:b", "beta banana", 2),
		snSpan("span:a", "alpha apple", 3),
	}
	budget := ctxplan.Budget{Tokens: 10}
	attribution := ctxplan.Attribution{"span:a": 0.9, "span:b": 0.0, "span:c": 0.0}
	faults := []string{"span:a"}
	session = []ctxplan.Turn{{Spans: spans, Budget: budget, Attribution: attribution, Faults: faults}}

	baseline = ctxplan.Forecast{Intents: []string{"beta", "gamma"}, Horizon: 1}
	basePlan := ctxplan.PlanCells(spans, baseline, budget, nil)
	lf, lw := ctxplan.LearnFromAttention(baseline, baseline.Weights, basePlan, spans, attribution, faults, ctxplan.DefaultHitThreshold, 3)
	lf.Weights = lw
	learned = lf
	return baseline, learned, session
}

// readRows parses every JSON line of a journal file into rsiloop.Row — the durable S/N
// trend ledger the loop writes.
func readRows(t *testing.T, path string) []rsiloop.Row {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	var rows []rsiloop.Row
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r rsiloop.Row
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("parse journal row %q: %v", line, err)
		}
		rows = append(rows, r)
	}
	return rows
}

// TestAttentionSNHarnessKeepsRealGain is the end-to-end closure proof for #867 acceptance
// #2/#3: a learned forecast that raises the witnessed attention-S/N over the lexical
// baseline, with a green suite and a truth-clean tree, is KEPT by the rsiloop keep-bit,
// and the journal records the turn-over-turn trend.
func TestAttentionSNHarnessKeepsRealGain(t *testing.T) {
	baseline, learned, session := controlledSession()
	doc := sessionDoc{Baseline: baseline, Candidates: []ctxplan.Forecast{learned}, Session: session, BaselineRef: "main"}

	journal := filepath.Join(t.TempDir(), "attn_sn.jsonl")
	j, err := rsiloop.NewJournal(journal)
	if err != nil {
		t.Fatalf("new journal: %v", err)
	}
	h := attentionSNHarness(doc, true, true)
	res, err := rsiloop.Run(h, j, 3, 0)
	j.Close()
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if res.Kept != 1 {
		t.Fatalf("want 1 keep, got %d", res.Kept)
	}
	if len(res.Rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(res.Rows))
	}
	row := res.Rows[0]
	if !row.Kept || row.Decision != "KEEP" {
		t.Fatalf("learned forecast not kept: kept=%v decision=%s", row.Kept, row.Decision)
	}
	if !row.Improved {
		t.Fatalf("kept row must show a strict S/N gain (Improved); got baseline=%.4f candidate=%.4f", row.Baseline, row.Candidate_)
	}
	if !(row.Candidate_ > row.Baseline) {
		t.Fatalf("candidate metric did not beat baseline: candidate=%.4f baseline=%.4f", row.Candidate_, row.Baseline)
	}
	if row.MetricName != "attention_sn" {
		t.Fatalf("metric mislabeled: %q", row.MetricName)
	}

	// The durable journal records the trend (acceptance #3 substrate).
	rows := readRows(t, journal)
	if len(rows) != 1 || !rows[0].Kept {
		t.Fatalf("journal did not record the kept S/N gain: %+v", rows)
	}
}

// TestAttentionSNHarnessRevertsNoGain proves the gate does NOT fire without a real gain:
// a candidate identical to the baseline (no S/N improvement) is REVERTED even with a green
// suite and a clean tree.
func TestAttentionSNHarnessRevertsNoGain(t *testing.T) {
	baseline, _, session := controlledSession()
	doc := sessionDoc{Baseline: baseline, Candidates: []ctxplan.Forecast{baseline}, Session: session}

	h := attentionSNHarness(doc, true, true)
	res, err := rsiloop.Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Kept != 0 {
		t.Fatalf("a no-gain candidate must not be kept; kept=%d", res.Kept)
	}
	if res.Rows[0].Kept || res.Rows[0].Decision != "REVERT" {
		t.Fatalf("no-gain candidate not reverted: kept=%v decision=%s", res.Rows[0].Kept, res.Rows[0].Decision)
	}
}

// TestAttentionSNHarnessTruthFloorBlocksGain is the "real, dos-verified S/N gain" property
// (acceptance #2) and the two-posture honesty split: a candidate with a REAL S/N gain
// (Improved=true) is still NOT kept when the truth-clean floor is unmet — the keep-bit
// folds the dos-verify witness, not just the metric, so a forecast cannot be adopted off a
// number alone on a dirty tree. The same holds when the suite is red.
func TestAttentionSNHarnessTruthFloorBlocksGain(t *testing.T) {
	baseline, learned, session := controlledSession()

	for _, tc := range []struct {
		name              string
		suiteGreen, truth bool
	}{
		{"dirty-tree", true, false},
		{"suite-red", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			doc := sessionDoc{Baseline: baseline, Candidates: []ctxplan.Forecast{learned}, Session: session}
			h := attentionSNHarness(doc, tc.suiteGreen, tc.truth)
			res, err := rsiloop.Run(h, nil, 3, 0)
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			row := res.Rows[0]
			if !row.Improved {
				t.Fatalf("fixture invalid: candidate must still show the S/N gain, baseline=%.4f candidate=%.4f", row.Baseline, row.Candidate_)
			}
			if row.Kept || row.Decision != "REVERT" {
				t.Fatalf("a real gain on an unverified tree must NOT be kept: kept=%v decision=%s", row.Kept, row.Decision)
			}
		})
	}
}

// TestAttentionSNHarnessBreakerEarlyExits proves the bad-streak early-exit (acceptance
// #3): with breaker k=2, two consecutive no-gain candidates ESCALATE and the loop stops
// BEFORE reaching the third (winning) candidate — the guard-loop pattern that aborts a
// degrading run instead of grinding the whole proposal list.
func TestAttentionSNHarnessBreakerEarlyExits(t *testing.T) {
	baseline, learned, session := controlledSession()
	doc := sessionDoc{
		Baseline:   baseline,
		Candidates: []ctxplan.Forecast{baseline, baseline, learned}, // two duds, then a winner never reached
		Session:    session,
	}
	h := attentionSNHarness(doc, true, true)
	res, err := rsiloop.Run(h, nil, 2, 0)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !res.Escalated {
		t.Fatalf("two consecutive non-keeps with k=2 must ESCALATE; res=%+v", res)
	}
	if res.Cycles != 2 {
		t.Fatalf("breaker must early-exit after the 2nd non-keep, not reach the winner; cycles=%d", res.Cycles)
	}
	if res.Kept != 0 {
		t.Fatalf("no candidate should have been kept before the breaker tripped; kept=%d", res.Kept)
	}
	if last := res.Rows[len(res.Rows)-1]; last.Decision != "ESCALATE" {
		t.Fatalf("final decision must be ESCALATE, got %s", last.Decision)
	}
}

// TestLoadSessionRoundTrips proves the on-disk session schema survives a marshal/unmarshal
// so the runnable driver replays exactly what a recorder wrote, and rejects unknown fields.
func TestLoadSessionRoundTrips(t *testing.T) {
	baseline, learned, session := controlledSession()
	doc := sessionDoc{Baseline: baseline, Candidates: []ctxplan.Forecast{learned}, Session: session, BaselineRef: "main"}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := loadSession(strings.NewReader(string(b)))
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if len(got.Candidates) != 1 || len(got.Session) != 1 || got.BaselineRef != "main" {
		t.Fatalf("round-trip lost structure: %+v", got)
	}
	// The witnessed reward (attention + faults) must survive the round-trip — it is what the
	// keep-bit closes on.
	if got.Session[0].Attribution["span:a"] != 0.9 {
		t.Fatalf("attribution reward lost in round-trip: %+v", got.Session[0].Attribution)
	}
	if _, err := loadSession(strings.NewReader(`{"bogus_field": 1}`)); err == nil {
		t.Fatalf("loadSession must reject unknown fields")
	}
}
