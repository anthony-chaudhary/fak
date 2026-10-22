package enginestep

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"
)

func specPositionCounts(t *testing.T, positions []SpecPositionStat) (verified, reached, accepted []uint64) {
	t.Helper()
	for i, row := range positions {
		if row.Position != i {
			t.Fatalf("positions[%d].Position = %d", i, row.Position)
		}
		verified = append(verified, row.Verified)
		reached = append(reached, row.Reached)
		accepted = append(accepted, row.Accepted)
	}
	return verified, reached, accepted
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativePerPositionAcceptance(t *testing.T) {
	r, _ := newFakeRecorder(8)
	idle := render(r)
	for _, fam := range []string{MetricSpecPositionVerifiedTotal, MetricSpecPositionReachedTotal, MetricSpecPositionAcceptedTotal} {
		if !strings.Contains(idle, "# TYPE "+fam+" counter\n") {
			t.Fatalf("idle render missing counter family %s", fam)
		}
	}
	if strings.Contains(idle, "{position=") {
		t.Fatal("idle render carries per-position samples")
	}

	r.ObserveSpeculativeRound(4, 3, 4, time.Millisecond)
	r.ObserveSpeculativeRound(4, 1, 2, time.Millisecond)
	r.ObserveSpeculativeRound(2, 2, 3, time.Millisecond)

	sp := r.Snapshot(0, "").Speculative
	if sp == nil || sp.DraftTokens != 10 || sp.AcceptedTokens != 6 {
		t.Fatalf("speculative = %+v, want 10 drafted / 6 accepted", sp)
	}
	verified, reached, accepted := specPositionCounts(t, sp.Positions)
	if !slices.Equal(verified, []uint64{3, 3, 2, 2}) || !slices.Equal(reached, []uint64{3, 3, 1, 1}) || !slices.Equal(accepted, []uint64{3, 2, 1, 0}) {
		t.Fatalf("verified/reached/accepted = %v/%v/%v, want [3 3 2 2]/[3 3 1 1]/[3 2 1 0]", verified, reached, accepted)
	}
	wantConditional := []float64{1, 2.0 / 3.0, 1, 0}
	wantRate := []float64{1, 2.0 / 3.0, 0.5, 0}
	for i, row := range sp.Positions {
		if row.ConditionalRate == nil || *row.ConditionalRate != wantConditional[i] || row.Rate == nil || *row.Rate != wantRate[i] {
			t.Fatalf("position %d rates = %v/%v, want conditional %v rate %v", i, row.ConditionalRate, row.Rate, wantConditional[i], wantRate[i])
		}
	}

	out := render(r)
	for _, w := range []string{
		MetricSpecDraftTokensTotal + " 10\n",
		MetricSpecAcceptedTokensTotal + " 6\n",
		MetricSpecPositionVerifiedTotal + `{position="0"} 3` + "\n",
		MetricSpecPositionVerifiedTotal + `{position="3"} 2` + "\n",
		MetricSpecPositionReachedTotal + `{position="1"} 3` + "\n",
		MetricSpecPositionReachedTotal + `{position="2"} 1` + "\n",
		MetricSpecPositionAcceptedTotal + `{position="1"} 2` + "\n",
		MetricSpecPositionAcceptedTotal + `{position="3"} 0` + "\n",
	} {
		if !strings.Contains(out, w) {
			t.Fatalf("render missing %q\n%s", w, out)
		}
	}
	if strings.Contains(out, `{position="4"}`) {
		t.Fatal("render carries an untouched position")
	}

	raw, err := json.Marshal(sp)
	if err != nil {
		t.Fatal(err)
	}
	var m struct {
		Positions []map[string]any `json:"positions"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if len(m.Positions) != 4 || m.Positions[1]["reached"] != float64(3) || m.Positions[3]["conditional_rate"] != float64(0) {
		t.Fatalf("positions JSON = %s", raw)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepSpeculativePositionsBoundedAtCap(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObserveSpeculativeRound(SpecPositionSlots+3, SpecPositionSlots+3, 1, time.Millisecond)
	r.ObserveSpeculativeRound(SpecPositionSlots+3, SpecPositionSlots, 1, time.Millisecond)
	sp := r.Snapshot(0, "").Speculative
	if sp == nil || len(sp.Positions) != SpecPositionSlots || sp.DraftTokens != 2*(SpecPositionSlots+3) {
		t.Fatalf("speculative = %+v, want %d positions and %d drafted", sp, SpecPositionSlots, 2*(SpecPositionSlots+3))
	}
	last := sp.Positions[SpecPositionSlots-1]
	if last.Verified != 8 || last.Reached != 6 || last.Accepted != 5 {
		t.Fatalf("folded last slot = %+v, want verified 8 reached 6 accepted 5", last)
	}
	var verified, accepted uint64
	for _, row := range sp.Positions {
		verified += row.Verified
		accepted += row.Accepted
	}
	if verified != sp.DraftTokens || accepted != sp.AcceptedTokens {
		t.Fatalf("per-position sums %d/%d, want totals %d/%d", verified, accepted, sp.DraftTokens, sp.AcceptedTokens)
	}
	if out := render(r); strings.Contains(out, fmt.Sprintf(`{position="%d"}`, SpecPositionSlots)) {
		t.Fatal("render exceeds the position slot cap")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepSpeculativePositionsIgnoreNoProposal(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObserveSpeculativeRound(0, 0, 1, time.Millisecond)
	r.ObserveSpeculativeRound(-3, 2, 2, time.Millisecond)
	if s := r.Snapshot(0, ""); s.Speculative != nil {
		t.Fatalf("no-proposal rounds leaked: %+v", s.Speculative)
	}
	if strings.Contains(render(r), "{position=") {
		t.Fatal("no-proposal rounds rendered per-position samples")
	}
	var nilRec *Recorder
	nilRec.ObserveSpeculativeRound(4, 2, 3, time.Millisecond)

	r.ObserveSpeculativeRound(3, 9, 4, time.Millisecond)
	r.ObserveSpeculativeRound(2, -5, 1, time.Millisecond)
	verified, reached, accepted := specPositionCounts(t, r.Snapshot(0, "").Speculative.Positions)
	if !slices.Equal(verified, []uint64{2, 2, 1}) || !slices.Equal(reached, []uint64{2, 1, 1}) || !slices.Equal(accepted, []uint64{1, 1, 1}) {
		t.Fatalf("clamped verified/reached/accepted = %v/%v/%v", verified, reached, accepted)
	}
}
