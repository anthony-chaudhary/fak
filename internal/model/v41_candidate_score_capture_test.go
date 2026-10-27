package model

import (
	"math"
	"slices"
	"testing"
)

// fak-test:runtime medium est=5s lane=default
func TestV41CandidateScoreSnapshotOwnership(t *testing.T) {
	var observed []string
	scores, err := v41IndexerScoreObserved([]float32{2}, []float32{1, 3}, []float32{1}, 1, 1, 2, func(arm string) {
		observed = append(observed, arm)
	})
	if err != nil {
		t.Fatal(err)
	}
	captured := v41CandidateScoreSnapshot(scores, true)
	if !slices.Equal(captured, []float32{2, 6}) || &captured[0] != &scores[0] {
		t.Fatal("fresh scalar scores were not transferred")
	}
	if !slices.Equal(observed, []string{"attention-indexer-score"}) {
		t.Fatalf("host observations = %v", observed)
	}
	if _, err := V41SelectIndexRows(scores, len(scores), 1, 0, nil); err != nil || !slices.Equal(captured, []float32{2, 6}) {
		t.Fatalf("selection changed captured scores: %v, %v", captured, err)
	}
	// A callback may reuse one backing row on every query. Its next result
	// must not alter the previously captured publication input.
	borrowed := []float32{2, 6}
	callbackCapture := v41CandidateScoreSnapshot(borrowed, false)
	borrowed[0], borrowed[1] = 9, 1
	if !slices.Equal(callbackCapture, []float32{2, 6}) || &callbackCapture[0] == &borrowed[0] {
		t.Fatal("callback capture aliases reusable output")
	}
	for _, owned := range []bool{false, true} {
		if v41CandidateScoreSnapshot(nil, owned) != nil || v41CandidateScoreSnapshot([]float32{}, owned) != nil {
			t.Fatal("empty capture changed its nil representation")
		}
	}

	// The capture wrapper must preserve the builtin's prepared-key authority,
	// without granting the same authority to an arbitrary scorer callback.
	m := v41IndexerTestFixture(t)
	dim := m.Cfg.IndexHeadDim
	qLat, hidden := make([]float32, m.Cfg.QLoraRank), make([]float32, m.Cfg.HiddenSize)
	for i := range qLat {
		qLat[i] = float32(i%7-3) / 8
	}
	for i := range hidden {
		hidden[i] = float32(i%5-2) / 8
	}
	keys := make([][]float32, 8)
	for row := range keys {
		keys[row] = make([]float32, dim)
		for i := range keys[row] {
			keys[row][i] = float32((row*7+i)%19-9) / 16
		}
	}
	panel := v41ColdIndexerKeyPanel(keys, dim)
	selectCaptured := func(prepared []float32, score v41IndexerScoreFunc) ([]int32, []float32, error) {
		var values []float32
		rows, err := m.v41IndexRowsWithPanelRowCapture(0, 15, qLat, hidden, keys, nil, newV41IndexerScoreBinding(score), nil, nil, prepared, nil, nil, v41IndexerPanelRow{}, &values, nil)
		return rows, values, err
	}
	wantRows, wantScores, err := selectCaptured(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// A shape-correct source mutation cannot alter the already-frozen panel.
	// The old capture wrapper rebuilt the flat input and would see this NaN.
	first := keys[0][0]
	keys[0][0] = float32(math.NaN())
	gotRows, gotScores, err := selectCaptured(panel, nil)
	keys[0][0] = first
	if err != nil || !slices.Equal(gotRows, wantRows) || !slices.Equal(gotScores, wantScores) {
		t.Fatalf("builtin capture did not use prepared keys: rows=%v want=%v err=%v", gotRows, wantRows, err)
	}
	var retained [][]float32
	reusedScores := make([]float32, len(keys))
	callback := func(_ int, q, flat, weights []float32, heads, width, rows int) ([]float32, error) {
		if &flat[0] == &panel[0] || !slices.Equal(flat, panel) {
			t.Fatal("ordinary capture borrowed prepared keys or reused a mutated input")
		}
		values, err := V41IndexerScore(q, flat, weights, heads, width, rows)
		copy(reusedScores, values)
		retained = append(retained, flat)
		flat[0] = float32(math.NaN())
		return reusedScores, err
	}
	for i := 0; i < 2; i++ {
		gotRows, gotScores, err = selectCaptured(panel, callback)
		if err != nil || !slices.Equal(gotRows, wantRows) || !slices.Equal(gotScores, wantScores) || &gotScores[0] == &reusedScores[0] {
			t.Fatalf("ordinary capture changed selection or retained output: %v", err)
		}
		reusedScores[0]++
		if !slices.Equal(gotScores, wantScores) {
			t.Fatal("callback output reuse changed the captured row")
		}
	}
	if &retained[0][0] == &retained[1][0] || panel[0] != first || keys[0][0] != first {
		t.Fatal("ordinary capture changed key ownership")
	}
}
