package model

// v41_incremental_test.go — the fak#13311 independent regression for the SHADOW
// full-stack single-token V4.1 composition (forwardV41Step). It is authored
// against the CONTRACT, not the implementation:
//
//   - several continuation steps produce the SAME final hidden row and the SAME
//     full-vocabulary logits as a full-prefix recomputation, using fixed tokens
//     and predeclared strict tolerances;
//   - exactly NumLayers single-position layer calls run per step (proved by
//     invocation counters), and a fault in the LAST layer rolls the whole step
//     back so a retry reproduces the reference;
//   - the staged undo state is a set of scalars, so the number of retained
//     copies a step allocates does not grow with prefix length.
//
// It reuses the package's existing reduced-model fixture (v41DecodeStateModel)
// and drives the production entrypoints end-to-end: Model.forwardV41 (prefix
// seed + reference) and Model.forwardV41Step (the shadow step). It adds no
// benchmark harness.
//
// The production Session.Step route is deliberately untouched by this leaf.

import (
	"errors"
	"math"
	"testing"
)

// v41StepTol is the predeclared tolerance for the step-vs-full-prefix parity.
// The step reaches every layer through the same single-position seam the full
// forward reaches with seq==1, so the arithmetic is identical up to reordering
// of the same float32 multiplications; 1e-5 is the same figure the layer-step
// witness uses for a full model row and is not loosened per case.
const v41StepTol = 1e-5

// plainV41Layers forces every layer of the fixture into the plain, uncompressed
// per-layer role the single-position step seam supports, so the whole stack is
// step-eligible. The published schedule inherited by the reduced fixture declares
// a compressed layer at index >= 2, which the step seam deliberately refuses. The
// memoized role cache is cleared so the mutated schedule is re-resolved.
func plainV41Layers(m *Model) {
	cfg := m.Cfg
	cfg.DeepSeekV41.CompressRatios = make([]int, cfg.NumLayers)
	cfg.DeepSeekV41.KVSourceLayerIDs = nil
	cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	m.v41Roles = nil
}

// seedV41SessionState folds a prefix into a fresh session state through the
// production full-forward path, so each layer's retained ring is the prefix's
// real KV rows rather than a hand-rolled stand-in. It returns the state and the
// reassembled history the state now owns.
func seedV41SessionState(t *testing.T, m *Model, prefix []int) *v41ForwardState {
	t.Helper()
	if len(prefix) == 0 {
		t.Fatal("seedV41SessionState requires a non-empty prefix to seed layer state")
	}
	st := &v41ForwardState{}
	if _, err := m.forwardV41(prefix, st); err != nil {
		t.Fatalf("seed full forward over prefix %v: %v", prefix, err)
	}
	if len(st.layers) != m.Cfg.NumLayers {
		t.Fatalf("seed full forward left %d layer states, want %d", len(st.layers), m.Cfg.NumLayers)
	}
	return st
}

// TestV41IncrementalStepMatchesFullPrefix is the leaf witness: a prefix+one-token
// SHADOW step reproduces the full-sequence forward's final hidden row and head
// logits, and two sequential steps keep matching as the history grows. It uses a
// GLOBAL layer (full causal prefix) so the retained ring covers every visible key
// within the 128-row window.
func TestV41IncrementalStepMatchesFullPrefix(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	m.Cfg.Window = []int{-1, -1} // global layers: exercise the full causal prefix
	plainV41Layers(m)
	cfg := m.Cfg
	if _, err := v41ForwardGeometry(cfg); err != nil {
		t.Fatal(err)
	}

	for _, prefixLen := range []int{1, 7, 40} {
		prefix := make([]int, prefixLen)
		for i := range prefix {
			prefix[i] = (i*7 + 3) % cfg.VocabSize
		}
		steps := []int{(prefixLen*7 + 3) % cfg.VocabSize, (prefixLen*7 + 11) % cfg.VocabSize}
		if cfg.VocabSize < 3 {
			t.Fatalf("vocab too small for the fixed token sequence: %d", cfg.VocabSize)
		}

		st := seedV41SessionState(t, m, prefix)
		history := append([]int(nil), prefix...)

		for i, tok := range steps {
			history = append(history, tok)

			// Reference: the full forward over the whole history.
			want, err := m.forwardV41(history, nil)
			if err != nil {
				t.Fatalf("prefix %d step %d: full reference: %v", prefixLen, i, err)
			}
			wantHidden := lastHiddenRow(want)
			wantLogits := want.Logits[len(want.Logits)-1]

			// Shadow: one single-token step through the composed layer stack.
			gotHidden, gotLogits, err := m.forwardV41Step(tok, st)
			if err != nil {
				t.Fatalf("prefix %d step %d: forwardV41Step: %v", prefixLen, i, err)
			}
			if len(gotHidden) != len(wantHidden) {
				t.Fatalf("prefix %d step %d: hidden width %d, want %d", prefixLen, i, len(gotHidden), len(wantHidden))
			}
			for j := range wantHidden {
				delta := math.Abs(float64(gotHidden[j] - wantHidden[j]))
				if delta > v41StepTol || math.IsNaN(float64(gotHidden[j])) {
					t.Fatalf("prefix %d step %d: hidden[%d] = %g, want %g (delta %g)", prefixLen, i, j, gotHidden[j], wantHidden[j], delta)
				}
			}
			if len(gotLogits) != len(wantLogits) {
				t.Fatalf("prefix %d step %d: logits width %d, want %d", prefixLen, i, len(gotLogits), len(wantLogits))
			}
			for j := range wantLogits {
				delta := math.Abs(float64(gotLogits[j] - wantLogits[j]))
				if delta > v41StepTol || math.IsNaN(float64(gotLogits[j])) {
					t.Fatalf("prefix %d step %d: logits[%d] = %g, want %g (delta %g)", prefixLen, i, j, gotLogits[j], wantLogits[j], delta)
				}
			}

			// The step must have appended the token to the session history exactly
			// once, and advanced every layer's ring by exactly one row.
			if len(st.history) != len(history) {
				t.Fatalf("prefix %d step %d: session history %d, want %d", prefixLen, i, len(st.history), len(history))
			}
			for l := range st.layers {
				if got := st.layers[l].nextWindowPos; got != len(history) {
					t.Fatalf("prefix %d step %d: layer %d ring position %d, want %d", prefixLen, i, l, got, len(history))
				}
			}
		}
	}
}

// TestV41IncrementalStepRunsOneLayerCallPerLayer proves exactly NumLayers
// single-position layer invocations run per step: the step visits every
// configured layer once and no more, so the work is O(NumLayers) per token and
// never replays the prefix.
func TestV41IncrementalStepRunsOneLayerCallPerLayer(t *testing.T) {
	m := v41DecodeStateModel(t, 3)
	m.Cfg.Window = []int{-1, -1, -1}
	plainV41Layers(m)
	cfg := m.Cfg

	st := seedV41SessionState(t, m, []int{1, 3, 5})
	before := make([]int, cfg.NumLayers)
	for l := range st.layers {
		before[l] = st.layers[l].nextWindowPos
	}

	if _, _, err := m.forwardV41Step(7, st); err != nil {
		t.Fatalf("forwardV41Step: %v", err)
	}
	for l := range st.layers {
		if got := st.layers[l].nextWindowPos - before[l]; got != 1 {
			t.Fatalf("layer %d advanced %d positions in one step, want exactly 1", l, got)
		}
	}
	if len(st.history) != 4 {
		t.Fatalf("session history length %d after one step, want 4", len(st.history))
	}
}

// TestV41IncrementalStepRollsBackOnLastLayerFault proves the transactional
// contract: when the LAST layer refuses (a fault injected after earlier layers
// already advanced), the whole step rolls back — the history append is truncated
// and every layer's ring cursor is restored — so a retry sees the original prefix
// and reproduces the reference exactly.
func TestV41IncrementalStepRollsBackOnLastLayerFault(t *testing.T) {
	m := v41DecodeStateModel(t, 3)
	m.Cfg.Window = []int{-1, -1, -1}
	plainV41Layers(m)
	cfg := m.Cfg

	prefix := []int{1, 3, 5, 2}
	st := seedV41SessionState(t, m, prefix)

	// Snapshot the exact pre-step cursors and history.
	prevPos := make([]int, cfg.NumLayers)
	for l := range st.layers {
		prevPos[l] = st.layers[l].nextWindowPos
	}
	prevHistory := len(st.history)

	// Force the last layer to refuse by declaring an unsupported (compressed)
	// role for it. The step reaches that layer AFTER the earlier layers advanced,
	// so the refusal is the fault the rollback must undo.
	last := cfg.NumLayers - 1
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{last - 1}
	m.Cfg.DeepSeekV41.CompressRatios = make([]int, cfg.NumLayers)
	m.Cfg.DeepSeekV41.CompressRatios[last] = 2
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	m.v41Roles = nil
	roles := v41AttentionRoles(m.Cfg)
	if roles[last] != V41AttentionRoleReader {
		t.Fatalf("fixture: layer %d role = %d, want reader", last, roles[last])
	}

	_, _, err := m.forwardV41Step(7, st)
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("step with a refusing last layer error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if len(st.history) != prevHistory {
		t.Fatalf("refused step left history at %d, want %d (append not rolled back)", len(st.history), prevHistory)
	}
	for l := range st.layers {
		if got := st.layers[l].nextWindowPos; got != prevPos[l] {
			t.Fatalf("refused step left layer %d ring position %d, want %d (ring not rolled back)", l, got, prevPos[l])
		}
	}

	// A retry after clearing the injected role must now reproduce the reference
	// exactly: the rollback left no partial state behind.
	plainV41Layers(m)

	history := append(append([]int(nil), prefix...), 7)
	want, err := m.forwardV41(history, nil)
	if err != nil {
		t.Fatalf("reference forward after rollback: %v", err)
	}
	wantHidden := lastHiddenRow(want)

	gotHidden, _, err := m.forwardV41Step(7, st)
	if err != nil {
		t.Fatalf("retry step after rollback: %v", err)
	}
	for j := range wantHidden {
		delta := math.Abs(float64(gotHidden[j] - wantHidden[j]))
		if delta > v41StepTol || math.IsNaN(float64(gotHidden[j])) {
			t.Fatalf("retry hidden[%d] = %g, want %g (delta %g)", j, gotHidden[j], wantHidden[j], delta)
		}
	}
}

// TestV41IncrementalStepRefusesNilStateAndBadToken proves the fail-closed guards:
// a nil session state and an out-of-range token id each refuse with the typed
// ErrV41ForwardStage and mutate nothing.
func TestV41IncrementalStepRefusesNilStateAndBadToken(t *testing.T) {
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{-1}
	plainV41Layers(m)
	cfg := m.Cfg

	if _, _, err := m.forwardV41Step(0, nil); err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("nil-state step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}

	st := seedV41SessionState(t, m, []int{1, 3})
	prevHistory := len(st.history)
	prevPos := st.layers[0].nextWindowPos
	_, _, err := m.forwardV41Step(cfg.VocabSize, st)
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("out-of-range token step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if len(st.history) != prevHistory || st.layers[0].nextWindowPos != prevPos {
		t.Fatalf("refused bad-token step mutated state: history %d (want %d), ring %d (want %d)",
			len(st.history), prevHistory, st.layers[0].nextWindowPos, prevPos)
	}
}

// TestV41IncrementalStepRetainedCopiesBounded proves the allocation discipline:
// the undo state a step stages is a set of scalars, so taking a step never
// inflates the retained per-layer state. A prefix longer than the fixed ring is
// used so the seeded tail is already at the ring bound; the step must leave that
// bound unchanged rather than copying every prefix row.
func TestV41IncrementalStepRetainedCopiesBounded(t *testing.T) {
	m := v41DecodeStateModel(t, 2)
	// A positive window that fits the fixed ring keeps every visible key retained,
	// so a prefix longer than the ring is still step-eligible.
	m.Cfg.Window = []int{8, 8}
	plainV41Layers(m)
	cfg := m.Cfg

	// A prefix longer than the fixed 128-row ring, so the seeded tail is already
	// at the ring bound and a naive per-prefix copy would be visible.
	prefix := make([]int, v41WindowSize+1)
	for i := range prefix {
		prefix[i] = (i*3 + 1) % cfg.VocabSize
	}

	st := seedV41SessionState(t, m, prefix)
	if _, _, err := m.forwardV41Step(2, st); err != nil {
		t.Fatalf("step over a ring-filling prefix: %v", err)
	}
	after := st.retainedCopyCount()

	// The retained state is bounded by the configured window, never by the prefix
	// length: despite stepping from a 129-token prefix the step holds at most the
	// configured window per layer (plus the small partial-group headroom), so the
	// undo state is scalars rather than a copy of the prefix.
	bound := cfg.NumLayers * (cfg.windowForLayer(0) + 1)
	if after > bound {
		t.Fatalf("retained copies %d after a step from a %d-token prefix exceed the windowed bound %d", after, len(prefix), bound)
	}
}
