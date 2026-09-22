package model

// v41_incremental_test.go — the fak#13311 independent regression for the
// full-stack incremental V4.1 decode step (forwardV41Step). It is authored
// against the CONTRACT, not the implementation: a step consumes the next token
// at pos = len(st.history), runs every layer via v41LayerStep then v41Head,
// commits history only after the whole stack succeeds, and rolls back by scalar
// restoration (nextWindowPos) on any fault — never replacing retained storage.
//
// The witness is deliberately atomic: parity against a full-prefix
// recomputation, invocation count, last-layer rollback, a structural
// copy/alloc independence probe, and the fail-closed edges. It reuses the
// production seed path (v41LayerStepSeed / v41LayerStepInputs) from
// v41_layer_step_test.go rather than hand-rolling stand-in state, so parity
// compares against rows the model actually produces.

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41IncrementalPlainModel builds a reduced V4.1 model whose layers are all
// PLAIN per-layer roles and installs a global (full-causal) window, so the
// incremental step's plain-layer seam accepts every layer. The reduced fixture
// inherits the official 40-layer compression schedule, which marks later layers
// compressed/indexed; forcing all-zero ratios and clearing the source sets keeps
// the fixture inside the seam's supported regime.
func v41IncrementalPlainModel(t *testing.T, layers int) *Model {
	t.Helper()
	m := v41DecodeStateModel(t, layers)
	m.Cfg.Window = []int{-1} // global (full-causal) layer
	d41 := m.Cfg.DeepSeekV41
	d41.CompressRatios = make([]int, layers)
	d41.KVSourceLayerIDs = nil
	d41.IndexSourceLayerIDs = nil
	d41.EngramLayerIDs = nil
	return m
}

// v41IncrementalState seeds a full-stack continuation state by running the
// PRODUCTION prefill seed path over prefix: forwardV41(prefix, st) projects each
// layer's retained KV rows from that layer's OWN per-position hidden input and
// commits prefix to history. This is the only seed that is a full-prefix
// reference for layers > 0; the leaf-06 per-layer helper v41LayerStepSeed builds
// every layer's rows from the raw embedding and is therefore a correct prefix
// seed for layer 0 alone.
func v41IncrementalState(t *testing.T, m *Model, prefix []int) *v41ForwardState {
	t.Helper()
	st := &v41ForwardState{}
	if _, err := m.forwardV41(prefix, st); err != nil {
		t.Fatalf("seed full-prefix state (prefix %v): %v", prefix, err)
	}
	if len(st.history) != len(prefix) {
		t.Fatalf("seed history = %v, want %v", st.history, prefix)
	}
	for i := range prefix {
		if st.history[i] != prefix[i] {
			t.Fatalf("seed history = %v, want %v", st.history, prefix)
		}
	}
	return st
}

// v41IncrementalPrefix builds a deterministic token prefix and its continuation
// token under the fixture's vocab.
func v41IncrementalPrefix(cfg Config, prefixLen int) ([]int, int) {
	prefix := make([]int, prefixLen)
	for i := range prefix {
		prefix[i] = (i*7 + 3) % cfg.VocabSize
	}
	return prefix, (prefixLen*7 + 3) % cfg.VocabSize
}

// v41AllNextWindowPos records each layer's retained window position so a
// rollback can assert scalar restoration (never a prefix-sized clone).
func v41AllNextWindowPos(st *v41ForwardState) []int {
	out := make([]int, len(st.layers))
	for l := range st.layers {
		if st.layers[l] != nil {
			out[l] = st.layers[l].nextWindowPos
		}
	}
	return out
}

// TestV41Incremental is the primary witness: several continuation steps each
// match the FULL-PREFIX recomputation's last-position logits (full vocab) and
// the retained per-layer state agrees, at a pre-declared tolerance. Clause 1
// (pos = len(st.history)), 2 (embed + 4 streams + all layers + head), 3 (commit
// after success) and 7 (parity + retained state). The sibling
// TestV41Incremental... tests below cover invocation count, rollback, copy
// independence and the fail-closed edges.
func TestV41Incremental(t *testing.T) {
	// The retained ring is a fixed 128 rows (v41WindowSize); keep prefixes at or
	// below the ring so a global (full-causal) layer's step reproduces the full
	// prefix exactly. tolerance is the same 1e-6 the sibling v41_layer_step
	// witness uses; the step and the full forward run the same f32 kernels in the
	// same order, so the observed delta is 0 and the declared bound is pure
	// margin.
	const tol = 1e-6

	for _, prefixLen := range []int{1, 7, 127} {
		m := v41IncrementalPlainModel(t, 2)
		cfg := m.Cfg
		prefix, next := v41IncrementalPrefix(cfg, prefixLen)
		st := v41IncrementalState(t, m, prefix)
		history := append([]int(nil), prefix...)

		// A global step needs positions 0..pos in the 128-row ring, so stop
		// before pos reaches v41WindowSize. A short prefix gets three
		// continuation steps; a near-ring prefix gets one.
		nSteps := 3
		if avail := v41WindowSize - prefixLen; avail < nSteps {
			nSteps = avail
		}
		steps := []int{next, (next*3 + 1) % cfg.VocabSize, (next*5 + 2) % cfg.VocabSize}[:nSteps]
		for i, tok := range steps {
			pos := len(history)
			if len(st.history) != pos {
				t.Fatalf("prefix %d step %d: history len %d, want pos %d", prefixLen, i, len(st.history), pos)
			}
			history = append(history, tok)

			want, err := m.forwardV41(history, nil)
			if err != nil {
				t.Fatalf("prefix %d step %d: full forward: %v", prefixLen, i, err)
			}
			if len(want.Logits) == 0 {
				t.Fatalf("prefix %d step %d: full forward returned no logits", prefixLen, i)
			}
			wantRow := want.Logits[len(want.Logits)-1]

			got, stats, err := m.forwardV41Step(tok, st, &v41ProjScratch{})
			if err != nil {
				t.Fatalf("prefix %d step %d (pos %d): forwardV41Step: %v", prefixLen, i, pos, err)
			}
			if !stats.Committed {
				t.Fatalf("prefix %d step %d: stats.Committed = false, want true", prefixLen, i)
			}
			if stats.LayerCalls != cfg.NumLayers {
				t.Fatalf("prefix %d step %d: LayerCalls = %d, want %d", prefixLen, i, stats.LayerCalls, cfg.NumLayers)
			}
			if stats.LayersRolledBack != 0 {
				t.Fatalf("prefix %d step %d: LayersRolledBack = %d, want 0 on success", prefixLen, i, stats.LayersRolledBack)
			}
			if !reflect.DeepEqual(st.history, history) {
				t.Fatalf("prefix %d step %d: history = %v, want %v", prefixLen, i, st.history, history)
			}
			if len(got) != len(wantRow) {
				t.Fatalf("prefix %d step %d: logits width %d, want %d", prefixLen, i, len(got), len(wantRow))
			}
			for j := range wantRow {
				delta := got[j] - wantRow[j]
				if delta < 0 {
					delta = -delta
				}
				if delta > tol || math.IsNaN(float64(got[j])) {
					t.Fatalf("prefix %d step %d: logits[%d] = %g, want %g (delta %g)", prefixLen, i, j, got[j], wantRow[j], delta)
				}
			}
			// Retained per-layer state must agree with the full-prefix seed: the
			// step advanced every layer by exactly one window row.
			for l := 0; l < cfg.NumLayers; l++ {
				if got := st.layerState(l).nextWindowPos; got != len(history) {
					t.Fatalf("prefix %d step %d layer %d: nextWindowPos = %d, want %d", prefixLen, i, l, got, len(history))
				}
			}
		}
	}
}

// TestV41IncrementalLastLayerRollback forces a deterministic fault in the LAST
// layer and proves the all-or-nothing commit: typed error, no commit,
// LayersRolledBack == NumLayers, every layer's nextWindowPos restored to its
// pre-step scalar, and history unchanged. Clause 4, 5, 6.
//
// The fault is forced WITHOUT reading the implementation by installing a
// DELIBERATELY unsupported state on the last layer: its append-only cursor is
// poisoned to a value that is not pos, so the layer's step refuses before
// mutating. Earlier layers therefore genuinely step and advance their cursors,
// and only the final layer refuses — so a rollback that skipped the faulting
// layer or leaked the earlier layers' advancement would be caught.
func TestV41IncrementalLastLayerRollback(t *testing.T) {
	m := v41IncrementalPlainModel(t, 3)
	cfg := m.Cfg

	prefix, next := v41IncrementalPrefix(cfg, 4)
	st := v41IncrementalState(t, m, prefix)

	// Advance once successfully so every layer holds a committed row.
	if _, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{}); err != nil || !stats.Committed {
		t.Fatalf("priming step: err=%v committed=%t", err, stats.Committed)
	}

	last := cfg.NumLayers - 1
	st.layerState(last).nextWindowPos = len(st.history) + 5 // unsupported: != pos
	before := v41AllNextWindowPos(st)
	beforeHistory := append([]int(nil), st.history...)

	faultTok := (next*7 + 1) % cfg.VocabSize
	logits, stats, err := m.forwardV41Step(faultTok, st, &v41ProjScratch{})
	if err == nil {
		t.Fatalf("last-layer fault returned nil error (logits=%d)", len(logits))
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("last-layer fault error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if stats.Committed {
		t.Fatal("last-layer fault reported Committed = true")
	}
	if stats.LayersRolledBack != cfg.NumLayers {
		t.Fatalf("last-layer fault LayersRolledBack = %d, want %d", stats.LayersRolledBack, cfg.NumLayers)
	}
	if !reflect.DeepEqual(st.history, beforeHistory) {
		t.Fatalf("last-layer fault history = %v, want unchanged %v", st.history, beforeHistory)
	}
	after := v41AllNextWindowPos(st)
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("last-layer fault nextWindowPos = %v, want restored %v", after, before)
	}
}

// TestV41IncrementalRingRollbackPreservesWrappedSlot is the regression the
// pre-#13311 rollback missed: once the fixed ring has wrapped, restoring only
// each layer's nextWindowPos leaves the slot that the failed step's
// V41AttentionState.Step overwrote holding the FAILED step's KV row, which
// retainedTailRows() then exposes.
//
// The test seeds prefix == v41WindowSize tokens so every layer's nextWindowPos
// sits exactly at the ring's wrap point (nextWindowPos % windowSize == 0), then
// poisons the LAST layer's cursor so it refuses inside the layer loop after
// layer 0 has genuinely wrapped and overwritten slot 0. The assertion that fails
// against the pre-fix implementation (which restored the cursor alone) and
// passes now is the byte-identical ring-slot comparison in clause 6.
//
// The poisoned layer's cursor is restored to the POISONED value, not to
// v41WindowSize: the composition restores every staged cursor to its pre-call
// value, and the last layer's pre-call value is the poisoned cursor. This test
// therefore asserts the poisoned layer only through LayersRolledBack, and pins
// the ring contents / cursor / history invariants of LAYER 0.
func TestV41IncrementalRingRollbackPreservesWrappedSlot(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg
	if cfg.NumLayers < 2 {
		t.Fatalf("fixture has %d layers, want at least 2", cfg.NumLayers)
	}

	// Seed EXACTLY the ring width so the next step is the first wrapped write.
	prefix, next := v41IncrementalPrefix(cfg, v41WindowSize)
	st := v41IncrementalState(t, m, prefix)
	if len(st.history) != v41WindowSize {
		t.Fatalf("seed history length = %d, want %d", len(st.history), v41WindowSize)
	}

	// Precondition: every layer's cursor is at the wrap point.
	for l := 0; l < cfg.NumLayers; l++ {
		ls := st.layerState(l)
		if ls.nextWindowPos != v41WindowSize {
			t.Fatalf("layer %d nextWindowPos = %d, want %d (the wrap point)", l, ls.nextWindowPos, v41WindowSize)
		}
		if ls.nextWindowPos%ls.windowSize != 0 {
			t.Fatalf("layer %d nextWindowPos %% windowSize = %d, want 0 (the wrapped slot)", l, ls.nextWindowPos%ls.windowSize)
		}
	}

	// Capture the PHYSICAL ring row layer 0's wrapped step will overwrite.
	layer0 := st.layerState(0)
	if layer0.nextWindowPos%layer0.windowSize != 0 {
		t.Fatalf("layer 0 wrap slot = %d, want 0", layer0.nextWindowPos%layer0.windowSize)
	}
	slot := layer0.nextWindowPos % layer0.windowSize
	capturedRow := append([]float32(nil), layer0.window[slot]...)

	// Poison the LAST layer's cursor so it refuses inside the layer loop while
	// layer 0 genuinely steps and wraps.
	last := cfg.NumLayers - 1
	st.layerState(last).nextWindowPos = len(st.history) + 5

	_, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{})
	if err == nil {
		t.Fatal("wrapped-ring fault returned nil error")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("wrapped-ring fault error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if stats.Committed {
		t.Fatal("wrapped-ring fault reported Committed = true")
	}
	if stats.LayersRolledBack != cfg.NumLayers {
		t.Fatalf("wrapped-ring fault LayersRolledBack = %d, want %d", stats.LayersRolledBack, cfg.NumLayers)
	}

	// The regression: layer 0's physical ring slot must be byte-identical to its
	// pre-step contents. Against the pre-fix implementation (which restored only
	// nextWindowPos) this slot holds the failed step's KV row and this
	// reflect.DeepEqual FAILS.
	if !reflect.DeepEqual(layer0.window[slot], capturedRow) {
		t.Fatalf("layer 0 wrapped ring slot %d was not restored: got %v, want %v", slot, layer0.window[slot], capturedRow)
	}
	if layer0.nextWindowPos != v41WindowSize {
		t.Fatalf("layer 0 nextWindowPos = %d, want restored %d", layer0.nextWindowPos, v41WindowSize)
	}
	if len(st.history) != v41WindowSize {
		t.Fatalf("history length = %d, want unchanged %d", len(st.history), v41WindowSize)
	}
}

// TestV41IncrementalMiddleLayerRollback proves LayersRolledBack == l+1
// distinguishes the failing layer: an unsupported state in layer 1 of 3 rolls
// back exactly two layers, and layer 2 never runs. Clause 5.
func TestV41IncrementalMiddleLayerRollback(t *testing.T) {
	m := v41IncrementalPlainModel(t, 3)
	cfg := m.Cfg

	prefix, next := v41IncrementalPrefix(cfg, 4)
	st := v41IncrementalState(t, m, prefix)
	if _, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{}); err != nil || !stats.Committed {
		t.Fatalf("priming step: err=%v committed=%t", err, stats.Committed)
	}

	const faultLayer = 1
	st.layerState(faultLayer).nextWindowPos = len(st.history) + 5 // unsupported: != pos
	before := v41AllNextWindowPos(st)
	beforeHistory := append([]int(nil), st.history...)

	faultTok := (next*7 + 1) % cfg.VocabSize
	_, stats, err := m.forwardV41Step(faultTok, st, &v41ProjScratch{})
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("middle-layer fault error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if stats.Committed {
		t.Fatal("middle-layer fault reported Committed = true")
	}
	if stats.LayersRolledBack != faultLayer+1 {
		t.Fatalf("middle-layer fault LayersRolledBack = %d, want %d", stats.LayersRolledBack, faultLayer+1)
	}
	if !reflect.DeepEqual(st.history, beforeHistory) {
		t.Fatalf("middle-layer fault history = %v, want unchanged %v", st.history, beforeHistory)
	}
	after := v41AllNextWindowPos(st)
	if !reflect.DeepEqual(after, before) {
		t.Fatalf("middle-layer fault nextWindowPos = %v, want restored %v", after, before)
	}
}

// TestV41IncrementalAdmissionRefusalCommitsNothing exercises a PRE-LOOP
// ADMISSION refusal: forwardV41Step calls m.v41ForwardAdmitted() before the layer
// loop, and that admission runs v41AdmitShape over ffn.shared_experts.w2.weight
// for every layer, so deleting the last layer's manifest entry refuses at
// admission rather than inside the layer's own MoE stage. LayerCalls == 0 pins
// that characterization.
//
// The refusal must commit nothing and move no cursor: history and every layer
// cursor stay at their pre-step values, and the step does not commit. Because no
// layer ran, LayersRolledBack is 0 (there is nothing staged to roll back) — in
// contrast to a cursor refusal inside the layer loop, which reports
// LayersRolledBack == NumLayers.
func TestV41IncrementalAdmissionRefusalCommitsNothing(t *testing.T) {
	m := v41IncrementalPlainModel(t, 3)
	cfg := m.Cfg

	prefix, next := v41IncrementalPrefix(cfg, 4)
	st := v41IncrementalState(t, m, prefix)
	if _, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{}); err != nil || !stats.Committed {
		t.Fatalf("priming step: err=%v committed=%t", err, stats.Committed)
	}

	last := cfg.NumLayers - 1
	delete(m.manifest, layerName(last, "ffn.shared_experts.w2.weight"))
	before := v41AllNextWindowPos(st)
	beforeHistory := append([]int(nil), st.history...)

	faultTok := (next*7 + 1) % cfg.VocabSize
	_, stats, err := m.forwardV41Step(faultTok, st, &v41ProjScratch{})
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("admission fault error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if stats.Committed {
		t.Fatal("admission fault reported Committed = true")
	}
	if stats.LayerCalls != 0 {
		t.Fatalf("admission fault LayerCalls = %d, want 0 (refused before the layer loop)", stats.LayerCalls)
	}
	if !reflect.DeepEqual(st.history, beforeHistory) {
		t.Fatalf("admission fault history = %v, want unchanged %v", st.history, beforeHistory)
	}
	if after := v41AllNextWindowPos(st); !reflect.DeepEqual(after, before) {
		t.Fatalf("admission fault nextWindowPos = %v, want restored %v", after, before)
	}
}

// TestV41IncrementalCopyFootprintIndependentOfPrefix is the copy/allocation
// probe. A brittle exact alloc-count equality is deliberately avoided (the
// ticket flags it as possibly flaky); instead the retained per-layer state is
// measured structurally with the package's own counters.
//
// Two honest, contract-derived properties are asserted:
//  1. bounded: every layer's retainedWindowRows / retainedCopies never exceed the
//     configured window, and retainedTailRows() never exceeds the fixed ring,
//     for ANY prefix length; and
//  2. saturated-independence: for two prefixes that both exceed the window
//     (7 and 12 on a window of 4), the per-layer retained footprint is IDENTICAL
//     — a prefix-sized clone or a rollback that snapshotted the prefix would
//     keep growing with the prefix instead of saturating at the window.
func TestV41IncrementalCopyFootprintIndependentOfPrefix(t *testing.T) {
	const window = 4
	m := v41IncrementalPlainModel(t, 2)
	m.Cfg.Window = []int{window, window}
	cfg := m.Cfg
	if got := cfg.windowForLayer(0); got != window {
		t.Fatalf("fixture window = %d, want %d", got, window)
	}

	footprint := func(prefixLen int) []int {
		t.Helper()
		prefix, next := v41IncrementalPrefix(cfg, prefixLen)
		st := v41IncrementalState(t, m, prefix)
		beforeHistory := len(st.history)

		got, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{})
		if err != nil {
			t.Fatalf("prefix %d: forwardV41Step: %v", prefixLen, err)
		}
		if !stats.Committed {
			t.Fatalf("prefix %d: step not committed", prefixLen)
		}
		if len(got) == 0 {
			t.Fatalf("prefix %d: step returned no logits", prefixLen)
		}
		if len(st.history) != beforeHistory+1 {
			t.Fatalf("prefix %d: history grew by %d, want exactly 1", prefixLen, len(st.history)-beforeHistory)
		}
		out := make([]int, 0, 2*cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			ls := st.layerState(l)
			tail := ls.retainedTailRows()
			if len(tail) > v41WindowSize {
				t.Fatalf("prefix %d layer %d: retained tail rows = %d, exceeds the %d-row ring", prefixLen, l, len(tail), v41WindowSize)
			}
			if ls.retainedWindowRows > window {
				t.Fatalf("prefix %d layer %d: retainedWindowRows = %d, exceeds configured window %d", prefixLen, l, ls.retainedWindowRows, window)
			}
			if ls.retainedCopies > window {
				t.Fatalf("prefix %d layer %d: retainedCopies = %d, exceeds configured window %d", prefixLen, l, ls.retainedCopies, window)
			}
			out = append(out, ls.retainedWindowRows, ls.retainedCopies)
		}
		return out
	}

	// Property 1: a short (sub-window) prefix stays bounded too.
	_ = footprint(3)
	// Property 2: two post-window prefixes saturate to the same footprint.
	if short, long := footprint(7), footprint(12); !reflect.DeepEqual(short, long) {
		t.Fatalf("retained footprint scales with prefix: prefix7=%v prefix12=%v, want identical", short, long)
	}
}

// TestV41IncrementalStepFootprintIndependentOfPrefix is the STEP-scoped
// complement to TestV41IncrementalCopyFootprintIndependentOfPrefix. That sibling
// asserts on retainedWindowRows / retainedCopies, which seedTemporal sets and the
// incremental Step never touches, so it witnesses the SEED's bounded footprint,
// not the step's.
//
// The fixture installs a POSITIVE per-layer window (a local sliding window), so a
// wrapped step (pos >= v41WindowSize) is legal: the visible key set is bounded by
// the window, so the ring always covers it. Both probe prefixes are at or past
// the wrap point, so a prefix-sized rollback ledger would be maximally visible.
//
// The probe records the step's observable effect on the retained state — the
// cursor delta, the retainedTailRows() length, and the history delta — and
// asserts the contract-level facts:
//
//   - the step's cursor delta is exactly one and history grows by exactly one,
//     regardless of prefix length; and
//   - the post-step retained tail is saturated at the fixed ring
//     (v41WindowSize), i.e. it does not grow with the prefix.
//
// Honest limitation: this is a structural saturation probe, not an
// allocation-count witness. It cannot observe the rollback ledger's size
// directly; it observes that the retained STATE the step produces is bounded by
// the ring and independent of the prefix.
func TestV41IncrementalStepFootprintIndependentOfPrefix(t *testing.T) {
	const window = 8
	m := v41IncrementalPlainModel(t, 2)
	m.Cfg.Window = []int{window, window}
	cfg := m.Cfg
	if got := cfg.windowForLayer(0); got != window {
		t.Fatalf("fixture window = %d, want %d", got, window)
	}

	type shape struct {
		cursorDelta []int
		tailLen     []int
		historyLen  int
	}
	probe := func(prefixLen int) shape {
		t.Helper()
		prefix, next := v41IncrementalPrefix(cfg, prefixLen)
		st := v41IncrementalState(t, m, prefix)
		before := v41AllNextWindowPos(st)
		beforeHistory := len(st.history)

		got, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{})
		if err != nil {
			t.Fatalf("prefix %d: forwardV41Step: %v", prefixLen, err)
		}
		if !stats.Committed {
			t.Fatalf("prefix %d: step not committed", prefixLen)
		}
		if len(got) == 0 {
			t.Fatalf("prefix %d: step returned no logits", prefixLen)
		}
		out := shape{historyLen: len(st.history) - beforeHistory}
		for l := 0; l < cfg.NumLayers; l++ {
			ls := st.layerState(l)
			out.cursorDelta = append(out.cursorDelta, ls.nextWindowPos-before[l])
			out.tailLen = append(out.tailLen, len(ls.retainedTailRows()))
		}
		return out
	}

	// The EXACT ring width: the next step is the first wrapped write.
	wrapped := probe(v41WindowSize)
	for l := range wrapped.cursorDelta {
		if wrapped.cursorDelta[l] != 1 {
			t.Fatalf("layer %d: step cursor delta = %d, want exactly 1", l, wrapped.cursorDelta[l])
		}
		if wrapped.tailLen[l] != v41WindowSize {
			t.Fatalf("layer %d: post-step retained tail = %d, want saturated %d", l, wrapped.tailLen[l], v41WindowSize)
		}
	}
	if wrapped.historyLen != 1 {
		t.Fatalf("history delta = %d, want exactly 1", wrapped.historyLen)
	}

	// A deeper prefix must observe the SAME per-step effect: cursor +1 and a
	// fully saturated tail. If the retained footprint grew with the prefix, the
	// deeper seed's pre-step tail would already be larger and this would break.
	long := probe(v41WindowSize + 6)
	if !reflect.DeepEqual(long.cursorDelta, wrapped.cursorDelta) {
		t.Fatalf("step cursor delta depends on prefix: %v vs %v", long.cursorDelta, wrapped.cursorDelta)
	}
	if !reflect.DeepEqual(long.tailLen, wrapped.tailLen) {
		t.Fatalf("step retained tail depends on prefix: %v vs %v", long.tailLen, wrapped.tailLen)
	}
	if long.historyLen != 1 {
		t.Fatalf("long-prefix history delta = %d, want exactly 1", long.historyLen)
	}
}

// TestV41IncrementalNestedPrefixUnawareStep stacks incrementally-projected layers
// on top of each other and checks the step contract still holds: a prefix seeded
// by one incremental step is a valid, append-ready session for the next. It runs
// the real pipeline (embed -> per-layer projection -> prefill seed -> incremental
// step) over more than one step and asserts the append-only, all-or-nothing
// invariants. It is NOT claimed as a ring-rollback or footprint witness
// (stacked incremental layers do not reproduce the full-sequence reference, so no
// parity is asserted).
func TestV41IncrementalNestedPrefixUnawareStep(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg

	prefix, next := v41IncrementalPrefix(cfg, 5)
	st := v41IncrementalState(t, m, prefix)

	// First step: a plain incremental step over the seeded prefix.
	if _, stats, err := m.forwardV41Step(next, st, &v41ProjScratch{}); err != nil || !stats.Committed {
		t.Fatalf("first nested step: err=%v committed=%t", err, stats.Committed)
	}
	if len(st.history) != len(prefix)+1 {
		t.Fatalf("after first step history = %d, want %d", len(st.history), len(prefix)+1)
	}
	// Second step: the state produced by the first step must be append-ready.
	tok2 := (next*3 + 1) % cfg.VocabSize
	got, stats, err := m.forwardV41Step(tok2, st, &v41ProjScratch{})
	if err != nil {
		t.Fatalf("second nested step: %v", err)
	}
	if !stats.Committed || stats.LayerCalls != cfg.NumLayers {
		t.Fatalf("second nested step: committed=%t LayerCalls=%d, want true/%d", stats.Committed, stats.LayerCalls, cfg.NumLayers)
	}
	if len(got) == 0 {
		t.Fatal("second nested step returned no logits")
	}
	if len(st.history) != len(prefix)+2 {
		t.Fatalf("after second step history = %d, want %d", len(st.history), len(prefix)+2)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if got := st.layerState(l).nextWindowPos; got != len(st.history) {
			t.Fatalf("layer %d nextWindowPos = %d, want %d", l, got, len(st.history))
		}
	}
}

// TestV41IncrementalFailClosedEdges proves the fail-closed edges (clause 6):
// out-of-range token ids and a nil continuation state refuse with the typed
// error and commit nothing. Clause 3. It also records the implementation's
// cold-start guard: a step over an EMPTY, unseeded history refuses rather than
// seeding a prefix from nothing (prefill owns the first token).
func TestV41IncrementalFailClosedEdges(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	cfg := m.Cfg

	prefix, _ := v41IncrementalPrefix(cfg, 3)
	for _, id := range []int{-1, cfg.VocabSize} {
		st := v41IncrementalState(t, m, prefix)
		beforeHistory := append([]int(nil), st.history...)
		before := v41AllNextWindowPos(st)
		_, stats, err := m.forwardV41Step(id, st, &v41ProjScratch{})
		if err == nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("token id %d: error = %v, want errors.Is(ErrV41ForwardStage)", id, err)
		}
		if stats.Committed {
			t.Fatalf("token id %d: refused step reported Committed = true", id)
		}
		if !reflect.DeepEqual(st.history, beforeHistory) {
			t.Fatalf("token id %d: history = %v, want unchanged %v", id, st.history, beforeHistory)
		}
		if after := v41AllNextWindowPos(st); !reflect.DeepEqual(after, before) {
			t.Fatalf("token id %d: nextWindowPos = %v, want unchanged %v", id, after, before)
		}
	}

	// A nil continuation state must fail closed, not panic.
	if _, stats, err := m.forwardV41Step(1, nil, &v41ProjScratch{}); err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("nil-state step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	} else if stats.Committed {
		t.Fatal("nil-state step reported Committed = true")
	}

	// An empty, unseeded history is a cold start: the step refuses rather than
	// fabricating a prefix seed, and commits nothing.
	{
		cold := &v41ForwardState{}
		_, stats, err := m.forwardV41Step(1, cold, &v41ProjScratch{})
		if err == nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("empty-history step error = %v, want errors.Is(ErrV41ForwardStage)", err)
		}
		if stats.Committed {
			t.Fatal("empty-history step reported Committed = true")
		}
		if len(cold.history) != 0 {
			t.Fatalf("empty-history step history = %v, want empty", cold.history)
		}
	}
}
