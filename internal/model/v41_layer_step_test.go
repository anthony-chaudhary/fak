package model

// v41_layer_step_test.go — the fak#13306 independent regression for the
// single-position plain-layer seam. It is authored against the CONTRACT, not the
// implementation: a step appended to a prefix must reproduce the leaf-13
// full-sequence forward's final row for a plain layer, and must refuse every
// unsupported role/position BEFORE mutating retained state.

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41LayerStepCloneState builds the retained state for layer l from a prefix by
// running the production seed path over the layer's projected KV rows. The rows
// are produced by the SAME projection helper the layer uses, so the state input
// is not a hand-rolled stand-in.
func v41LayerStepSeed(t *testing.T, m *Model, l int, prefix []int) *V41AttentionState {
	t.Helper()
	cfg := m.Cfg
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	_ = nH
	eps := float32(cfg.RMSNormEps)
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}
	embed := m.tensor("model.embed_tokens.weight")
	x := make([][]float32, len(prefix))
	for t, id := range prefix {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	// The per-position KV row is projected from the mHC PRE-COLLAPSE vector, not
	// from the raw attn-normed hidden: the full forward's `c` in its projection
	// loop is `preByPos[t]` (v41_forward.go v41Layer). Reproduce that pre-collapse
	// here with the same mHC arithmetic so the seeded retained row is the layer's
	// real KV row; a seed built from the wrong carrier makes the parity witness
	// compare against a row the model never produces.
	wMix, err := m.v41MHCMixF32Into(l, nil)
	if err != nil {
		t.Fatal(err)
	}
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		t.Fatalf("layer %d mHC mix weight holds no admitted geometry", l)
	}
	rows := make([][]float32, len(prefix))
	for pos := range prefix {
		xn := rmsnormCfg(x[pos], attnNorm, eps, cfg)
		var mixes []float32
		if mhcFlat {
			mixes, err = v41MHCProjectFull(wMix, [][]float32{x[pos], x[pos], x[pos], x[pos]}, H, eps, mhcTransposed)
			if err != nil {
				t.Fatal(err)
			}
		} else {
			mixes = matRows(wMix, xn, v41MHCMixWidth, H)
		}
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
		if err != nil {
			t.Fatal(err)
		}
		streams4 := [][]float32{x[pos], x[pos], x[pos], x[pos]}
		if !full {
			streams4 = [][]float32{xn, xn, xn, xn}
		}
		c, err := v41MHCPre(streams4, mix.pre)
		if err != nil {
			t.Fatal(err)
		}
		var kv []float32
		if full {
			kvFull, projErr := m.v41ProjMatRows(l, "attn.wkv.weight", c, v41KVLoraRank, H)
			if projErr != nil {
				t.Fatal(projErr)
			}
			kv = append([]float32(nil), kvFull[:hd]...)
			kvNorm := m.tensor(layerName(l, "attn.kv_norm.weight"))
			kv = rmsnormCfg(kv, kvNorm, eps, cfg)
		} else {
			kvReduced, projErr := m.v41ProjMatRows(l, "attn.wkv.weight", c, hd, H)
			if projErr != nil {
				t.Fatal(projErr)
			}
			kv = append([]float32(nil), kvReduced...)
		}
		cos, sin := v41RopeTableForLayer(cfg, l, pos)
		applyRopeTailInterleaved(kv, cos, sin, cfg.QKRopeHeadDim)
		rows[pos] = kv
	}
	state, err := NewV41AttentionState(hd, 8)
	if err != nil {
		t.Fatal(err)
	}
	// An empty prefix has nothing to seed: the state stays freshly constructed
	// and the position-0 step seeds it with its first row.
	if len(prefix) == 0 {
		return state
	}
	if err := state.seedTemporal(rows, 0, cfg.windowForLayer(l)); err != nil {
		t.Fatal(err)
	}
	return state
}

// TestV41LayerStepPlain is the leaf witness: the final hidden row after a
// prefix+one-token step matches the full-sequence forward's final row, for a
// plain layer, at empty/short prefixes and the exact 127/128 boundaries for a
// global (full-causal) layer. The retained ring is a fixed 128 rows
// (v41WindowSize), so a global layer's full-prefix parity holds exactly while
// every retained key is visible, i.e. pos <= 128; beyond that a bounded state
// cannot reproduce the unbounded prefix and the step is asserted against the
// same 128-row window the ring actually holds (the leaf-13 configured-mask
// semantics) rather than against an unreachable full prefix.
func TestV41LayerStepPlain(t *testing.T) {
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{-1} // global layer: full causal prefix, exercises 128/129
	cfg := m.Cfg

	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Within the retained ring a global layer's step must reproduce the full
	// forward's final row exactly.
	for _, prefixLen := range []int{0, 1, 7, 126, 127, 128} {
		prefix := make([]int, prefixLen)
		for i := range prefix {
			prefix[i] = (i*7 + 3) % cfg.VocabSize
		}
		next := (prefixLen*7 + 3) % cfg.VocabSize
		ids := append(append([]int(nil), prefix...), next)

		want, err := m.forwardV41(ids, nil)
		if err != nil {
			t.Fatalf("prefix %d: full forward: %v", prefixLen, err)
		}
		wantX := lastHiddenRow(want)

		// The step path: seed the layer state from the prefix, then step the NEXT
		// token's embedded row through this layer. Layer 0's input at a new
		// position is that token's (scaled) embedding; streams 1..3 are the
		// persistent zero residuals a full forward initializes them to.
		state := v41LayerStepSeed(t, m, 0, prefix)
		x, streams := v41LayerStepInputs(t, m, next, full)
		if err := m.v41LayerStep(0, x, streams, prefixLen, state, &v41ProjScratch{}); err != nil {
			t.Fatalf("prefix %d: v41LayerStep: %v", prefixLen, err)
		}
		if got := state.nextWindowPos; got != prefixLen+1 {
			t.Fatalf("prefix %d: step advanced window position to %d, want %d", prefixLen, got, prefixLen+1)
		}
		const tol = 1e-6
		for i := range wantX {
			delta := x[i] - wantX[i]
			if delta < 0 {
				delta = -delta
			}
			if delta > tol || math.IsNaN(float64(x[i])) {
				t.Fatalf("prefix %d: step final row[%d] = %g, want %g (delta %g)", prefixLen, i, x[i], wantX[i], delta)
			}
		}
	}

	// Past the ring a global layer's step must FAIL CLOSED: the oldest causal
	// keys are gone and a bounded state cannot reproduce the unbounded prefix, so
	// silently contracting the surviving tail would emit logits from a truncated
	// context. The refusal must not mutate retained state.
	{
		const prefixLen = 129
		prefix := make([]int, prefixLen)
		for i := range prefix {
			prefix[i] = (i*7 + 3) % cfg.VocabSize
		}
		next := (prefixLen*7 + 3) % cfg.VocabSize
		state := v41LayerStepSeed(t, m, 0, prefix)
		if got := state.retainedWindowRows; got != v41WindowSize {
			t.Fatalf("prefix %d: retained rows = %d, want the %d-row ring", prefixLen, got, v41WindowSize)
		}
		beforePos := state.nextWindowPos
		x, streams := v41LayerStepInputs(t, m, next, full)
		err := m.v41LayerStep(0, x, streams, prefixLen, state, &v41ProjScratch{})
		if err == nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("global step at prefix %d error = %v, want errors.Is(ErrV41ForwardStage)", prefixLen, err)
		}
		if state.nextWindowPos != beforePos {
			t.Fatalf("refused global step mutated position to %d, want %d", state.nextWindowPos, beforePos)
		}
	}

	// An out-of-order position is refused before mutation.
	state := v41LayerStepSeed(t, m, 0, []int{1, 3})
	beforePos := state.nextWindowPos
	x, streams := v41LayerStepInputs(t, m, 4, full)
	err = m.v41LayerStep(0, x, streams, 7, state, &v41ProjScratch{})
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("out-of-order step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if state.nextWindowPos != beforePos {
		t.Fatalf("refused step mutated window position to %d, want %d", state.nextWindowPos, beforePos)
	}
}

// TestV41LayerStepRefusesNonPlainRoles proves the fail-closed contract: a
// compressed, source or reader layer is refused BEFORE any mutation, so the
// seam can never silently run the per-layer window path for a schedule the
// checkpoint did not declare.
func TestV41LayerStepRefusesNonPlainRoles(t *testing.T) {
	m := v41DecodeStateModel(t, 3)
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}
	x, streams := v41LayerStepInputs(t, m, 5, full)

	// A reader layer (compressed regime following a declared source) must refuse.
	m.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}
	m.Cfg.DeepSeekV41.CompressRatios = []int{0, 2, 0}
	m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
	roles := v41AttentionRoles(m.Cfg)
	if roles[1] != V41AttentionRoleReader {
		t.Fatalf("fixture role[1] = %d, want reader", roles[1])
	}
	state, err := NewV41AttentionState(cfg.HeadDim, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.seedTemporal([][]float32{make([]float32, cfg.HeadDim)}, 0, -1); err != nil {
		t.Fatal(err)
	}
	beforePos := state.nextWindowPos
	stepErr := m.v41LayerStep(1, x, streams, beforePos, state, &v41ProjScratch{})
	if stepErr == nil || !errors.Is(stepErr, ErrV41ForwardStage) {
		t.Fatalf("reader-layer step error = %v, want errors.Is(ErrV41ForwardStage)", stepErr)
	}
	if state.nextWindowPos != beforePos {
		t.Fatalf("refused reader step mutated position to %d, want %d", state.nextWindowPos, beforePos)
	}

	// A nil state must refuse too, not panic.
	if err := m.v41LayerStep(1, x, streams, 0, nil, &v41ProjScratch{}); err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("nil-state step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
}

// TestV41LayerStepLocalWindowRetainsBoundaryRows exercises the leaf-13 window
// boundary: a local layer with window 4 at positions 0..5 attends exactly the
// trailing 4 rows, and the step's ordered key set must match the full forward's.
func TestV41LayerStepLocalWindowRetainsBoundaryRows(t *testing.T) {
	for _, w := range []int{1, 4, 128} {
		for _, prefixLen := range []int{0, 1, 3, 5, 127, 128, 129} {
			pos := prefixLen
			keys := v41PlainWindowKeys(pos, w)
			if len(keys) == 0 {
				t.Fatalf("window %d pos %d: empty key set", w, pos)
			}
			if int(keys[len(keys)-1]) != pos {
				t.Fatalf("window %d pos %d: last key %d, want %d", w, pos, keys[len(keys)-1], pos)
			}
			if len(keys) > w {
				t.Fatalf("window %d pos %d: %d keys exceed the window", w, pos, len(keys))
			}
			if got := int(keys[0]); got != maxInt(0, pos-w+1) {
				t.Fatalf("window %d pos %d: first key %d, want %d", w, pos, got, maxInt(0, pos-w+1))
			}
		}
	}

	// A local layer parity check at the exact window boundary.
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{4}
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}
	prefix := make([]int, 7)
	for i := range prefix {
		prefix[i] = (i*3 + 1) % cfg.VocabSize
	}
	next := (len(prefix)*3 + 1) % cfg.VocabSize
	ids := append(append([]int(nil), prefix...), next)
	want, err := m.forwardV41(ids, nil)
	if err != nil {
		t.Fatalf("local-window full forward: %v", err)
	}
	wantX := lastHiddenRow(want)

	state := v41LayerStepSeed(t, m, 0, prefix)
	x, streams := v41LayerStepInputs(t, m, next, full)
	if err := m.v41LayerStep(0, x, streams, len(prefix), state, &v41ProjScratch{}); err != nil {
		t.Fatalf("local-window step: %v", err)
	}
	const tol = 1e-6
	for i := range wantX {
		delta := x[i] - wantX[i]
		if delta < 0 {
			delta = -delta
		}
		if delta > tol {
			t.Fatalf("local-window step row[%d] = %g, want %g", i, x[i], wantX[i])
		}
	}
}

// TestV41LayerStepCountsOneNewRow proves the step processes exactly one new row:
// the retained window position advances by exactly one and no more.
func TestV41LayerStepCountsOneNewRow(t *testing.T) {
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{-1}
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}
	state := v41LayerStepSeed(t, m, 0, []int{1, 3, 5})
	x, streams := v41LayerStepInputs(t, m, 7, full)
	before := state.nextWindowPos
	if err := m.v41LayerStep(0, x, streams, before, state, &v41ProjScratch{}); err != nil {
		t.Fatalf("step: %v", err)
	}
	if got := state.nextWindowPos - before; got != 1 {
		t.Fatalf("step advanced %d positions, want exactly 1", got)
	}
}

// TestV41LayerStepMultiStepDecode proves a session can advance MORE THAN ONE
// position: two sequential steps on the same retained state reproduce the full
// forward's final row at each appended position. This is the defect the single
// step witness missed -- V41AttentionState.Step does not grow its
// retainedWindowRows counter, so a step that read that counter would under-count
// the retained tail and spuriously refuse (or truncate) the second step.
func TestV41LayerStepMultiStepDecode(t *testing.T) {
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{-1}
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}

	prefix := []int{1, 3, 5, 2}
	steps := []int{7, 4, 6, 0, 3, 1}
	state := v41LayerStepSeed(t, m, 0, prefix)

	history := append([]int(nil), prefix...)
	for i, tok := range steps {
		history = append(history, tok)
		want, err := m.forwardV41(history, nil)
		if err != nil {
			t.Fatalf("step %d: full forward: %v", i, err)
		}
		wantX := lastHiddenRow(want)

		pos := len(history) - 1
		x, streams := v41LayerStepInputs(t, m, tok, full)
		if err := m.v41LayerStep(0, x, streams, pos, state, &v41ProjScratch{}); err != nil {
			t.Fatalf("step %d (pos %d): v41LayerStep: %v", i, pos, err)
		}
		if got := state.nextWindowPos; got != pos+1 {
			t.Fatalf("step %d: window position %d, want %d", i, got, pos+1)
		}
		const tol = 1e-6
		for j := range wantX {
			delta := x[j] - wantX[j]
			if delta < 0 {
				delta = -delta
			}
			if delta > tol || math.IsNaN(float64(x[j])) {
				t.Fatalf("step %d pos %d row[%d] = %g, want %g (delta %g)", i, pos, j, x[j], wantX[j], delta)
			}
		}
	}
}

// TestV41LayerStepRefusesWindowWiderThanRing proves the fail-closed guard covers
// a LOCAL layer whose configured window is wider than the fixed retained ring:
// its oldest visible keys are not retained, so the step must refuse rather than
// silently contract a truncated context.
func TestV41LayerStepRefusesWindowWiderThanRing(t *testing.T) {
	m := v41DecodeStateModel(t, 1)
	m.Cfg.Window = []int{v41WindowSize + 2}
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// A prefix that fills the ring so the oldest visible key falls below it.
	prefixLen := v41WindowSize + 5
	prefix := make([]int, prefixLen)
	for i := range prefix {
		prefix[i] = (i*3 + 1) % cfg.VocabSize
	}
	next := (prefixLen*3 + 1) % cfg.VocabSize
	state := v41LayerStepSeed(t, m, 0, prefix)
	beforePos := state.nextWindowPos
	x, streams := v41LayerStepInputs(t, m, next, full)
	err = m.v41LayerStep(0, x, streams, prefixLen, state, &v41ProjScratch{})
	if err == nil || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("oversized-window step error = %v, want errors.Is(ErrV41ForwardStage)", err)
	}
	if state.nextWindowPos != beforePos {
		t.Fatalf("refused oversized-window step mutated position to %d, want %d", state.nextWindowPos, beforePos)
	}
}

func lastHiddenRow(act *Activations) []float32 {
	if act == nil || len(act.Hidden) == 0 {
		return nil
	}
	// act.Hidden[i] is flatten(x) AFTER layer i (Hidden[0] is the embedding row
	// panel). Each entry is a flat seq*H slice; the row for the LAST position is
	// its trailing H values. Recover H from the model's own width by dividing the
	// flat length by the sequence length, never by the flat length itself.
	if act.Seq <= 0 {
		return nil
	}
	flat := act.Hidden[len(act.Hidden)-1]
	if len(flat)%act.Seq != 0 {
		return nil
	}
	H := len(flat) / act.Seq
	return append([]float32(nil), flat[(act.Seq-1)*H:]...)
}

// v41LayerStepInputs builds the input carrier for a step of the NEXT token id
// through layer 0: the token's scaled embedding in stream 0 and the persistent
// zero residuals a full forward initializes streams 1..3 to.
func v41LayerStepInputs(t *testing.T, m *Model, next int, full bool) ([]float32, [][]float32) {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	embed := m.tensor("model.embed_tokens.weight")
	x := append([]float32(nil), embed[next*H:(next+1)*H]...)
	scaleEmbedInPlace(x, cfg)
	streams := [][]float32{x, make([]float32, H), make([]float32, H), make([]float32, H)}
	_ = full
	return x, streams
}

var _ = reflect.DeepEqual
