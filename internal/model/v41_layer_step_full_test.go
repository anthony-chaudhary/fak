package model

// v41_layer_step_full_test.go — the fak#13433 witness for the FULL-geometry
// (full=true) branches of the #13306 single-position plain-layer decode seam.
//
// Every existing witness for v41LayerStep (TestV41LayerStepPlain and siblings in
// v41_layer_step_test.go) runs on the REDUCED fixture (v41ForwardGeometry returns
// full=false: HiddenSize=64, NumHeads=2, HeadDim=32). The full-only branches of
// the seam are therefore compiled but never executed:
//
//   - v41MHCProjectFull — the flattened four-stream mHC mix projection (the
//     artifact's stored [4H,24] transpose), not the reduced 24 x H matmul;
//   - the q-lora latent norm (attn.wq_a_norm) applied to the wq_a output before
//     wq_b (v41_layer_step.go:204-211); and
//   - the KV latent norm (attn.kv_norm) applied at the published v41KVLoraRank
//     (512) width before the rope tail (v41_layer_step.go:217-240).
//
// This file adds an independent full-geometry witness that asserts the SAME
// parity the reduced witness asserts — prefix + one-token v41LayerStep equals
// m.forwardV41(prefix + token)'s final row — on a small but internally consistent
// full model built by the existing #13258 fixture (v41RawFullFlattenedMHC: a
// materialized synthetic raw checkpoint whose mhc.mixes.weight is the stored
// [4H,24] transpose).
//
// NOTE on the seed. The existing v41LayerStepSeed helper builds the retained
// state from "four identical copies of the raw input", which is the REDUCED
// stand-in (v41Layer's `streams4 = {xn,xn,xn,xn}` branch). The full path seeds
// the persistent streams as `{x[t], 0, 0, 0}` (stream 0 the live hidden, streams
// 1..3 the zero residual), so this file carries its own full-correct seed
// (v41FullStepSeed) rather than reusing the reduced-only helper; the reduced
// helper and its witnesses are left untouched. (On this tiny fixture the two
// initializations happen to agree within tolerance because the KV latent norm
// compresses the pre-collapse difference; the distinct-stream seed is still the
// faithful one and is what the forward actually runs.)
//
// Branch coverage is demonstrated non-vacuously:
//   - the flattened mHC path is gated by v41MHCWeightLayout reporting the stored
//     [4H,24] transpose, which is exactly the condition v41LayerStep branches on
//     to call v41MHCProjectFull; and
//   - the two latent norms are shown to be READ by the step: a second fixture
//     with nonuniform, non-unit q/kv norm gains produces a different step row
//     than the unit-gain fixture, and still matches forwardV41. A step that
//     dropped, mis-ordered or unit-substituted either norm would break the parity.
//
// Test-only: no production behavior change; the reduced witness is untouched.

import (
	"errors"
	"math"
	"testing"
)

// v41FullStepModel is the full-geometry fixture with an explicit global
// (full-causal) window so the retained-ring boundary is reachable, mirroring the
// reduced witness's m.Cfg.Window = []int{-1}.
func v41FullStepModel(t *testing.T) *Model {
	t.Helper()
	m := v41RawFullFlattenedMHC(t)
	m.Cfg.Window = []int{-1}
	return m
}

// v41FullStepPatchedNorms is v41FullStepModel with deterministic nonuniform,
// non-unit q-lora and KV latent gain vectors written into the two norm tensors.
// The gains are a smooth ramp strictly inside (0.5,1.5), so a step that ignores,
// reorders or unit-substitutes either norm cannot coincidentally agree.
func v41FullStepPatchedNorms(t *testing.T) *Model {
	t.Helper()
	m := v41RawFullFlattenedMHC(t)
	m.Cfg.Window = []int{-1}
	cfg := m.Cfg
	qGain := make([]float32, cfg.QLoraRank)
	for i := range qGain {
		qGain[i] = float32(0.5 + float64(i)/float64(cfg.QLoraRank))
	}
	kvGain := make([]float32, v41KVLoraRank)
	for i := range kvGain {
		kvGain[i] = float32(0.5 + float64(i)/float64(v41KVLoraRank))
	}
	v41WriteTensorF32(t, m, layerName(0, "attn.wq_a_norm.weight"), qGain)
	v41WriteTensorF32(t, m, layerName(0, "attn.kv_norm.weight"), kvGain)
	if v41AllOnes(qGain) || v41AllOnes(kvGain) {
		t.Fatal("patched latent-norm gains are unit; the non-vacuity assertion would be meaningless")
	}
	return m
}

// v41FullStepSeed builds layer 0's retained decode state from a prefix, mirroring
// the FULL forward's per-position KV row: the row is projected from the mHC
// PRE-COLLAPSE of the persistent streams `{x[pos], 0, 0, 0}`, through the
// flattened projection, then the KV latent norm at the published width, then the
// rope tail. An empty prefix yields a freshly constructed state (position 0's
// step seeds it), matching the production append-only contract.
func v41FullStepSeed(t *testing.T, m *Model, l int, prefix []int) *V41AttentionState {
	t.Helper()
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	eps := float32(cfg.RMSNormEps)
	if full, err := v41ForwardGeometry(cfg); err != nil || !full {
		t.Fatalf("v41FullStepSeed requires full geometry, got (%v,%v)", full, err)
	}
	state, err := NewV41AttentionState(hd, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(prefix) == 0 {
		return state
	}

	embed := m.tensor("model.embed_tokens.weight")
	x := make([][]float32, len(prefix))
	for t, id := range prefix {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	wMix, err := m.v41MHCMixF32Into(l, nil)
	if err != nil {
		t.Fatal(err)
	}
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK || !mhcFlat {
		t.Fatalf("layer %d mHC weight is not the flattened full geometry (flat=%v ok=%v)", l, mhcFlat, mhcOK)
	}
	kvNorm := m.tensor(layerName(l, "attn.kv_norm.weight"))
	rows := make([][]float32, len(prefix))
	for pos := range prefix {
		// The full forward's persistent streams: stream 0 is the live hidden and
		// streams 1..3 are the zero-initialized residual (distinct, not copies of
		// stream 0). This is the initialization v41LayerStepSeed's reduced stand-in
		// does NOT reproduce, which is why this seed is authored here.
		streams := [][]float32{x[pos], make([]float32, H), make([]float32, H), make([]float32, H)}
		mixes, err := v41MHCProjectFull(wMix, streams, H, eps, mhcTransposed)
		if err != nil {
			t.Fatal(err)
		}
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
		if err != nil {
			t.Fatal(err)
		}
		collapsed, err := v41MHCPre(streams, mix.pre)
		if err != nil {
			t.Fatal(err)
		}
		kvFull, err := m.v41ProjMatRows(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H)
		if err != nil {
			t.Fatal(err)
		}
		kv := rmsnormCfg(append([]float32(nil), kvFull[:hd]...), kvNorm, eps, cfg)
		cos, sin := v41RopeTableForLayer(cfg, l, pos)
		applyRopeTailInterleaved(kv, cos, sin, cfg.QKRopeHeadDim)
		rows[pos] = kv
	}
	if err := state.seedTemporal(rows, 0, cfg.windowForLayer(l)); err != nil {
		t.Fatal(err)
	}
	return state
}

// v41RunFullStep seeds layer 0's retained state from prefix and steps the NEXT
// token through the full-geometry seam, returning the advanced hidden row. It
// asserts the step committed exactly one position.
func v41RunFullStep(t *testing.T, m *Model, prefix []int, next int) []float32 {
	t.Helper()
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	state := v41FullStepSeed(t, m, 0, prefix)
	x, streams := v41LayerStepInputs(t, m, next, full)
	if err := m.v41LayerStep(0, x, streams, len(prefix), state, &v41ProjScratch{}); err != nil {
		t.Fatalf("prefix %d: full-geometry v41LayerStep: %v", len(prefix), err)
	}
	if got := state.nextWindowPos; got != len(prefix)+1 {
		t.Fatalf("prefix %d: step advanced window position to %d, want %d", len(prefix), got, len(prefix)+1)
	}
	return x
}

// assertV41RowsClose fails when any element of got differs from want by more
// than tol or is non-finite.
func assertV41RowsClose(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: row width %d, want %d", what, len(got), len(want))
	}
	for i := range want {
		delta := math.Abs(float64(got[i] - want[i]))
		if delta > tol || math.IsNaN(float64(got[i])) {
			t.Fatalf("%s: row[%d] = %g, want %g (delta %g, tol %.0e)", what, i, got[i], want[i], delta, tol)
		}
	}
}

// TestV41LayerStepFullGeometry is the #13433 acceptance witness. It drives the
// real full-geometry (full=true) branches of v41LayerStep and requires the same
// parity the reduced fixture proves, plus explicit evidence that the q/kv latent
// norms and the flattened mHC projection are engaged.
func TestV41LayerStepFullGeometry(t *testing.T) {
	const tol = 1e-6

	t.Run("full geometry is selected with the flattened mHC transpose", func(t *testing.T) {
		m := v41FullStepModel(t)
		if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
			t.Fatalf("fixture geometry = (%v,%v), want (true,nil)", full, err)
		}
		// v41MHCWeightLayout's flat+transposed verdict is exactly the condition
		// v41LayerStep branches on to call v41MHCProjectFull (v41_layer_step.go:164,
		// :171-172). A reduced [24,H] weight would report flat=false and take the
		// matRows stand-in instead, so this assertion is the branch gate.
		flat, transposed, ok := m.v41MHCWeightLayout(0)
		if !ok || !flat || !transposed {
			t.Fatalf("mHC layout = (flat=%v, transposed=%v, ok=%v), want the stored [4H,24] transpose that selects v41MHCProjectFull",
				flat, transposed, ok)
		}
		// The full-only latent-norm leaves must be present for the step to read.
		for _, leaf := range []string{"attn.wq_a_norm.weight", "attn.kv_norm.weight"} {
			if !m.has(layerName(0, leaf)) {
				t.Fatalf("full fixture is missing %s", layerName(0, leaf))
			}
		}
	})

	t.Run("step matches forwardV41 at empty and short prefixes and the retained-ring boundary", func(t *testing.T) {
		m := v41FullStepModel(t)
		cfg := m.Cfg
		for _, prefixLen := range []int{0, 1, 7, v41WindowSize - 1, v41WindowSize} {
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

			got := v41RunFullStep(t, m, prefix, next)
			assertV41RowsClose(t, "prefix "+itoa(prefixLen), got, wantX, tol)
		}
	})

	t.Run("past the retained ring the full step fails closed without mutating state", func(t *testing.T) {
		m := v41FullStepModel(t)
		cfg := m.Cfg
		const prefixLen = v41WindowSize + 1
		prefix := make([]int, prefixLen)
		for i := range prefix {
			prefix[i] = (i*7 + 3) % cfg.VocabSize
		}
		next := (prefixLen*7 + 3) % cfg.VocabSize
		full, err := v41ForwardGeometry(cfg)
		if err != nil {
			t.Fatal(err)
		}
		state := v41FullStepSeed(t, m, 0, prefix)
		if got := state.retainedWindowRows; got != v41WindowSize {
			t.Fatalf("precision: retained rows = %d, want the %d-row ring", got, v41WindowSize)
		}
		beforePos := state.nextWindowPos
		x, streams := v41LayerStepInputs(t, m, next, full)
		err = m.v41LayerStep(0, x, streams, prefixLen, state, &v41ProjScratch{})
		if err == nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("global full step at prefix %d error = %v, want errors.Is(ErrV41ForwardStage)", prefixLen, err)
		}
		if state.nextWindowPos != beforePos {
			t.Fatalf("refused full step mutated position to %d, want %d", state.nextWindowPos, beforePos)
		}
	})

	t.Run("q and kv latent norms are read by the step", func(t *testing.T) {
		// Same prefix/next on a unit-gain model and a patched non-unit-gain model:
		// the step rows must DIFFER, proving the norm weights are read (non-vacuous).
		// The parity subtest above then proves they are applied at the pinned width
		// and order.
		base := v41FullStepModel(t)
		patched := v41FullStepPatchedNorms(t)
		prefix := []int{1, 3, 5}
		next := 7
		baseRow := v41RunFullStep(t, base, prefix, next)
		patchedRow := v41RunFullStep(t, patched, prefix, next)
		if v41RowsAgree(baseRow, patchedRow, tol) {
			t.Fatal("step row is identical with unit and non-unit latent-norm gains; the norm path is not read by v41LayerStep")
		}
		// The patched model must ALSO match forwardV41, so the divergent row is the
		// correct one, not just a different one.
		ids := append(append([]int(nil), prefix...), next)
		want, err := patched.forwardV41(ids, nil)
		if err != nil {
			t.Fatalf("patched full forward: %v", err)
		}
		assertV41RowsClose(t, "patched-norm parity", patchedRow, lastHiddenRow(want), tol)
	})

	t.Run("flattened mHC projection is load-bearing", func(t *testing.T) {
		// Perturb the flattened mhc.mixes.weight and confirm the step row MOVES
		// while STILL matching forwardV41. That proves v41MHCProjectFull's output
		// feeds the step result (the branch is not dead) and that the step applies
		// the same projection the full forward does.
		m := v41FullStepModel(t)
		prefix := []int{1, 3, 5}
		next := 7
		baseRow := v41RunFullStep(t, m, prefix, next)

		mixName := layerName(0, "mhc.mixes.weight")
		w := append([]float32(nil), m.tensor(mixName)...)
		for i := 0; i < len(w); i += 97 {
			w[i] += 0.5
		}
		v41WriteTensorF32(t, m, mixName, w)

		pertRow := v41RunFullStep(t, m, prefix, next)
		if v41RowsAgree(baseRow, pertRow, tol) {
			t.Fatal("perturbing the flattened mHC weight did not move the step row; v41MHCProjectFull is not load-bearing")
		}
		ids := append(append([]int(nil), prefix...), next)
		want, err := m.forwardV41(ids, nil)
		if err != nil {
			t.Fatalf("perturbed-mHC full forward: %v", err)
		}
		assertV41RowsClose(t, "perturbed-mHC parity", pertRow, lastHiddenRow(want), tol)
	})
}

// v41RowsAgree reports whether a and b are within tol on every element.
func v41RowsAgree(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tol {
			return false
		}
	}
	return true
}
