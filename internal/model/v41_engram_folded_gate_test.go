package model

// v41_engram_folded_gate_test.go is the #14121 acceptance witness: the DS4.1
// Engram gate may consume its per-cell coefficient either UNFOLDED (the two
// F32 norm tensors engram_q_norm.weight / engram_k_norm.weight, whose product
// the host recomputes at runtime) or FOLDED (one F32 tensor
// engram_folded_gate.weight carrying the pre-composed product, the form the upstream
// DS4.1 converter/runtime publishes — the reference keeps engram_q and
// engram_k apart but only ever uses their product). Exactly one form per
// Engram layer is admitted; co-presence is refused as ambiguous, and every
// unimplemented combination fails closed with ErrV41NativeUnsupported before
// retrieval, projection, or publication.
//
// Arms:
//  1. folded-only reduced + full injection matches independent f64 oracles
//     that consume the folded cells directly (never reconstructing q/k);
//  2. bit-identity: unfolded q=g, k=ones publishes byte-identical results to
//     folded coef=g (g[i]*1.0 is exact), proving no BF16 boundary moved;
//  3. fail-closed: non-finite folded cell (both entry routes), mis-shaped
//     folded tensor, co-presence, and neither-form-present are all typed
//     refusals with zero projection calls, no cache movement, and no partial
//     publication;
//  4. admission: v41EngramForwardAdmitted admits folded-only and refuses the
//     mixed and missing combinations.
//
// Loader aliasing of the upstream GGUF spelling onto engram_folded_gate.weight is a
// follow-up leaf (ggufload); this file pins the model-side contract.

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41FoldedTestCoefficients builds hc*H deterministic coefficients, strictly
// varying per (stream, index) inside (-0.9, 0.9) and never exactly zero
// ((i*7+s*3)%17 is an integer while the zero of the ramp sits at 8.5), so a
// dropped, reordered, or unit-substituted coefficient cannot coincidentally
// agree with the oracle.
func v41FoldedTestCoefficients(hc, H int) []float32 {
	coef := make([]float32, hc*H)
	for s := 0; s < hc; s++ {
		for i := 0; i < H; i++ {
			coef[s*H+i] = float32(0.5*float64(s+1)) * float32(float64((i*7+s*3)%17)/float64(17)-0.5)
		}
	}
	return coef
}

// v41SetGateForms reshapes the layer's gate-coefficient forms on a fixture
// model: removeQK deletes the unfolded q/k norm tensors, and a non-nil coef
// appends a folded engram_folded_gate.weight tensor to the raw blob and manifest
// (mirroring how a future upstream GGUF would publish the pre-composed
// product). coef == nil leaves the layer with whatever q/k form remains.
func v41SetGateForms(t *testing.T, m *Model, layer int, removeQK bool, coef []float32) {
	t.Helper()
	if removeQK {
		delete(m.manifest, layerName(layer, "engram_q_norm.weight"))
		delete(m.manifest, layerName(layer, "engram_k_norm.weight"))
	}
	if coef == nil {
		return
	}
	name := layerName(layer, v41EngramFoldedGateLeaf)
	if _, dup := m.manifest[name]; dup {
		t.Fatalf("fixture: %s already present", name)
	}
	off := len(m.raw)
	m.raw = append(m.raw, make([]byte, len(coef)*4)...)
	for i, v := range coef {
		binary.LittleEndian.PutUint32(m.raw[off+i*4:], math.Float32bits(v))
	}
	m.manifest[name] = tensorMeta{Dtype: "F32", Shape: []int{len(coef)}, Offset: off, Nbytes: len(coef) * 4}
}

// v41FoldedGateGeometry asserts the layer resolved the folded form: coef
// populated, q/k empty, cells == hc*H.
func v41FoldedGateGeometry(t *testing.T, m *Model, l, hc, H int) v41EngramGeometry {
	t.Helper()
	g, err := m.v41EngramInjectionGeometry(l, 1)
	if err != nil {
		t.Fatalf("v41EngramInjectionGeometry(folded) error = %v, want nil", err)
	}
	if g.coef == nil || len(g.qNorm) != 0 || len(g.kNorm) != 0 || len(g.coef) != hc*H {
		t.Fatalf("geometry did not resolve the folded form: coef=%d qNorm=%d kNorm=%d, want coef=%d qNorm=0 kNorm=0",
			len(g.coef), len(g.qNorm), len(g.kNorm), hc*H)
	}
	return g
}

// v41OracleFoldedEngramInject is the independent scalar transcription of the
// reduced Engram add for one in-range layer, consuming the folded coefficient
// cells directly (never reconstructing q/k). It mirrors v41OracleEngramInject
// except the dot product multiplies by coef[s,i] instead of qNorm[k]*kNorm[k];
// h2/k2/dot accumulate in float64, matching the f32 forward within
// cpuOracleTol. It mutates x in place.
func v41OracleFoldedEngramInject(t *testing.T, m *Model, layout V41EngramLayout, l int, x [][]float32, ids []int, eps float32, coef []float32) {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := cfg.DeepSeekV41.EngramHeadDim
	hc := 4
	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("oracle: Engram stage not wired")
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		t.Fatal(err)
	}
	allRows, err := hash.Hash(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	layers := len(layout.Rows)
	cacheIdx := stage.cacheIndex(l)
	layerRows := make([]uint32, 0, len(ids)*cols)
	for tt := range ids {
		base := tt*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		t.Fatal(err)
	}

	wKV := cpuOracleTensor(t, m, layerName(l, "engram_kv.weight"))

	for tt := range ids {
		rowVec := make([]float32, cols*dim)
		for c := 0; c < cols; c++ {
			copy(rowVec[c*dim:], v41OracleEngramDequant(t, gathered[tt*cols+c], dim))
		}
		projected := cpuOracleMatVec(wKV, rowVec, (hc+1)*H, cols*dim)
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41OracleBF16(projected[hc*H+i])
		}
		h := x[tt]
		acc := make([]float64, H)
		for s := 0; s < hc; s++ {
			key := make([]float32, H)
			var h2, k2, dot float64
			for i := 0; i < H; i++ {
				key[i] = v41OracleBF16(projected[s*H+i])
				hh := float64(h[i])
				kk := float64(key[i])
				h2 += hh * hh
				k2 += kk * kk
				dot += hh * float64(coef[s*H+i]) * kk
			}
			dot *= 1 / math.Sqrt(h2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(k2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(float64(H))
			gate := 1 / (1 + math.Exp(-math.Copysign(math.Sqrt(math.Max(math.Abs(dot), 1e-6)), dot)))
			for i := 0; i < H; i++ {
				acc[i] += float64(v41OracleBF16(float32(gate) * value[i]))
			}
		}
		for i := 0; i < H; i++ {
			h[i] += float32(acc[i])
		}
	}
}

// v41OracleFoldedFullEngramUpdates independently transcribes the full
// per-stream schedule WITHOUT calling v41EngramInject, consuming the folded
// coefficient cells directly. It mirrors v41OracleFullEngramUpdates with the
// dot product multiplied by coef[s,i]; h2/k2/dot accumulate in float64. It
// returns the expected post-call stream values as [position][stream][dim].
func v41OracleFoldedFullEngramUpdates(t *testing.T, m *Model, layout V41EngramLayout, l int, streams [][][]float32, ids []int, eps float32, coef []float32) [][][]float32 {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := cfg.DeepSeekV41.EngramHeadDim
	hc := 4
	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("oracle: Engram stage not wired")
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		t.Fatal(err)
	}
	allRows, err := hash.Hash(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	layers := len(layout.Rows)
	cacheIdx := stage.cacheIndex(l)
	layerRows := make([]uint32, 0, len(ids)*cols)
	for tt := range ids {
		base := tt*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		t.Fatal(err)
	}

	wKV := cpuOracleTensor(t, m, layerName(l, "engram_kv.weight"))

	want := make([][][]float32, len(ids))
	for tt := range ids {
		rowVec := make([]float32, cols*dim)
		for c := 0; c < cols; c++ {
			copy(rowVec[c*dim:], v41OracleEngramDequant(t, gathered[tt*cols+c], dim))
		}
		projected := cpuOracleMatVec(wKV, rowVec, (hc+1)*H, cols*dim)
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41OracleBF16(projected[hc*H+i])
		}
		want[tt] = make([][]float32, hc)
		for s := 0; s < hc; s++ {
			h := streams[tt][s]
			key := make([]float32, H)
			var h2, k2, dot float64
			for i := 0; i < H; i++ {
				key[i] = v41OracleBF16(projected[s*H+i])
				hh := float64(h[i])
				kk := float64(key[i])
				h2 += hh * hh
				k2 += kk * kk
				dot += hh * float64(coef[s*H+i]) * kk
			}
			dot *= 1 / math.Sqrt(h2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(k2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(float64(H))
			gate := 1 / (1 + math.Exp(-math.Copysign(math.Sqrt(math.Max(math.Abs(dot), 1e-6)), dot)))
			row := make([]float32, H)
			for i := 0; i < H; i++ {
				delta := float32(float32(gate) * value[i])
				row[i] = v41OracleBF16(float32(h[i] + delta))
			}
			want[tt][s] = row
		}
	}
	return want
}

// TestV41EngramFoldedGate is the #14121 witness: folded Engram gate
// coefficients are consumed directly (no q/k reconstruction), the unfolded
// path stays bit-identical, and every unimplemented combination fails closed.
//
// fak-test:runtime medium est=2s lane=default
func TestV41EngramFoldedGate(t *testing.T) {
	t.Parallel()

	const hc = 4
	ids := []int{1, 3, 5}

	t.Run("folded_only_reduced_matches_oracle", func(t *testing.T) {
		t.Parallel()
		m, layout := v41ReducedEngramModel(t)
		l, H := 1, m.Cfg.HiddenSize
		coef := v41FoldedTestCoefficients(hc, H)
		v41SetGateForms(t, m, l, true, coef)
		v41FoldedGateGeometry(t, m, l, hc, H)

		eps := float32(m.Cfg.RMSNormEps)
		x0 := make([][]float32, len(ids))
		for tt, id := range ids {
			x0[tt] = make([]float32, H)
			for i := range x0[tt] {
				x0[tt][i] = v41OracleBF16(float32(0.05*float64((id+i*3)%11) - 5))
			}
		}
		xProd := make([][]float32, len(ids))
		for tt := range ids {
			xProd[tt] = append([]float32(nil), x0[tt]...)
		}
		xWant := make([][]float32, len(ids))
		for tt := range ids {
			xWant[tt] = append([]float32(nil), x0[tt]...)
		}

		if err := m.v41EngramInject(l, xProd, nil, false, ids, eps); err != nil {
			t.Fatalf("v41EngramInject(folded, reduced) error = %v, want nil", err)
		}
		v41OracleFoldedEngramInject(t, m, layout, l, xWant, ids, eps, coef)
		for tt := range ids {
			for i := 0; i < H; i++ {
				if d := math.Abs(float64(xProd[tt][i] - xWant[tt][i])); d > cpuOracleTol {
					t.Fatalf("reduced x[%d][%d] = %v, want %v (|delta| = %.3e > tol %.0e)",
						tt, i, xProd[tt][i], xWant[tt][i], d, cpuOracleTol)
				}
			}
		}
		// Non-vacuity: the injection moved the hidden state.
		moved := false
		for tt := range ids {
			for i := 0; i < H; i++ {
				if xProd[tt][i] != x0[tt][i] {
					moved = true
				}
			}
		}
		if !moved {
			t.Fatal("folded reduced injection left the hidden state untouched; the witness is vacuous")
		}
	})

	t.Run("folded_only_full_matches_oracle", func(t *testing.T) {
		t.Parallel()
		m, layout := v41FullEngramModel(t)
		l, H := 0, m.Cfg.HiddenSize
		coef := v41FoldedTestCoefficients(hc, H)
		v41SetGateForms(t, m, l, true, coef)
		v41FoldedGateGeometry(t, m, l, hc, H)

		eps := float32(m.Cfg.RMSNormEps)
		x, streams := v41FullEngramAsymmetricStreams(ids, H)
		want := v41OracleFoldedFullEngramUpdates(t, m, layout, l, streams, ids, eps, coef)

		if err := m.v41EngramInject(l, x, streams, true, ids, eps); err != nil {
			t.Fatalf("v41EngramInject(folded, full) error = %v, want nil", err)
		}
		for tt := range ids {
			for s := 0; s < hc; s++ {
				for i := 0; i < H; i++ {
					if d := math.Abs(float64(streams[tt][s][i] - want[tt][s][i])); d > cpuOracleTol {
						t.Fatalf("full streams[%d][%d][%d] = %v, want %v (|delta| = %.3e > tol %.0e)",
							tt, s, i, streams[tt][s][i], want[tt][s][i], d, cpuOracleTol)
					}
				}
			}
			for i := 0; i < H; i++ {
				if x[tt][i] != streams[tt][0][i] {
					t.Fatalf("x[%d][%d] = %v, want stream 0 value %v", tt, i, x[tt][i], streams[tt][0][i])
				}
			}
		}
	})

	t.Run("bit_identity_unfolded_vs_folded", func(t *testing.T) {
		t.Parallel()
		// Unfolded q=g, k=ones: the runtime product is g[i]*1.0 == g[i] exactly,
		// so the unfolded path must publish BYTE-IDENTICAL results to the folded
		// path with coef=g — proving no BF16 boundary moved.
		buildX := func() [][]float32 {
			H := 64
			x := make([][]float32, len(ids))
			for tt, id := range ids {
				x[tt] = make([]float32, H)
				for i := range x[tt] {
					x[tt][i] = v41OracleBF16(float32(0.05*float64((id+i*3)%11) - 5))
				}
			}
			return x
		}
		runUnfolded := func() [][]float32 {
			m, _ := v41ReducedEngramModel(t)
			H := m.Cfg.HiddenSize
			g := v41FoldedTestCoefficients(hc, H)
			ones := make([]float32, hc*H)
			for i := range ones {
				ones[i] = 1
			}
			v41WriteTensorF32(t, m, layerName(1, "engram_q_norm.weight"), g)
			v41WriteTensorF32(t, m, layerName(1, "engram_k_norm.weight"), ones)
			x := buildX()
			if err := m.v41EngramInject(1, x, nil, false, ids, float32(m.Cfg.RMSNormEps)); err != nil {
				t.Fatalf("v41EngramInject(unfolded) error = %v, want nil", err)
			}
			return x
		}
		runFolded := func() [][]float32 {
			m, _ := v41ReducedEngramModel(t)
			H := m.Cfg.HiddenSize
			v41SetGateForms(t, m, 1, true, v41FoldedTestCoefficients(hc, H))
			x := buildX()
			if err := m.v41EngramInject(1, x, nil, false, ids, float32(m.Cfg.RMSNormEps)); err != nil {
				t.Fatalf("v41EngramInject(folded) error = %v, want nil", err)
			}
			return x
		}
		xUnfolded, xFolded := runUnfolded(), runFolded()
		if !reflect.DeepEqual(xUnfolded, xFolded) {
			t.Fatal("unfolded(q=g,k=ones) and folded(coef=g) publications differ; a BF16 boundary moved")
		}
	})

	t.Run("fail_closed_nonfinite_folded", func(t *testing.T) {
		t.Parallel()
		for _, fault := range []struct {
			name  string
			value float32
		}{
			{"nan", float32(math.NaN())},
			{"positive_inf", float32(math.Inf(1))},
			{"negative_inf", float32(math.Inf(-1))},
		} {
			fault := fault
			t.Run(fault.name, func(t *testing.T) {
				m, _ := v41ReducedEngramModel(t)
				l, H := 1, m.Cfg.HiddenSize
				coef := v41FoldedTestCoefficients(hc, H)
				coef[len(coef)-1] = fault.value // final stream: earlier streams must not move
				v41SetGateForms(t, m, l, true, coef)

				eps := float32(m.Cfg.RMSNormEps)
				x := make([][]float32, len(ids))
				for tt := range ids {
					x[tt] = make([]float32, H)
					for i := range x[tt] {
						x[tt][i] = 1
					}
				}
				beforeX := sysFlatten(x)
				stage := m.v41EngramStageFor()
				cache := stage.caches[stage.cacheIndex(l)]
				beforeCache, calls := cache.Stats(), 0
				project := func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
					calls++
					values := make([]float32, out)
					for i := range values {
						values[i] = 1
					}
					return values, v41ProjectionHandled, nil
				}

				err := m.v41EngramInjectWithProjection(l, x, nil, false, ids, eps, project)
				var attributed *V41ForwardError
				if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != l {
					t.Fatalf("nonfinite folded coefficient err=%v, want typed Engram refusal at layer %d", err, l)
				}
				if calls != 0 || cache.Stats() != beforeCache {
					t.Errorf("nonfinite folded coefficient reached retrieval/projection: calls=%d", calls)
				}
				if !reflect.DeepEqual(beforeX, sysFlatten(x)) {
					t.Error("nonfinite folded coefficient partially published hidden positions")
				}
			})
		}
	})

	t.Run("fail_closed_prepared_route_defense_in_depth", func(t *testing.T) {
		t.Parallel()
		// A geometry built OUTSIDE v41EngramInjectionGeometry (hand-assembled
		// with a non-finite folded cell) must still be refused by the
		// defense-in-depth recheck before retrieval or publication.
		m, _ := v41ReducedEngramModel(t)
		l, H := 1, m.Cfg.HiddenSize
		coef := v41FoldedTestCoefficients(hc, H)
		v41SetGateForms(t, m, l, true, coef)
		good := v41FoldedGateGeometry(t, m, l, hc, H)

		bad := good
		badCoef := append([]float32(nil), good.coef...)
		badCoef[len(badCoef)-1] = float32(math.NaN())
		bad.coef = badCoef

		eps := float32(m.Cfg.RMSNormEps)
		x := make([][]float32, len(ids))
		for tt := range ids {
			x[tt] = make([]float32, H)
		}
		beforeX := sysFlatten(x)
		stage := m.v41EngramStageFor()
		cache := stage.caches[stage.cacheIndex(l)]
		beforeCache, calls := cache.Stats(), 0
		project := func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			return make([]float32, out), v41ProjectionHandled, nil
		}
		bad.project = project

		_, err := m.v41EngramInjectPrepared(l, x, nil, false, len(ids), make([]uint32, len(ids)*good.cols), eps, bad)
		var attributed *V41ForwardError
		if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != l {
			t.Fatalf("hand-built folded geometry err=%v, want typed Engram refusal at layer %d", err, l)
		}
		if calls != 0 || cache.Stats() != beforeCache {
			t.Errorf("hand-built folded geometry reached retrieval/projection: calls=%d", calls)
		}
		if !reflect.DeepEqual(beforeX, sysFlatten(x)) {
			t.Error("hand-built folded geometry partially published hidden positions")
		}
	})

	t.Run("fail_closed_misshaped_folded", func(t *testing.T) {
		t.Parallel()
		m, _ := v41ReducedEngramModel(t)
		l, H := 1, m.Cfg.HiddenSize
		v41SetGateForms(t, m, l, true, make([]float32, hc*H-1)) // one cell short

		_, err := m.v41EngramInjectionGeometry(l, 1)
		var attributed *V41ForwardError
		if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != l {
			t.Fatalf("mis-shaped folded tensor err=%v, want typed Engram refusal at layer %d", err, l)
		}
		if admErr := m.v41EngramForwardAdmitted(); admErr == nil {
			t.Fatal("admission accepted a mis-shaped folded tensor, want refusal")
		}
	})

	t.Run("fail_closed_copresence", func(t *testing.T) {
		t.Parallel()
		m, _ := v41ReducedEngramModel(t)
		l, H := 1, m.Cfg.HiddenSize
		v41SetGateForms(t, m, l, false, v41FoldedTestCoefficients(hc, H)) // keep q/k AND add folded

		_, err := m.v41EngramInjectionGeometry(l, 1)
		var attributed *V41ForwardError
		if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != l {
			t.Fatalf("co-present forms err=%v, want typed Engram ambiguity refusal at layer %d", err, l)
		}
		if admErr := m.v41EngramForwardAdmitted(); admErr == nil {
			t.Fatal("admission accepted co-present coefficient forms, want refusal")
		}
	})

	t.Run("fail_closed_neither_form", func(t *testing.T) {
		t.Parallel()
		m, _ := v41ReducedEngramModel(t)
		l := 1
		v41SetGateForms(t, m, l, true, nil) // strip q/k, install nothing

		_, err := m.v41EngramInjectionGeometry(l, 1)
		var attributed *V41ForwardError
		if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != l {
			t.Fatalf("absent forms err=%v, want the historical typed Engram refusal at layer %d", err, l)
		}
		if admErr := m.v41EngramForwardAdmitted(); admErr == nil {
			t.Fatal("admission accepted a layer with no gate-coefficient form, want refusal")
		}
	})

	t.Run("admission_folded_only_admitted", func(t *testing.T) {
		t.Parallel()
		m, _ := v41ReducedEngramModel(t)
		l, H := 1, m.Cfg.HiddenSize
		v41SetGateForms(t, m, l, true, v41FoldedTestCoefficients(hc, H))
		if err := m.v41EngramForwardAdmitted(); err != nil {
			t.Fatalf("v41EngramForwardAdmitted(folded-only) error = %v, want nil", err)
		}
	})
}

// Admission must inspect every declared Engram layer, whether the first layer
// uses unfolded norms or the upstream folded coefficient. A valid first layer
// must not hide a missing or malformed later layer before retrieval begins.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41EngramAdmissionChecksEveryLayer(t *testing.T) {
	t.Parallel()
	for _, first := range []string{"unfolded", "folded"} {
		for _, fault := range []string{"missing_q", "short_k"} {
			t.Run(first+"/"+fault, func(t *testing.T) {
				m, _ := v41IncrementalEngramFixture(t, false)
				if first == "folded" {
					v41SetGateForms(t, m, 0, true, v41FoldedTestCoefficients(4, m.Cfg.HiddenSize))
				}
				if err := m.v41EngramForwardAdmitted(); err != nil {
					t.Fatalf("valid two-layer fixture refused: %v", err)
				}
				const later = 2
				if fault == "missing_q" {
					delete(m.manifest, layerName(later, "engram_q_norm.weight"))
				} else {
					name := layerName(later, "engram_k_norm.weight")
					meta := m.manifest[name]
					meta.Shape = []int{4*m.Cfg.HiddenSize - 1}
					m.manifest[name] = meta
				}
				before := v41IncrementalEngramRequests(m)
				err := m.v41EngramForwardAdmitted()
				var attributed *V41ForwardError
				if !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != later {
					t.Fatalf("later Engram layer was not refused at admission: %v", err)
				}
				if after := v41IncrementalEngramRequests(m); after != before {
					t.Fatalf("admission read Engram rows: requests %d -> %d", before, after)
				}
			})
		}
	}
}

// A folded coefficient plus either legacy norm is ambiguous, not a complete
// folded-only publication. Both admission and geometry must reject it before I/O.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=1s lane=default
func TestV41EngramFoldedGateRejectsPartialNorms(t *testing.T) {
	t.Parallel()
	for _, remaining := range []string{"engram_q_norm.weight", "engram_k_norm.weight"} {
		t.Run(remaining, func(t *testing.T) {
			m, _ := v41FullEngramModel(t)
			v41SetGateForms(t, m, 0, false, v41FoldedTestCoefficients(4, m.Cfg.HiddenSize))
			for _, leaf := range []string{"engram_q_norm.weight", "engram_k_norm.weight"} {
				if leaf != remaining {
					delete(m.manifest, layerName(0, leaf))
				}
			}
			before := v41IncrementalEngramRequests(m)
			_, geometryErr := m.v41EngramInjectionGeometry(0, 1)
			for _, err := range []error{m.v41EngramForwardAdmitted(), geometryErr} {
				var attributed *V41ForwardError
				if !errors.Is(err, ErrV41NativeUnsupported) || !errors.As(err, &attributed) || attributed.Stage != v41StageEngram || attributed.Layer != 0 {
					t.Fatalf("folded coefficient plus %s was not refused: %v", remaining, err)
				}
			}
			if after := v41IncrementalEngramRequests(m); after != before {
				t.Fatalf("ambiguous mixture read Engram rows: requests %d -> %d", before, after)
			}
		})
	}
}
