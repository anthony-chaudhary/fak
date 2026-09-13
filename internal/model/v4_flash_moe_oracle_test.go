package model

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This file is an independent scalar oracle for the published DeepSeek V4 Flash
// 0731 MoE execution. Every expected value below is transcribed from the pinned
// reference equations, NOT read from or copied out of the production helpers.
//
// Pinned reference: DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062
// inference/model.py Gate/Expert/MoE with config n_routed_experts=256,
// n_shared_experts=1, num_experts_per_tok=6, routed_scaling_factor=1.5,
// swiglu_limit=10, scoring_func=sqrtsoftplus.
//
// It exercises the existing seams (v4HashRoute, v4ScoredRoute,
// decodeV4ExpertQuant, the live v4ExpertForward wiring, the v4 expert runtime).
// One isolated function references the shared-expert execution symbol; see
// TestV4FlashSharedExpertExecutionSymbol.

const (
	moeFlashExperts    = 256
	moeFlashTopK       = 6
	moeFlashRouteScale = float32(1.5)
)

// moeOracleScore is the pinned sqrt(softplus(z)) scoring function written from
// the formula, including the overflow-safe softplus branch.
func moeOracleScore(z float32) float64 {
	zf := float64(z)
	return math.Sqrt(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf))))
}

// moeOracleTopK is an insertion top-k over the full width ranking by descending
// score+bias with the lower expert index winning ties. Written from scratch so
// the expected selection is independent of the production sort.
func moeOracleTopK(logits, bias []float32, k int) []int {
	selected := make([]int, 0, k)
	for i := range logits {
		var bi float32
		if len(bias) != 0 {
			bi = bias[i]
		}
		si := moeOracleScore(logits[i]) + float64(bi)
		pos := len(selected)
		for j, e := range selected {
			var be float32
			if len(bias) != 0 {
				be = bias[e]
			}
			se := moeOracleScore(logits[e]) + float64(be)
			if si > se || (si == se && i < e) {
				pos = j
				break
			}
		}
		if pos < k {
			selected = append(selected, 0)
			copy(selected[pos+1:], selected[pos:])
			selected[pos] = i
			if len(selected) > k {
				selected = selected[:k]
			}
		}
	}
	return selected
}

// moeE2M1Oracle decodes one OCP MX E2M1 nibble from its bit fields:
// 1 sign bit, 2 exponent bits with bias 1, 1 mantissa bit. It is a from-first-
// principles decode, and it returns a signed zero for the sign-set zero code.
func moeE2M1Oracle(code int) float32 {
	sign := code >> 3 & 1
	exponent := code >> 1 & 3
	mantissa := code & 1
	var magnitude float64
	if exponent == 0 {
		magnitude = float64(mantissa) / 2
	} else {
		magnitude = math.Ldexp(1+float64(mantissa)/2, exponent-1)
	}
	if sign == 1 {
		if magnitude == 0 {
			return float32(math.Copysign(0, -1))
		}
		return float32(-magnitude)
	}
	return float32(magnitude)
}

// moeE8M0ScaleOracle decodes one F8_E8M0 byte to its power of two. The bias is
// 127, so byte 127 is 2^0 and byte 128 is 2^1.
func moeE8M0ScaleOracle(b byte) float64 {
	return math.Ldexp(1, int(b)-127)
}

func moeBitsEqual32(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b)
}

// TestDeepSeekV4FlashMOEHashRoute256IndependentOracle covers deliverable 1: the
// first-three-layer table route for 256 logits and a chosen token's six IDs.
func TestDeepSeekV4FlashMOEHashRoute256IndependentOracle(t *testing.T) {
	logits := make([]float32, moeFlashExperts)
	for i := range logits {
		logits[i] = float32(i%17)/4 - 2.5
	}
	ids := []int{255, 1, 3, 5, 7, 9}

	got, err := v4HashRoute(logits, ids, moeFlashRouteScale)
	if err != nil {
		t.Fatalf("256-expert Flash table route rejected: %v", err)
	}
	if len(got) != moeFlashTopK {
		t.Fatalf("picks=%d want %d", len(got), moeFlashTopK)
	}
	var denom float64
	for _, id := range ids {
		denom += moeOracleScore(logits[id])
	}
	if denom <= 0 {
		t.Fatalf("independent denominator=%v, want positive", denom)
	}
	for i, id := range ids {
		if got[i].expert != id {
			t.Fatalf("pick[%d].expert=%d want table id %d", i, got[i].expert, id)
		}
		want := float32(moeOracleScore(logits[id]) / denom * float64(moeFlashRouteScale))
		if diff := math.Abs(float64(got[i].weight - want)); diff > 2e-6 {
			t.Fatalf("id %d weight=%g want=%g diff=%g", id, got[i].weight, want, diff)
		}
	}
}

// TestDeepSeekV4FlashMOEScoredRouteBiasSelectionUnbiasedWeights covers
// deliverable 2: selection uses score+bias while the returned weights stay the
// UNBIASED scores, with and without a correction bias.
func TestDeepSeekV4FlashMOEScoredRouteBiasSelectionUnbiasedWeights(t *testing.T) {
	makeLogits := func() []float32 {
		logits := make([]float32, moeFlashExperts)
		for i := range logits {
			logits[i] = -12
		}
		// A natural six-way block tied on unbiased score.
		for i := 100; i <= 105; i++ {
			logits[i] = 4
		}
		// Bias will lift expert 200 above the block; without bias it is not
		// selected, proving selection reads score+bias.
		logits[200] = 0.5
		return logits
	}

	t.Run("biased", func(t *testing.T) {
		logits := makeLogits()
		bias := make([]float32, moeFlashExperts)
		bias[200] = 6
		got, err := v4ScoredRoute(logits, bias, moeFlashTopK, moeFlashRouteScale)
		if err != nil {
			t.Fatalf("biased scored route: %v", err)
		}
		order := moeOracleTopK(logits, bias, moeFlashTopK)
		if order[0] != 200 {
			t.Fatalf("independent biased order=%v, want expert 200 first", order)
		}
		var denom float64
		for _, e := range order {
			denom += moeOracleScore(logits[e])
		}
		for i, p := range got {
			if p.expert != order[i] {
				t.Fatalf("pick[%d].expert=%d want oracle %d (order=%v)", i, p.expert, order[i], order)
			}
			want := float32(moeOracleScore(logits[p.expert]) / denom * float64(moeFlashRouteScale))
			if diff := math.Abs(float64(p.weight - want)); diff > 2e-6 {
				t.Fatalf("biased pick expert=%d weight=%g want unbiased=%g diff=%g", p.expert, p.weight, want, diff)
			}
		}
	})

	t.Run("unbiased", func(t *testing.T) {
		logits := makeLogits()
		got, err := v4ScoredRoute(logits, nil, moeFlashTopK, moeFlashRouteScale)
		if err != nil {
			t.Fatalf("unbiased scored route: %v", err)
		}
		order := moeOracleTopK(logits, nil, moeFlashTopK)
		// Expert 200 must NOT appear without the bias: it is below the block.
		for _, e := range order {
			if e == 200 {
				t.Fatalf("expert 200 selected without bias: order=%v", order)
			}
		}
		var denom float64
		for _, e := range order {
			denom += moeOracleScore(logits[e])
		}
		for i, p := range got {
			if p.expert != order[i] {
				t.Fatalf("pick[%d].expert=%d want oracle %d (order=%v)", i, p.expert, order[i], order)
			}
			want := float32(moeOracleScore(logits[p.expert]) / denom * float64(moeFlashRouteScale))
			if diff := math.Abs(float64(p.weight - want)); diff > 2e-6 {
				t.Fatalf("unbiased pick expert=%d weight=%g want=%g diff=%g", p.expert, p.weight, want, diff)
			}
		}
	})
}

// TestDeepSeekV4FlashMOESelectedWeightNormalizationSumRouteScale covers
// deliverable 3: exactly six weights summing to route scale 1.5.
func TestDeepSeekV4FlashMOESelectedWeightNormalizationSumRouteScale(t *testing.T) {
	logits := make([]float32, moeFlashExperts)
	for i := range logits {
		logits[i] = -30
	}
	controlled := map[int]float32{5: 2, 40: 0.5, 77: -1, 150: 3, 200: -2, 255: 1}
	for i, z := range controlled {
		logits[i] = z
	}
	got, err := v4ScoredRoute(logits, nil, moeFlashTopK, moeFlashRouteScale)
	if err != nil {
		t.Fatalf("scored route: %v", err)
	}
	if len(got) != moeFlashTopK {
		t.Fatalf("picks=%d want %d", len(got), moeFlashTopK)
	}
	var sum float64
	for _, p := range got {
		sum += float64(p.weight)
	}
	if math.Abs(sum-float64(moeFlashRouteScale)) > 2e-6 {
		t.Fatalf("sum(weights)=%g want route_scale=%g", sum, moeFlashRouteScale)
	}
	order := moeOracleTopK(logits, nil, moeFlashTopK)
	var denom float64
	for _, e := range order {
		denom += moeOracleScore(logits[e])
	}
	for i, p := range got {
		if p.expert != order[i] {
			t.Fatalf("pick[%d].expert=%d want %d (order=%v)", i, p.expert, order[i], order)
		}
		want := float32(moeOracleScore(logits[p.expert]) / denom * float64(moeFlashRouteScale))
		if diff := math.Abs(float64(p.weight - want)); diff > 2e-6 {
			t.Fatalf("expert %d weight=%g want=%g diff=%g", p.expert, p.weight, want, diff)
		}
	}
}

// TestV4FlashSharedExpertAddedOnEveryToken covers deliverable 4 through the LIVE
// wiring: it drives v4ExpertForward end to end with a fake routed accumulator so
// the combined output must equal routed + the independently recomputed shared
// expert, and the all-zero routed case must survive as the shared term alone.
func TestV4FlashSharedExpertAddedOnEveryToken(t *testing.T) {
	cfg, m := v4SharedExpertTestFixture(t, true)
	be := compute.Default()
	s := m.NewSession()
	s.Backend = be
	routed := sharedRoutedProbe(cfg, 0.25)
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: routed}
	x := sharedTestProbe(cfg)
	normalized := s.uploadHostF32([]int{cfg.HiddenSize}, x, compute.MemoryActivation, "v4-moe-oracle-norm")
	defer be.Free(normalized)
	gotTensor, err := s.v4ExpertForward(sharedTestLayer, 1, normalized)
	if err != nil {
		t.Fatalf("v4ExpertForward: %v", err)
	}
	defer be.Free(gotTensor)
	got := be.Read(gotTensor)
	w1 := m.tensor(v4SharedExpertName(sharedTestLayer, "w1"))
	w2 := m.tensor(v4SharedExpertName(sharedTestLayer, "w2"))
	w3 := m.tensor(v4SharedExpertName(sharedTestLayer, "w3"))
	shared := sharedExpertOracle(x, w1, w2, w3, cfg.MoEIntermediateSize, cfg.HiddenSize, float32(cfg.SwigluLimit))
	want := make([]float32, cfg.HiddenSize)
	for i := range want {
		want[i] = routed[i] + shared[i]
	}
	if !closeSlice(got, want, 1e-4) {
		t.Fatalf("live shared add mismatch: got[0:3]=%v want[0:3]=%v", got[:3], want[:3])
	}
	if closeSlice(got, routed, 1e-3) {
		t.Fatalf("routed-only unchanged; shared not added: got[0:3]=%v", got[:3])
	}
	s.v4Expert = &sharedExpertFakeRouted{hidden: cfg.HiddenSize, out: make([]float32, cfg.HiddenSize)}
	zerosTensor, err := s.v4ExpertForward(sharedTestLayer, 2, normalized)
	if err != nil {
		t.Fatalf("v4ExpertForward zero-routed: %v", err)
	}
	defer be.Free(zerosTensor)
	if gotZeros := be.Read(zerosTensor); !closeSlice(gotZeros, shared, 1e-4) {
		t.Fatalf("zero-routed wiring[0:3]=%v want shared-only[0:3]=%v", gotZeros[:3], shared[:3])
	}
	s.Close()
}

// TestV4FlashSharedExpertExecutionSymbol is the single isolated reference to
// the shared-expert execution seam. It proves the symbol is wired and fails
// closed on an unready session; the resident-weight execution is covered by
// v4_expert_shared_test.go. This one function is the reconciliation point if
// the seam is named or shaped differently.
func TestV4FlashSharedExpertExecutionSymbol(t *testing.T) {
	var nilSession *Session
	_, err := nilSession.v4SharedExpertForward(0, nil, 10, 4, 8)
	if !errors.Is(err, ErrV4SharedExpert) {
		t.Fatalf("nil-session shared expert error=%v want ErrV4SharedExpert", err)
	}
}

// TestV4FlashMOEQuantNegativeZeroNibbleDecode covers deliverable 5: E2M1 low
// nibble=val0 / high nibble=val1 ordering, the sign bit, signed zero, and a
// zero nibble decoding to zero. It reads the Flash w1 packed shape through
// decodeV4ExpertQuant and compares exact float32 bits to the scalar E2M1 oracle.
func TestV4FlashMOEQuantNegativeZeroNibbleDecode(t *testing.T) {
	const rows, packedCols, scaleCols = 2048, 2048, 128
	weights := make([]byte, rows*packedCols)
	scales := make([]byte, rows*scaleCols)
	for r := 0; r < rows; r++ {
		scales[r*scaleCols] = 127 // unit scale for the first 32 unpacked K
	}
	// Row 0, packed byte 0: low nibble 0x0 (+0), high nibble 0x1 (0.5).
	weights[0] = 0x10
	// Row 0, packed byte 1: low nibble 0x8 (-0), high nibble 0x0 (+0).
	weights[1] = 0x08
	// Row 0, packed byte 2: both nibbles 0x9 (-0.5), proving the sign bit.
	weights[2] = 0x99
	// Row 0, packed byte 3: both nibbles 0xf (-6).
	weights[3] = 0xff
	// Row 0, packed byte 4: low 0x8 (-0), high 0x7 (+6).
	weights[4] = 0x78

	name := "layers.0.ffn.experts.0.w1.weight"
	scaleName := "layers.0.ffn.experts.0.w1.scale"
	decoded, shape, err := decodeV4ExpertQuant(
		name, scaleName,
		stEntry{Dtype: "I8", Shape: []int{rows, packedCols}},
		stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}},
		weights, scales,
	)
	if err != nil {
		t.Fatalf("Flash w1 decode: %v", err)
	}
	if !sameShape(shape, []int{rows, packedCols * 2}) {
		t.Fatalf("decoded shape=%v want [%d %d]", shape, rows, packedCols*2)
	}
	value := func(unpacked int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(decoded[unpacked*4:]))
	}

	cases := []struct {
		k    int
		code int
	}{
		{0, 0x0}, // low nibble of byte 0 -> val0 = +0
		{1, 0x1}, // high nibble of byte 0 -> val1 = 0.5
		{2, 0x8}, // low nibble of byte 1 -> -0
		{3, 0x0}, // high nibble of byte 1 -> +0
		{4, 0x9}, // -0.5
		{5, 0x9}, // -0.5
		{6, 0xf}, // -6
		{7, 0xf}, // -6
		{8, 0x8}, // -0
		{9, 0x7}, // +6
	}
	for _, tc := range cases {
		want := float32(math.Ldexp(float64(moeE2M1Oracle(tc.code)), 0))
		if !moeBitsEqual32(value(tc.k), want) {
			t.Fatalf("K=%d code=%#x got=%v bits=%08x want=%v bits=%08x",
				tc.k, tc.code, value(tc.k), math.Float32bits(value(tc.k)), want, math.Float32bits(want))
		}
	}
	for _, k := range []int{10, 31, 100, 4095} {
		if !moeBitsEqual32(value(k), 0) {
			t.Fatalf("zero nibble at K=%d decoded to %v, want +0", k, value(k))
		}
	}
}

// TestV4FlashMOEQuantE8M0ScaleBoundaryPer32 covers deliverable 6: one E8M0
// scale applies to each 32 unpacked K values, with an exact boundary witnessed
// against the independent ldexp oracle. Each packed byte covers unpacked K
// (packedCol*2, packedCol*2+1) and reads scale[packedCol/16], so packed byte 15
// is K30/K31 under scale 0 and packed byte 16 is K32/K33 under scale 1.
func TestV4FlashMOEQuantE8M0ScaleBoundaryPer32(t *testing.T) {
	const rows, packedCols, scaleCols = 2048, 2048, 128
	weights := make([]byte, rows*packedCols)
	scales := make([]byte, rows*scaleCols)
	scales[0] = 127 // 2^0 covers packed cols 0..15 -> unpacked K 0..31
	scales[1] = 128 // 2^1 covers packed cols 16..31 -> unpacked K 32..63
	// Code 2 (1.0) at the last K of group 0 and the first pair of group 1.
	weights[15] = 0x22 // K30 and K31, scale 2^0
	weights[16] = 0x22 // K32 and K33, scale 2^1

	name := "layers.0.ffn.experts.0.w1.weight"
	scaleName := "layers.0.ffn.experts.0.w1.scale"
	decoded, _, err := decodeV4ExpertQuant(
		name, scaleName,
		stEntry{Dtype: "I8", Shape: []int{rows, packedCols}},
		stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}},
		weights, scales,
	)
	if err != nil {
		t.Fatalf("Flash w1 decode: %v", err)
	}
	value := func(unpacked int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(decoded[unpacked*4:]))
	}
	checks := []struct {
		k    int
		code int
		byte byte
	}{
		{k: 30, code: 0x2, byte: 127}, // last pair of group 0
		{k: 31, code: 0x2, byte: 127},
		{k: 32, code: 0x2, byte: 128}, // first pair of group 1
		{k: 33, code: 0x2, byte: 128},
	}
	for _, tc := range checks {
		want := float32(math.Ldexp(float64(moeE2M1Oracle(tc.code)), int(tc.byte)-127))
		if !moeBitsEqual32(value(tc.k), want) {
			t.Fatalf("K=%d scale byte=%d got=%v bits=%08x want=%v bits=%08x",
				tc.k, tc.byte, value(tc.k), math.Float32bits(value(tc.k)), want, math.Float32bits(want))
		}
	}
	// The group boundary flips the scale: K31 * 2^0 = 1, K32 * 2^1 = 2.
	if value(31) == value(32) {
		t.Fatalf("scale did not change across the 32-value boundary: K31=%v K32=%v", value(31), value(32))
	}
	if math.Abs(float64(moeE8M0ScaleOracle(128)/moeE8M0ScaleOracle(127)-2)) > 1e-12 {
		t.Fatal("independent E8M0 oracle disagree on the 2^0/2^1 boundary")
	}
}

// TestDeepSeekV4FlashMOEInvalidIDsFailClosed covers deliverable 7: hash ID
// >= 256, duplicate IDs, wrong logits widths, and non-finite logits/bias all
// fail closed. The 256-wide contract for scored layers lives in the runtime
// (forwardScored/forwardHash), so it is witnessed through newV4ExpertRuntime.
func TestDeepSeekV4FlashMOEInvalidIDsFailClosed(t *testing.T) {
	valid := make([]float32, moeFlashExperts)
	for i := range valid {
		valid[i] = float32(i%7) / 2
	}

	wantRouteError := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("expected a fail-closed error, got nil")
		}
		var typed *v4RouteError
		if !errors.As(err, &typed) {
			t.Fatalf("error %T %v is not *v4RouteError", err, err)
		}
	}

	t.Run("hash id 256", func(t *testing.T) {
		_, err := v4HashRoute(valid, []int{0, 1, 2, 3, 4, 256}, moeFlashRouteScale)
		wantRouteError(t, err)
	})
	t.Run("hash duplicate ids", func(t *testing.T) {
		_, err := v4HashRoute(valid, []int{0, 1, 2, 3, 4, 4}, moeFlashRouteScale)
		wantRouteError(t, err)
	})
	t.Run("hash id negative", func(t *testing.T) {
		_, err := v4HashRoute(valid, []int{-1, 1, 2, 3, 4, 5}, moeFlashRouteScale)
		wantRouteError(t, err)
	})
	for _, width := range []int{255, 257} {
		t.Run("hash logits width "+itoa(width), func(t *testing.T) {
			_, err := v4HashRoute(make([]float32, width), []int{0, 1, 2, 3, 4, 5}, moeFlashRouteScale)
			wantRouteError(t, err)
		})
	}
	t.Run("hash non-finite logit", func(t *testing.T) {
		logits := append([]float32(nil), valid...)
		logits[9] = float32(math.NaN())
		_, err := v4HashRoute(logits, []int{0, 1, 2, 3, 4, 5}, moeFlashRouteScale)
		wantRouteError(t, err)
	})
	t.Run("scored non-finite logit", func(t *testing.T) {
		logits := append([]float32(nil), valid...)
		logits[3] = float32(math.Inf(1))
		_, err := v4ScoredRoute(logits, nil, moeFlashTopK, moeFlashRouteScale)
		wantRouteError(t, err)
	})
	t.Run("scored non-finite bias", func(t *testing.T) {
		bias := make([]float32, moeFlashExperts)
		bias[5] = float32(math.NaN())
		_, err := v4ScoredRoute(valid, bias, moeFlashTopK, moeFlashRouteScale)
		wantRouteError(t, err)
	})

	// Runtime width contract: the Flash runtime selects exactly its configured
	// 256 experts, so a 255- or 257-wide scored/hash vector fails closed.
	restoreSpecs := useTinyV4RuntimeQuantSpecs()
	defer restoreSpecs()
	dir, _ := writeV4RuntimeFixture(t)
	_, flash := readDeepSeekV4FlashConfig(t)
	be := compute.Default()
	runtime, err := newV4ExpertRuntime(dir, flash, be, 16384, 2)
	if err != nil {
		t.Fatalf("construct Flash runtime: %v", err)
	}
	defer runtime.Close()
	x := be.Upload(compute.NewF32(be, []int{32}, make([]float32, 32)), compute.F32)
	defer be.Free(x)
	for _, width := range []int{255, 257} {
		bad := make([]float32, width)
		if _, err := runtime.forwardHash(0, 0, bad, x); !errors.Is(err, ErrV4ExpertRuntime) {
			t.Fatalf("forwardHash width %d error=%v, want ErrV4ExpertRuntime", width, err)
		}
		if _, err := runtime.forwardScored(3, bad, nil, x); !errors.Is(err, ErrV4ExpertRuntime) {
			t.Fatalf("forwardScored width %d error=%v, want ErrV4ExpertRuntime", width, err)
		}
	}
}
