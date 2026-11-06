package compute

import (
	"fmt"
	"math"
	"testing"
)

// CPU-oracle witness for the fused sqrt(softplus) V4.1 router top-k contract
// (device pass: k_v41_router_argmax_topk in cuda_v41_router.cu, the vLLM
// topkGatingSoftplusSqrt / dsv4HashTopkSoftplusSqrt pattern pinned there;
// compiled-kernel twin: TestCUDAV41RouterSoftplusSqrtDeviceParity in
// cuda_v41_router_test.go, which runs the same fixtures through the kernel on
// the HW fleet). The device kernel builds only under the CUDA tag, so this
// untagged test pins the numeric contract on every host: a float32
// transcription of the device scoring, selection and in-order normalization
// is checked against the host reference (v41RouterReferenceForTest) across z
// in [-40,40] including the exp-underflow region, and against hand-computed
// closed-contract values for the score form, exact-tie ordering and
// hash-table weight normalization.

// v41rtScoreHost transcribes the device v41rt_score (cuda_v41_router.cu):
// sqrt(softplus(z)) in F32 in the stable log1p(exp(-|z|)) form, with the
// reduced geometry's exp(z/2) fast path below -20. Each operation rounds to
// F32 as the __f*_rn intrinsics do; transcendentals take one extra F64 round
// trip, bounded by one ulp of any libm expf/logf.
func v41rtScoreHost(z float32, full bool) float32 {
	if !full && z < -20 {
		return float32(math.Exp(0.5 * float64(z)))
	}
	e := float32(math.Exp(-math.Abs(float64(z))))
	u := 1 + e
	log1p := e
	if u != 1 {
		log1p = float32(math.Log(float64(u))) * (e / (u - 1))
	}
	s := z
	if s < 0 {
		s = 0
	}
	return float32(math.Sqrt(float64(s + log1p)))
}

// v41RouterSoftplusSqrtRouteHost transcribes one row of the device pass:
// per-expert score in the (logit, then score) fault order, corrected
// (score+bias) insertion-ordered top-k keeping the lower expert first on
// corrected ties, hash-table picks verbatim, then the in-order F32 sum
// (+1e-20 when full) and the two-rounding weight (score/denominator)*scale.
// Callers must pass topK <= len(z) with all entries finite, as the validated
// backend geometries guarantee.
func v41RouterSoftplusSqrtRouteHost(z, bias []float32, hash []int32, topK int, scale float32, full, hasBias bool) (ids []int32, weights []float32, stage V41RouterTopKStage) {
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	type held struct {
		e    int
		c, s float32
	}
	var top []held
	for e := 0; e < len(z); e++ {
		v := z[e]
		if !finite(v) {
			return nil, nil, V41RouterTopKStageLogit
		}
		s := v41rtScoreHost(v, full)
		if !finite(s) {
			return nil, nil, V41RouterTopKStageScore
		}
		if hash != nil {
			continue
		}
		c := s
		if hasBias {
			b := bias[e]
			if !finite(b) {
				return nil, nil, V41RouterTopKStageBias
			}
			c = s + b
			if !finite(c) {
				return nil, nil, V41RouterTopKStageSelection
			}
		}
		insert := len(top)
		for i := range top {
			// Strict > keeps earlier (lower) experts first on corrected ties.
			if c > top[i].c {
				insert = i
				break
			}
		}
		if insert < topK {
			top = append(top, held{})
			copy(top[insert+1:], top[insert:len(top)-1])
			top[insert] = held{e: e, c: c, s: s}
			if len(top) > topK {
				top = top[:topK]
			}
		}
	}
	pick := make([]int, topK)
	score := make([]float32, topK)
	if hash != nil {
		for slot := 0; slot < topK; slot++ {
			pick[slot] = int(hash[slot])
			score[slot] = v41rtScoreHost(z[pick[slot]], full)
		}
	} else {
		for slot := 0; slot < topK; slot++ {
			pick[slot] = top[slot].e
			score[slot] = top[slot].s
		}
	}
	sum := float32(0)
	for slot := 0; slot < topK; slot++ {
		sum += score[slot]
	}
	if !finite(sum) || sum < 0 || (!full && sum == 0) {
		return nil, nil, V41RouterTopKStageNormalization
	}
	denominator := sum
	if full {
		denominator = sum + 1e-20
	}
	ids = make([]int32, topK)
	weights = make([]float32, topK)
	for slot := 0; slot < topK; slot++ {
		w := (score[slot] / denominator) * scale
		if !finite(w) {
			return nil, nil, V41RouterTopKStageWeight
		}
		ids[slot] = int32(pick[slot])
		weights[slot] = w
	}
	return ids, weights, V41RouterTopKStageSuccess
}

// requireExactPicks pins the picked expert ids slot for slot.
func requireExactPicks(t *testing.T, label string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: pick extent %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: slot %d expert %d, want %d (all %v vs %v)", label, i, got[i], want[i], got, want)
		}
	}
}

// requireRelativeWeights pins weights by relative error only: a score that a
// degenerate softplus form would publish as zero must not round away to zero,
// which requireV41RouterParity's absolute floor would swallow.
func requireRelativeWeights(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: weight extent %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		w := float64(want[i])
		if w == 0 {
			if got[i] != 0 {
				t.Fatalf("%s: slot %d weight %g, want exactly 0", label, i, got[i])
			}
			continue
		}
		d := math.Abs(float64(got[i]) - w)
		if d > 4e-6*math.Abs(w) {
			t.Fatalf("%s: slot %d weight %g, want %g (rel %g)", label, i, got[i], want[i], d/math.Abs(w))
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RouterSoftplusSqrtMatchesHostReference(t *testing.T) {
	// Score form, hand-computed: the stable log1p(exp(-|z|)) form must keep
	// underflow-region scores nonzero where the naive log(1+exp(z)) form
	// rounds 1+exp(z) to 1 and publishes zero.
	for _, tc := range []struct {
		z    float32
		full bool
		want float32
	}{
		{-40, false, 2.0611537e-09}, // exp(-20): the reduced fast path
		{-40, true, 2.0611537e-09},  // F32-clamped softplus then sqrt
		{-20.5, false, 3.53575e-05}, // just inside the fast path
		{-20, false, 4.539993e-05},  // the boundary row takes the general path
		{0, false, 0.83255464},      // sqrt(log 2)
		{3, true, 1.7460204},        // sqrt(3 + log1p(exp(-3)))
		{40, false, 6.3245554},      // sqrt(40): exp(-40) vanishes in F32
	} {
		got := v41rtScoreHost(tc.z, tc.full)
		if d := math.Abs(float64(got) - float64(tc.want)); d > 1e-7*math.Abs(float64(tc.want)) {
			t.Fatalf("score form z=%v full=%v = %g, want ~%g", tc.z, tc.full, got, tc.want)
		}
	}

	// (a) One row per z in [-40,40] at half-unit steps: expert 0 carries z,
	// expert 1 a fixed 3.0 anchor, so each weight pair pins the device score
	// against the host reference across the whole range. Half-steps from -40
	// are exact in F32, so the reduced geometry's -20 fast-path boundary
	// lands on a row, and z <= -20 exercises the exp-underflow region.
	const sweepRows = 161
	sweep := make([]float32, sweepRows*2)
	for i := 0; i < sweepRows; i++ {
		sweep[i*2] = float32(-40 + 0.5*float64(i))
		sweep[i*2+1] = 3
	}
	for _, full := range []bool{false, true} {
		label := fmt.Sprintf("sweep full=%v", full)
		wantIDs, wantW, stage := v41RouterReferenceForTest(sweep, nil, nil, sweepRows, 2, 2, 1.5, full)
		if stage != 0 {
			t.Fatalf("%s: reference faulted at %v", label, stage)
		}
		for row := 0; row < sweepRows; row++ {
			z := sweep[row*2 : row*2+2]
			gotIDs, gotW, st := v41RouterSoftplusSqrtRouteHost(z, nil, nil, 2, 1.5, full, false)
			if st != 0 {
				t.Fatalf("%s row %d: transcription faulted at %v", label, row, st)
			}
			// Closed contract: the higher-corrected expert leads; the tie at
			// z = 3 keeps the lower expert first.
			wantLead := int32(1)
			if z[0] >= 3 {
				wantLead = 0
			}
			if gotIDs[0] != wantLead || gotIDs[1] != 1-wantLead {
				t.Fatalf("%s row %d (z=%v): picks %v, want lead %d", label, row, z[0], gotIDs, wantLead)
			}
			rowLabel := fmt.Sprintf("%s row %d", label, row)
			requireExactPicks(t, rowLabel, gotIDs, wantIDs[row*2:row*2+2])
			requireRelativeWeights(t, rowLabel, gotW, wantW[row*2:row*2+2])
		}
	}

	// (b) Exact ties keep the lower expert first: the higher tied block leads
	// in id order, then the lower block's lowest ids; an all-equal row picks
	// 0..topK-1 with equal weights.
	const tieExperts = 32
	ties := make([]float32, 2*tieExperts)
	for e := 0; e < tieExperts; e++ {
		switch {
		case e < 4:
			ties[e] = 5
		case e < 8:
			ties[e] = 7
		case e < 12:
			ties[e] = 5
		default:
			ties[e] = -30
		}
		ties[tieExperts+e] = 1
	}
	tieBias := make([]float32, tieExperts)
	for i := range tieBias {
		tieBias[i] = 0.5
	}
	for _, hasBias := range []bool{false, true} {
		hb := []float32(nil)
		if hasBias {
			hb = tieBias
		}
		label := fmt.Sprintf("ties bias=%v", hasBias)
		wantIDs, wantW, stage := v41RouterReferenceForTest(ties, hb, nil, 2, tieExperts, 6, 1.5, false)
		if stage != 0 {
			t.Fatalf("%s: reference faulted at %v", label, stage)
		}
		got0, w0, st := v41RouterSoftplusSqrtRouteHost(ties[:tieExperts], hb, nil, 6, 1.5, false, hasBias)
		if st != 0 {
			t.Fatalf("%s row 0: transcription faulted at %v", label, st)
		}
		got1, w1, st := v41RouterSoftplusSqrtRouteHost(ties[tieExperts:], hb, nil, 6, 1.5, false, hasBias)
		if st != 0 {
			t.Fatalf("%s row 1: transcription faulted at %v", label, st)
		}
		// Closed contract, hand-computed: the z=7 block (experts 4..7) leads
		// in id order, then the two lowest z=5 experts; the all-equal row
		// picks 0..5. A uniform bias shifts every corrected value equally, so
		// the picks are unchanged and the unbiased weights are identical.
		requireExactPicks(t, label+" row 0", got0, []int32{4, 5, 6, 7, 0, 1})
		requireExactPicks(t, label+" row 1", got1, []int32{0, 1, 2, 3, 4, 5})
		for slot := range w1 {
			if w1[slot] != w1[0] {
				t.Fatalf("%s row 1: unequal tied weights %v", label, w1)
			}
		}
		if d := math.Abs(float64(w1[0]) - 0.25); d > 1e-7 {
			t.Fatalf("%s row 1: tied weight %g, want ~0.25 (each score over six times itself times 1.5)", label, w1[0])
		}
		requireExactPicks(t, label+" row 0 ref", got0, wantIDs[:6])
		requireRelativeWeights(t, label+" row 0 ref", w0, wantW[:6])
		requireExactPicks(t, label+" row 1 ref", got1, wantIDs[6:])
		requireRelativeWeights(t, label+" row 1 ref", w1, wantW[6:])
	}

	// (c) Hash-table layers: picks come from the table verbatim; weights are
	// the unbiased scores over the in-order table sum times routeScale,
	// spanning the underflow region (-40) and the large-positive side (40).
	hashLogits := []float32{
		40, -40, 0, 3, -25, 1, 0.5, -5,
		-3, 2, -30, 40, 0.5, -40, 1, 2.5,
		0, 0, 5, -20, -20, 0, 5, 5,
	}
	hashIDs := []int32{
		7, 0, 3, 1,
		5, 3, 2, 0,
		4, 6, 1, 2,
	}
	for _, tc := range []struct {
		full bool
		rs   float32
	}{
		{false, 1.5},
		{true, 2},
	} {
		label := fmt.Sprintf("hash full=%v scale=%g", tc.full, tc.rs)
		wantIDs, wantW, stage := v41RouterReferenceForTest(hashLogits, nil, hashIDs, 3, 8, 4, tc.rs, tc.full)
		if stage != 0 {
			t.Fatalf("%s: reference faulted at %v", label, stage)
		}
		for row := 0; row < 3; row++ {
			z := hashLogits[row*8 : (row+1)*8]
			ids := hashIDs[row*4 : (row+1)*4]
			gotIDs, gotW, st := v41RouterSoftplusSqrtRouteHost(z, nil, ids, 4, tc.rs, tc.full, false)
			if st != 0 {
				t.Fatalf("%s row %d: transcription faulted at %v", label, row, st)
			}
			rowLabel := fmt.Sprintf("%s row %d", label, row)
			requireExactPicks(t, rowLabel, gotIDs, ids)
			requireExactPicks(t, rowLabel+" ref", gotIDs, wantIDs[row*4:(row+1)*4])
			requireRelativeWeights(t, rowLabel+" ref", gotW, wantW[row*4:(row+1)*4])
		}
	}
	// Row 0, hand-computed bands: the picked ids [7,0,3,1] read z =
	// [-5,40,3,-40], so the underflow-region score stays nonzero while the
	// in-order sum times routeScale normalizes the large-positive side.
	for _, tc := range []struct {
		full bool
		rs   float32
		band [4][2]float64
	}{
		{false, 1.5, [4][2]float64{{0.0145, 0.0157}, {1.15, 1.18}, {0.318, 0.325}, {3.2e-10, 4.4e-10}}},
		{true, 2, [4][2]float64{{0.0195, 0.0207}, {1.54, 1.565}, {0.425, 0.432}, {4.5e-10, 5.6e-10}}},
	} {
		_, gotW, st := v41RouterSoftplusSqrtRouteHost(hashLogits[:8], nil, hashIDs[:4], 4, tc.rs, tc.full, false)
		if st != 0 {
			t.Fatalf("hash row 0 full=%v: transcription faulted at %v", tc.full, st)
		}
		for slot := 0; slot < 4; slot++ {
			w := float64(gotW[slot])
			if w < tc.band[slot][0] || w > tc.band[slot][1] {
				t.Fatalf("hash row 0 full=%v scale=%g slot %d weight %g outside [%g, %g]", tc.full, tc.rs, slot, w, tc.band[slot][0], tc.band[slot][1])
			}
		}
	}
}
