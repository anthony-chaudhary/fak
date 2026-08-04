package model

import (
	"math"
	"testing"
)

// rope_yarn_longcontext_test.go — the LONG-OFFSET positional witness for the 262K context
// claim (#4874). rope_yarn_shapes_test.go already proves a classic-key yarn block reaches
// the yarn path and moves the numbers; that is a decode-shape witness on a small full-rotary
// qwen2 config. It says nothing about whether the resulting table is CORRECT at a quarter-
// million positions, which is the part the 262K claim actually rests on.
//
// The reference these tests measure against is the UNSCALED (pre-extension) rope at the
// equivalent in-window position. That is the whole point of a position-interpolating
// scheme: a token at offset 262143 must rotate to an angle the model already saw during
// its 32768-position training, or the model is reading positions it was never taught.
//
// Every fixture here is Bonsai-SHAPED, not the real Bonsai checkpoint: the Qwen3.6 hybrid
// geometry (head_dim 256, partial_rotary_factor 0.25, layer_types 3:1 linear:full,
// rope_theta 1e7) with a 262144 window extended 8x off a 32768 base.
//
// INVALIDATING ASSUMPTION — read this before trusting the numbers: nobody here has read
// the real Bonsai config.json. These tests assume Bonsai extends context with STOCK yarn
// under the classic rope_scaling key at factor 8 off 32768. If the real checkpoint ships a
// bespoke scheme, different beta_fast/beta_slow, an explicit attention_factor, or the
// nested rope_parameters shape with different numbers, the pipeline assertions below still
// hold (they are config-driven) but the specific factor-8/32768 arithmetic is fiction and
// the "262K is correct for Bonsai" conclusion does not follow. Re-pin against the published
// config before promoting this out of gen/next.

// bonsaiShaped262K is the extended long-context config: classic-key yarn, 8x off 32768.
const bonsaiShaped262K = `{
	"model_type": "qwen3_5_text",
	"hidden_size": 6144, "num_hidden_layers": 8, "num_attention_heads": 24,
	"num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 17408,
	"vocab_size": 248320, "rms_norm_eps": 1e-6,
	"layer_types": [
		"linear_attention", "linear_attention", "linear_attention", "full_attention",
		"linear_attention", "linear_attention", "linear_attention", "full_attention"
	],
	"partial_rotary_factor": 0.25,
	"max_position_embeddings": 262144,
	"rope_theta": 10000000,
	"rope_scaling": {
		"rope_type": "yarn", "factor": 8.0,
		"original_max_position_embeddings": 32768
	}
}`

// bonsaiShapedBase is the SAME geometry with the rope_scaling block removed: the unscaled
// 32768-window reference. Identical head_dim / partial_rotary_factor / rope_theta /
// layer_types is load bearing — it makes invFreqDenom and rope_theta agree, so the bare
// table is bit-for-bit the pre-scaling table applyRopeScaling starts from.
const bonsaiShapedBase = `{
	"model_type": "qwen3_5_text",
	"hidden_size": 6144, "num_hidden_layers": 8, "num_attention_heads": 24,
	"num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 17408,
	"vocab_size": 248320, "rms_norm_eps": 1e-6,
	"layer_types": [
		"linear_attention", "linear_attention", "linear_attention", "full_attention",
		"linear_attention", "linear_attention", "linear_attention", "full_attention"
	],
	"partial_rotary_factor": 0.25,
	"max_position_embeddings": 32768,
	"rope_theta": 10000000
}`

const (
	yarnLongCtx      = 262144 // the extended window
	yarnOrigCtx      = 32768  // the trained window
	yarnLongCtxScale = 8.0    // yarnLongCtx / yarnOrigCtx
)

// TestYarnLongContextConfigReachesYarnAt262K is the precondition every other test here
// leans on: a 262K Bonsai-shaped config lands on the yarn path with the right params, on
// the PARTIAL-rotary hybrid geometry (rotary_dim 64 of head_dim 256) rather than the small
// full-rotary shape the existing shape test uses.
func TestYarnLongContextConfigReachesYarnAt262K(t *testing.T) {
	cfg := decodeRopeCfg(t, bonsaiShaped262K)

	if cfg.RopeScaling != "yarn" {
		t.Fatalf("RopeScaling = %q, want %q — a 262K config on a bare inv_freq mis-rotates every long offset", cfg.RopeScaling, "yarn")
	}
	if cfg.RopeFactor != yarnLongCtxScale {
		t.Fatalf("RopeFactor = %v, want %v", cfg.RopeFactor, yarnLongCtxScale)
	}
	if cfg.RopeOrigContext != yarnOrigCtx {
		t.Fatalf("RopeOrigContext = %d, want %d", cfg.RopeOrigContext, yarnOrigCtx)
	}
	if cfg.MaxPositionEmbeddings != yarnLongCtx {
		t.Fatalf("MaxPositionEmbeddings = %d, want %d", cfg.MaxPositionEmbeddings, yarnLongCtx)
	}
	// factor * original == the declared window. A config where these disagree is extending
	// to a different length than it claims, which no amount of correct yarn math fixes.
	if got := float64(cfg.RopeOrigContext) * cfg.RopeFactor; got != float64(cfg.MaxPositionEmbeddings) {
		t.Fatalf("original_max_position_embeddings * factor = %v, want max_position_embeddings %d", got, cfg.MaxPositionEmbeddings)
	}

	if !cfg.IsQwen35Hybrid() {
		t.Fatal("fixture is not recognized as a Qwen3.5/3.6 hybrid — the partial-rotary and KV-layer branches under test would not fire")
	}
	// partial_rotary_factor 0.25 of head_dim 256: 64 rotated dims, 32 frequencies. The
	// hybrid branch of invFreqDenom must denominate by the rotary dim, not head_dim —
	// denominating by 256 would make every frequency too high and drift with position.
	if got := cfg.rotaryDim(); got != 64 {
		t.Fatalf("rotaryDim = %d, want 64", got)
	}
	if got := cfg.invFreqDenom(); got != 64 {
		t.Fatalf("invFreqDenom = %d, want the rotary dim 64 (head_dim 256 would over-scale every frequency)", got)
	}

	// yarn's attention temperature must be live too: sqrt-log mscale at factor 8.
	wantAttn := 0.1*math.Log(yarnLongCtxScale) + 1
	if got := cfg.ropeAttentionFactor(); math.Abs(got-wantAttn) > 1e-12 {
		t.Fatalf("ropeAttentionFactor = %v, want %v (0.1*ln(factor)+1)", got, wantAttn)
	}
}

// yarnRefInvFreq is an INDEPENDENT transcription of HF's _compute_yarn_parameters. It
// exists so the tests below measure applyRopeScaling against an outside statement of the
// formula rather than against itself — a self-consistent wrong table would sail through a
// test written only in terms of fak's own helpers.
func yarnRefInvFreq(headDim int, partial, theta, factor float64, origCtx int, betaFast, betaSlow float64) []float64 {
	dim := int(float64(headDim) * partial)
	half := dim / 2

	// HF: find_correction_dim in FULL-dim units, then the ramp is evaluated over dim//2
	// points. The mixed units are HF's, and fak must reproduce them to match.
	correctionDim := func(rot float64) float64 {
		return (float64(dim) * math.Log(float64(origCtx)/(rot*2*math.Pi))) / (2 * math.Log(theta))
	}
	low := math.Max(math.Floor(correctionDim(betaFast)), 0)
	high := math.Min(math.Ceil(correctionDim(betaSlow)), float64(dim-1))

	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		posFreq := math.Pow(theta, float64(2*j)/float64(dim))
		extrapolation := 1.0 / posFreq
		interpolation := 1.0 / (factor * posFreq)
		ramp := (float64(j) - low) / (high - low)
		ramp = math.Min(math.Max(ramp, 0), 1)
		extrapolationFactor := 1 - ramp
		inv[j] = interpolation*(1-extrapolationFactor) + extrapolation*extrapolationFactor
	}
	return inv
}

func TestYarnLongContextMatchesTheHFReferenceTable(t *testing.T) {
	cfg := decodeRopeCfg(t, bonsaiShaped262K)
	got := invFreq(cfg, 0)
	want := yarnRefInvFreq(256, 0.25, 1e7, yarnLongCtxScale, yarnOrigCtx, 32, 1)

	if len(got) != len(want) {
		t.Fatalf("inv_freq length = %d, want %d", len(got), len(want))
	}
	if len(got) != 32 {
		t.Fatalf("inv_freq length = %d, want 32 (rotary_dim 64 / 2)", len(got))
	}
	for j := range got {
		// Relative tolerance: the two transcriptions order the same float ops differently.
		if d := math.Abs(got[j] - want[j]); d > 1e-15*math.Abs(want[j])+1e-18 {
			t.Fatalf("inv_freq[%d] = %.17g, HF reference = %.17g (delta %g)", j, got[j], want[j], d)
		}
	}
}

// bandsOf splits the frequency indices by where the yarn ramp puts them, using the
// reference implementation's own low/high rather than fak's, so the split is not derived
// from the code under test.
func bandsOf(t *testing.T, headDim int, partial, theta float64, origCtx int, betaFast, betaSlow float64) (extrapolated, interpolated []int) {
	t.Helper()
	dim := int(float64(headDim) * partial)
	correctionDim := func(rot float64) float64 {
		return (float64(dim) * math.Log(float64(origCtx)/(rot*2*math.Pi))) / (2 * math.Log(theta))
	}
	low := math.Max(math.Floor(correctionDim(betaFast)), 0)
	high := math.Min(math.Ceil(correctionDim(betaSlow)), float64(dim-1))
	for j := 0; j < dim/2; j++ {
		switch {
		case float64(j) <= low:
			extrapolated = append(extrapolated, j)
		case float64(j) >= high:
			interpolated = append(interpolated, j)
		}
	}
	if len(extrapolated) == 0 || len(interpolated) == 0 {
		t.Fatalf("degenerate ramp for this geometry: low=%v high=%v — the band assertions would be vacuous", low, high)
	}
	return extrapolated, interpolated
}

// TestYarnLongContextKeepsHighOffsetsInTheTrainedBand is the done-condition witness:
// positional parity at a high offset against the unscaled reference.
//
// yarn is NTK-by-parts, so "parity" means two different things in the two bands and the
// test checks each on its own terms:
//
//   - INTERPOLATED (low-frequency, long-wavelength) dims are divided by the full factor.
//     A token at the top of the 262K window must rotate these to exactly the angle the
//     unscaled rope gives at the top of the 32768 window. Anything past that angle is a
//     position the model never saw.
//   - EXTRAPOLATED (high-frequency, short-wavelength) dims are deliberately left alone,
//     which is only safe because they complete at least beta_fast=32 full revolutions
//     INSIDE the trained window — every phase was already seen, so running them past the
//     window revisits known angles instead of extrapolating into new ones.
func TestYarnLongContextKeepsHighOffsetsInTheTrainedBand(t *testing.T) {
	cfg := decodeRopeCfg(t, bonsaiShaped262K)
	base := decodeRopeCfg(t, bonsaiShapedBase)
	if base.RopeScaling != "" {
		t.Fatalf("reference config picked up RopeScaling=%q; it must be the unscaled table", base.RopeScaling)
	}

	inv := invFreq(cfg, 0)
	bare := invFreq(base, 0)
	if len(inv) != len(bare) {
		t.Fatalf("table lengths differ: %d vs %d", len(inv), len(bare))
	}
	extrapolated, interpolated := bandsOf(t, 256, 0.25, 1e7, yarnOrigCtx, 32, 1)

	// --- interpolated band: exact angular parity at the window edge ---------------------
	//
	// Exact (not approximate) equality is legitimate here because factor 8 and the two
	// window sizes are powers of two: x/8 and 262144*(x/8) == 32768*x are both exponent
	// shifts with no mantissa loss. A non-power-of-two factor would need a tolerance.
	for _, j := range interpolated {
		if want := bare[j] / yarnLongCtxScale; inv[j] != want {
			t.Fatalf("inv_freq[%d] = %.17g, want the unscaled %.17g divided by factor %v = %.17g",
				j, inv[j], bare[j], yarnLongCtxScale, want)
		}
		longAngle := float64(yarnLongCtx) * inv[j]
		refAngle := float64(yarnOrigCtx) * bare[j]
		if longAngle != refAngle {
			t.Fatalf("dim %d: angle at extended offset %d = %.17g, want the unscaled angle at offset %d = %.17g",
				j, yarnLongCtx, longAngle, yarnOrigCtx, refAngle)
		}
	}

	// The same parity at the cos/sin level, bit-for-bit: the rotation an interpolated dim
	// applies at the top of the extended window IS the rotation the unscaled model applies
	// at the top of its trained window. This is the claim in the form the attention kernel
	// actually consumes.
	cosLong, sinLong := ropeRowFromInv(inv, yarnLongCtx)
	cosRef, sinRef := ropeRowFromInv(bare, yarnOrigCtx)
	for _, j := range interpolated {
		if cosLong[j] != cosRef[j] || sinLong[j] != sinRef[j] {
			t.Fatalf("dim %d: (cos,sin) at offset %d = (%v,%v), want the unscaled row at offset %d = (%v,%v)",
				j, yarnLongCtx, cosLong[j], sinLong[j], yarnOrigCtx, cosRef[j], sinRef[j])
		}
	}

	// The LAST REAL position (the window is 0..262143) must land strictly inside the
	// trained window once mapped back through the unscaled table.
	const lastPos = yarnLongCtx - 1
	for _, j := range interpolated {
		effective := float64(lastPos) * inv[j] / bare[j]
		if effective >= float64(yarnOrigCtx) {
			t.Fatalf("dim %d: offset %d maps to effective position %.4f, want < %d (the trained window)",
				j, lastPos, effective, yarnOrigCtx)
		}
	}

	// --- extrapolated band: untouched, and safe only because it is periodic -------------
	for _, j := range extrapolated {
		if inv[j] != bare[j] {
			t.Fatalf("inv_freq[%d] = %.17g, want the unscaled %.17g — high-frequency dims must be left alone by NTK-by-parts",
				j, inv[j], bare[j])
		}
		// revolutions completed within the TRAINED window; beta_fast=32 is the threshold
		// that made leaving this dim unscaled safe in the first place.
		revs := float64(yarnOrigCtx) * bare[j] / (2 * math.Pi)
		if revs < 32 {
			t.Fatalf("dim %d is unscaled but completes only %.2f revolutions inside the trained %d-window (want >= 32): "+
				"running it to offset %d extrapolates into angles training never covered", j, revs, yarnOrigCtx, lastPos)
		}
	}

	// --- the witness must have teeth ----------------------------------------------------
	//
	// If the yarn promotion regressed to a bare table, every assertion above about the
	// interpolated band would be comparing a table to itself. Prove the two tables really
	// do disagree at a long offset.
	var maxDelta float64
	cosBareLong, _ := ropeRowFromInv(bare, lastPos)
	cosYarnLong, _ := ropeRowFromInv(inv, lastPos)
	for j := range cosYarnLong {
		if d := math.Abs(float64(cosYarnLong[j] - cosBareLong[j])); d > maxDelta {
			maxDelta = d
		}
	}
	if maxDelta < 0.1 {
		t.Fatalf("yarn and unscaled cos rows at offset %d differ by only %g — the scaling is not running and this test is vacuous", lastPos, maxDelta)
	}
	t.Logf("offset %d: %d interpolated dims at exact trained-window parity, %d extrapolated dims periodic; max|cos delta| vs unscaled = %g",
		lastPos, len(interpolated), len(extrapolated), maxDelta)
}

// TestYarnLongContext262KKVBudgetIsBounded closes the second half of the done condition on
// the SAME 262K config the positional witness uses: at a quarter-million positions the
// full-attention KV must fit a stated budget at 4 bits.
//
// Two separate savings compose here and the test pins both, because conflating them is how
// a 262K memory claim gets overstated: (1) only 1/4 of the hybrid's layers hold a KV cache
// at all, and (2) those layers can be held at the KVQuant4 rate.
func TestYarnLongContext262KKVBudgetIsBounded(t *testing.T) {
	cfg := decodeRopeCfg(t, bonsaiShaped262K)

	kvLayers := cfg.KVQuantLayers()
	if len(kvLayers) != 2 || kvLayers[0] != 3 || kvLayers[1] != 7 {
		t.Fatalf("KV-holding layers = %v, want [3 7] — the ~75%% linear-attention layers hold a recurrent state, not KV", kvLayers)
	}

	// 4 kv heads * 256 head_dim = 1024 elements per row, K and V, on 2 layers, 262144 positions.
	const perRowElems = 4 * 256
	f32 := cfg.KVCacheBytesAtBits(yarnLongCtx, 32)
	q4 := cfg.KVCacheBytesAtBits(yarnLongCtx, 4)

	wantF32 := int64(2) * 2 * perRowElems * 4 * yarnLongCtx // layers * (K+V) * elems * 4B
	if f32 != wantF32 {
		t.Fatalf("f32 KV at %d positions = %d bytes, want %d", yarnLongCtx, f32, wantF32)
	}
	if wantF32 != 4<<30 {
		t.Fatalf("fixture sanity: f32 KV = %d bytes, want exactly 4 GiB", wantF32)
	}
	// 6 bits/element honest rate (4-bit payload + f32 scale and min per 32-element group),
	// so the ratio is 32/6, NOT the 8x a bare-nibble count would advertise.
	if want := int64(768) << 20; q4 != want {
		t.Fatalf("4-bit KV at %d positions = %d bytes, want %d (768 MiB)", yarnLongCtx, q4, want)
	}
	if got := float64(f32) / float64(q4); math.Abs(got-32.0/6.0) > 1e-9 {
		t.Fatalf("f32:4-bit ratio = %v, want 32/6 — the group metadata is being dropped from the accounting", got)
	}

	// Sizing the cache over ALL layers instead of the KV-holding ones would overstate the
	// hybrid's budget 4x. Pin the gap so a regression that reintroduces it is visible.
	dense := cfg
	dense.LayerTypes = nil
	if got := dense.KVCacheBytesAtBits(yarnLongCtx, 4); got != 4*q4 {
		t.Fatalf("all-layers 4-bit KV = %d bytes, want 4x the hybrid's %d", got, q4)
	}

	// The codec's own error ceiling is what bounds the decode-quality loss this budget buys.
	// It must be reported from the stored scales, not assumed.
	row := make([]float32, perRowElems)
	for i := range row {
		row[i] = float32(math.Sin(float64(i) * 0.37))
	}
	q := QuantizeKV4(row)
	bound := q.ErrorBound()
	if bound <= 0 {
		t.Fatalf("ErrorBound = %v on a non-constant row, want > 0", bound)
	}
	back := q.Dequantize()
	for i := range row {
		if d := float32(math.Abs(float64(back[i] - row[i]))); d > bound {
			t.Fatalf("element %d round-tripped with error %v, exceeding the stated bound %v", i, d, bound)
		}
	}
	t.Logf("262K KV: %d of %d layers hold KV; f32 %d MiB -> 4-bit %d MiB; round-trip error bound %g",
		len(kvLayers), cfg.NumLayers, f32>>20, q4>>20, bound)
}
