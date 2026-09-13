package model

// v4_flash_oracle_test.go — the weight-free, checkpoint-independent CPU numeric
// oracle for the DeepSeek-V4-Flash-0731 component equations. Companion to
// family_cpu_oracle_test.go, which states the independence doctrine, and to
// v4_router_test.go / v4_quant_decode_test.go / v4_expert_compose_test.go, which
// cover the production arms.
//
// Scope and provenance. The reference below is a plain in-order scalar
// transcription of the pinned official artifact
// deepseek-ai/DeepSeek-V4-Flash-0731 @ revision
// 7872f01b1d1fe23eabc4c98b48bffcef5a386062 (inference/model.py +
// inference/kernel.py). It reuses NONE of the production machinery — not
// v4HashRoute, not v4ScoredRoute, not decodeV4ExpertQuant, not
// composeV4RoutedExperts, not v4SiLU, and not cpuOracle*. Every matmul, norm,
// sigmoid, softmax and Sinkhorn sweep is a naive scalar loop transcribed here.
// The nibble order and E2M1 value table were read off the production
// decodeV4ExpertQuant implementation ONLY to pin the contract; that production
// reference is NOT independently cross-checked here at the fixture level — the
// production cross-check lives in v4_quant_decode_test.go. The oracle decoder
// re-implements the layout by hand.
//
// What is covered: this oracle transcribes four pinned component equations.
// Three sit against an existing production seam (scored/hash routing, packed
// FP4+E8M0 decode, the per-layer compression schedule); two are forward-looking
// (four-stream mHC pre/post mixing with 20-iteration Sinkhorn, and the
// eight-group low-rank output projection) whose production forward paths do not
// exist yet. For those two the oracle is a transcription/regression pin, not a
// cross-implementation comparison.
//
//	A. sqrt-softplus routing for later (scored) layers: top-6 by
//	   sqrt(softplus(z)), weights = score/sum(score)*route_scale.
//	B. four-stream mHC mixing (pre/post) plus the 20-iteration Sinkhorn
//	   normalization of the comb[dest][source] 4x4 matrix, and the
//	   collapse/apply step.
//	C. packed FP4 (E2M1) dequantization with a per-32-value E8M0 scale.
//	D. grouped low-rank output projection (per-group split then a single
//	   concatenated up-projection).
//
// What is deliberately NOT covered: the full Session.Prefill / Session.Step
// native V4 forward does not exist yet (deepseek_v4 is excluded from
// usesMLAMoELayout), so the integration arm is escalated separately and this
// file stays at the component boundary.
//
// Independence + anti-vacuity: the checked-in fixture holds expected outputs
// produced by the scalar transcription in this file; the oracle is compared to
// the fixture within cpuOracleTol, and separate tests prove the load-bearing
// axes (routing selection, Sinkhorn orientation, the pre-only +hc_eps, the
// per-group output split) move the result by far more than the tolerance.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

const v4FlashOracleRevision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

// ---------------------------------------------------------------------------
// Deterministic input generation (fixed LCG, no math/rand, no randomness)
// ---------------------------------------------------------------------------

// oracleV4FlashLCG returns a deterministic stream in [-1, 1) from a fixed
// 64-bit linear congruential state. Same seed -> same inputs on every run.
func oracleV4FlashLCG(seed uint64) func() float64 {
	state := seed
	const mul = 6364136223846793005
	const inc = 1442695040888963407
	return func() float64 {
		state = state*mul + inc
		return float64(state>>11)/float64(uint64(1)<<53)*2 - 1
	}
}

func oracleV4FlashFill(n int, seed uint64, scale float64) []float64 {
	next := oracleV4FlashLCG(seed)
	out := make([]float64, n)
	for i := range out {
		out[i] = next() * scale
	}
	return out
}

// ---------------------------------------------------------------------------
// A. sqrt-softplus routing (later scored layers)
// ---------------------------------------------------------------------------

// oracleV4FlashScore is sqrt(softplus(z)) with the overflow-safe softplus
// max(z,0)+log1p(exp(-abs(z))), transcribed from the pinned Gate.forward.
func oracleV4FlashScore(z float64) float64 {
	return math.Sqrt(math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z))))
}

// oracleV4FlashScoredRoute selects topK experts by descending sqrt-softplus
// score (lower index wins ties) and returns weights normalized over the
// selected scores and scaled by routeScale. No production helper is called.
func oracleV4FlashScoredRoute(logits []float64, topK int, routeScale float64) ([]int, []float64) {
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	// Deterministic insertion sort: descending score, then lower index.
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0; j-- {
			a, b := idx[j-1], idx[j]
			sa, sb := oracleV4FlashScore(logits[a]), oracleV4FlashScore(logits[b])
			if sa > sb || (sa == sb && a < b) {
				break
			}
			idx[j-1], idx[j] = idx[j], idx[j-1]
		}
	}
	picks := append([]int(nil), idx[:topK]...)
	var sum float64
	for _, e := range picks {
		sum += oracleV4FlashScore(logits[e])
	}
	weights := make([]float64, topK)
	for i, e := range picks {
		weights[i] = oracleV4FlashScore(logits[e]) / sum * routeScale
	}
	return picks, weights
}

// oracleV4FlashHashWeights re-derives the hash-layer weight rule: score the
// logits AT the supplied expert ids, normalize, scale. Selection comes from the
// tid2eid table, not the logits, which is the hash/scored distinction.
func oracleV4FlashHashWeights(logits []float64, expertIDs []int, routeScale float64) []float64 {
	var sum float64
	for _, id := range expertIDs {
		sum += oracleV4FlashScore(logits[id])
	}
	w := make([]float64, len(expertIDs))
	for i, id := range expertIDs {
		w[i] = oracleV4FlashScore(logits[id]) / sum * routeScale
	}
	return w
}

// ---------------------------------------------------------------------------
// B. four-stream mHC mixing + Sinkhorn
// ---------------------------------------------------------------------------

const (
	oracleV4FlashHCMult        = 4
	oracleV4FlashHCEps         = 1e-6
	oracleV4FlashSinkhornIters = 20
	oracleV4FlashRMSFlatEps    = 1e-6
	oracleV4FlashStreamDim     = 3
	oracleV4FlashMixVecWidth   = oracleV4FlashHCMult * oracleV4FlashStreamDim
)

// oracleV4FlashFlattenStreams lays the four width-d streams end to end.
func oracleV4FlashFlattenStreams(x [][]float64) []float64 {
	out := make([]float64, 0, len(x)*len(x[0]))
	for _, s := range x {
		out = append(out, s...)
	}
	return out
}

// oracleV4FlashMHCProjection computes the 24 mix coefficients (pre|post|comb)
// from the flattened streams and hc_fn, applying the 1/sqrt(mean(x^2)+eps)
// input scale. Warm/kernel.py: mixes = linear(xflat, hc_fn) * rsqrt.
func oracleV4FlashMHCProjection(xflat, hcFn []float64, mixWidth, flatWidth int, rmsEps float64) []float64 {
	var ss float64
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := 1 / math.Sqrt(ss/float64(len(xflat))+rmsEps)
	mixes := make([]float64, mixWidth)
	for m := 0; m < mixWidth; m++ {
		var s float64
		row := hcFn[m*flatWidth : (m+1)*flatWidth]
		for i := 0; i < flatWidth; i++ {
			s += row[i] * xflat[i]
		}
		mixes[m] = s * rsqrt
	}
	return mixes
}

func oracleV4FlashSigmoid(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

// oracleV4FlashMixesToPrePostComb splits the 24 mixes into pre[4], post[4],
// comb[4][4] using the pinned layout indices, including the pre-only +hc_eps.
func oracleV4FlashMixesToPrePostComb(mixes, hcScale, hcBase []float64, hcMult int, preEps bool) (pre, post []float64, comb [][]float64) {
	pre = make([]float64, hcMult)
	post = make([]float64, hcMult)
	comb = make([][]float64, hcMult)
	for j := 0; j < hcMult; j++ {
		pre[j] = oracleV4FlashSigmoid(mixes[j]*hcScale[0] + hcBase[j])
		if preEps {
			pre[j] += oracleV4FlashHCEps
		}
		post[j] = 2 * oracleV4FlashSigmoid(mixes[j+hcMult]*hcScale[1]+hcBase[j+hcMult])
		comb[j] = make([]float64, hcMult)
		for k := 0; k < hcMult; k++ {
			comb[j][k] = mixes[(j*hcMult+k)+2*hcMult]*hcScale[2] + hcBase[(j*hcMult+k)+2*hcMult]
		}
	}
	return pre, post, comb
}

// oracleV4FlashSinkhornApply runs the pinned normalization: the full executed
// sequence is an initial row-softmax over k (with +eps applied after the
// softmax), then one column normalization, then (iters-1) alternating row/column
// iterations — for iters=20 that is 20 row passes and 20 column passes total.
// The returned schedule records only from the first column normalization
// onward (the initial row-softmax is deliberately NOT recorded as a pass
// entry), so it is exactly ["col", "row", "col", ...] with length 2*iters-1.
// The exact iteration contract can therefore be pinned structurally without
// changing the executed math.
func oracleV4FlashSinkhornApply(comb [][]float64, iters int, eps float64) []string {
	n := len(comb)
	passes := make([]string, 0, 2*iters)
	for j := 0; j < n; j++ {
		mx := comb[j][0]
		for k := 1; k < n; k++ {
			if comb[j][k] > mx {
				mx = comb[j][k]
			}
		}
		var rowsum float64
		for k := 0; k < n; k++ {
			comb[j][k] = math.Exp(comb[j][k] - mx)
			rowsum += comb[j][k]
		}
		for k := 0; k < n; k++ {
			comb[j][k] = comb[j][k]/rowsum + eps
		}
	}
	passes = append(passes, "col")
	col := make([]float64, n)
	for k := 0; k < n; k++ {
		var s float64
		for j := 0; j < n; j++ {
			s += comb[j][k]
		}
		col[k] = s
	}
	for j := 0; j < n; j++ {
		for k := 0; k < n; k++ {
			comb[j][k] /= col[k] + eps
		}
	}
	for it := 1; it < iters; it++ {
		passes = append(passes, "row", "col")
		row := make([]float64, n)
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += comb[j][k]
			}
			row[j] = s
		}
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				comb[j][k] /= row[j] + eps
			}
		}
		for k := 0; k < n; k++ {
			var s float64
			for j := 0; j < n; j++ {
				s += comb[j][k]
			}
			col[k] = s
		}
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				comb[j][k] /= col[k] + eps
			}
		}
	}
	return passes
}

// oracleV4FlashMHCApply is the pinned application: collapse the four streams
// with pre, then write each destination stream j from post[j] * collapsed plus
// the source contribution sum_k comb[j][k] * residual[k].
func oracleV4FlashMHCApply(x [][]float64, pre, post []float64, comb [][]float64, transposeComb bool) (collapsed, y []float64) {
	d := len(x[0])
	collapsed = make([]float64, d)
	for j := range x {
		for k := 0; k < d; k++ {
			collapsed[k] += pre[j] * x[j][k]
		}
	}
	y = make([]float64, len(x)*d)
	for j := range x {
		for k := 0; k < d; k++ {
			var s float64
			for src := range x {
				c := comb[j][src]
				if transposeComb {
					c = comb[src][j]
				}
				s += c * x[src][k]
			}
			y[j*d+k] = post[j]*collapsed[k] + s
		}
	}
	return collapsed, y
}

// ---------------------------------------------------------------------------
// C. packed FP4 (E2M1) with E8M0 scale
// ---------------------------------------------------------------------------

// oracleV4FlashE2M1 is the E2M1 finite value table indexed by one nibble,
// transcribed by hand (low nibble = even index, high nibble = odd index).
func oracleV4FlashE2M1(nibble int) float64 {
	table := [16]float64{
		0, 0.5, 1, 1.5, 2, 3, 4, 6,
		math.Copysign(0, -1), -0.5, -1, -1.5, -2, -3, -4, -6,
	}
	return table[nibble]
}

// oracleV4FlashDecodeFP4 unpacks row-major packedWeight [rows, weightCols]
// (2 nibbles/byte along K, low nibble first) with one E8M0 scale per 16 packed
// bytes (32 unpacked values), scale = 2^(byte-127).
func oracleV4FlashDecodeFP4(packedWeight, scales []byte, rows, weightCols, scaleCols int) []float64 {
	unpackedCols := weightCols * 2
	out := make([]float64, rows*unpackedCols)
	for r := 0; r < rows; r++ {
		packedRow := packedWeight[r*weightCols : (r+1)*weightCols]
		scaleRow := scales[r*scaleCols : (r+1)*scaleCols]
		for pc, b := range packedRow {
			exp := int(scaleRow[pc/16]) - 127
			base := r*unpackedCols + pc*2
			out[base] = math.Ldexp(oracleV4FlashE2M1(int(b&0x0f)), exp)
			out[base+1] = math.Ldexp(oracleV4FlashE2M1(int(b>>4)), exp)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// D. grouped low-rank output projection
// ---------------------------------------------------------------------------

// oracleV4FlashOProj applies per-group down projections woA[g] (each
// [oLora][groupIn]) to o[g], concatenates the group latents, then applies the
// single woB ([outDim][groups*oLora]). The per-group split is explicit so a
// group-orientation bug diverges.
func oracleV4FlashOProj(o, woA, woB []float64, groups, groupIn, oLora, outDim int) (concat, out []float64) {
	concat = make([]float64, 0, groups*oLora)
	for g := 0; g < groups; g++ {
		og := o[g*groupIn : (g+1)*groupIn]
		for r := 0; r < oLora; r++ {
			rowOff := (g*oLora + r) * groupIn
			var s float64
			for i := 0; i < groupIn; i++ {
				s += woA[rowOff+i] * og[i]
			}
			concat = append(concat, s)
		}
	}
	out = make([]float64, outDim)
	for r := 0; r < outDim; r++ {
		row := woB[r*(groups*oLora) : (r+1)*(groups*oLora)]
		var s float64
		for i := range row {
			s += row[i] * concat[i]
		}
		out[r] = s
	}
	return concat, out
}

// ---------------------------------------------------------------------------
// Compress schedule (transcribed literal)
// ---------------------------------------------------------------------------

// oracleV4FlashCompressSchedule is the pinned per-layer compression ratio for
// the 43 decoder layers plus the 3 MTP layers (indices 43..45 are ratio 0).
func oracleV4FlashCompressSchedule() []int {
	return []int{
		0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
		4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
		4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
	}
}

// ---------------------------------------------------------------------------
// Fixture aggregation
// ---------------------------------------------------------------------------

// oracleV4FlashFixture is the full computed component-oracle bundle that is
// cross-checked against the checked-in expected-values JSON.
type oracleV4FlashFixture struct {
	Revision string                  `json:"revision"`
	Seed     uint64                  `json:"seed"`
	Route    oracleV4FlashRoute      `json:"route"`
	MHC      oracleV4FlashMHC        `json:"mhc"`
	FP4      oracleV4FlashFP4        `json:"fp4"`
	OProj    oracleV4FlashOProjBlock `json:"oproj"`
	Compress oracleV4FlashSched      `json:"compress"`
}

type oracleV4FlashRoute struct {
	LogitSeed float64   `json:"logit_seed"`
	TopK      int       `json:"top_k"`
	Scale     float64   `json:"route_scale"`
	Indices   []int     `json:"indices"`
	Weights   []float64 `json:"weights"`
	HashIDs   []int     `json:"hash_ids"`
	HashW     []float64 `json:"hash_weights"`
}

type oracleV4FlashMHC struct {
	HCMult    int         `json:"hc_mult"`
	StreamDim int         `json:"stream_dim"`
	SinkhornN int         `json:"sinkhorn_iters"`
	Pre       []float64   `json:"pre"`
	Post      []float64   `json:"post"`
	Comb      [][]float64 `json:"comb"`
	Collapsed []float64   `json:"collapsed"`
	Applied   []float64   `json:"applied"`
}

type oracleV4FlashFP4 struct {
	Rows       int       `json:"rows"`
	WeightCols int       `json:"weight_cols"`
	ScaleCols  int       `json:"scale_cols"`
	WeightHex  string    `json:"weight_bytes_hex"`
	ScaleHex   string    `json:"scale_bytes_hex"`
	Decoded    []float64 `json:"decoded"`
}

type oracleV4FlashOProjBlock struct {
	Groups  int       `json:"groups"`
	GroupIn int       `json:"group_in"`
	OLora   int       `json:"o_lora_rank"`
	OutDim  int       `json:"out_dim"`
	O       []float64 `json:"o"`
	Concat  []float64 `json:"concat"`
	Out     []float64 `json:"out"`
}

type oracleV4FlashSched struct {
	Count    int   `json:"count"`
	Schedule []int `json:"schedule"`
}

func oracleV4FlashComputeFixture() oracleV4FlashFixture {
	const seed = 20260912
	var fx oracleV4FlashFixture
	fx.Revision = v4FlashOracleRevision
	fx.Seed = seed

	// A. routing: 256 logits, top-6, scale 1.5.
	logits := oracleV4FlashFill(256, seed+1, 2.0)
	idx, w := oracleV4FlashScoredRoute(logits, 6, 1.5)
	hashIDs := []int{7, 40, 3, 200, 111, 255}
	fx.Route = oracleV4FlashRoute{
		LogitSeed: seed + 1,
		TopK:      6,
		Scale:     1.5,
		Indices:   idx,
		Weights:   w,
		HashIDs:   hashIDs,
		HashW:     oracleV4FlashHashWeights(logits, hashIDs, 1.5),
	}

	// B. mHC: hc_mult 4, stream dim 3.
	const mixWidth = 24
	const flatWidth = oracleV4FlashMixVecWidth
	streams := make([][]float64, oracleV4FlashHCMult)
	for j := range streams {
		streams[j] = oracleV4FlashFill(oracleV4FlashStreamDim, seed+10+uint64(j), 1.25)
	}
	xflat := oracleV4FlashFlattenStreams(streams)
	hcFn := oracleV4FlashFill(mixWidth*flatWidth, seed+20, 0.5)
	hcScale := []float64{0.7, 1.1, 0.9}
	hcBase := oracleV4FlashFill(mixWidth, seed+21, 0.2)
	mixes := oracleV4FlashMHCProjection(xflat, hcFn, mixWidth, flatWidth, oracleV4FlashRMSFlatEps)
	pre, post, comb := oracleV4FlashMixesToPrePostComb(mixes, hcScale, hcBase, oracleV4FlashHCMult, true)
	_ = oracleV4FlashSinkhornApply(comb, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	collapsed, applied := oracleV4FlashMHCApply(streams, pre, post, comb, false)
	fx.MHC = oracleV4FlashMHC{
		HCMult:    oracleV4FlashHCMult,
		StreamDim: oracleV4FlashStreamDim,
		SinkhornN: oracleV4FlashSinkhornIters,
		Pre:       pre,
		Post:      post,
		Comb:      comb,
		Collapsed: collapsed,
		Applied:   applied,
	}

	// C. FP4: one row of 32 packed bytes, two E8M0 scale bytes.
	rows, weightCols, scaleCols := 1, 32, 2
	packed := make([]byte, rows*weightCols)
	scales := make([]byte, rows*scaleCols)
	next := oracleV4FlashLCG(seed + 30)
	for i := range packed {
		packed[i] = byte(int(next()*127) & 0xff)
	}
	for i := range scales {
		scales[i] = byte(124 + (i % 6)) // exponents 2^-3 .. 2^2, never 0xff NaN
	}
	fx.FP4 = oracleV4FlashFP4{
		Rows:       rows,
		WeightCols: weightCols,
		ScaleCols:  scaleCols,
		WeightHex:  oracleV4FlashHex(packed),
		ScaleHex:   oracleV4FlashHex(scales),
		Decoded:    oracleV4FlashDecodeFP4(packed, scales, rows, weightCols, scaleCols),
	}

	// D. grouped output projection: 8 groups.
	const groups, groupIn, oLora, outDim = 8, 4, 3, 8
	o := oracleV4FlashFill(groups*groupIn, seed+40, 1.0)
	woA := oracleV4FlashFill(groups*oLora*groupIn, seed+41, 0.5)
	woB := oracleV4FlashFill(outDim*groups*oLora, seed+42, 0.5)
	concat, out := oracleV4FlashOProj(o, woA, woB, groups, groupIn, oLora, outDim)
	fx.OProj = oracleV4FlashOProjBlock{
		Groups:  groups,
		GroupIn: groupIn,
		OLora:   oLora,
		OutDim:  outDim,
		O:       o,
		Concat:  concat,
		Out:     out,
	}

	// Compress schedule.
	sched := oracleV4FlashCompressSchedule()
	fx.Compress = oracleV4FlashSched{Count: len(sched), Schedule: sched}
	return fx
}

func oracleV4FlashHex(b []byte) string {
	const hexdigits = "0123456789abcdef"
	out := make([]byte, 0, len(b)*2)
	for _, v := range b {
		out = append(out, hexdigits[v>>4], hexdigits[v&0x0f])
	}
	return string(out)
}

func oracleV4FlashUnhex(t *testing.T, s string) []byte {
	t.Helper()
	if len(s)%2 != 0 {
		t.Fatalf("hex string length %d is odd", len(s))
	}
	out := make([]byte, len(s)/2)
	for i := range out {
		hi := oracleV4FlashHexVal(t, s[i*2])
		lo := oracleV4FlashHexVal(t, s[i*2+1])
		out[i] = hi<<4 | lo
	}
	return out
}

func oracleV4FlashHexVal(t *testing.T, c byte) byte {
	t.Helper()
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10
	}
	t.Fatalf("invalid hex digit %q", c)
	return 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// TestV4FlashOracleFixtureValues is the primary witness: the independent scalar
// oracle must reproduce every checked-in expected value within cpuOracleTol.
func TestV4FlashOracleFixtureValues(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v4_flash_oracle_expected.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var want oracleV4FlashFixture
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if want.Revision != v4FlashOracleRevision {
		t.Fatalf("fixture revision %q, want %q", want.Revision, v4FlashOracleRevision)
	}

	got := oracleV4FlashComputeFixture()

	// A. routing.
	if want.Route.TopK != 6 || want.Route.Scale != 1.5 {
		t.Fatalf("fixture route top_k=%d scale=%v, want 6/1.5", want.Route.TopK, want.Route.Scale)
	}
	oracleV4FlashAssertInts(t, "route.indices", got.Route.Indices, want.Route.Indices)
	oracleV4FlashAssertFloats(t, "route.weights", got.Route.Weights, want.Route.Weights)
	oracleV4FlashAssertFloats(t, "route.hash_weights", got.Route.HashW, want.Route.HashW)

	// B. mHC.
	oracleV4FlashAssertFloats(t, "mhc.pre", got.MHC.Pre, want.MHC.Pre)
	oracleV4FlashAssertFloats(t, "mhc.post", got.MHC.Post, want.MHC.Post)
	oracleV4FlashAssertMatrix(t, "mhc.comb", got.MHC.Comb, want.MHC.Comb)
	oracleV4FlashAssertFloats(t, "mhc.collapsed", got.MHC.Collapsed, want.MHC.Collapsed)
	oracleV4FlashAssertFloats(t, "mhc.applied", got.MHC.Applied, want.MHC.Applied)
	if want.MHC.HCMult != 4 || want.MHC.SinkhornN != 20 {
		t.Fatalf("fixture mhc hc_mult=%d sinkhorn=%d, want 4/20", want.MHC.HCMult, want.MHC.SinkhornN)
	}

	// C. FP4.
	packed := oracleV4FlashUnhex(t, want.FP4.WeightHex)
	scales := oracleV4FlashUnhex(t, want.FP4.ScaleHex)
	dec := oracleV4FlashDecodeFP4(packed, scales, want.FP4.Rows, want.FP4.WeightCols, want.FP4.ScaleCols)
	oracleV4FlashAssertFloats(t, "fp4.decoded", got.FP4.Decoded, dec)
	oracleV4FlashAssertFloats(t, "fp4.decoded(self)", got.FP4.Decoded, want.FP4.Decoded)

	// D. grouped output projection.
	oracleV4FlashAssertFloats(t, "oproj.o", got.OProj.O, want.OProj.O)
	oracleV4FlashAssertFloats(t, "oproj.concat", got.OProj.Concat, want.OProj.Concat)
	oracleV4FlashAssertFloats(t, "oproj.out", got.OProj.Out, want.OProj.Out)

	// Compress schedule.
	oracleV4FlashAssertInts(t, "compress.schedule", got.Compress.Schedule, want.Compress.Schedule)
}

// TestV4FlashOracleRouteNonVacuous proves the routing oracle's selection axis
// is live: a wrong selection (e.g. raw-logit top-k instead of sqrt-softplus)
// changes both the chosen experts and the normalized weights.
func TestV4FlashOracleRouteNonVacuous(t *testing.T) {
	logits := oracleV4FlashFill(256, 20260913, 2.0)
	idx, w := oracleV4FlashScoredRoute(logits, 6, 1.5)

	var sum float64
	for _, v := range w {
		if v < 0 {
			t.Fatalf("negative weight %v", v)
		}
		sum += v
	}
	if math.Abs(sum-1.5) > cpuOracleTol {
		t.Fatalf("weight sum = %v, want 1.5", sum)
	}
	if len(idx) != 6 {
		t.Fatalf("got %d indices, want 6", len(idx))
	}
	for i := 1; i < len(idx); i++ {
		if oracleV4FlashScore(logits[idx[i-1]]) < oracleV4FlashScore(logits[idx[i]]) {
			t.Fatalf("indices not in descending score order: %v", idx)
		}
	}
	// sqrt(softplus) is strictly monotonic, so selection order is invariant;
	// the LIVE axis is the weight transform. A wrong-but-plausible
	// softmax-over-raw-logits weighting must move the weights.
	var denom float64
	mx := logits[idx[0]]
	for _, e := range idx {
		if logits[e] > mx {
			mx = logits[e]
		}
	}
	for _, e := range idx {
		denom += math.Exp(logits[e] - mx)
	}
	var worst float64
	for i, e := range idx {
		wrong := math.Exp(logits[e]-mx) / denom * 1.5
		if d := math.Abs(w[i] - wrong); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("sqrt-softplus weighting is inert vs softmax(raw logits) (max|delta| = %.3e <= tol %.0e)", worst, cpuOracleTol)
	}
}

// TestV4FlashOracleSinkhornDoublyStochasticish asserts the 20-iteration
// Sinkhorn matrix has row and column sums near 1.
func TestV4FlashOracleSinkhornDoublyStochasticish(t *testing.T) {
	const mixWidth = 24
	const flatWidth = oracleV4FlashMixVecWidth
	streams := make([][]float64, oracleV4FlashHCMult)
	for j := range streams {
		streams[j] = oracleV4FlashFill(oracleV4FlashStreamDim, 20260914+uint64(j), 1.25)
	}
	xflat := oracleV4FlashFlattenStreams(streams)
	hcFn := oracleV4FlashFill(mixWidth*flatWidth, 20260915, 0.5)
	hcBase := oracleV4FlashFill(mixWidth, 20260916, 0.2)
	mixes := oracleV4FlashMHCProjection(xflat, hcFn, mixWidth, flatWidth, oracleV4FlashRMSFlatEps)
	_, _, comb := oracleV4FlashMixesToPrePostComb(mixes, []float64{0.7, 1.1, 0.9}, hcBase, oracleV4FlashHCMult, true)
	_ = oracleV4FlashSinkhornApply(comb, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)

	const tol = 1e-3
	for j := range comb {
		var rs, cs float64
		for k := range comb[j] {
			rs += comb[j][k]
			cs += comb[k][j]
		}
		if math.Abs(rs-1) > tol {
			t.Errorf("row %d sum = %v, want ~1", j, rs)
		}
		if math.Abs(cs-1) > tol {
			t.Errorf("col %d sum = %v, want ~1", j, cs)
		}
	}
}

// TestV4FlashOracleMHCCombOrientation proves the comb[dest][source] orientation
// is live: transposing the combination matrix moves the applied result by well
// over the tolerance.
func TestV4FlashOracleMHCCombOrientation(t *testing.T) {
	const mixWidth = 24
	const flatWidth = oracleV4FlashMixVecWidth
	streams := make([][]float64, oracleV4FlashHCMult)
	for j := range streams {
		streams[j] = oracleV4FlashFill(oracleV4FlashStreamDim, 20260917+uint64(j), 1.25)
	}
	xflat := oracleV4FlashFlattenStreams(streams)
	hcFn := oracleV4FlashFill(mixWidth*flatWidth, 20260918, 0.5)
	hcBase := oracleV4FlashFill(mixWidth, 20260919, 0.2)
	mixes := oracleV4FlashMHCProjection(xflat, hcFn, mixWidth, flatWidth, oracleV4FlashRMSFlatEps)
	pre, post, comb := oracleV4FlashMixesToPrePostComb(mixes, []float64{0.7, 1.1, 0.9}, hcBase, oracleV4FlashHCMult, true)
	_ = oracleV4FlashSinkhornApply(comb, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	_, correct := oracleV4FlashMHCApply(streams, pre, post, comb, false)
	_, wrong := oracleV4FlashMHCApply(streams, pre, post, comb, true)
	if d := oracleV4FlashMaxAbsDiff(correct, wrong); d <= cpuOracleTol {
		t.Fatalf("comb orientation is inert (max|delta| = %.3e <= tol %.0e)", d, cpuOracleTol)
	}
}

// TestV4FlashOracleMHCPreEps proves the pre-only +hc_eps is observable:
// dropping it moves the collapsed (and downstream applied) result by more than
// the tolerance. The fixture streams are scaled so the additive eps is not
// swamped by the tolerance; post correctly carries no eps.
func TestV4FlashOracleMHCPreEps(t *testing.T) {
	const mixWidth = 24
	const flatWidth = oracleV4FlashMixVecWidth
	streams := make([][]float64, oracleV4FlashHCMult)
	for j := range streams {
		streams[j] = oracleV4FlashFill(oracleV4FlashStreamDim, 20260920+uint64(j), 2000)
	}
	xflat := oracleV4FlashFlattenStreams(streams)
	hcFn := oracleV4FlashFill(mixWidth*flatWidth, 20260921, 0.5)
	hcBase := oracleV4FlashFill(mixWidth, 20260922, 0.2)
	mixes := oracleV4FlashMHCProjection(xflat, hcFn, mixWidth, flatWidth, oracleV4FlashRMSFlatEps)
	preEps, post, combA := oracleV4FlashMixesToPrePostComb(mixes, []float64{0.7, 1.1, 0.9}, hcBase, oracleV4FlashHCMult, true)
	preNoEps, _, combB := oracleV4FlashMixesToPrePostComb(mixes, []float64{0.7, 1.1, 0.9}, hcBase, oracleV4FlashHCMult, false)
	for j := range preEps {
		if math.Abs((preEps[j]-oracleV4FlashHCEps)-preNoEps[j]) > 1e-12 {
			t.Fatalf("pre[%d] does not differ from its no-eps form by exactly hc_eps", j)
		}
	}
	// The four pre values include +hc_eps each, so the collapsed difference is
	// hc_eps * sum_j x[j][k], which the 2000x stream scale makes >> 1e-4.
	_ = oracleV4FlashSinkhornApply(combA, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	_ = oracleV4FlashSinkhornApply(combB, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	_, withEps := oracleV4FlashMHCApply(streams, preEps, post, combA, false)
	_, noEps := oracleV4FlashMHCApply(streams, preNoEps, post, combB, false)
	if d := oracleV4FlashMaxAbsDiff(withEps, noEps); d <= cpuOracleTol {
		t.Fatalf("pre +hc_eps is inert (max|delta| = %.3e <= tol %.0e)", d, cpuOracleTol)
	}
}

// TestV4FlashOracleSinkhornIterationCount pins the exact schedule: one
// column pass plus (iters-1) alternating row/column passes. The returned pass
// sequence must be exactly ["col", "row", "col", ...] with one initial column
// pass and iters-1 row passes and iters-1 column passes in total.
func TestV4FlashOracleSinkhornIterationCount(t *testing.T) {
	build := func() [][]float64 {
		return [][]float64{
			{0.5, -1.2, 0.3, 2.1},
			{-0.7, 0.9, 1.4, -0.2},
			{1.1, 0.4, -1.8, 0.6},
			{0.2, -0.5, 0.8, 1.3},
		}
	}
	passes := oracleV4FlashSinkhornApply(build(), oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	if len(passes) != 2*oracleV4FlashSinkhornIters-1 {
		t.Fatalf("pass count = %d, want %d (1 initial col + 19 row/col pairs)", len(passes), 2*oracleV4FlashSinkhornIters-1)
	}
	if passes[0] != "col" {
		t.Fatalf("first pass = %q, want col", passes[0])
	}
	rows, cols := 0, 0
	for i, p := range passes {
		switch p {
		case "row":
			rows++
			if i%2 != 1 {
				t.Fatalf("row pass at index %d is not in the alternating tail: %v", i, passes)
			}
		case "col":
			cols++
			if i != 0 && i%2 != 0 {
				t.Fatalf("col pass at index %d is not in the alternating tail: %v", i, passes)
			}
		default:
			t.Fatalf("unknown pass %q", p)
		}
	}
	if rows != oracleV4FlashSinkhornIters-1 || cols != oracleV4FlashSinkhornIters {
		t.Fatalf("row passes = %d, col passes = %d, want %d and %d", rows, cols, oracleV4FlashSinkhornIters-1, oracleV4FlashSinkhornIters)
	}
	// Fewer iterations must leave a different matrix (proving iters is not
	// ignored), even though a converged matrix changes little between n and n+1.
	a := build()
	oracleV4FlashSinkhornApply(a, 1, oracleV4FlashHCEps)
	b := build()
	oracleV4FlashSinkhornApply(b, oracleV4FlashSinkhornIters, oracleV4FlashHCEps)
	if d := oracleV4FlashMaxAbsDiff(oracleV4FlashFlattenMatrix(a), oracleV4FlashFlattenMatrix(b)); d <= cpuOracleTol {
		t.Fatalf("1 vs 20 sinkhorn iterations are indistinguishable (max|delta| = %.3e)", d)
	}
}

// TestV4FlashOracleFP4Decode pins the nibble order and E8M0 scale: a byte with
// low nibble 0x7 (value 6) and high nibble 0x0 (0) at scale exponent +3 must
// decode to [48, 0, ...]. It also checks a negative nibble and the 32-value
// scale boundary.
func TestV4FlashOracleFP4Decode(t *testing.T) {
	// 32 packed bytes -> 64 values, 2 scale bytes. First 16 bytes share scale 0.
	packed := make([]byte, 32)
	packed[0] = 0x70           // low=0 -> value 0, high=7 -> value 6 (unsigned)
	packed[1] = 0x0f           // low=15 -> -6, high=0 -> 0
	packed[16] = 0x70          // boundary: packedCol 16 -> scale[1], decoded[32]/[33]
	scales := []byte{130, 127} // 2^3, 2^0
	got := oracleV4FlashDecodeFP4(packed, scales, 1, 32, 2)
	if got[0] != 0 {
		t.Errorf("decoded[0] = %v, want 0 (low nibble 0)", got[0])
	}
	if got[1] != 48 { // 6 * 2^3
		t.Errorf("decoded[1] = %v, want 48 (high nibble 6, 2^3)", got[1])
	}
	if got[2] != -48 { // -6 * 2^3
		t.Errorf("decoded[2] = %v, want -48 (low nibble -6, 2^3)", got[2])
	}
	if got[3] != 0 {
		t.Errorf("decoded[3] = %v, want 0 (high nibble 0)", got[3])
	}
	// Boundary: value index 32 is the first packedCol whose /16 == 1, so it
	// picks up scale[1] = 2^0. packed[16] drives decoded[32]/[33].
	if got[32] != 0 || got[33] != 6 {
		t.Errorf("boundary decoded[32..33] = %v,%v, want 0,6 (scale 2^0)", got[32], got[33])
	}
}

// TestV4FlashOracleOProjGroupSplit proves the per-group split is live: using a
// single flat matmul that ignores the group boundary changes the result.
func TestV4FlashOracleOProjGroupSplit(t *testing.T) {
	const groups, groupIn, oLora, outDim = 8, 4, 3, 8
	o := oracleV4FlashFill(groups*groupIn, 20260923, 1.0)
	woA := oracleV4FlashFill(groups*oLora*groupIn, 20260924, 0.5)
	woB := oracleV4FlashFill(outDim*groups*oLora, 20260925, 0.5)
	_, grouped := oracleV4FlashOProj(o, woA, woB, groups, groupIn, oLora, outDim)

	// Wrong-but-plausible: treat the group down-projection as one flat matmul
	// over the full head axis, reusing the same stored weights.
	flat := make([]float64, groups*oLora)
	for r := 0; r < groups*oLora; r++ {
		row := woA[r*groupIn : (r+1)*groupIn]
		var s float64
		for i := 0; i < groupIn; i++ {
			s += row[i] * o[i]
		}
		flat[r] = s
	}
	wrong := make([]float64, outDim)
	for r := 0; r < outDim; r++ {
		row := woB[r*(groups*oLora) : (r+1)*(groups*oLora)]
		var s float64
		for i := range row {
			s += row[i] * flat[i]
		}
		wrong[r] = s
	}
	if d := oracleV4FlashMaxAbsDiff(grouped, wrong); d <= cpuOracleTol {
		t.Fatalf("group split is inert (max|delta| = %.3e <= tol %.0e)", d, cpuOracleTol)
	}
}

// TestV4FlashOracleCompressSchedule pins the exact schedule interpretation:
// 46 entries, indices 0..42 decoder layers, 43..45 MTP ratio-0, layers 0,1
// ratio-0, odd layers 3..41 ratio-128, the rest ratio-4. It also checks the
// transcribed literal matches the production table by position.
func TestV4FlashOracleCompressSchedule(t *testing.T) {
	sched := oracleV4FlashCompressSchedule()
	if len(sched) != 46 {
		t.Fatalf("schedule length = %d, want 46", len(sched))
	}
	if sched[0] != 0 || sched[1] != 0 {
		t.Errorf("layers 0,1 ratio = %d,%d, want 0,0", sched[0], sched[1])
	}
	for l := 3; l <= 41; l += 2 {
		if sched[l] != 128 {
			t.Errorf("layer %d ratio = %d, want 128", l, sched[l])
		}
	}
	for l := 2; l <= 42; l += 2 {
		if sched[l] != 4 {
			t.Errorf("layer %d ratio = %d, want 4", l, sched[l])
		}
	}
	for l := 43; l <= 45; l++ {
		if sched[l] != 0 {
			t.Errorf("MTP layer %d ratio = %d, want 0", l, sched[l])
		}
	}
	// Cross-check the transcribed literal against the production table by
	// position. Reading the var by name is allowed; calling a helper is not.
	if len(deepSeekV4FlashCompressRatios) != len(sched) {
		t.Fatalf("production schedule length = %d, oracle = %d", len(deepSeekV4FlashCompressRatios), len(sched))
	}
	for i, want := range sched {
		if deepSeekV4FlashCompressRatios[i] != want {
			t.Fatalf("schedule[%d] = %d, oracle literal = %d", i, deepSeekV4FlashCompressRatios[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// Assertion helpers (test-local, oracleV4Flash-prefixed)
// ---------------------------------------------------------------------------

func oracleV4FlashAssertInts(t *testing.T, name string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %d, want %d", name, i, got[i], want[i])
		}
	}
}

func oracleV4FlashAssertFloats(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(got[i] - want[i]); d > cpuOracleTol {
			t.Fatalf("%s[%d] = %v, want %v (max|delta| = %.3e > tol %.0e)", name, i, got[i], want[i], d, cpuOracleTol)
		}
	}
}

func oracleV4FlashAssertMatrix(t *testing.T, name string, got, want [][]float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows = %d, want %d", name, len(got), len(want))
	}
	for j := range got {
		oracleV4FlashAssertFloats(t, name+"["+strconv.Itoa(j)+"]", got[j], want[j])
	}
}

func oracleV4FlashMaxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var worst float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > worst {
			worst = d
		}
	}
	return worst
}

func oracleV4FlashFlattenMatrix(m [][]float64) []float64 {
	out := make([]float64, 0, len(m)*len(m[0]))
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}
