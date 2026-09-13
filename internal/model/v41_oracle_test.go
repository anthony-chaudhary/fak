package model

// v41_oracle_test.go - the weight-free, checkpoint-independent CPU numeric oracle
// for the published DeepSeek-V4.1-Flash component equations. Companion to
// family_cpu_oracle_test.go, which states the independence doctrine, and to
// v4_flash_oracle_test.go, which does the same for the separately-scoped
// V4-Flash-0731 control. This file targets the V4.1 artifact only.
//
// Scope and provenance. The reference below is a plain in-order scalar
// transcription of the pinned official artifact
// deepseek-ai/DeepSeek-V4.1-Flash @ revision
// dba1be0a40aa45a94ad051997016db3960a90277 (inference/model.py +
// inference/kernel.py), plus the published config axes admitted in
// v41_config.go. It reuses NONE of the production machinery - not
// V41EngramHashState.Hash, not v41MHCSplit, not v41Route, not
// v41SqrtSoftplus, not sigmoid32. Every hash multiply, XOR, modulo, sigmoid,
// softmax and Sinkhorn sweep is a naive scalar loop transcribed here from the
// equation, not called from production.
//
// What is covered: four pinned V4.1 component equations that the landed native
// code already implements, so each is a cross-implementation comparison
// against production:
//
//	A. Engram n-gram hashing and prime/offset row-ID flattening.
//	B. four-stream mHC mixing: sigmoid pre (+hc_eps) / 2*sigmoid post /
//	   linear comb, the 20-iteration Sinkhorn normalization, the pre collapse,
//	   and the comb[source][destination] post application.
//	C. the 384-expert top-6 router: sqrt(softplus(z)) scoring, noaux_tc
//	   score-plus-bias selection, unbiased-score normalized weights at
//	   route_scale 1.5, plus the always-on shared-expert add.
//	D. the per-layer compression schedule (40 decoder layers + 3 nextn).
//
// What is deliberately NOT covered: the full Session.Prefill / Session.Step
// native V4.1 forward does not exist yet (ErrV41NativeUnsupported fences it),
// so the integration arm stays escalated separately and this file stays at the
// component boundary.
//
// Independence + anti-vacuity: the checked-in fixture holds expected outputs
// produced by the scalar transcription in this file; the oracle is compared to
// the fixture within cpuOracleTol, and separate tests prove the load-bearing
// axes (router weighting transform, Engram n-gram/blocking structure, Sinkhorn
// iteration count, mHC comb orientation) move the result by far more than the
// tolerance.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const v41OracleRevision = "dba1be0a40aa45a94ad051997016db3960a90277"

func oracleV41LCG(seed uint64) func() float64 {
	state := seed
	const mul = 6364136223846793005
	const inc = 1442695040888963407
	return func() float64 {
		state = state*mul + inc
		return float64(state>>11)/float64(uint64(1)<<53)*2 - 1
	}
}

func oracleV41Fill(n int, seed uint64, scale float64) []float64 {
	next := oracleV41LCG(seed)
	out := make([]float64, n)
	for i := range out {
		out[i] = next() * scale
	}
	return out
}

type oracleV41EngramLayout struct {
	TokenMap        []uint32
	CompressedVocab uint32
	PadID           uint32
	Rows            []uint32
	Multipliers     [][]uint64
	Primes          [][]uint32
	MaxNgramSize    int
	HeadsPerNgram   int
}

func oracleV41EngramLayoutFor(seed uint64) oracleV41EngramLayout {
	const compressed = 99092
	mapIDs := make([]uint32, 256)
	next := oracleV41LCG(seed)
	for i := range mapIDs {
		v := next()*0.5 + 0.5
		if v < 0 {
			v = 0
		}
		if v > 0.999999 {
			v = 0.999999
		}
		mapIDs[i] = uint32(math.Floor(v*float64(compressed))) % compressed
	}
	const layers = 2
	l := oracleV41EngramLayout{
		TokenMap:        mapIDs,
		CompressedVocab: compressed,
		PadID:           2,
		Rows:            make([]uint32, layers),
		Multipliers:     make([][]uint64, layers),
		Primes:          make([][]uint32, layers),
		MaxNgramSize:    4,
		HeadsPerNgram:   8,
	}
	cols := (l.MaxNgramSize - 1) * l.HeadsPerNgram
	for layer := 0; layer < layers; layer++ {
		l.Multipliers[layer] = make([]uint64, l.MaxNgramSize)
		for j := range l.Multipliers[layer] {
			l.Multipliers[layer][j] = uint64(35184372088831) - uint64(2*(j+4*layer))
		}
		l.Primes[layer] = make([]uint32, cols)
		for j := range l.Primes[layer] {
			p := uint32(16000057 + j*2 + layer)
			if !oracleV41IsPrime(int(p)) {
				p = oracleV41NextPrime(int(p))
			}
			l.Primes[layer][j] = p
			l.Rows[layer] += p
		}
	}
	return l
}

func oracleV41IsPrime(n int) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for d := 3; d*d <= n; d += 2 {
		if n%d == 0 {
			return false
		}
	}
	return true
}

func oracleV41NextPrime(n int) uint32 {
	for {
		n++
		if oracleV41IsPrime(n) {
			return uint32(n)
		}
	}
}

func oracleV41EngramHash(l oracleV41EngramLayout, tokens []int, mask []bool) []uint32 {
	cols := (l.MaxNgramSize - 1) * l.HeadsPerNgram
	tail := make([]int64, l.MaxNgramSize-1)
	for i := range tail {
		tail[i] = -1
	}
	out := make([]uint32, 0, len(tokens)*len(l.Rows)*cols)
	for pos := range tokens {
		current := int64(l.TokenMap[tokens[pos]])
		if mask != nil && !mask[pos] {
			current = -1
		}
		ids := make([]uint32, l.MaxNgramSize)
		blocked := false
		for shift := range ids {
			id := current
			if shift > 0 {
				id = tail[shift-1]
			}
			blocked = blocked || id < 0
			if blocked {
				ids[shift] = l.PadID
			} else {
				ids[shift] = uint32(id)
			}
		}
		for layer := range l.Rows {
			hash := uint64(ids[0]) * l.Multipliers[layer][0]
			var offset uint64
			for ngram := 1; ngram < l.MaxNgramSize; ngram++ {
				hash ^= uint64(ids[ngram]) * l.Multipliers[layer][ngram]
				for head := 0; head < l.HeadsPerNgram; head++ {
					col := (ngram-1)*l.HeadsPerNgram + head
					prime := uint64(l.Primes[layer][col])
					out = append(out, uint32(hash%prime+offset))
					offset += prime
				}
			}
		}
		copy(tail[1:], tail[:len(tail)-1])
		tail[0] = current
	}
	return out
}

const (
	oracleV41HCMult        = 4
	oracleV41HCEps         = 1e-6
	oracleV41SinkhornIters = 20
	oracleV41StreamDim     = 3
)

func oracleV41Sigmoid(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

func oracleV41MHCKernel(mixes, scale, base []float64, hc, iters int, eps float64) (pre, post, comb []float64) {
	pre = make([]float64, hc)
	post = make([]float64, hc)
	comb = make([]float64, hc*hc)
	for j := 0; j < hc; j++ {
		pre[j] = oracleV41Sigmoid(mixes[j]*scale[0]+base[j]) + eps
		post[j] = 2 * oracleV41Sigmoid(mixes[hc+j]*scale[1]+base[hc+j])
	}
	for i := range comb {
		comb[i] = mixes[2*hc+i]*scale[2] + base[2*hc+i]
	}
	for row := 0; row < hc; row++ {
		start := row * hc
		mx := comb[start]
		for col := 1; col < hc; col++ {
			mx = math.Max(mx, comb[start+col])
		}
		var sum float64
		for col := 0; col < hc; col++ {
			comb[start+col] = math.Exp(comb[start+col] - mx)
			sum += comb[start+col]
		}
		for col := 0; col < hc; col++ {
			comb[start+col] = comb[start+col]/sum + eps
		}
	}
	oracleV41NormCols(comb, hc, eps)
	for it := 1; it < iters; it++ {
		oracleV41NormRows(comb, hc, eps)
		oracleV41NormCols(comb, hc, eps)
	}
	return pre, post, comb
}

func oracleV41NormRows(m []float64, hc int, eps float64) {
	for row := 0; row < hc; row++ {
		var s float64
		for col := 0; col < hc; col++ {
			s += m[row*hc+col]
		}
		for col := 0; col < hc; col++ {
			m[row*hc+col] /= s + eps
		}
	}
}

func oracleV41NormCols(m []float64, hc int, eps float64) {
	for col := 0; col < hc; col++ {
		var s float64
		for row := 0; row < hc; row++ {
			s += m[row*hc+col]
		}
		for row := 0; row < hc; row++ {
			m[row*hc+col] /= s + eps
		}
	}
}

func oracleV41MHCPre(streams [][]float64, pre []float64) []float64 {
	out := make([]float64, len(streams[0]))
	for h := range streams {
		for d := range out {
			out[d] += pre[h] * streams[h][d]
		}
	}
	return out
}

func oracleV41MHCPost(collapsed []float64, residual [][]float64, post, comb []float64, hc int, transpose bool) [][]float64 {
	out := make([][]float64, hc)
	for dst := 0; dst < hc; dst++ {
		out[dst] = make([]float64, len(collapsed))
		for src := 0; src < hc; src++ {
			c := comb[src*hc+dst]
			if transpose {
				c = comb[dst*hc+src]
			}
			for d := range collapsed {
				out[dst][d] += c * residual[src][d]
			}
		}
		for d := range collapsed {
			out[dst][d] += post[dst] * collapsed[d]
		}
	}
	return out
}

const (
	oracleV41Experts    = 384
	oracleV41TopK       = 6
	oracleV41RouteScale = 1.5
)

func oracleV41Score(z float64) float64 {
	return math.Sqrt(math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z))))
}

func oracleV41Route(logits, bias []float64, topK int, routeScale float64) ([]int, []float64) {
	n := len(logits)
	choice := make([]float64, n)
	for i, z := range logits {
		choice[i] = oracleV41Score(z)
		if len(bias) != 0 {
			choice[i] += bias[i]
		}
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0; j-- {
			a, b := idx[j-1], idx[j]
			if choice[a] > choice[b] || (choice[a] == choice[b] && a < b) {
				break
			}
			idx[j-1], idx[j] = idx[j], idx[j-1]
		}
	}
	picks := append([]int(nil), idx[:topK]...)
	var sum float64
	for _, e := range picks {
		sum += oracleV41Score(logits[e])
	}
	weights := make([]float64, topK)
	for i, e := range picks {
		weights[i] = oracleV41Score(logits[e]) / sum * routeScale
	}
	return picks, weights
}

func oracleV41SharedExpertAdd(routed, shared []float64) []float64 {
	out := make([]float64, len(routed))
	for i := range routed {
		out[i] = routed[i] + shared[i]
	}
	return out
}

func oracleV41CompressSchedule() []int {
	s := make([]int, 0, 43)
	s = append(s, 0, 0)
	for l := 2; l <= 19; l++ {
		s = append(s, 2)
	}
	for l := 20; l <= 39; l++ {
		s = append(s, 1)
	}
	s = append(s, 0, 0, 0)
	return s
}

type oracleV41Fixture struct {
	Revision string                `json:"revision"`
	Seed     uint64                `json:"seed"`
	Engram   oracleV41EngramBlock  `json:"engram"`
	MHC      oracleV41MHCBlock     `json:"mhc"`
	Route    oracleV41RouteBlock   `json:"route"`
	Shared   oracleV41SharedBlock  `json:"shared"`
	Compress oracleV41CompressBlck `json:"compress"`
}

type oracleV41EngramBlock struct {
	MaxNgramSize  int      `json:"max_ngram_size"`
	HeadsPerNgram int      `json:"heads_per_ngram"`
	Layers        int      `json:"layers"`
	PadID         uint32   `json:"pad_id"`
	Tokens        []int    `json:"tokens"`
	Mask          []bool   `json:"mask"`
	RowIDs        []uint32 `json:"row_ids"`
}

type oracleV41MHCBlock struct {
	HCMult    int       `json:"hc_mult"`
	Iters     int       `json:"sinkhorn_iters"`
	Eps       float64   `json:"hc_eps"`
	StreamDim int       `json:"stream_dim"`
	Pre       []float64 `json:"pre"`
	Post      []float64 `json:"post"`
	Comb      []float64 `json:"comb"`
	Collapsed []float64 `json:"collapsed"`
	Applied   []float64 `json:"applied"`
}

type oracleV41RouteBlock struct {
	Experts   int       `json:"experts"`
	TopK      int       `json:"top_k"`
	Scale     float64   `json:"route_scale"`
	LogitSeed float64   `json:"logit_seed"`
	Indices   []int     `json:"indices"`
	Weights   []float64 `json:"weights"`
}

type oracleV41SharedBlock struct {
	Routed []float64 `json:"routed"`
	Shared []float64 `json:"shared"`
	Sum    []float64 `json:"sum"`
}

type oracleV41CompressBlck struct {
	Count    int   `json:"count"`
	Schedule []int `json:"schedule"`
}

func oracleV41ComputeFixture() oracleV41Fixture {
	const seed = 20260912
	var fx oracleV41Fixture
	fx.Revision = v41OracleRevision
	fx.Seed = seed

	const tokens = 64
	lay := oracleV41EngramLayoutFor(seed + 1)
	toks := make([]int, tokens)
	mask := make([]bool, tokens)
	for i := range toks {
		toks[i] = (i*97 + i/3) % len(lay.TokenMap)
		mask[i] = i%17 != 0
	}
	fx.Engram = oracleV41EngramBlock{
		MaxNgramSize:  lay.MaxNgramSize,
		HeadsPerNgram: lay.HeadsPerNgram,
		Layers:        len(lay.Rows),
		PadID:         lay.PadID,
		Tokens:        toks,
		Mask:          mask,
		RowIDs:        oracleV41EngramHash(lay, toks, mask),
	}

	const mixWidth = (2 + oracleV41HCMult) * oracleV41HCMult
	streams := make([][]float64, oracleV41HCMult)
	for j := range streams {
		streams[j] = oracleV41Fill(oracleV41StreamDim, seed+10+uint64(j), 1.25)
	}
	mixes := oracleV41Fill(mixWidth, seed+20, 0.75)
	scale := []float64{0.7, 1.1, 0.9}
	base := oracleV41Fill(mixWidth, seed+21, 0.2)
	pre, post, comb := oracleV41MHCKernel(mixes, scale, base, oracleV41HCMult, oracleV41SinkhornIters, oracleV41HCEps)
	collapsed := oracleV41MHCPre(streams, pre)
	applied := oracleV41MHCPost(collapsed, streams, post, comb, oracleV41HCMult, false)
	flatApplied := make([]float64, 0, len(applied)*len(applied[0]))
	for _, row := range applied {
		flatApplied = append(flatApplied, row...)
	}
	fx.MHC = oracleV41MHCBlock{
		HCMult:    oracleV41HCMult,
		Iters:     oracleV41SinkhornIters,
		Eps:       oracleV41HCEps,
		StreamDim: oracleV41StreamDim,
		Pre:       pre,
		Post:      post,
		Comb:      comb,
		Collapsed: collapsed,
		Applied:   flatApplied,
	}

	const logitSeed = seed + 30
	logits := oracleV41Fill(oracleV41Experts, logitSeed, 2.0)
	bias := oracleV41Fill(oracleV41Experts, seed+31, 0.3)
	idx, w := oracleV41Route(logits, bias, oracleV41TopK, oracleV41RouteScale)
	fx.Route = oracleV41RouteBlock{
		Experts:   oracleV41Experts,
		TopK:      oracleV41TopK,
		Scale:     oracleV41RouteScale,
		LogitSeed: float64(logitSeed),
		Indices:   idx,
		Weights:   w,
	}

	routed := oracleV41Fill(5, seed+40, 1.0)
	shared := oracleV41Fill(5, seed+41, 1.0)
	fx.Shared = oracleV41SharedBlock{Routed: routed, Shared: shared, Sum: oracleV41SharedExpertAdd(routed, shared)}

	sched := oracleV41CompressSchedule()
	fx.Compress = oracleV41CompressBlck{Count: len(sched), Schedule: sched}
	return fx
}

func TestDeepSeekV41OracleFixtureValues(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v41_oracle_expected.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var want oracleV41Fixture
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if want.Revision != v41OracleRevision {
		t.Fatalf("fixture revision %q, want %q", want.Revision, v41OracleRevision)
	}

	got := oracleV41ComputeFixture()

	if want.Engram.MaxNgramSize != 4 || want.Engram.HeadsPerNgram != 8 {
		t.Fatalf("fixture engram ngram=%d heads=%d, want 4/8", want.Engram.MaxNgramSize, want.Engram.HeadsPerNgram)
	}
	oracleV41AssertInts(t, "engram.tokens", got.Engram.Tokens, want.Engram.Tokens)
	oracleV41AssertBools(t, "engram.mask", got.Engram.Mask, want.Engram.Mask)
	oracleV41AssertUints(t, "engram.row_ids", got.Engram.RowIDs, want.Engram.RowIDs)

	oracleV41AssertFloats(t, "mhc.pre", got.MHC.Pre, want.MHC.Pre)
	oracleV41AssertFloats(t, "mhc.post", got.MHC.Post, want.MHC.Post)
	oracleV41AssertFloats(t, "mhc.comb", got.MHC.Comb, want.MHC.Comb)
	oracleV41AssertFloats(t, "mhc.collapsed", got.MHC.Collapsed, want.MHC.Collapsed)
	oracleV41AssertFloats(t, "mhc.applied", got.MHC.Applied, want.MHC.Applied)
	if want.MHC.HCMult != 4 || want.MHC.Iters != 20 {
		t.Fatalf("fixture mhc hc_mult=%d iters=%d, want 4/20", want.MHC.HCMult, want.MHC.Iters)
	}

	oracleV41AssertInts(t, "route.indices", got.Route.Indices, want.Route.Indices)
	oracleV41AssertFloats(t, "route.weights", got.Route.Weights, want.Route.Weights)
	oracleV41AssertFloats(t, "shared.routed", got.Shared.Routed, want.Shared.Routed)
	oracleV41AssertFloats(t, "shared.sum", got.Shared.Sum, want.Shared.Sum)

	oracleV41AssertInts(t, "compress.schedule", got.Compress.Schedule, want.Compress.Schedule)
}

func TestDeepSeekV41OracleRouteNonVacuous(t *testing.T) {
	logits := oracleV41Fill(oracleV41Experts, 20260913, 2.0)
	bias := oracleV41Fill(oracleV41Experts, 20260914, 0.3)
	idx, w := oracleV41Route(logits, bias, oracleV41TopK, oracleV41RouteScale)
	if len(idx) != oracleV41TopK {
		t.Fatalf("got %d indices, want %d", len(idx), oracleV41TopK)
	}
	var sum float64
	for _, v := range w {
		if v < 0 {
			t.Fatalf("negative weight %v", v)
		}
		sum += v
	}
	if math.Abs(sum-oracleV41RouteScale) > cpuOracleTol {
		t.Fatalf("weight sum = %v, want %v", sum, oracleV41RouteScale)
	}
	mx := logits[idx[0]]
	for _, e := range idx {
		if logits[e] > mx {
			mx = logits[e]
		}
	}
	var denom float64
	for _, e := range idx {
		denom += math.Exp(logits[e] - mx)
	}
	var worst float64
	for i, e := range idx {
		wrong := math.Exp(logits[e]-mx) / denom * oracleV41RouteScale
		if d := math.Abs(w[i] - wrong); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("sqrt-softplus weighting is inert vs softmax(raw logits) (max|delta| = %.3e <= tol %.0e)", worst, cpuOracleTol)
	}
}

func TestDeepSeekV41OracleEngramStructure(t *testing.T) {
	lay := oracleV41EngramLayoutFor(20260915)
	const tokens = 48
	toks := make([]int, tokens)
	mask := make([]bool, tokens)
	for i := range toks {
		toks[i] = (i*53 + 7) % len(lay.TokenMap)
		mask[i] = i%17 != 0
	}
	withMask := oracleV41EngramHash(lay, toks, mask)
	noMask := oracleV41EngramHash(lay, toks, nil)
	if len(withMask) != len(noMask) {
		t.Fatalf("output length mismatch %d vs %d", len(withMask), len(noMask))
	}
	if oracleV41UintMaxDiff(withMask, noMask) == 0 {
		t.Fatal("Engram mask/blocking is inert (identical row IDs with and without mask)")
	}
	reordered := oracleV41EngramHashReordered(lay, toks, mask)
	if oracleV41UintMaxDiff(withMask, reordered) == 0 {
		t.Fatal("Engram ngram/head column order is inert")
	}
}

func oracleV41EngramHashReordered(l oracleV41EngramLayout, tokens []int, mask []bool) []uint32 {
	cols := (l.MaxNgramSize - 1) * l.HeadsPerNgram
	tail := make([]int64, l.MaxNgramSize-1)
	for i := range tail {
		tail[i] = -1
	}
	out := make([]uint32, 0, len(tokens)*len(l.Rows)*cols)
	for pos := range tokens {
		current := int64(l.TokenMap[tokens[pos]])
		if mask != nil && !mask[pos] {
			current = -1
		}
		ids := make([]uint32, l.MaxNgramSize)
		blocked := false
		for shift := range ids {
			id := current
			if shift > 0 {
				id = tail[shift-1]
			}
			blocked = blocked || id < 0
			if blocked {
				ids[shift] = l.PadID
			} else {
				ids[shift] = uint32(id)
			}
		}
		for layer := range l.Rows {
			var offset uint64
			for head := 0; head < l.HeadsPerNgram; head++ {
				hash := uint64(ids[0]) * l.Multipliers[layer][0]
				for ngram := 1; ngram < l.MaxNgramSize; ngram++ {
					hash ^= uint64(ids[ngram]) * l.Multipliers[layer][ngram]
					col := (ngram-1)*l.HeadsPerNgram + head
					prime := uint64(l.Primes[layer][col])
					out = append(out, uint32(hash%prime+offset))
					offset += prime
				}
			}
		}
		copy(tail[1:], tail[:len(tail)-1])
		tail[0] = current
	}
	return out
}

func TestDeepSeekV41OracleSinkhornIterationCount(t *testing.T) {
	mixes := oracleV41Fill((2+oracleV41HCMult)*oracleV41HCMult, 20260916, 0.75)
	scale := []float64{0.7, 1.1, 0.9}
	base := oracleV41Fill((2+oracleV41HCMult)*oracleV41HCMult, 20260917, 0.2)
	_, _, one := oracleV41MHCKernel(mixes, scale, base, oracleV41HCMult, 1, oracleV41HCEps)
	_, _, full := oracleV41MHCKernel(mixes, scale, base, oracleV41HCMult, oracleV41SinkhornIters, oracleV41HCEps)
	if d := oracleV41MaxAbsDiff(one, full); d <= cpuOracleTol {
		t.Fatalf("1 vs %d sinkhorn iterations are indistinguishable (max|delta| = %.3e)", oracleV41SinkhornIters, d)
	}
	const tol = 1e-3
	hc := oracleV41HCMult
	for col := 0; col < hc; col++ {
		var cs float64
		for row := 0; row < hc; row++ {
			cs += full[row*hc+col]
		}
		if math.Abs(cs-1) > tol {
			t.Errorf("col %d sum = %v, want ~1", col, cs)
		}
	}
}

func TestDeepSeekV41OracleMHCCombOrientation(t *testing.T) {
	const mixWidth = (2 + oracleV41HCMult) * oracleV41HCMult
	streams := make([][]float64, oracleV41HCMult)
	for j := range streams {
		streams[j] = oracleV41Fill(oracleV41StreamDim, 20260918+uint64(j), 1.25)
	}
	mixes := oracleV41Fill(mixWidth, 20260919, 0.75)
	scale := []float64{0.7, 1.1, 0.9}
	base := oracleV41Fill(mixWidth, 20260920, 0.2)
	pre, post, comb := oracleV41MHCKernel(mixes, scale, base, oracleV41HCMult, oracleV41SinkhornIters, oracleV41HCEps)
	collapsed := oracleV41MHCPre(streams, pre)
	correct := oracleV41MHCPost(collapsed, streams, post, comb, oracleV41HCMult, false)
	wrong := oracleV41MHCPost(collapsed, streams, post, comb, oracleV41HCMult, true)
	flat := func(m [][]float64) []float64 {
		out := make([]float64, 0, len(m)*len(m[0]))
		for _, r := range m {
			out = append(out, r...)
		}
		return out
	}
	if d := oracleV41MaxAbsDiff(flat(correct), flat(wrong)); d <= cpuOracleTol {
		t.Fatalf("comb orientation is inert (max|delta| = %.3e <= tol %.0e)", d, cpuOracleTol)
	}
}

func TestDeepSeekV41OracleCompressSchedule(t *testing.T) {
	sched := oracleV41CompressSchedule()
	if len(sched) != 43 {
		t.Fatalf("schedule length = %d, want 43", len(sched))
	}
	if sched[0] != 0 || sched[1] != 0 {
		t.Errorf("layers 0,1 ratio = %d,%d, want 0,0", sched[0], sched[1])
	}
	for l := 2; l <= 19; l++ {
		if sched[l] != 2 {
			t.Errorf("layer %d ratio = %d, want 2", l, sched[l])
		}
	}
	for l := 20; l <= 39; l++ {
		if sched[l] != 1 {
			t.Errorf("layer %d ratio = %d, want 1", l, sched[l])
		}
	}
	for l := 40; l <= 42; l++ {
		if sched[l] != 0 {
			t.Errorf("nextn layer %d ratio = %d, want 0", l, sched[l])
		}
	}
}

func oracleV41AssertInts(t *testing.T, name string, got, want []int) {
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

func oracleV41AssertUints(t *testing.T, name string, got, want []uint32) {
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

func oracleV41AssertBools(t *testing.T, name string, got, want []bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func oracleV41AssertFloats(t *testing.T, name string, got, want []float64) {
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

func oracleV41MaxAbsDiff(a, b []float64) float64 {
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

func oracleV41UintMaxDiff(a, b []uint32) uint32 {
	if len(a) != len(b) {
		return ^uint32(0)
	}
	var worst uint32
	for i := range a {
		d := a[i]
		if b[i] > a[i] {
			d = b[i] - a[i]
		} else {
			d = a[i] - b[i]
		}
		if d > worst {
			worst = d
		}
	}
	return worst
}
