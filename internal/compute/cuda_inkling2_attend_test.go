package compute

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// serialSoftmaxFP64 computes the serial FP64 reference softmax denominator and
// float32-normalized weights matching k_ink2_attend_exp + serial k_ink2_attend_out.
func serialSoftmaxFP64(scores []float64) (denom float64, weights []float32) {
	m := math.Inf(-1)
	for _, s := range scores {
		if s > m {
			m = s
		}
	}
	if m < 0 {
		m = 0
	}
	exps := make([]float64, len(scores))
	denom = math.Exp(-m)
	for j, s := range scores {
		e := math.Exp(s - m)
		exps[j] = e
		denom += e
	}
	weights = make([]float32, len(scores))
	for j, e := range exps {
		weights[j] = float32(e / denom)
	}
	return denom, weights
}

// treeSoftmaxFP64 models k_ink2_attend_out's parallel warp-shuffle + shared-memory
// tree-sum reduction: blockDim.x = groups*32 threads stride across keys, each warp
// reduces via __shfl_xor_sync(16,8,4,2,1), and thread 0 sums exp(-m) + wpart[0..groups-1].
func treeSoftmaxFP64(scores []float64, groups int) (denom float64, weights []float32) {
	m := math.Inf(-1)
	for _, s := range scores {
		if s > m {
			m = s
		}
	}
	if m < 0 {
		m = 0
	}
	exps := make([]float64, len(scores))
	for j, s := range scores {
		exps[j] = math.Exp(s - m)
	}

	threads := groups * 32
	threadPart := make([]float64, threads)
	for tid := 0; tid < threads; tid++ {
		var p float64
		for j := tid; j < len(exps); j += threads {
			p += exps[j]
		}
		threadPart[tid] = p
	}

	wpart := make([]float64, groups)
	for w := 0; w < groups; w++ {
		var lane [32]float64
		copy(lane[:], threadPart[w*32:(w+1)*32])
		for o := 16; o > 0; o >>= 1 {
			var next [32]float64
			for l := 0; l < 32; l++ {
				next[l] = lane[l] + lane[l^o]
			}
			lane = next
		}
		wpart[w] = lane[0]
	}

	denom = math.Exp(-m)
	for w := 0; w < groups; w++ {
		denom += wpart[w]
	}

	weights = make([]float32, len(exps))
	for j, e := range exps {
		weights[j] = float32(e / denom)
	}
	return denom, weights
}

// fak-test:runtime fast est=1s
func TestCUDAInk2AttendSoftmaxTreeSumParity(t *testing.T) {
	rng := rand.New(rand.NewSource(1946))

	makeScores := func(kind string, n int) []float64 {
		s := make([]float64, n)
		switch kind {
		case "peaked":
			for i := range s {
				s[i] = -6.0 + 0.5*rng.NormFloat64()
			}
			s[n-1] = 8.5
			if n > 10 {
				s[n/3] = 5.2
			}
		case "heavy_tail":
			for i := range s {
				s[i] = -2.0 + 2.5*rng.NormFloat64()
			}
		case "uniform":
			for i := range s {
				s[i] = 0.01*rng.Float64() - 0.005
			}
		case "extreme_neg":
			for i := range s {
				s[i] = -50.0 + 10.0*rng.NormFloat64()
			}
		case "extreme_pos":
			for i := range s {
				s[i] = 40.0 + 4.0*rng.NormFloat64()
			}
		}
		return s
	}

	depths := []int{1, 31, 32, 33, 128, 1946, 4096}
	groupCounts := []int{2, 4, 8} // headDim 256, 128, 64
	dists := []string{"peaked", "heavy_tail", "uniform", "extreme_neg", "extreme_pos"}

	for _, dist := range dists {
		for _, n := range depths {
			scores := makeScores(dist, n)
			refDenom, refW := serialSoftmaxFP64(scores)
			for _, groups := range groupCounts {
				treeDenom, treeW := treeSoftmaxFP64(scores, groups)
				relErr := math.Abs(treeDenom-refDenom) / refDenom
				if relErr > 1e-13 {
					t.Fatalf("%s n=%d groups=%d: denom relErr=%g > 1e-13 (ref=%g tree=%g)",
						dist, n, groups, relErr, refDenom, treeDenom)
				}
				var maxWDiff float64
				var kl float64
				for j := range refW {
					d := math.Abs(float64(treeW[j]) - float64(refW[j]))
					if d > maxWDiff {
						maxWDiff = d
					}
					p := float64(refW[j])
					q := float64(treeW[j])
					if p > 1e-12 && q > 1e-12 {
						kl += p * math.Log(p/q)
					}
				}
				if maxWDiff > 1e-7 {
					t.Fatalf("%s n=%d groups=%d: max weight diff=%g > 1e-7", dist, n, groups, maxWDiff)
				}
				if kl > 1e-12 {
					t.Fatalf("%s n=%d groups=%d: KL=%g > 1e-12", dist, n, groups, kl)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestCUDAInk2AttendOutUsesParallelDenominatorContract(t *testing.T) {
	srcBytes, err := os.ReadFile(filepath.Join("cuda", "inkling2_attend_split.cuh"))
	if err != nil {
		t.Fatalf("read inkling2_attend_split.cuh: %v", err)
	}
	src := string(srcBytes)
	start := strings.Index(src, "k_ink2_attend_out(")
	if start < 0 {
		t.Fatal("k_ink2_attend_out not found in inkling2_attend_split.cuh")
	}
	end := strings.Index(src[start:], "k_ink2_attend_out_tree(")
	if end < 0 {
		t.Fatal("k_ink2_attend_out_tree not found after k_ink2_attend_out")
	}
	body := src[start : start+end]

	if strings.Contains(body, "for (int j = 0; j < n; ++j) denom +=") {
		t.Fatal("k_ink2_attend_out still uses serial FP64 softmax denominator loop")
	}
	if !strings.Contains(body, "__shfl_xor_sync") {
		t.Fatal("k_ink2_attend_out missing __shfl_xor_sync warp reduction for denominator")
	}
	if !strings.Contains(body, "wpart[") {
		t.Fatal("k_ink2_attend_out missing wpart shared-memory warp-partial reduction")
	}
}

// TestCUDAInk2AttendDepth1946DeviceHook is the -tags cuda hardware verification hook
// for aegis3: it runs Ink2Attend at depth 1946 and checks output against the CPU FP64
// softmax reference. On non-CUDA hosts it skips cleanly.
// fak-test:runtime fast est=1s
func TestCUDAInk2AttendDepth1946DeviceHook(t *testing.T) {
	be := Pick("cuda")
	if be == nil || be.Name() != "cuda" {
		t.Skip("cuda backend not registered (run with -tags cuda on CUDA host)")
	}
	dev, ok := be.(Inkling2Device)
	if !ok {
		t.Skip("cuda backend does not implement Inkling2Device")
	}

	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	const capLen = 4096
	const depth = 1946
	kvDim := p.KVHeads * p.HeadDim

	alloc := func(n int) Tensor {
		tt, err := dev.Ink2Alloc(n)
		if err != nil {
			t.Fatalf("Ink2Alloc(%d): %v", n, err)
		}
		return tt
	}
	rng := rand.New(rand.NewSource(1946))
	fill := func(tt Tensor, n int, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(sd * rng.NormFloat64())
		}
		if err := dev.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatalf("Ink2Write: %v", err)
		}
		return v
	}

	q := alloc(p.Heads * p.HeadDim)
	r := alloc(p.Heads * p.RelLen)
	sinks := alloc(p.Heads)
	relBias := alloc(capLen * p.RelLen)
	kc := alloc(capLen * kvDim)
	vc := alloc(capLen * kvDim)
	scores := alloc(p.Heads * capLen * 2) // float64 slots
	out := alloc(p.Heads * p.HeadDim)
	meta := alloc(2)

	qh := fill(q, p.Heads*p.HeadDim, 0.5)
	rh := fill(r, p.Heads*p.RelLen, 0.4)
	sinksh := fill(sinks, p.Heads, 0.3)
	rbh := fill(relBias, capLen*p.RelLen, 0.2)
	kch := fill(kc, capLen*kvDim, 0.4)
	vch := fill(vc, capLen*kvDim, 0.4)
	if err := dev.Ink2Write(Ink2At(meta, 0), []float32{math.Float32frombits(uint32(depth - 1)), 0}); err != nil {
		t.Fatalf("Ink2Write meta: %v", err)
	}

	if err := dev.Ink2Attend(
		Ink2At(q, 0), Ink2At(r, 0), Ink2At(sinks, 0), Ink2At(relBias, 0),
		Ink2At(kc, 0), Ink2At(vc, 0), Ink2At(scores, 0), Ink2At(out, 0), Ink2At(meta, 0),
		p, capLen, 0, capLen,
	); err != nil {
		t.Fatalf("Ink2Attend: %v", err)
	}
	got, err := dev.Ink2Read(Ink2At(out, 0), p.Heads*p.HeadDim)
	if err != nil {
		t.Fatalf("Ink2Read: %v", err)
	}

	// Compare each head against CPU FP64 softmax reference.
	gqa := p.Heads / p.KVHeads
	var maxDiff float64
	for h := 0; h < p.Heads; h++ {
		kvh := h / gqa
		sc := make([]float64, depth)
		for j := 0; j < depth; j++ {
			var dot float32
			for i := 0; i < p.HeadDim; i++ {
				dot += qh[h*p.HeadDim+i] * kch[j*kvDim+kvh*p.HeadDim+i]
			}
			var rdot float32
			dist := (depth - 1) - j
			for i := 0; i < p.RelLen; i++ {
				rdot += rbh[dist*p.RelLen+i] * rh[h*p.RelLen+i]
			}
			sc[j] = float64(dot*p.Scale + (-sinksh[h] + rdot))
		}
		_, w := serialSoftmaxFP64(sc)
		for d := 0; d < p.HeadDim; d++ {
			var acc float32
			for j := 0; j < depth; j++ {
				acc += w[j] * vch[j*kvDim+kvh*p.HeadDim+d]
			}
			diff := math.Abs(float64(got[h*p.HeadDim+d]) - float64(acc))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	if maxDiff > 1e-4 {
		t.Fatalf("depth-1946 Ink2Attend max diff vs CPU FP64 ref = %g > 1e-4", maxDiff)
	}
}
