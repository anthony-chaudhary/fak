//go:build darwin && arm64 && cgo

package compute

import (
	"math"
	"strconv"
	"testing"
)

// metal_test.go — the on-box witness that the Metal backend runs a real forward pass on the
// Apple-Silicon GPU and matches the cpuref Reference within the Approx gate (argmax-exact +
// logit-cosine >= threshold), NOT bit-identity (RequireReference keeps the device off the
// exact rungs). Compiled by default on Apple Silicon when cgo is enabled; skips cleanly if no Metal
// device is registered. Reuses the package test helpers cosine/lcg/randVec/argmaxF32
// (compute_test.go). The synth harness is mtl-prefixed so it never collides with the
// cuda-tagged copy in a hypothetical combined-tags build.

func metalOrSkip(t *testing.T) *metalBackend {
	be := Pick("metal")
	mb, ok := be.(*metalBackend)
	if !ok {
		t.Skip("metal backend not registered (no reachable Metal device)")
	}
	if RequireReference(mb) {
		t.Fatal("metal backend must be Approx, not Reference")
	}
	return mb
}

func mtlRscale(s *lcg, n int, scale float32) []float32 {
	v := randVec(s, n) // ~[-0.5,0.5)
	for i := range v {
		v[i] *= scale
	}
	return v
}

// mtlMkResident puts host data onto a backend uniformly: cpuref.Upload is identity over the
// host slice; metal.Upload copies to a device buffer. So the same []float32 feeds both.
func mtlMkResident(be Backend, shape []int, data []float32) Tensor {
	return be.Upload(NewF32(be, shape, data), F32)
}

type mtlSynthCfg struct {
	H, L, nH, nKV, hd, I, vocab int
	eps                         float32
	theta                       float64
}

func (c mtlSynthCfg) grp() int       { return c.nH / c.nKV }
func (c mtlSynthCfg) scale() float32 { return float32(1.0 / math.Sqrt(float64(c.hd))) }

// mtlSynthModel holds one backend's resident weights + a live KV cache, and steps tokens
// through the exact Llama-decode op chain the model's tokenHidden uses — but expressed
// entirely through the Backend interface, so cpuref and metal run the SAME chain.
type mtlSynthModel struct {
	be  Backend
	cfg mtlSynthCfg
	W   map[string]Tensor
	kv  KVStore
	pos int
	emb []float32 // host embedding rows [vocab*H] (gather stays host)
}

func mtlNewSynth(be Backend, cfg mtlSynthCfg, host map[string][]float32) *mtlSynthModel {
	W := map[string]Tensor{
		"norm": mtlMkResident(be, []int{cfg.H}, host["norm"]),
		"head": mtlMkResident(be, []int{cfg.vocab, cfg.H}, host["head"]),
	}
	for l := 0; l < cfg.L; l++ {
		s := strconv.Itoa(l)
		W["in"+s] = mtlMkResident(be, []int{cfg.H}, host["in"+s])
		W["post"+s] = mtlMkResident(be, []int{cfg.H}, host["post"+s])
		W["q"+s] = mtlMkResident(be, []int{cfg.nH * cfg.hd, cfg.H}, host["q"+s])
		W["k"+s] = mtlMkResident(be, []int{cfg.nKV * cfg.hd, cfg.H}, host["k"+s])
		W["v"+s] = mtlMkResident(be, []int{cfg.nKV * cfg.hd, cfg.H}, host["v"+s])
		W["o"+s] = mtlMkResident(be, []int{cfg.H, cfg.nH * cfg.hd}, host["o"+s])
		W["gate"+s] = mtlMkResident(be, []int{cfg.I, cfg.H}, host["gate"+s])
		W["up"+s] = mtlMkResident(be, []int{cfg.I, cfg.H}, host["up"+s])
		W["down"+s] = mtlMkResident(be, []int{cfg.H, cfg.I}, host["down"+s])
	}
	kv := be.NewKV(KVConfig{NumLayers: cfg.L, NumKVHeads: cfg.nKV, HeadDim: cfg.hd, RopeTheta: cfg.theta})
	return &mtlSynthModel{be: be, cfg: cfg, W: W, kv: kv, pos: 0, emb: host["emb"]}
}

// step runs one token through all layers (decode path), appends K/V, returns next logits.
func (m *mtlSynthModel) step(id int) []float32 {
	be, c := m.be, m.cfg
	pos := m.pos
	row := m.emb[id*c.H : (id+1)*c.H]
	x := mtlMkResident(be, []int{c.H}, append([]float32(nil), row...))
	for l := 0; l < c.L; l++ {
		s := strconv.Itoa(l)
		xn := be.RMSNorm(x, m.W["in"+s], c.eps)
		q := be.MatMul(m.W["q"+s], xn)
		k := be.MatMul(m.W["k"+s], xn)
		v := be.MatMul(m.W["v"+s], xn)
		qr := be.RoPE(q, pos, c.nH, c.hd, c.theta)
		kr := be.RoPE(k, pos, c.nKV, c.hd, c.theta)
		m.kv.AppendKV(l, k, kr, v, pos) // kRaw=k (pre-RoPE), kRoPE=kr
		attn := be.Attention(qr, m.kv, l, true, c.grp(), c.scale())
		o := be.MatMul(m.W["o"+s], attn)
		be.AddInPlace(x, o)
		xn2 := be.RMSNorm(x, m.W["post"+s], c.eps)
		g := be.MatMul(m.W["gate"+s], xn2)
		u := be.MatMul(m.W["up"+s], xn2)
		gg := be.SwiGLU(g, u)
		down := be.MatMul(m.W["down"+s], gg)
		be.AddInPlace(x, down)
	}
	xf := be.RMSNorm(x, m.W["norm"], c.eps)
	logits := be.MatMul(m.W["head"], xf)
	m.pos++
	return be.Read(logits)
}

func mtlSynthHostWeights(cfg mtlSynthCfg) map[string][]float32 {
	var seed lcg = 0x1234567
	g := &seed
	h := map[string][]float32{
		"emb":  mtlRscale(g, cfg.vocab*cfg.H, 0.20),
		"norm": mtlRscale(g, cfg.H, 0.20),
		"head": mtlRscale(g, cfg.vocab*cfg.H, 0.20),
	}
	for i := range h["norm"] { // norm gains ~1, like a real model
		h["norm"][i] += 1.0
	}
	for l := 0; l < cfg.L; l++ {
		s := strconv.Itoa(l)
		in := mtlRscale(g, cfg.H, 0.2)
		post := mtlRscale(g, cfg.H, 0.2)
		for i := range in {
			in[i] += 1.0
			post[i] += 1.0
		}
		h["in"+s] = in
		h["post"+s] = post
		h["q"+s] = mtlRscale(g, cfg.nH*cfg.hd*cfg.H, 0.10)
		h["k"+s] = mtlRscale(g, cfg.nKV*cfg.hd*cfg.H, 0.10)
		h["v"+s] = mtlRscale(g, cfg.nKV*cfg.hd*cfg.H, 0.10)
		h["o"+s] = mtlRscale(g, cfg.H*cfg.nH*cfg.hd, 0.10)
		h["gate"+s] = mtlRscale(g, cfg.I*cfg.H, 0.10)
		h["up"+s] = mtlRscale(g, cfg.I*cfg.H, 0.10)
		h["down"+s] = mtlRscale(g, cfg.H*cfg.I, 0.10)
	}
	return h
}

// TestMetalMatMulApproxMatchesRef — op-level first light: a single device MPS SGEMM vs the
// cpuref fdot matmul on the same random W,x. Approx (MPS reduction order differs), so the
// gate is cosine, not equality.
func TestMetalMatMulApproxMatchesRef(t *testing.T) {
	mb := metalOrSkip(t)
	ref := Default() // cpu-ref
	var seed lcg = 99
	g := &seed
	out, in := 257, 192
	w := mtlRscale(g, out*in, 0.2)
	x := mtlRscale(g, in, 1.0)
	yRef := ref.Read(ref.MatMul(mtlMkResident(ref, []int{out, in}, w), mtlMkResident(ref, []int{in}, x)))
	yMt := mb.Read(mb.MatMul(mtlMkResident(mb, []int{out, in}, w), mtlMkResident(mb, []int{in}, x)))
	if len(yRef) != out || len(yMt) != out {
		t.Fatalf("shape: ref=%d metal=%d want %d", len(yRef), len(yMt), out)
	}
	c := cosine(yRef, yMt)
	if c < 0.9999 {
		t.Fatalf("matmul cosine %.6f < 0.9999", c)
	}
	var maxAbs float64
	for i := range yRef {
		if d := math.Abs(float64(yRef[i] - yMt[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("MatMul: cosine=%.8f maxAbs=%.2e (device=%s tier=%s class=%s)", c, maxAbs, mb.Name(), mb.Tier(), mb.Class())
}

func TestMetalDeviceMemoryInfoReportsWorkingSet(t *testing.T) {
	mb := metalOrSkip(t)
	total, free, known := DeviceMemoryInfo(mb)
	if !known {
		t.Skip("Metal recommended working-set size unavailable")
	}
	if total <= 0 || free != FreeUnknown {
		t.Fatalf("DeviceMemoryInfo = total=%d free=%d known=%v, want positive total/free unknown/known", total, free, known)
	}
	if !mb.Caps().CapacityProbe {
		t.Fatal("known Metal working-set size must advertise CapacityProbe")
	}
	if v, avail := FitsOnDevice(mb, total+1, 0); v != FitTooBig || avail != total {
		t.Fatalf("oversize Metal fit = %s avail=%d, want FitTooBig avail=%d", v, avail, total)
	}
}

// TestMetalForwardMatchesRef — the headline: a full multi-layer Llama decode forward,
// greedily run for several tokens, on the GPU vs the CPU reference. The Approx gate: every
// step's argmax must be EXACT (same next token) and the logit cosine >= 0.999.
func TestMetalForwardMatchesRef(t *testing.T) {
	mb := metalOrSkip(t)
	ref := Default()
	cfg := mtlSynthCfg{H: 64, L: 3, nH: 8, nKV: 2, hd: 8, I: 172, vocab: 96, eps: 1e-5, theta: 10000}
	host := mtlSynthHostWeights(cfg)

	mRef := mtlNewSynth(ref, cfg, host)
	mMt := mtlNewSynth(mb, cfg, host)

	prompt := []int{5, 17, 42, 3, 88, 11}
	const nGen = 8

	var lref, lmt []float32
	for _, id := range prompt {
		lref = mRef.step(id)
		lmt = mMt.step(id)
	}
	checkPair := func(tag string) {
		c := cosine(lref, lmt)
		if c < 0.999 {
			t.Fatalf("%s logit cosine %.6f < 0.999", tag, c)
		}
		aMt := mb.Argmax(mtlMkResident(mb, []int{len(lmt)}, lmt)) // device argmax kernel
		hRef, hMt := argmaxF32(lref), argmaxF32(lmt)
		if aMt != hMt {
			t.Fatalf("%s metal Argmax kernel %d != host %d", tag, aMt, hMt)
		}
		if hRef != hMt {
			t.Fatalf("%s next-token mismatch: ref=%d metal=%d (cosine=%.6f)", tag, hRef, hMt, c)
		}
	}
	checkPair("prefill")

	for step := 0; step < nGen; step++ {
		next := argmaxF32(lref)
		lref = mRef.step(next)
		lmt = mMt.step(next)
		checkPair("gen" + strconv.Itoa(step))
	}
	t.Logf("Metal forward parity: %d prompt + %d greedy steps, argmax-exact, final cosine=%.8f", len(prompt), nGen, cosine(lref, lmt))
}
