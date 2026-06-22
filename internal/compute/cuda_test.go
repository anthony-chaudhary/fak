//go:build cuda

package compute

import (
	"math"
	"testing"
)

// cuda_test.go — the on-box witness that the CUDA backend runs a real forward pass on the
// GPU and matches the cpuref Reference within the Approx gate (argmax-exact + logit-cosine
// ≥ threshold), NOT bit-identity (RequireReference keeps the device off the exact rungs).
// Compiled only under -tags cuda; skips cleanly if no CUDA device is registered. Reuses
// the package test helpers cosine/lcg/randVec (compute_test.go) and itoaC (cuda.go).

func cudaOrSkip(t *testing.T) *cudaBackend {
	be := Pick("cuda")
	cb, ok := be.(*cudaBackend)
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	if RequireReference(cb) {
		t.Fatal("cuda backend must be Approx, not Reference")
	}
	return cb
}

func rscale(s *lcg, n int, scale float32) []float32 {
	v := randVec(s, n) // ~[-0.5,0.5)
	for i := range v {
		v[i] *= scale
	}
	return v
}

// mkResident puts host data onto a backend uniformly: cpuref.Upload is identity over the
// host slice; cuda.Upload H2D-copies to VRAM. So the same []float32 feeds both backends.
func mkResident(be Backend, shape []int, data []float32) Tensor {
	return be.Upload(NewF32(be, shape, data), F32)
}

type synthCfg struct {
	H, L, nH, nKV, hd, I, vocab int
	eps                         float32
	theta                       float64
}

func (c synthCfg) grp() int       { return c.nH / c.nKV }
func (c synthCfg) scale() float32 { return float32(1.0 / math.Sqrt(float64(c.hd))) }

// synthModel holds one backend's resident weights + a live KV cache, and steps tokens
// through the exact Llama-decode op chain the model's tokenHidden uses — but expressed
// entirely through the Backend interface, so cpuref and cuda run the SAME chain.
type synthModel struct {
	be  Backend
	cfg synthCfg
	W   map[string]Tensor
	kv  KVStore
	pos int
	emb []float32 // host embedding rows [vocab*H] (gather stays host)
}

func newSynth(be Backend, cfg synthCfg, host map[string][]float32) *synthModel {
	W := map[string]Tensor{
		"norm": mkResident(be, []int{cfg.H}, host["norm"]),
		"head": mkResident(be, []int{cfg.vocab, cfg.H}, host["head"]),
	}
	for l := 0; l < cfg.L; l++ {
		s := itoaC(l)
		W["in"+s] = mkResident(be, []int{cfg.H}, host["in"+s])
		W["post"+s] = mkResident(be, []int{cfg.H}, host["post"+s])
		W["q"+s] = mkResident(be, []int{cfg.nH * cfg.hd, cfg.H}, host["q"+s])
		W["k"+s] = mkResident(be, []int{cfg.nKV * cfg.hd, cfg.H}, host["k"+s])
		W["v"+s] = mkResident(be, []int{cfg.nKV * cfg.hd, cfg.H}, host["v"+s])
		W["o"+s] = mkResident(be, []int{cfg.H, cfg.nH * cfg.hd}, host["o"+s])
		W["gate"+s] = mkResident(be, []int{cfg.I, cfg.H}, host["gate"+s])
		W["up"+s] = mkResident(be, []int{cfg.I, cfg.H}, host["up"+s])
		W["down"+s] = mkResident(be, []int{cfg.H, cfg.I}, host["down"+s])
	}
	kv := be.NewKV(KVConfig{NumLayers: cfg.L, NumKVHeads: cfg.nKV, HeadDim: cfg.hd, RopeTheta: cfg.theta})
	return &synthModel{be: be, cfg: cfg, W: W, kv: kv, pos: 0, emb: host["emb"]}
}

// embedDev uploads token id's embedding row into a fresh device-resident hidden vector. It is
// the per-token INPUT stage (a cudaMalloc + H2D), kept OUTSIDE any CUDA-graph capture by the
// graph path (#483) — a cudaMalloc/H2D mid-capture is illegal. Gather stays on the host.
func (m *synthModel) embedDev(id int) Tensor {
	c := m.cfg
	row := m.emb[id*c.H : (id+1)*c.H]
	return mkResident(m.be, []int{c.H}, append([]float32(nil), row...))
}

// decodeChain runs the resident hidden x through every layer (RMSNorm→QKV→RoPE→Attention→
// o_proj→FFN), appending this position's K/V, then the final RMSNorm→head, and returns the
// DEVICE-resident logits. This is the FIXED per-token decode op sequence #483 captures into a
// CUDA graph; both the eager (stepDev) and graph-replayed (stepDevGraph) paths call it, so the
// graph witness compares the very same ops captured vs not. It does not touch m.pos (the caller
// owns position bookkeeping) and does NOT Read — the caller fences via Read or Argmax.
func (m *synthModel) decodeChain(x Tensor, pos int) Tensor {
	be, c := m.be, m.cfg
	for l := 0; l < c.L; l++ {
		s := itoaC(l)
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
	return be.MatMul(m.W["head"], xf)
}

// stepDev runs one token through all layers (decode path), appends K/V, and returns the
// DEVICE-resident logits tensor WITHOUT a host Read — so a caller can fence it either way:
// Read (full vector host-ward) or Argmax (on-device reduction, only the id crosses). step() is
// exactly Read(stepDev()); the #482 async witness uses stepDev()+Argmax, and the #483 graph
// witness compares stepDev (eager) against stepDevGraph (captured) over the shared decodeChain.
func (m *synthModel) stepDev(id int) Tensor {
	logits := m.decodeChain(m.embedDev(id), m.pos)
	m.pos++
	return logits
}

// step runs one token through all layers (decode path), appends K/V, returns next logits.
func (m *synthModel) step(id int) []float32 { return m.be.Read(m.stepDev(id)) }

func synthHostWeights(cfg synthCfg) map[string][]float32 {
	var seed lcg = 0x1234567
	g := &seed
	h := map[string][]float32{
		"emb":  rscale(g, cfg.vocab*cfg.H, 0.20),
		"norm": rscale(g, cfg.H, 0.20),
		"head": rscale(g, cfg.vocab*cfg.H, 0.20),
	}
	for i := range h["norm"] { // norm gains ~1, like a real model
		h["norm"][i] += 1.0
	}
	for l := 0; l < cfg.L; l++ {
		s := itoaC(l)
		in := rscale(g, cfg.H, 0.2)
		post := rscale(g, cfg.H, 0.2)
		for i := range in {
			in[i] += 1.0
			post[i] += 1.0
		}
		h["in"+s] = in
		h["post"+s] = post
		h["q"+s] = rscale(g, cfg.nH*cfg.hd*cfg.H, 0.10)
		h["k"+s] = rscale(g, cfg.nKV*cfg.hd*cfg.H, 0.10)
		h["v"+s] = rscale(g, cfg.nKV*cfg.hd*cfg.H, 0.10)
		h["o"+s] = rscale(g, cfg.H*cfg.nH*cfg.hd, 0.10)
		h["gate"+s] = rscale(g, cfg.I*cfg.H, 0.10)
		h["up"+s] = rscale(g, cfg.I*cfg.H, 0.10)
		h["down"+s] = rscale(g, cfg.H*cfg.I, 0.10)
	}
	return h
}

// TestCUDAMatMulApproxMatchesRef — op-level first light: a single device SGEMM vs the
// cpuref fdot matmul on the same random W,x. Approx (cuBLAS reduction order differs), so
// the gate is cosine, not equality.
func TestCUDAMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default() // cpu-ref
	var seed lcg = 99
	g := &seed
	out, in := 257, 192
	w := rscale(g, out*in, 0.2)
	x := rscale(g, in, 1.0)
	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))
	yCu := cb.Read(cb.MatMul(mkResident(cb, []int{out, in}, w), mkResident(cb, []int{in}, x)))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape: ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}
	c := cosine(yRef, yCu)
	if c < 0.9999 {
		t.Fatalf("matmul cosine %.6f < 0.9999", c)
	}
	var maxAbs float64
	for i := range yRef {
		if d := math.Abs(float64(yRef[i] - yCu[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("MatMul: cosine=%.8f maxAbs=%.2e (device=%s tier=%s class=%s)", c, maxAbs, cb.Name(), cb.Tier(), cb.Class())
}

// TestCUDAForwardMatchesRef — the headline: a full multi-layer Llama decode forward,
// greedily run for several tokens, on the GPU vs the CPU reference. The Approx gate:
// every step's argmax must be EXACT (same next token) and the logit cosine ≥ the recorded
// floor. The per-layer Attention is now the fused flash/online-softmax kernel (#486,
// Caps.FusedAttn), so this end-to-end witness IS the flash kernel's forward-pass gate; the
// recorded floor is cudaFlashAttnCosineMin (the same 0.999 the forward witness has used).
func TestCUDAForwardMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	cfg := synthCfg{H: 64, L: 3, nH: 8, nKV: 2, hd: 8, I: 172, vocab: 96, eps: 1e-5, theta: 10000}
	host := synthHostWeights(cfg)

	mRef := newSynth(ref, cfg, host)
	mCu := newSynth(cb, cfg, host)

	prompt := []int{5, 17, 42, 3, 88, 11}
	const nGen = 8

	var lref, lcu []float32
	for _, id := range prompt {
		lref = mRef.step(id)
		lcu = mCu.step(id)
	}
	checkPair := func(tag string) {
		c := cosine(lref, lcu)
		if c < cudaFlashAttnCosineMin {
			t.Fatalf("%s logit cosine %.6f < %.4f (flash-attn forward gate)", tag, c, cudaFlashAttnCosineMin)
		}
		aCu := cb.Argmax(mkResident(cb, []int{len(lcu)}, lcu)) // device argmax kernel
		hRef, hCu := argmaxF32(lref), argmaxF32(lcu)
		if aCu != hCu {
			t.Fatalf("%s cuda Argmax kernel %d != host %d", tag, aCu, hCu)
		}
		if hRef != hCu {
			t.Fatalf("%s next-token mismatch: ref=%d cuda=%d (cosine=%.6f)", tag, hRef, hCu, c)
		}
	}
	checkPair("prefill")

	for step := 0; step < nGen; step++ {
		next := argmaxF32(lref)
		lref = mRef.step(next)
		lcu = mCu.step(next)
		checkPair("gen" + itoaC(step))
	}
	t.Logf("CUDA forward parity (fused flash attention, #486): %d prompt + %d greedy steps, argmax-exact, final cosine=%.8f gate=%.4f", len(prompt), nGen, cosine(lref, lcu), cudaFlashAttnCosineMin)
}
