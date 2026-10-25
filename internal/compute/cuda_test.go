//go:build cuda && cgo

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

// TestCUDAUploadCacheRetainsHostOwner guards the pointer-keyed weight cache against a
// same-address false hit. A uintptr is not a GC root: without the retained HostBuffer, Go may
// recycle a dead same-shaped weight slice at the same address while stale VRAM remains cached.
func TestCUDAUploadCacheRetainsHostOwner(t *testing.T) {
	cb := cudaOrSkip(t)
	host := NewF32(Default(), []int{2, 2}, []float32{1, 2, 3, 4})
	assertOwner := func(as Dtype) {
		resident := cb.Upload(host, as)
		db, ok := resident.buf.(*cudaBuf)
		if !ok {
			t.Fatalf("%s cached upload buffer = %T, want *cudaBuf", as, resident.buf)
		}
		if db.host == 0 {
			t.Fatalf("%s cached upload has no host key", as)
		}
		if db.hostKeep != host.buf {
			t.Fatalf("%s pointer-keyed CUDA upload cache did not retain its source HostBuffer", as)
		}
		cb.Free(resident)
	}
	assertOwner(F32)
	assertOwner(F16)
}

func TestCUDADeviceMemoryInfoReportsCurrentFree(t *testing.T) {
	cb := cudaOrSkip(t)
	total, free, known := DeviceMemoryInfo(cb)
	if !known || total <= 0 {
		t.Fatalf("DeviceMemoryInfo = total=%d free=%d known=%v, want positive total/known", total, free, known)
	}
	if free == FreeUnknown {
		t.Fatalf("CUDA cudaMemGetInfo should report current free bytes, got FreeUnknown with total=%d", total)
	}
	if free < 0 || free > total {
		t.Fatalf("DeviceMemoryInfo free=%d outside [0,total=%d]", free, total)
	}
}

// TestCUDAUploadRankOneCopiesCurrentContents captures the #3186 failure at the cache seam.
// Upload is a copy operation: mutating and re-uploading a same-address rank-one host slice must
// stage the new bytes, never return the first resident tensor from the immutable-weight cache.
// Before the fix, the second Upload aliases first because the cache key is the host address;
// multi-token forward then feeds CUDA a stale token embedding and collapses its logits.
func TestCUDAUploadRankOneCopiesCurrentContents(t *testing.T) {
	cb := cudaOrSkip(t)
	src := []float32{1, 2, 3, 4}
	first := cb.Upload(NewF32(cb, []int{len(src)}, src), F32)
	if got := cb.Read(first); len(got) != len(src) || got[0] != 1 || got[3] != 4 {
		t.Fatalf("first Upload = %v, want original rank-one contents", got)
	}

	copy(src, []float32{-4, -3, -2, -1})
	second := cb.Upload(NewF32(cb, []int{len(src)}, src), F32)
	t.Cleanup(func() {
		cb.Free(first)
		if first.buf != second.buf {
			cb.Free(second)
		}
	})
	if got := cb.Read(second); len(got) != len(src) || got[0] != -4 || got[3] != -1 {
		t.Fatalf("second Upload = %v, want current rank-one contents; stale upload-cache alias", got)
	}
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
	// The synthetic dense model is capture-eligible, as model.cudaGraphCaptureEligible rules
	// for a dense config, so under graphs its KV takes the fixed capacity a capture needs.
	kv := be.NewKV(KVConfig{NumLayers: cfg.L, NumKVHeads: cfg.nKV, HeadDim: cfg.hd, RopeTheta: cfg.theta, GraphCapture: true})
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

// TestCUDAMatMulNonAlignedShapes — the regression set #972 asked for: op-level GEMV (P=1) and
// batched GEMM (P>1) across output dims that are NOT multiples of the device block width (64/256).
// The first sm_80 / CUDA-13.0 witness saw out=257 come back as 64 rows (a one-block write), so the
// grid/launch math has to cover ceil(out/BLK) rows for every dtype path. Each case checks the
// exact result length (the structural gate that fired on the GPU) then the cosine vs the cpuref.
func TestCUDAMatMulNonAlignedShapes(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	// out dims straddling the 64- and 256-block boundaries; in dims off the boundary too.
	cases := []struct{ out, in, P int }{
		{257, 192, 1},  // the original #972 witness shape (out just over 4*64)
		{65, 128, 1},   // out just over one 64-block
		{255, 256, 1},  // out just under a 256-block
		{1000, 320, 1}, // large non-aligned out
		{257, 192, 3},  // batched: non-aligned out, P>1
		{130, 96, 5},   // batched: out just over 2*64, P>1
	}
	var seed lcg = 99
	g := &seed
	for _, tc := range cases {
		w := rscale(g, tc.out*tc.in, 0.2)
		wRefT := mkResident(ref, []int{tc.out, tc.in}, w)
		wCuT := mkResident(cb, []int{tc.out, tc.in}, w)
		if tc.P == 1 {
			x := rscale(g, tc.in, 1.0)
			yRef := ref.Read(ref.MatMul(wRefT, mkResident(ref, []int{tc.in}, x)))
			yCu := cb.Read(cb.MatMul(wCuT, mkResident(cb, []int{tc.in}, x)))
			if len(yCu) != tc.out {
				t.Fatalf("MatMul out=%d in=%d: got %d rows want %d", tc.out, tc.in, len(yCu), tc.out)
			}
			if c := cosine(yRef, yCu); c < 0.9999 {
				t.Fatalf("MatMul out=%d in=%d: cosine %.6f < 0.9999", tc.out, tc.in, c)
			}
		} else {
			x := rscale(g, tc.P*tc.in, 1.0)
			yRef := ref.Read(ref.BatchedMatMul(wRefT, mkResident(ref, []int{tc.P, tc.in}, x), tc.P))
			yCu := cb.Read(cb.BatchedMatMul(wCuT, mkResident(cb, []int{tc.P, tc.in}, x), tc.P))
			if len(yCu) != tc.P*tc.out {
				t.Fatalf("BatchedMatMul out=%d in=%d P=%d: got %d want %d", tc.out, tc.in, tc.P, len(yCu), tc.P*tc.out)
			}
			if c := cosine(yRef, yCu); c < 0.9999 {
				t.Fatalf("BatchedMatMul out=%d in=%d P=%d: cosine %.6f < 0.9999", tc.out, tc.in, tc.P, c)
			}
		}
	}
	t.Logf("MatMul non-aligned shapes: %d cases pass (device=%s)", len(cases), cb.Name())
}

// cudaForwardLogitRelErrMax bounds TestCUDAForwardMatchesRef's per-step logit error,
// max|cuda-ref| / max|ref|. Measured on unmutated kernels (RTX 4070 sm_89, CUDA 12.8,
// 2026-10-23): worst 1.4e-6 over these prompts and 2.0e-6 over 576 steps of 64 prompts, so 2e-5
// leaves ~10x for another arch's cuBLAS reduction order. The weakest kernel mutant it kills,
// RoPE theta x1.001, reaches 5.2e-4; the 0.999 cosine floor passes it and the swiglu, rmsnorm
// eps and attention-scale mutants. FAK_CUDA_TF32=1 left these P=1 GEMMs bit-identical on sm_89.
const cudaForwardLogitRelErrMax = 2e-5

// cudaForwardParityPrompts span a one-token prefill, the original witness prompt, vocab edges
// with repeats, and a window long enough that every flash-attention head streams 30+ keys.
var cudaForwardParityPrompts = [][]int{
	{5, 17, 42, 3, 88, 11},
	{0},
	{95, 1, 94, 2, 93, 3, 47, 48, 47, 48, 0, 95},
	{7, 61, 23, 80, 14, 39, 52, 66, 9, 31, 77, 45, 12, 90, 28, 58, 3, 71, 36, 84, 19, 50, 63, 26},
}

// forwardParityHostWeights draws zero-mean weights at fan-in scale so every op in the chain
// moves the logits. randVec spans [-1.5, 0.5), so synthHostWeights' draws carry a -0.5 mean:
// that fixture's residual is a near-constant vector, its logits barely depend on the prompt,
// and a RoPE theta x1.001 or attention scale x1.001 kernel mutant left them inside f32 noise.
func forwardParityHostWeights(cfg synthCfg) map[string][]float32 {
	var seed lcg = 0x5eed1234
	g := &seed
	centered := func(n int, std float64) []float32 {
		v := randVec(g, n)
		a := float32(std * math.Sqrt(3))
		for i := range v {
			v[i] = (v[i] + 0.5) * a // uniform [-a, a): mean 0, standard deviation std
		}
		return v
	}
	gain := func(n int) []float32 {
		v := centered(n, 0.1)
		for i := range v {
			v[i]++
		}
		return v
	}
	fan := func(in int) float64 { return 1 / math.Sqrt(float64(in)) }
	q, kv := cfg.nH*cfg.hd, cfg.nKV*cfg.hd
	h := map[string][]float32{
		"emb":  centered(cfg.vocab*cfg.H, 0.05),
		"norm": gain(cfg.H),
		"head": centered(cfg.vocab*cfg.H, fan(cfg.H)),
	}
	for l := 0; l < cfg.L; l++ {
		s := itoaC(l)
		h["in"+s] = gain(cfg.H)
		h["post"+s] = gain(cfg.H)
		h["q"+s] = centered(q*cfg.H, fan(cfg.H))
		h["k"+s] = centered(kv*cfg.H, fan(cfg.H))
		h["v"+s] = centered(kv*cfg.H, fan(cfg.H))
		h["o"+s] = centered(cfg.H*q, fan(q))
		h["gate"+s] = centered(cfg.I*cfg.H, fan(cfg.H))
		h["up"+s] = centered(cfg.I*cfg.H, fan(cfg.H))
		h["down"+s] = centered(cfg.H*cfg.I, fan(cfg.I))
	}
	return h
}

func cudaForwardLogitRelErr(ref, got []float32) float64 {
	var maxRef float64
	for _, v := range ref {
		maxRef = math.Max(maxRef, math.Abs(float64(v)))
	}
	if maxRef == 0 {
		return math.Inf(1)
	}
	// maxAbsDelta compares len(ref) entries and skips NaN, so a short slice or a
	// non-finite device logit in the compared range must fail here.
	if len(got) < len(ref) {
		return math.Inf(1)
	}
	for _, v := range got[:len(ref)] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return math.Inf(1)
		}
	}
	return maxAbsDelta(ref, got) / maxRef
}

// cudaForwardParityRun steps prompt plus nGen greedy tokens through fresh cpuref and cuda synth
// models fed the same reference-chosen tokens, and hands every step's logits pair to check.
func cudaForwardParityRun(cb *cudaBackend, cfg synthCfg, host map[string][]float32, prompt []int, nGen int, check func(tag string, ref, got []float32)) {
	mRef := newSynth(Default(), cfg, host)
	mCu := newSynth(cb, cfg, host)
	defer mRef.kv.Free()
	defer mCu.kv.Free()
	var lref, lcu []float32
	for _, id := range prompt {
		lref, lcu = mRef.step(id), mCu.step(id)
	}
	check("prefill", lref, lcu)
	for step := 0; step < nGen; step++ {
		next := argmaxF32(lref)
		lref, lcu = mRef.step(next), mCu.step(next)
		check("gen"+itoaC(step), lref, lcu)
	}
}

// TestCUDAForwardMatchesRef — the headline: a full multi-layer Llama decode forward,
// greedily run for several tokens, on the GPU vs the CPU reference. The Approx gate:
// every step's argmax must be EXACT (same next token), the logit cosine ≥ the recorded
// floor, and the scale-relative max logit error within the measured CUDA error budget. The
// per-layer Attention is the fused flash/online-softmax kernel (#486, Caps.FusedAttn), so this
// end-to-end witness IS the flash kernel's forward-pass gate.
func TestCUDAForwardMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	cfg := synthCfg{H: 64, L: 3, nH: 8, nKV: 2, hd: 8, I: 172, vocab: 96, eps: 1e-5, theta: 10000}
	host := forwardParityHostWeights(cfg)
	const nGen = 8

	var worst float64
	var worstTag string
	steps := 0
	for pi, prompt := range cudaForwardParityPrompts {
		cudaForwardParityRun(cb, cfg, host, prompt, nGen, func(tag string, lref, lcu []float32) {
			tag = "prompt" + itoaC(pi) + "/" + tag
			c := cosine(lref, lcu)
			if c < cudaFlashAttnCosineMin {
				t.Fatalf("%s logit cosine %.6f < %.4f (flash-attn forward gate)", tag, c, cudaFlashAttnCosineMin)
			}
			rel := cudaForwardLogitRelErr(lref, lcu)
			if !(rel <= cudaForwardLogitRelErrMax) {
				t.Fatalf("%s logit max|Δ|/max|ref| %.3e > budget %.3e (tf32=%v cosine=%.8f)", tag, rel, cudaForwardLogitRelErrMax, tf32Enabled, c)
			}
			if rel > worst {
				worst, worstTag = rel, tag
			}
			aCu := cb.Argmax(mkResident(cb, []int{len(lcu)}, lcu)) // device argmax kernel
			hRef, hCu := argmaxF32(lref), argmaxF32(lcu)
			if aCu != hCu {
				t.Fatalf("%s cuda Argmax kernel %d != host %d", tag, aCu, hCu)
			}
			if hRef != hCu {
				t.Fatalf("%s next-token mismatch: ref=%d cuda=%d (cosine=%.6f)", tag, hRef, hCu, c)
			}
			steps++
		})
	}
	t.Logf("CUDA forward parity (fused flash attention, #486): %d prompts, %d steps, argmax-exact, worst max|Δ|/max|ref|=%.3e at %s, budget=%.3e (tf32=%v), cosine gate=%.4f",
		len(cudaForwardParityPrompts), steps, worst, worstTag, cudaForwardLogitRelErrMax, tf32Enabled, cudaFlashAttnCosineMin)
}
