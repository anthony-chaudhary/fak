//go:build darwin && arm64 && cgo

package compute

import (
	"errors"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestMetalMatMulTraceEventHostTiming(t *testing.T) {
	started := time.Date(2026, time.August, 30, 12, 0, 0, 123, time.FixedZone("test", -7*60*60))
	w := Tensor{Dtype: F16, Shape: []int{3, 4}}
	x := Tensor{Dtype: BF16, Shape: []int{4}}
	y := Tensor{Dtype: I8, Shape: []int{3}}

	e := metalMatMulTraceEvent(started, metalCommandReceipt{WaitMilliseconds: 1.25}, w, x, y, 3, 4, 1)
	if e.Operation != "matmul" || e.Phase != "kernel" || e.Backend != "metal" || e.Device != "metal:0" || e.Kernel != "mps_f32_matmul" || e.Route != "device" {
		t.Fatalf("trace identity = operation=%q phase=%q backend=%q device=%q kernel=%q route=%q", e.Operation, e.Phase, e.Backend, e.Device, e.Kernel, e.Route)
	}
	if e.StartedAt != started.UTC() || e.DurationNS != 1_250_000 || e.DeviceDurationNS != 0 || e.TimerDomain != "host_monotonic" {
		t.Fatalf("trace timing = started=%v duration=%d device_duration=%d domain=%q", e.StartedAt, e.DurationNS, e.DeviceDurationNS, e.TimerDomain)
	}
	if e.WeightDType != "f16" || e.InputDType != "bf16" || e.OutputDType != "i8" {
		t.Fatalf("trace dtypes = weight=%q input=%q output=%q", e.WeightDType, e.InputDType, e.OutputDType)
	}
	if e.BytesRead != 32 || e.BytesWritten != 3 {
		t.Fatalf("trace bytes = read=%d written=%d, want 32/3", e.BytesRead, e.BytesWritten)
	}
	if e.EstimatedFLOPs != 24 || !reflect.DeepEqual(e.Shapes, [][]int{{3, 4}, {4}}) {
		t.Fatalf("trace work = flops=%d shapes=%v", e.EstimatedFLOPs, e.Shapes)
	}
	if e.ProvenanceDigest != computetrace.Digest("metal", "mps_f32_matmul") {
		t.Fatalf("trace provenance = %q", e.ProvenanceDigest)
	}
}

func TestMetalMatMulTraceEventDeviceTiming(t *testing.T) {
	e := metalMatMulTraceEvent(time.Time{}, metalCommandReceipt{
		WaitMilliseconds: 2.5,
		TimingAvailable:  true,
		GPUMilliseconds:  0.75,
	}, Tensor{Dtype: F32, Shape: []int{2, 2}}, Tensor{Dtype: F32, Shape: []int{2}}, Tensor{Dtype: F32, Shape: []int{2}}, 2, 2, 1)
	if e.DurationNS != 2_500_000 || e.DeviceDurationNS != 750_000 || e.TimerDomain != "metal_command_buffer" {
		t.Fatalf("trace timing = duration=%d device_duration=%d domain=%q", e.DurationNS, e.DeviceDurationNS, e.TimerDomain)
	}
}

func TestMetalMatMulRejectsEitherNonF32OperandBeforeDeviceUse(t *testing.T) {
	backend := &metalBackend{}
	for _, tc := range []struct {
		name string
		call func()
		want string
	}{
		{
			name: "matmul weight",
			call: func() { backend.MatMul(Tensor{Dtype: F16}, Tensor{Dtype: F32}) },
			want: "metal MatMul supports only F32 inputs today (weight=f16 input=f32)",
		},
		{
			name: "matmul input",
			call: func() { backend.MatMul(Tensor{Dtype: F32}, Tensor{Dtype: BF16}) },
			want: "metal MatMul supports only F32 inputs today (weight=f32 input=bf16)",
		},
		{
			name: "batched input",
			call: func() { backend.BatchedMatMul(Tensor{Dtype: F32}, Tensor{Dtype: I8}, 2) },
			want: "metal BatchedMatMul supports only F32 inputs today (weight=f32 input=i8)",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				got := recover()
				var message string
				if err, ok := got.(error); ok {
					message = err.Error()
				} else if str, ok := got.(string); ok {
					message = str
				}
				if !strings.Contains(message, tc.want) {
					t.Fatalf("panic = %v, want text %q", got, tc.want)
				}
			}()
			tc.call()
		})
	}
}

// metal_test.go ΓÇö the on-box witness that the Metal backend runs a real forward pass on the
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
// through the exact Llama-decode op chain the model's tokenHidden uses ΓÇö but expressed
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

// TestMetalMatMulApproxMatchesRef ΓÇö op-level first light: a single device MPS SGEMM vs the
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

// TestMetalCommandOwnerBatchesTwoMatMuls proves the caller-owned seam can encode
// independent operations and cross exactly one commit/wait boundary before readback.
func TestMetalCommandOwnerBatchesTwoMatMuls(t *testing.T) {
	mb := metalOrSkip(t)
	ref := Default()
	var seed lcg = 9259
	g := &seed

	const in = 64
	type testCase struct {
		out        int
		w, x       []float32
		wDev, xDev Tensor
	}
	cases := []testCase{
		{out: 37, w: mtlRscale(g, 37*in, 0.2), x: mtlRscale(g, in, 1)},
		{out: 53, w: mtlRscale(g, 53*in, 0.2), x: mtlRscale(g, in, 1)},
	}
	for i := range cases {
		cases[i].wDev = mtlMkResident(mb, []int{cases[i].out, in}, cases[i].w)
		cases[i].xDev = mtlMkResident(mb, []int{in}, cases[i].x)
	}

	var outputs []Tensor
	var owner *metalCommandOwner
	var receipt metalCommandReceipt
	func() {
		metalMu.Lock()
		defer metalMu.Unlock()
		var err error
		owner, err = beginMetalCommand()
		if err != nil {
			t.Fatal(err)
		}
		outputs = make([]Tensor, len(cases))
		for i, tc := range cases {
			y, _ := mb.devTr([]int{tc.out}, F32)
			outputs[i] = y
			if err := owner.encodeMatMul(mb.mb(tc.wDev), mb.mb(tc.xDev), mb.mb(y), tc.out, in, 1); err != nil {
				_ = owner.abort()
				t.Fatal(err)
			}
		}
		if owner.lifecycle.encoders != 2 {
			_ = owner.abort()
			t.Fatalf("encoded operations = %d, want 2 before the single submit", owner.lifecycle.encoders)
		}
		receipt, err = owner.finish()
		if err != nil {
			t.Fatal(err)
		}
	}()
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 2 {
		t.Fatalf("receipt = %+v, want one committed/completed submit containing 2 encoders", receipt)
	}
	for i, tc := range cases {
		got := mb.Read(outputs[i])
		want := ref.Read(ref.MatMul(mtlMkResident(ref, []int{tc.out, in}, tc.w), mtlMkResident(ref, []int{in}, tc.x)))
		if c := cosine(want, got); c < 0.9999 {
			t.Fatalf("matmul %d cosine %.6f < 0.9999", i, c)
		}
	}
	if err := owner.encodeMatMul(nil, nil, nil, 1, 1, 1); !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("post-finish encode = %v, want terminal", err)
	}
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

// TestMetalForwardMatchesRef ΓÇö the headline: a full multi-layer Llama decode forward,
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

func TestMetalTypedErrorsAndRecovery(t *testing.T) {
	// 1. Pure Go verification of requireMetalMatMulF32 typed error.
	t.Run("typed_unsupported_dtype", func(t *testing.T) {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on non-F32 MatMul input")
			}
			var dtypeErr *MetalInputDtypeError
			if !errors.As(r.(error), &dtypeErr) {
				t.Fatalf("expected *MetalInputDtypeError, got %T: %v", r, r)
			}
			if dtypeErr.Op != "MatMul" || dtypeErr.Weight != Q8_0 || dtypeErr.Input != F32 {
				t.Fatalf("unexpected error payload: %+v", dtypeErr)
			}
			if !strings.Contains(dtypeErr.Error(), "compute: metal MatMul supports only F32 inputs") {
				t.Fatalf("error string mismatch: %s", dtypeErr.Error())
			}
		}()
		invalidW := Tensor{Shape: []int{4, 4}, Dtype: Q8_0}
		validX := Tensor{Shape: []int{4}, Dtype: F32}
		requireMetalMatMulF32("MatMul", invalidW, validX)
	})

	// 2. Hardware-gated execution on real Metal device.
	t.Run("on_device_recovery", func(t *testing.T) {
		mb := metalOrSkip(t)
		ref := Default()

		func() {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected panic on non-F32 MatMul input")
				}
				var dtypeErr *MetalInputDtypeError
				if !errors.As(r.(error), &dtypeErr) {
					t.Fatalf("expected *MetalInputDtypeError, got %T: %v", r, r)
				}
			}()

			invalidW := Tensor{Shape: []int{4, 4}, Dtype: Q8_0}
			validX := mtlMkResident(mb, []int{4}, []float32{1, 2, 3, 4})
			_ = mb.MatMul(invalidW, validX)
		}()

		w := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		x := []float32{2, 3, 5, 7}
		dw := mtlMkResident(mb, []int{4, 4}, w)
		dx := mtlMkResident(mb, []int{4}, x)
		got := mb.Read(mb.MatMul(dw, dx))
		want := ref.Read(ref.MatMul(mtlMkResident(ref, []int{4, 4}, w), mtlMkResident(ref, []int{4}, x)))

		if c := cosine(want, got); c < 0.9999 {
			t.Fatalf("subsequent MatMul failed: cosine %.6f < 0.9999", c)
		}
	})
}

const metalQ2CosineMin = 0.999

func mtlDominantRow(w []float32, out, in int) int {
	best, bestNorm := 0, float32(-1)
	for o := 0; o < out; o++ {
		var n float32
		for _, v := range w[o*in : o*in+in] {
			n += v * v
		}
		if n > bestNorm {
			bestNorm, best = n, o
		}
	}
	return best
}

func mtlAlignActToRow(w []float32, out, in, target int) []float32 {
	x := make([]float32, in)
	copy(x, w[target*in:target*in+in])
	return x
}

func mtlNonTarget(v []float32, out, target int) []float32 {
	r := make([]float32, 0, out-1)
	for o := 0; o < out; o++ {
		if o != target {
			r = append(r, v[o])
		}
	}
	return r
}

func mtlMaxAbsDelta(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestMetalQ2_0MatMulApproxMatchesRef verifies that packed-ternary Q2_0 decode GEMV (P=1)
// runs on Apple Silicon Metal and matches the cpuref reference within the approx gate
// (cosine >= 0.999 and argmax-exact).
func TestMetalQ2_0MatMulApproxMatchesRef(t *testing.T) {
	mb := metalOrSkip(t)
	ref := Default() // cpu-ref
	defer metalgemm.ResetQ2_0()
	const out, in = 320, 256 // in divisible by 32 (q2Block)
	packed, scale := randTernaryWeight(0x4872, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)
	target := mtlDominantRow(dense, out, in)
	x := mtlAlignActToRow(dense, out, in, target)

	yRef := ref.Read(ref.MatMul(mtlMkResident(ref, []int{out, in}, dense), mtlMkResident(ref, []int{in}, x)))

	hostQ2 := NewQ2(mb, []int{out, in}, packed, scale, q2Block)
	wQ2 := mb.Upload(hostQ2, Q2_0)
	if wQ2.Dtype != Q2_0 {
		t.Fatalf("Upload(_, Q2_0) produced Dtype %s, want q2_0", wQ2.Dtype)
	}

	// Verify upload caching: same host buffer returns cached resident tensor.
	wQ2Cached := mb.Upload(hostQ2, Q2_0)
	if wQ2.buf != wQ2Cached.buf {
		t.Fatalf("Q2_0 upload cache miss on identical host tensor")
	}

	yMt := mb.Read(mb.MatMul(wQ2, mtlMkResident(mb, []int{in}, x)))
	if len(yRef) != out || len(yMt) != out {
		t.Fatalf("shape ref=%d metal=%d want %d", len(yRef), len(yMt), out)
	}

	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aMt := mb.Argmax(mtlMkResident(mb, []int{out}, yMt))
	if aMt != argmaxF32(yMt) || argmaxF32(yMt) != argmaxF32(yRef) {
		t.Fatalf("Q2_0 argmax-exact failed: ref=%d metalHost=%d metalKernel=%d", argmaxF32(yRef), argmaxF32(yMt), aMt)
	}
	c := cosine(mtlNonTarget(yRef, out, target), mtlNonTarget(yMt, out, target))
	if c < metalQ2CosineMin {
		t.Fatalf("Q2_0 MatMul cosine %.6f < recorded Q2_0 gate %.6f (metalQ2CosineMin)", c, metalQ2CosineMin)
	}
	t.Logf("Metal Q2_0 MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact (device=%s tier=%s class=%s)",
		c, mtlMaxAbsDelta(yRef, yMt), metalQ2CosineMin, mb.Name(), mb.Tier(), mb.Class())

	mb.Free(wQ2)
}

// TestMetalQ2_0BatchedMatMulApproxMatchesRef verifies that packed-ternary Q2_0 prefill GEMM (P>1)
// runs on Apple Silicon Metal and matches the cpuref reference with cosine >= 0.999.
func TestMetalQ2_0BatchedMatMulApproxMatchesRef(t *testing.T) {
	mb := metalOrSkip(t)
	ref := Default() // cpu-ref
	defer metalgemm.ResetQ2_0()
	const out, in, P = 320, 256, 8
	packed, scale := randTernaryWeight(0x4872b, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)
	var seed lcg = 0x4872c
	X := mtlRscale(&seed, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mtlMkResident(ref, []int{out, in}, dense), mtlMkResident(ref, []int{P, in}, X), P))
	wQ2 := mb.Upload(NewQ2(mb, []int{out, in}, packed, scale, q2Block), Q2_0)
	YMt := mb.Read(mb.BatchedMatMul(wQ2, mtlMkResident(mb, []int{P, in}, X), P))
	if len(YRef) != P*out || len(YMt) != P*out {
		t.Fatalf("shape ref=%d metal=%d want %d", len(YRef), len(YMt), P*out)
	}
	c := cosine(YRef, YMt)
	if c < metalQ2CosineMin {
		t.Fatalf("Q2_0 BatchedMatMul cosine %.6f < recorded Q2_0 gate %.6f", c, metalQ2CosineMin)
	}
	t.Logf("Metal Q2_0 BatchedMatMul (P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f", P, c, mtlMaxAbsDelta(YRef, YMt), metalQ2CosineMin)

	mb.Free(wQ2)
}

// TestMetalQ2_0UploadCacheAndFree verifies upload caching and cache eviction on Free for Q2_0 weights.
func TestMetalQ2_0UploadCacheAndFree(t *testing.T) {
	mb := metalOrSkip(t)
	defer metalgemm.ResetQ2_0()
	const out, in = 64, 32
	packed, scale := randTernaryWeight(0x1234, out, in, q2Block)
	hostQ2 := NewQ2(mb, []int{out, in}, packed, scale, q2Block)

	res1 := mb.Upload(hostQ2, Q2_0)
	res2 := mb.Upload(hostQ2, Q2_0)
	if res1.buf != res2.buf {
		t.Fatalf("Q2_0 upload should return cached resident tensor on identical host buffer")
	}

	mb.Free(res1)
	if b, ok := res1.buf.(*metalBuf); ok && b.q2w != nil {
		t.Fatalf("Free did not clear resident q2w handle")
	}

	res3 := mb.Upload(hostQ2, Q2_0)
	if res3.buf == res1.buf {
		t.Fatalf("re-upload after Free should allocate fresh buffer, not reuse evicted cache")
	}
	mb.Free(res3)
}
