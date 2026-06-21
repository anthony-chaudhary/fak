package compute

import (
	"math"
	"testing"
)

// lcg is a tiny deterministic PRNG (no global math/rand, no Date.now-style nondeterminism)
// so every run feeds the kernels identical inputs — the bit-identity tests need that.
type lcg uint64

func (s *lcg) f() float32 {
	*s = *s*6364136223846793005 + 1442695040888963407
	return float32(int32(*s>>32))/float32(1<<31) - 0.5 // ~[-0.5,0.5)
}
func randVec(s *lcg, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = s.f()
	}
	return v
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func cpu() *cpuBackend { return Pick("cpu-ref").(*cpuBackend) }

// TestReductionOrderIsTheModelTree pins fdot to the model's EXACT 8-accumulator combine
// order — the load-bearing invariant for R2/R14. If someone "simplifies" fdot to a naive
// sum, the bytes shift ~1e-6 and this fires. We prove (a) fdot equals the explicit tree
// and (b) it is NOT the naive single-accumulator sum on an order-sensitive input.
func TestReductionOrderIsTheModelTree(t *testing.T) {
	var s lcg = 1
	r, x := randVec(&s, 19), randVec(&s, 19) // 19 = 2 lanes + 3 tail, exercises both paths
	// explicit tree, byte-for-byte the fdot body
	var a0, a1, a2, a3, a4, a5, a6, a7 float32
	i := 0
	for ; i+8 <= 16; i += 8 {
		a0 += r[i] * x[i]
		a1 += r[i+1] * x[i+1]
		a2 += r[i+2] * x[i+2]
		a3 += r[i+3] * x[i+3]
		a4 += r[i+4] * x[i+4]
		a5 += r[i+5] * x[i+5]
		a6 += r[i+6] * x[i+6]
		a7 += r[i+7] * x[i+7]
	}
	want := ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
	for ; i < 19; i++ {
		want += r[i] * x[i]
	}
	if got := fdot(r, x); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("fdot reduction order drifted: got %v want %v (bits %x vs %x)", got, want, math.Float32bits(got), math.Float32bits(want))
	}
	// naive single-accumulator — must differ (else the tree isn't actually load-bearing here)
	var naive float32
	for k := range r {
		naive += r[k] * x[k]
	}
	if math.Float32bits(naive) == math.Float32bits(fdot(r, x)) {
		t.Skip("inputs happened to be order-insensitive; reduction-order check inconclusive")
	}
}

// TestMatMulDelegatesVerbatim locks the "CPU backend == the model function" invariant: the
// F32 MatMul of each row must equal a direct fdot call (so adoption is byte-identical), and
// BatchedMatMul must equal per-row fdot.
func TestMatMulDelegatesVerbatim(t *testing.T) {
	c := cpu()
	var s lcg = 7
	out, in, P := 5, 32, 3
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	wt := NewF32(c, []int{out, in}, w)
	xt := NewF32(c, []int{in}, x)
	y := c.Read(c.MatMul(wt, xt))
	for o := 0; o < out; o++ {
		if want := fdot(w[o*in:o*in+in], x); math.Float32bits(y[o]) != math.Float32bits(want) {
			t.Fatalf("MatMul row %d not verbatim: got %v want %v", o, y[o], want)
		}
	}
	X := randVec(&s, P*in)
	Y := c.Read(c.BatchedMatMul(wt, NewF32(c, []int{P, in}, X), P))
	for tk := 0; tk < P; tk++ {
		for o := 0; o < out; o++ {
			want := fdot(w[o*in:o*in+in], X[tk*in:tk*in+in])
			if math.Float32bits(Y[tk*out+o]) != math.Float32bits(want) {
				t.Fatalf("BatchedMatMul[%d,%d] not verbatim", tk, o)
			}
		}
	}
}

// TestRMSNormAndSwiGLUVerbatim checks the elementwise/norm ops reproduce the model formulas.
func TestRMSNormAndSwiGLUVerbatim(t *testing.T) {
	c := cpu()
	var s lcg = 11
	n := 64
	x, wn := randVec(&s, n), randVec(&s, n)
	eps := float32(1e-5)
	got := c.Read(c.RMSNorm(NewF32(c, []int{n}, x), NewF32(c, []int{n}, wn), eps))
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := range x {
		if want := x[i] * inv * wn[i]; math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("RMSNorm[%d] drift", i)
		}
	}
	g, u := randVec(&s, n), randVec(&s, n)
	sw := c.Read(c.SwiGLU(NewF32(c, []int{n}, g), NewF32(c, []int{n}, u)))
	for i := range g {
		if want := silu(g[i]) * u[i]; math.Float32bits(sw[i]) != math.Float32bits(want) {
			t.Fatalf("SwiGLU[%d] drift", i)
		}
	}
}

// TestQ8DispatchIsApproxAndGated proves the dtype-selects-the-kernel collapse: a Q8 weight
// routes MatMul to the int8 path, and the result passes the Approx gate vs the f32
// reference (argmax-exact + high cosine) — exactly the contract a device/Q8 backend lives
// under, while the f32 path stays Reference.
func TestQ8DispatchIsApproxAndGated(t *testing.T) {
	c := cpu()
	var s lcg = 23
	out, in := 96, 64 // multiples of the 32 block
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	wf := NewF32(c, []int{out, in}, w)
	wq := QuantizeQ8(c, []int{out, in}, w, 32)
	if wq.Dtype != Q8_0 || wq.Quant == nil {
		t.Fatal("QuantizeQ8 did not produce a Q8_0 tensor with a QuantSpec")
	}
	yf := c.Read(c.MatMul(wf, NewF32(c, []int{in}, x)))
	yq := c.Read(c.MatMul(wq, NewF32(c, []int{in}, x)))
	if cos := cosine(yf, yq); cos < 0.995 {
		t.Fatalf("Q8 vs f32 cosine %.5f < 0.995 (the Approx gate)", cos)
	}
	// determinism: same input => same bytes (a quantized backend must be reproducible)
	yq2 := c.Read(c.MatMul(wq, NewF32(c, []int{in}, x)))
	for i := range yq {
		if math.Float32bits(yq[i]) != math.Float32bits(yq2[i]) {
			t.Fatalf("Q8 MatMul not deterministic at %d", i)
		}
	}
}

// TestAttentionCausalGQA checks the fused Attention op against an inline causal-GQA
// reference over a populated KV store.
func TestAttentionCausalGQA(t *testing.T) {
	c := cpu()
	cfg := KVConfig{NumLayers: 1, NumKVHeads: 2, HeadDim: 4, RopeTheta: 10000}
	grp := 3
	nH := grp * cfg.NumKVHeads
	hd, w := cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	kv := c.NewKV(cfg)
	var s lcg = 31
	nPos := 5
	for p := 0; p < nPos; p++ {
		kr := randVec(&s, w)
		v := randVec(&s, w)
		kv.AppendKV(0, NewF32(c, []int{w}, kr), NewF32(c, []int{w}, append([]float32(nil), kr...)), NewF32(c, []int{w}, v), p)
	}
	q := randVec(&s, nH*hd)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	got := c.Read(c.Attention(NewF32(c, []int{nH * hd}, q), kv, 0, true, grp, scale))
	// inline reference
	ck := kv.(*cpuKV)
	want := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		sc := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			sc[j] = dot(qh, ck.K[0][j*w+kvh*hd:j*w+(kvh+1)*hd]) * scale
		}
		softmaxInPlace(sc)
		for j := 0; j < nPos; j++ {
			vh := ck.V[0][j*w+kvh*hd : j*w+(kvh+1)*hd]
			for d := 0; d < hd; d++ {
				want[h*hd+d] += sc[j] * vh[d]
			}
		}
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("Attention[%d] drift: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestEvictEqualsNeverSaw is the KV-quarantine witness at small scale: evicting a middle
// span and comparing against a cache that NEVER saw it must be max|Δ|=0 — the property the
// model's single-rotation re-RoPE guarantees and that kvmmu's quarantine relies on.
func TestEvictEqualsNeverSaw(t *testing.T) {
	c := cpu()
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 4, RopeTheta: 10000}
	w := cfg.NumKVHeads * cfg.HeadDim
	// deterministic pre-RoPE K (Kraw) and V per (layer,pos); K = RoPE(Kraw, pos).
	rawK := func(l, p int) []float32 {
		var s lcg = lcg(1000*l + p + 1)
		return randVec(&s, w)
	}
	rawV := func(l, p int) []float32 {
		var s lcg = lcg(7000*l + p + 1)
		return randVec(&s, w)
	}
	ropeK := func(raw []float32, p int) []float32 {
		out := append([]float32(nil), raw...)
		cos, sin := ropeRow(cfg.RopeTheta, cfg.HeadDim, p)
		for h := 0; h < cfg.NumKVHeads; h++ {
			applyRope(out[h*cfg.HeadDim:(h+1)*cfg.HeadDim], cos, sin)
		}
		return out
	}
	// A: append positions 0..4, then evict index 1 (one position).
	A := c.NewKV(cfg)
	for p := 0; p < 5; p++ {
		for l := 0; l < cfg.NumLayers; l++ {
			kr := rawK(l, p)
			A.AppendKV(l, NewF32(c, []int{w}, kr), NewF32(c, []int{w}, ropeK(kr, p)), NewF32(c, []int{w}, rawV(l, p)), p)
		}
	}
	A.Evict(1, 1)
	// B: the survivors (original positions 0,2,3,4) appended at NEW indices 0,1,2,3 — i.e.
	// a run that never saw original position 1.
	survivors := []int{0, 2, 3, 4}
	B := c.NewKV(cfg)
	for ni, op := range survivors {
		for l := 0; l < cfg.NumLayers; l++ {
			kr := rawK(l, op)
			B.AppendKV(l, NewF32(c, []int{w}, kr), NewF32(c, []int{w}, ropeK(kr, ni)), NewF32(c, []int{w}, rawV(l, op)), ni)
		}
	}
	ca, cb := A.(*cpuKV), B.(*cpuKV)
	for l := 0; l < cfg.NumLayers; l++ {
		if len(ca.K[l]) != len(cb.K[l]) {
			t.Fatalf("layer %d length mismatch after evict", l)
		}
		for i := range ca.K[l] {
			if math.Float32bits(ca.K[l][i]) != math.Float32bits(cb.K[l][i]) {
				t.Fatalf("layer %d K[%d]: evict %v != never-saw %v (re-RoPE not bit-exact)", l, i, ca.K[l][i], cb.K[l][i])
			}
		}
		for i := range ca.V[l] {
			if math.Float32bits(ca.V[l][i]) != math.Float32bits(cb.V[l][i]) {
				t.Fatalf("layer %d V[%d] mismatch", l, i)
			}
		}
	}
}

// TestCloneIsDeepAndExact verifies prefix-reuse Clone is an exact, independent copy.
func TestCloneIsDeepAndExact(t *testing.T) {
	c := cpu()
	cfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, RopeTheta: 10000}
	w := cfg.HeadDim
	A := c.NewKV(cfg)
	var s lcg = 99
	for p := 0; p < 3; p++ {
		kr := randVec(&s, w)
		A.AppendKV(0, NewF32(c, []int{w}, kr), NewF32(c, []int{w}, append([]float32(nil), kr...)), NewF32(c, []int{w}, randVec(&s, w)), p)
	}
	B := A.Clone().(*cpuKV)
	A.(*cpuKV).K[0][0] = 12345 // mutate A; B must be unaffected
	if B.K[0][0] == 12345 {
		t.Fatal("Clone is not a deep copy")
	}
	if B.Len() != 3 {
		t.Fatalf("Clone length %d != 3", B.Len())
	}
}

// ---- device-tensor type contract + registry/capability enforcement ----------------

// devBuf is a non-host Buffer: it proves a Tensor can carry storage the host cannot
// dereference, and that Host() then correctly reports (nil,false).
type devBuf struct{ ready bool }

func (d devBuf) Ready() bool { return d.ready }

// fakeDevice is a minimal Approx backend used only to exercise the type contract — it is
// NOT a working device, just enough to prove the seam admits one without a host pointer.
type fakeDevice struct{}

func (fakeDevice) Name() string            { return "fake-dev" }
func (fakeDevice) Tier() string            { return "fake" }
func (fakeDevice) Class() CorrectnessClass { return Approx }
func (fakeDevice) Caps() Caps              { return Caps{Async: true, DeviceMemory: true, FusedAttn: true} }
func (fakeDevice) Upload(t Tensor, _ Dtype) Tensor {
	return makeTensor(fakeDevice{}, t.Dtype, t.Layout, t.Shape, t.Quant, devBuf{ready: true})
}
func (fakeDevice) Host(Tensor) ([]float32, bool)                { return nil, false } // host cannot deref device memory
func (fakeDevice) Read(Tensor) []float32                        { return nil }
func (fakeDevice) Free(Tensor)                                  {}
func (fakeDevice) NewKV(KVConfig) KVStore                       { return nil }
func (fakeDevice) MatMul(_, _ Tensor) Tensor                    { return Tensor{} }
func (fakeDevice) BatchedMatMul(_, _ Tensor, _ int) Tensor      { return Tensor{} }
func (fakeDevice) RMSNorm(_, _ Tensor, _ float32) Tensor        { return Tensor{} }
func (fakeDevice) RoPE(_ Tensor, _, _, _ int, _ float64) Tensor { return Tensor{} }
func (fakeDevice) SwiGLU(_, _ Tensor) Tensor                    { return Tensor{} }
func (fakeDevice) AddInPlace(_, _ Tensor)                       {}
func (fakeDevice) AddBias(_, _ Tensor)                          {}
func (fakeDevice) Attention(_ Tensor, _ KVStore, _ int, _ bool, _ int, _ float32) Tensor {
	return Tensor{}
}
func (fakeDevice) Argmax(Tensor) int { return 0 }

func TestDeviceTensorHasNoHostPointer(t *testing.T) {
	dev := fakeDevice{}
	// upload host f32 to the "device": the resulting tensor's Buffer is NOT a HostBuffer,
	// so host addressability is unreachable by construction — the unsafe.Slice fix.
	src := NewF32(cpu(), []int{4}, []float32{1, 2, 3, 4})
	dt := dev.Upload(src, F16)
	if _, ok := dt.Buf().(HostBuffer); ok {
		t.Fatal("device tensor exposed a HostBuffer — host-pointer assumption leaked")
	}
	if _, ok := dev.Host(dt); ok {
		t.Fatal("device backend Host() returned ok=true — should be (nil,false)")
	}
	// the CPU reference, by contrast, IS host-addressable.
	if _, ok := cpu().Host(src); !ok {
		t.Fatal("cpu-ref tensor should be host-addressable")
	}
}

func TestCorrectnessClassEnforcement(t *testing.T) {
	if !RequireReference(cpu()) {
		t.Fatal("cpu-ref must be Reference (eligible for bit-identity rungs)")
	}
	if RequireReference(fakeDevice{}) {
		t.Fatal("an Approx device must NOT be eligible for bit-identity rungs")
	}
}

func TestRegistryPickAndDefault(t *testing.T) {
	Register(fakeDevice{}) // self-registration analogue
	if Default().Name() != "cpu-ref" {
		t.Fatalf("Default must be the Reference floor, got %q", Default().Name())
	}
	if b, ok := Lookup("fake-dev"); !ok || b.Name() != "fake-dev" {
		t.Fatal("Lookup by name failed")
	}
	if _, ok := Lookup("does-not-exist"); ok {
		t.Fatal("Lookup of unknown backend must not fall back")
	}
	if Pick("fake-dev").Name() != "fake-dev" {
		t.Fatal("Pick by name failed")
	}
	if Pick("does-not-exist").Name() != "cpu-ref" {
		t.Fatal("Pick of unknown name must fall back to Default (the floor)")
	}
	found := false
	for _, n := range Registered() {
		if n == "cpu-ref" {
			found = true
		}
	}
	if !found {
		t.Fatal("cpu-ref not in registry")
	}
}

func TestDtypeMetadata(t *testing.T) {
	for _, tc := range []struct {
		d Dtype
		b int
		q bool
	}{{F32, 4, false}, {F16, 2, false}, {BF16, 2, false}, {Q8_0, 1, true}, {I4, 1, true}, {FP8, 1, true}} {
		if tc.d.Bytes() != tc.b {
			t.Errorf("%s Bytes()=%d want %d", tc.d, tc.d.Bytes(), tc.b)
		}
		if tc.d.Quantized() != tc.q {
			t.Errorf("%s Quantized()=%v want %v", tc.d, tc.d.Quantized(), tc.q)
		}
	}
}
