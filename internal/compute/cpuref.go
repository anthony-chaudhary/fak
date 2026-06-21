package compute

import "math"

// cpuref.go — the day-1 CPU *reference* backend. It is the only Backend that exists
// yet, and it is the Reference (held to max|Δ|=0): every method reproduces the exact
// arithmetic the model package uses today — the same 8-accumulator fdot tree, the same
// serial RMSNorm sum-of-squares, the same single-accumulator attention score dot, the
// same per-block Q8 reduction order, the same single-rotation eviction re-RoPE, the
// same first-max argmax. So when the model adopts this seam, its bit-identity rungs (R2
// max|Δ|=0, R14 d==0, the HF argmax oracle) see byte-identical results: the only change
// is that the loop reaches the kernel through an interface method instead of a direct
// call.
//
// It is deliberately pure Go, scalar, stdlib-only — no unsafe, no asm, no cgo, no
// os.Getenv — so it is ALSO the portable floor every other backend degrades to (it
// compiles to wasm unchanged). A real CPU backend may later expose the model's x86
// AVX2/AVX-512 Q8 kernels through Tier(); that is a private acceleration of this same
// reference contract, selected by the registry, not a fork of the loop.

func init() { Register(&cpuBackend{}) }

// hostBuf is the CPU reference Buffer: host-resident, always Ready, holding the bytes as
// an f32 and/or int8 slice. It is the sole HostBuffer implementation, which is what makes
// host addressability unreachable on a device tensor by construction.
type hostBuf struct {
	f32 []float32
	i8  []int8
}

func (h *hostBuf) Ready() bool    { return true }
func (h *hostBuf) F32() []float32 { return h.f32 }
func (h *hostBuf) I8() []int8     { return h.i8 }

type cpuBackend struct{}

func (c *cpuBackend) Name() string            { return "cpu-ref" }
func (c *cpuBackend) Tier() string            { return "scalar" } // a real cpu backend probes CPUID here
func (c *cpuBackend) Class() CorrectnessClass { return Reference }
func (c *cpuBackend) Caps() Caps              { return Caps{Collective: true} } // synchronous, host-memory, no fusion; single-box exact collectives (collective.go)

// ---- host data entry + residency ------------------------------------------------

// NewF32 wraps host f32 data as a Tensor owned by be. This is the host-data entry point
// (used to feed the reference backend, or as the source argument to a device Upload).
func NewF32(be Backend, shape []int, data []float32) Tensor {
	return makeTensor(be, F32, RowMajor, shape, nil, &hostBuf{f32: data})
}

// NewQ8 wraps prequantized Q8_0 weight codes + per-block scales as a Tensor. shape is
// [out, in]; len(codes)==out*in; len(scale)==out*(in/block).
func NewQ8(be Backend, shape []int, codes []int8, scale []float32, block int) Tensor {
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true, Scale: scale}
	return makeTensor(be, Q8_0, RowMajor, shape, q, &hostBuf{i8: codes})
}

// QuantizeQ8 builds a Q8_0 weight Tensor from an f32 [out,in] matrix using the exact
// per-block scheme model.quantizeQ8 uses (d=amax/127, code=q8round(w/d)). The result's
// MatMul is Approx vs the f32 reference — exactly the model's Q8 lane.
func QuantizeQ8(be Backend, shape []int, w []float32, block int) Tensor {
	out, in := shape[0], shape[1]
	nblk := in / block
	codes := make([]int8, out*in)
	scale := make([]float32, out*nblk)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := w[o*in+b*block : o*in+b*block+block]
			var amax float32
			for _, v := range blk {
				if a := absf(v); a > amax {
					amax = a
				}
			}
			d := amax / 127
			scale[o*nblk+b] = d
			if d == 0 {
				continue
			}
			inv := 1.0 / d
			for i := 0; i < block; i++ {
				codes[o*in+b*block+i] = q8round(blk[i] * inv)
			}
		}
	}
	return NewQ8(be, shape, codes, scale, block)
}

func (c *cpuBackend) Upload(t Tensor, as Dtype) Tensor { return t } // identity: host data already resident
func (c *cpuBackend) Free(Tensor)                      {}           // host GC owns it

func (c *cpuBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(*hostBuf); ok && hb.f32 != nil {
		return hb.f32, true
	}
	return nil, false
}

func (c *cpuBackend) Read(t Tensor) []float32 {
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32
	}
	return nil
}

// f32 / i8 are the backend-private accessors: a device backend's equivalents fence and
// DMA; the reference's are zero-copy host views (so the proven path gains no copies).
func (c *cpuBackend) f32(t Tensor) []float32 { return t.buf.(*hostBuf).f32 }
func (c *cpuBackend) i8(t Tensor) []int8     { return t.buf.(*hostBuf).i8 }

func (c *cpuBackend) result(shape []int, data []float32) Tensor {
	return makeTensor(c, F32, RowMajor, shape, nil, &hostBuf{f32: data})
}

// ---- matmul family (dtype-dispatched on the WEIGHT — the f32/Q8 twin collapse) -----

// MatMul: y[o] = Σ_i W[o,i]·x[i]. F32 reproduces matRows/parMatRows (fdot reduction);
// Q8_0 reproduces qMatRows (quantize the activation, per-block int8 dot). One method,
// two dtypes — the duplication the audit ranked hardest, expressed as dispatch.
func (c *cpuBackend) MatMul(w, x Tensor) Tensor {
	out, in := w.Shape[0], w.Shape[1]
	xf := c.f32(x)
	y := make([]float32, out)
	switch w.Dtype {
	case F32:
		wf := c.f32(w)
		for o := 0; o < out; o++ {
			y[o] = fdot(wf[o*in:o*in+in], xf)
		}
	case Q8_0:
		blk := w.Quant.Block
		nblk := in / blk
		qx, dx := quantizeVecQ8(xf, blk)
		wc, ws := c.i8(w), w.Quant.Scale
		for o := 0; o < out; o++ {
			y[o] = qdot8scalar(wc[o*in:o*in+in], ws[o*nblk:o*nblk+nblk], qx, dx, blk)
		}
	default:
		panic("compute: cpu-ref MatMul unsupported weight dtype " + w.Dtype.String())
	}
	return c.result([]int{out}, y)
}

// BatchedMatMul: Y[t,o] = Σ_i W[o,i]·X[t,i] — the prefill GEMM. F32 reproduces
// matMulBatch; Q8_0 reproduces qGemm8scalar (qgemm8cell deferred-reduction, lanes=16).
func (c *cpuBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	out, in := w.Shape[0], w.Shape[1]
	Xf := c.f32(X)
	Y := make([]float32, P*out)
	switch w.Dtype {
	case F32:
		wf := c.f32(w)
		for o := 0; o < out; o++ {
			r := wf[o*in : o*in+in]
			for t := 0; t < P; t++ {
				Y[t*out+o] = fdot(r, Xf[t*in:t*in+in])
			}
		}
	case Q8_0:
		blk := w.Quant.Block
		nblk := in / blk
		wc, ws := c.i8(w), w.Quant.Scale
		qx := make([]int8, P*in)
		dx := make([]float32, P*nblk)
		for t := 0; t < P; t++ {
			qxr, dxr := quantizeVecQ8(Xf[t*in:t*in+in], blk)
			copy(qx[t*in:], qxr)
			copy(dx[t*nblk:], dxr)
		}
		for o := 0; o < out; o++ {
			for t := 0; t < P; t++ {
				Y[t*out+o] = qgemm8cell(wc[o*in:o*in+in], ws[o*nblk:o*nblk+nblk], qx[t*in:t*in+in], dx[t*nblk:t*nblk+nblk], nblk, blk)
			}
		}
	default:
		panic("compute: cpu-ref BatchedMatMul unsupported weight dtype " + w.Dtype.String())
	}
	return makeTensor(c, F32, RowMajor, []int{P, out}, nil, &hostBuf{f32: Y})
}

// RMSNorm reproduces the model's serial in-order sum-of-squares (the one marked
// load-bearing for R2/R14 — NOT the fdot twin), then x·inv·weight.
func (c *cpuBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	xf, wf := c.f32(x), c.f32(weight)
	var ss float32
	for _, v := range xf {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(xf))+eps)))
	out := make([]float32, len(xf))
	for i, v := range xf {
		out[i] = v * inv * wf[i]
	}
	return c.result(append([]int(nil), x.Shape...), out)
}

// RoPE rotates each of nHeads head-vectors (length headDim) at absolute position pos,
// using HF's non-interleaved rotate_half — reproducing ropeRow + applyRopeRow. Returns a
// new tensor (functional/SSA value semantics, the shape a graph backend wants).
func (c *cpuBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	xf := c.f32(x)
	out := append([]float32(nil), xf...)
	cos, sin := ropeRow(theta, headDim, pos)
	half := headDim / 2
	for h := 0; h < nHeads; h++ {
		hv := out[h*headDim : (h+1)*headDim]
		for j := 0; j < half; j++ {
			a, b := hv[j], hv[j+half]
			hv[j] = a*cos[j] - b*sin[j]
			hv[j+half] = b*cos[j] + a*sin[j]
		}
	}
	return c.result(append([]int(nil), x.Shape...), out)
}

// SwiGLU: silu(gate)·up, elementwise (reproduces the MLP activation).
func (c *cpuBackend) SwiGLU(gate, up Tensor) Tensor {
	g, u := c.f32(gate), c.f32(up)
	out := make([]float32, len(g))
	for i := range g {
		out[i] = silu(g[i]) * u[i]
	}
	return c.result(append([]int(nil), gate.Shape...), out)
}

func (c *cpuBackend) AddInPlace(dst, src Tensor) {
	d, s := c.f32(dst), c.f32(src)
	for i := range d {
		d[i] += s[i]
	}
}

func (c *cpuBackend) AddBias(dst, bias Tensor) {
	d, b := c.f32(dst), c.f32(bias)
	for i := range d {
		d[i] += b[i]
	}
}

// Attention is the fused causal GQA op: for each query head, score against the cached
// keys of its KV group (single-accumulator dot · scale — matching tokenHidden), softmax,
// then the in-order ΣwV. q is one position's [nH*hd]; returns [nH*hd]. It downcasts the
// KVStore to the reference's own cpuKV (a backend and its KV are a matched pair).
func (c *cpuBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	ck := kv.(*cpuKV)
	hd, nKV := ck.cfg.HeadDim, ck.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	qf := c.f32(q)
	K := ck.K[layer]
	V := ck.V[layer]
	nPos := len(K) / w
	out := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := qf[h*hd : (h+1)*hd]
		scores := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			kh := K[j*w+kvh*hd : j*w+(kvh+1)*hd]
			scores[j] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		o := out[h*hd : (h+1)*hd]
		for j := 0; j < nPos; j++ {
			vh := V[j*w+kvh*hd : j*w+(kvh+1)*hd]
			wj := scores[j]
			for d := 0; d < hd; d++ {
				o[d] += wj * vh[d]
			}
		}
	}
	return c.result([]int{nH * hd}, out)
}

func (c *cpuBackend) Argmax(logits Tensor) int { return argmaxF32(c.f32(logits)) }

// ---- KV store (verbatim wrap of the model's flat row-major cache) ----------------

func (c *cpuBackend) NewKV(cfg KVConfig) KVStore {
	return &cpuKV{
		be:   c,
		cfg:  cfg,
		K:    make([][]float32, cfg.NumLayers),
		Kraw: make([][]float32, cfg.NumLayers),
		V:    make([][]float32, cfg.NumLayers),
	}
}

// cpuKV reproduces model.KVCache: flat row-major K/Kraw/V per layer, a shared pos[]. The
// numerics of AppendKV/Evict/Clone are byte-for-byte the model's, so the kvmmu witness
// (evict == never-saw, max|Δ|=0) is unaffected when the model adopts this.
type cpuKV struct {
	be   Backend
	cfg  KVConfig
	K    [][]float32
	Kraw [][]float32
	V    [][]float32
	pos  []int
}

func (k *cpuKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *cpuKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	hb := k.be.(*cpuBackend)
	k.Kraw[layer] = append(k.Kraw[layer], hb.f32(kRaw)...)
	k.K[layer] = append(k.K[layer], hb.f32(kRoPE)...)
	k.V[layer] = append(k.V[layer], hb.f32(v)...)
	if layer == 0 { // pos is shared across layers; record once per position
		k.pos = append(k.pos, pos)
	}
}

func (k *cpuKV) Len() int   { return len(k.pos) }
func (k *cpuKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *cpuKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := len(k.K[layer]) / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &hostBuf{f32: k.K[layer]})
}
func (k *cpuKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := len(k.V[layer]) / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &hostBuf{f32: k.V[layer]})
}

// Evict reproduces model.KVCache.Evict exactly: compact survivors out of every layer,
// then re-derive the post-RoPE K of any survivor whose absolute position changed from
// its pre-RoPE Kraw in a single rotation at the new index (bit-exact to never-saw).
func (k *cpuKV) Evict(from, n int) int {
	if from < 0 || n <= 0 || from >= len(k.pos) {
		return 0
	}
	end := from + n
	if end > len(k.pos) {
		end = len(k.pos)
	}
	w := k.stride()
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	for l := 0; l < k.cfg.NumLayers; l++ {
		k.K[l] = append(k.K[l][:from*w], k.K[l][end*w:]...)
		k.Kraw[l] = append(k.Kraw[l][:from*w], k.Kraw[l][end*w:]...)
		k.V[l] = append(k.V[l][:from*w], k.V[l][end*w:]...)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		if k.pos[i] != i {
			cos, sin := ropeRow(k.cfg.RopeTheta, hd, i)
			for l := 0; l < k.cfg.NumLayers; l++ {
				for h := 0; h < nKV; h++ {
					dst := k.K[l][i*w+h*hd : i*w+(h+1)*hd]
					copy(dst, k.Kraw[l][i*w+h*hd:i*w+(h+1)*hd])
					applyRope(dst, cos, sin)
				}
			}
		}
		k.pos[i] = i
	}
	return end - from
}

func (k *cpuKV) Clone() KVStore {
	n := &cpuKV{be: k.be, cfg: k.cfg,
		K: make([][]float32, len(k.K)), Kraw: make([][]float32, len(k.Kraw)), V: make([][]float32, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	for l := range k.K {
		n.K[l] = append([]float32(nil), k.K[l]...)
		n.Kraw[l] = append([]float32(nil), k.Kraw[l]...)
		n.V[l] = append([]float32(nil), k.V[l]...)
	}
	return n
}

// ---- reference kernels (reproduce model numerics byte-for-byte) ------------------

// fdot: 8-accumulator inner product in the model's exact fixed combine order.
func fdot(r, x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(r)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += r[i] * x[i]
		s1 += r[i+1] * x[i+1]
		s2 += r[i+2] * x[i+2]
		s3 += r[i+3] * x[i+3]
		s4 += r[i+4] * x[i+4]
		s5 += r[i+5] * x[i+5]
		s6 += r[i+6] * x[i+6]
		s7 += r[i+7] * x[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += r[i] * x[i]
	}
	return s
}

// dot: single-accumulator in-order (attention scores use this in the model).
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func softmaxInPlace(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

func silu(z float32) float32 { return z / (1 + float32(math.Exp(float64(-z)))) }

func argmaxF32(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// ropeRow / applyRope reproduce the model's RoPE cos/sin table + non-interleaved rotation.
func ropeRow(theta float64, headDim, pos int) (cos, sin []float32) {
	half := headDim / 2
	cos = make([]float32, half)
	sin = make([]float32, half)
	for j := 0; j < half; j++ {
		inv := 1.0 / math.Pow(theta, float64(2*j)/float64(headDim))
		a := float64(pos) * inv
		cos[j] = float32(math.Cos(a))
		sin[j] = float32(math.Sin(a))
	}
	return
}

func applyRope(hv, cos, sin []float32) {
	half := len(cos)
	for j := 0; j < half; j++ {
		a, b := hv[j], hv[j+half]
		hv[j] = a*cos[j] - b*sin[j]
		hv[j+half] = b*cos[j] + a*sin[j]
	}
}

// q8round: truncate-then-exact-fractional round, half away from zero — byte-identical to
// model.q8round (and to math.Round over the int8 code range).
func q8round(x float32) int8 {
	t := int32(x)
	f := x - float32(t)
	if f >= 0.5 {
		t++
	} else if f <= -0.5 {
		t--
	}
	if t > 127 {
		return 127
	}
	if t < -127 {
		return -127
	}
	return int8(t)
}

// quantizeVecQ8 quantizes one activation vector to Q8_0 (per-block d=amax/127), matching
// model.quantizeVecQ8.
func quantizeVecQ8(x []float32, block int) (q []int8, d []float32) {
	nblk := len(x) / block
	q = make([]int8, len(x))
	d = make([]float32, nblk)
	for b := 0; b < nblk; b++ {
		blk := x[b*block : b*block+block]
		var amax float32
		for _, v := range blk {
			if a := absf(v); a > amax {
				amax = a
			}
		}
		dd := amax / 127
		d[b] = dd
		if dd == 0 {
			continue
		}
		inv := 1.0 / dd
		for i := 0; i < block; i++ {
			q[b*block+i] = q8round(blk[i] * inv)
		}
	}
	return
}

// qdot8scalar: Q8_0 inner product, 4 int32 accumulators per block, fixed float combine —
// matching model.qdot8scalar (the decode kernel reference).
func qdot8scalar(qw []int8, dw []float32, qx []int8, dx []float32, block int) float32 {
	nblk := len(qw) / block
	var acc float32
	for b := 0; b < nblk; b++ {
		wb := qw[b*block:]
		xb := qx[b*block:]
		var s0, s1, s2, s3 int32
		for i := 0; i < block; i += 4 {
			s0 += int32(wb[i]) * int32(xb[i])
			s1 += int32(wb[i+1]) * int32(xb[i+1])
			s2 += int32(wb[i+2]) * int32(xb[i+2])
			s3 += int32(wb[i+3]) * int32(xb[i+3])
		}
		acc += float32((s0+s1)+(s2+s3)) * dw[b] * dx[b]
	}
	return acc
}

// qgemm8cell: the deferred-reduction Q8_0 dot the prefill tile kernel is pinned to
// (lanes=16), matching model.qgemm8cell — block partials kept in 16 float lanes, reduced
// once via the same pairwise tree.
func qgemm8cell(qw []int8, dw []float32, qx []int8, dx []float32, nblk, block int) float32 {
	const lanes = 16
	var acc [16]float32
	for b := 0; b < nblk; b++ {
		wb := qw[b*block : b*block+block]
		xb := qx[b*block : b*block+block]
		var p [16]int32
		for kk := 0; kk < lanes; kk++ {
			p[kk] = int32(wb[2*kk])*int32(xb[2*kk]) + int32(wb[2*kk+1])*int32(xb[2*kk+1])
		}
		s := dw[b] * dx[b]
		for kk := 0; kk < lanes; kk++ {
			acc[kk] += float32(p[kk]) * s
		}
	}
	for kk := 0; kk < 8; kk++ {
		acc[kk] += acc[kk+8]
	}
	for kk := 0; kk < 4; kk++ {
		acc[kk] += acc[kk+4]
	}
	acc[0] += acc[2]
	acc[1] += acc[3]
	return acc[0] + acc[1]
}
