//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var _ Inkling2Device = (*metalBackend)(nil)

var metalInk2 struct {
	once      sync.Once
	dev       *metalgemm.Ink2Device
	err       error
	profiling bool
	fallback  *metalBackend
}

// metalInk2Device is the process-wide deferred encoder behind every Ink2 op.
func metalInk2Device() (*metalgemm.Ink2Device, error) {
	metalInk2.once.Do(func() {
		metalInk2.dev, metalInk2.err = metalgemm.NewInk2Device()
		metalInk2.profiling = metalInk2.err == nil && metalInk2.dev.ProfileMode() != metalgemm.Ink2ProfileOff
	})
	return metalInk2.dev, metalInk2.err
}

// ink2MetalScope labels the dispatches of the op that follows for the profiler.
func ink2MetalScope(op string) {
	if _, err := metalInk2Device(); err == nil && metalInk2.profiling {
		metalInk2.dev.SetProfileScope(op)
	}
}

// MetalInkling2Stats reports the process-wide inkling2 device's counters; ok is
// false without a Metal device.
func MetalInkling2Stats() (metalgemm.Ink2Stats, bool) {
	d, err := metalInk2Device()
	if err != nil {
		return metalgemm.Ink2Stats{}, false
	}
	return d.Stats(), true
}

// MetalInkling2Profile returns the per-dispatch profile of the process-wide
// inkling2 device, which is empty unless FAK_INK2_PROFILE selected a mode.
func MetalInkling2Profile() (metalgemm.Ink2Profile, bool) {
	d, err := metalInk2Device()
	if err != nil {
		return metalgemm.Ink2Profile{}, false
	}
	return d.Profile(), true
}

// MetalInkling2ProfileReset starts a new profile window at a fence.
func MetalInkling2ProfileReset() error {
	d, err := metalInk2Device()
	if err != nil {
		return err
	}
	return d.ResetProfile()
}

// MetalInkling2Backend returns the Metal backend that carries the inkling2
// device ops. They are pure MSL, so when MPS is off and no "metal" backend is
// registered this still returns one; only its Ink2 ops and Free are usable then.
func MetalInkling2Backend() (Backend, bool) {
	if metalDev != nil {
		return metalDev, true
	}
	if _, err := metalInk2Device(); err != nil {
		return nil, false
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	if metalInk2.fallback == nil {
		metalInk2.fallback = &metalBackend{name: "metal", tier: metalgemm.DeviceName(), totalMem: metalDeviceMemoryTotal()}
	}
	return metalInk2.fallback, true
}

// metalInk2Weight is a weight the packed kernels read in place from the mapped
// GGUF: one expert of m, or (expert < 0) its whole stack.
type metalInk2Weight struct {
	m      *metalgemm.PackedGGUFMatrix
	expert int
}

func (*metalInk2Weight) Ready() bool { return true }

var metalInk2Dtype = map[uint32]Dtype{0: F32, 1: F16, 8: Q8_0, 11: Q3_K, 12: Q4_K, 13: Q5_K, 14: Q6_K,
	17: IQ2_XS, 18: IQ3_XXS, 20: IQ4_NL, 21: IQ3_S, 22: IQ2_S, 30: BF16, 40: NVFP4}

// MetalInkling2Weight wraps a packed GGUF matrix as a resident weight of the
// Metal inkling2 ops: expert e as a [rows, cols] tensor, or with expert < 0 the
// whole stack as [experts*rows, cols]. The matrix's store keeps the bytes;
// Free on the tensor is a no-op.
func MetalInkling2Weight(m *metalgemm.PackedGGUFMatrix, expert int) (Tensor, error) {
	be, ok := MetalInkling2Backend()
	if !ok {
		return Tensor{}, metalgemm.ErrMetalUnavailable
	}
	typ, rows, cols, experts := m.Dims()
	dt, ok := metalInk2Dtype[typ]
	if !ok || expert >= experts {
		return Tensor{}, fmt.Errorf("metal inkling2 weight: GGML type %d, expert %d of %d", typ, expert, experts)
	}
	shape := []int{rows, cols}
	if expert < 0 {
		expert, shape[0] = -1, experts*rows
	}
	return makeTensor(be, dt, RowMajor, shape, nil, &metalInk2Weight{m: m, expert: expert}), nil
}

func ink2MetalErr(op string, err error) error {
	if err != nil {
		return fmt.Errorf("metal inkling2 %s: %w", op, err)
	}
	return nil
}

// ink2MetalRef resolves a Ref to its MTLBuffer; caller holds metalMu.
func ink2MetalRef(r Ink2Ref) (metalgemm.Ink2Ref, error) {
	if r.T.buf == nil {
		return metalgemm.Ink2Ref{}, nil
	}
	b, ok := r.T.buf.(*metalBuf)
	if !ok || b.ptr == nil {
		return metalgemm.Ink2Ref{}, fmt.Errorf("operand %T is not a live Metal buffer", r.T.buf)
	}
	return metalgemm.Ink2Ref{Buf: metalgemm.Ink2AdoptBuffer(b.ptr), Off: r.Off}, nil
}

// ink2MetalBind returns the device and the resolved operands of one op; caller
// holds metalMu.
func ink2MetalBind(refs ...Ink2Ref) (*metalgemm.Ink2Device, []metalgemm.Ink2Ref, error) {
	d, err := metalInk2Device()
	if err != nil {
		return nil, nil, err
	}
	out := make([]metalgemm.Ink2Ref, len(refs))
	for i, r := range refs {
		if out[i], err = ink2MetalRef(r); err != nil {
			return nil, nil, fmt.Errorf("operand %d: %w", i, err)
		}
	}
	return d, out, nil
}

func ink2MetalRefs(refs []Ink2Ref) ([]metalgemm.Ink2Ref, error) {
	if refs == nil {
		return nil, nil
	}
	_, out, err := ink2MetalBind(refs...)
	return out, err
}

// ink2MetalWeight resolves a weight tensor: a packed GGUF view, or dense F32
// rows the resident forward uploaded into a device buffer. stack selects a
// packed matrix's whole expert stack and reports false when a device id cannot
// index it.
func ink2MetalWeight(w Tensor, stack bool) (metalgemm.Ink2Weight, bool, error) {
	if len(w.Shape) != 2 {
		return metalgemm.Ink2Weight{}, false, fmt.Errorf("weight shape %v", w.Shape)
	}
	switch b := w.buf.(type) {
	case *metalInk2Weight:
		if stack != (b.expert < 0) {
			return metalgemm.Ink2Weight{}, false, fmt.Errorf("expert view %d used as stack=%v", b.expert, stack)
		}
		if stack {
			v, ok := b.m.Ink2Stack()
			return v, ok, nil
		}
		v, err := b.m.Ink2Weight(b.expert)
		return v, true, err
	case *metalBuf:
		if w.Dtype != F32 || b.ptr == nil {
			break
		}
		return metalgemm.Ink2DenseF32(metalgemm.Ink2Ref{Buf: metalgemm.Ink2AdoptBuffer(b.ptr)}, w.Shape[0], w.Shape[1]), true, nil
	}
	return metalgemm.Ink2Weight{}, false, fmt.Errorf("unsupported %s weight in %T", w.Dtype, w.buf)
}

func (c *metalBackend) Ink2Alloc(n int) (t Tensor, err error) {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, err := metalInk2Device()
	if err != nil {
		return Tensor{}, ink2MetalErr("alloc", err)
	}
	if n < 0 {
		return Tensor{}, fmt.Errorf("metal inkling2 alloc: %d floats", n)
	}
	defer func() {
		if r := recover(); r != nil {
			t, err = Tensor{}, fmt.Errorf("metal inkling2 alloc of %d floats: %v", n, r)
		}
	}()
	// A metalBuf, so the driver's Backend.Free releases it like any other tensor.
	b := c.dallocClass(n*4, MemoryActivation, "inkling2-resident")
	if err := d.Zero(metalgemm.Ink2Ref{Buf: metalgemm.Ink2AdoptBuffer(b.ptr)}, n); err != nil {
		return Tensor{}, ink2MetalErr("zero", err)
	}
	return makeTensor(c, F32, RowMajor, []int{n}, nil, b), nil
}

func (c *metalBackend) Ink2Write(dst Ink2Ref, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(dst)
	if err != nil {
		return ink2MetalErr("write", err)
	}
	return ink2MetalErr("write", d.Write(r[0], src))
}

func (c *metalBackend) Ink2Read(src Ink2Ref, n int) ([]float32, error) {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(src)
	if err != nil {
		return nil, ink2MetalErr("read", err)
	}
	out, err := d.Read(r[0], n)
	return out, ink2MetalErr("read", err)
}

func (c *metalBackend) Ink2ReadInts(src Ink2Ref, n int) ([]int32, error) {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(src)
	if err != nil {
		return nil, ink2MetalErr("read-ints", err)
	}
	out, err := d.ReadInts(r[0], n)
	return out, ink2MetalErr("read-ints", err)
}

func (c *metalBackend) Ink2MatVec(w Tensor, x, y Ink2Ref) error {
	ink2MetalScope("Ink2MatVec")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, y)
	if err != nil {
		return ink2MetalErr("matvec", err)
	}
	wv, _, err := ink2MetalWeight(w, false)
	if err != nil {
		return ink2MetalErr("matvec", err)
	}
	return ink2MetalErr("matvec", d.MatVec(wv, r[0], r[1]))
}

func (c *metalBackend) Ink2MatVecQ8(w Tensor, x, gate, q Ink2Ref, quantize bool, y Ink2Ref) (bool, error) {
	ink2MetalScope("Ink2MatVecQ8")
	if w.Dtype != Q8_0 || len(w.Shape) != 2 {
		return false, nil
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, gate, q, y)
	if err != nil {
		return false, ink2MetalErr("matvec q8_0 shared", err)
	}
	wv, _, err := ink2MetalWeight(w, false)
	if err != nil {
		return false, ink2MetalErr("matvec q8_0 shared", err)
	}
	ok, err := d.MatVecQ8(wv, r[0], r[1], r[2], quantize, r[3])
	return ok, ink2MetalErr("matvec q8_0 shared", err)
}

func (c *metalBackend) Ink2ExpertMatVec(stack Tensor, expertRows int, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref, situ *Ink2Situ) (bool, error) {
	ink2MetalScope("Ink2ExpertMatVec")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(ids, x, y, scale)
	if err != nil {
		return false, ink2MetalErr("expert matvec", err)
	}
	wv, indexed, err := ink2MetalWeight(stack, true)
	if err != nil || !indexed {
		return false, ink2MetalErr("expert matvec", err)
	}
	if wv.Experts == 1 && expertRows > 0 && wv.Rows%expertRows == 0 {
		// dense rows carry no expert count of their own
		wv.Experts, wv.Rows = wv.Rows/expertRows, expertRows
	}
	if wv.Rows != expertRows {
		return false, fmt.Errorf("metal inkling2 expert matvec: %d rows per expert, stack has %d", expertRows, wv.Rows)
	}
	ok, err := d.ExpertMatVec(wv, r[0], r[1], xStride, r[2], k, r[3], (*metalgemm.Ink2Situ)(situ))
	return ok, ink2MetalErr("expert matvec", err)
}

func (c *metalBackend) Ink2RMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	ink2MetalScope("Ink2RMS")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, y)
	if err != nil {
		return ink2MetalErr("rms", err)
	}
	return ink2MetalErr("rms", d.RMS(r[0], r[1], r[2], rows, n, eps))
}

func (c *metalBackend) Ink2CtrlPair(cur, msg, pair Ink2Ref, n int) error {
	ink2MetalScope("Ink2CtrlPair")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(cur, msg, pair)
	if err != nil {
		return ink2MetalErr("ctrl pair", err)
	}
	return ink2MetalErr("ctrl pair", d.CtrlPair(r[0], r[1], r[2], n))
}

func (c *metalBackend) Ink2Mix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n int) error {
	ink2MetalScope("Ink2Mix")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(queries, out)
	if err != nil {
		return ink2MetalErr("mix", err)
	}
	ss, err := ink2MetalRefs(states)
	if err != nil {
		return ink2MetalErr("mix states", err)
	}
	ns, err := ink2MetalRefs(norms)
	if err != nil {
		return ink2MetalErr("mix norms", err)
	}
	return ink2MetalErr("mix", d.Mix(ss, r[0], nq, ns, eps, r[1], n))
}

func (c *metalBackend) Ink2Control(ctl Ink2Controls, g, out Ink2Ref, n, rank int, norms []Ink2Ref, eps float32) error {
	ink2MetalScope("Ink2Control")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(g, out)
	if err != nil {
		return ink2MetalErr("control", err)
	}
	var mc metalgemm.Ink2Controls
	for _, f := range []struct {
		dst *[]metalgemm.Ink2Ref
		src []Ink2Ref
	}{{&mc.Cur, ctl.Cur}, {&mc.Msg, ctl.Msg}, {&mc.Down, ctl.Down}, {&mc.Gain, ctl.Gain}, {&mc.Forget, ctl.Forget}} {
		if *f.dst, err = ink2MetalRefs(f.src); err != nil {
			return ink2MetalErr("control", err)
		}
	}
	ns, err := ink2MetalRefs(norms)
	if err != nil {
		return ink2MetalErr("control norms", err)
	}
	return ink2MetalErr("control", d.Control(mc, r[0], r[1], n, rank, ns, eps))
}

func (c *metalBackend) Ink2Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error {
	ink2MetalScope("Ink2Conv")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dyn, hist, out, resid)
	if err != nil {
		return ink2MetalErr("conv", err)
	}
	return ink2MetalErr("conv", d.Conv(r[0], r[1], r[2], r[3], r[4], r[5], n, heads, kn, norm))
}

func (c *metalBackend) Ink2CombineConv(down, wts Ink2Ref, k int, w, hist, out, resid Ink2Ref, n, kn int, norm float32) error {
	ink2MetalScope("Ink2CombineConv")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, w, hist, out, resid)
	if err != nil {
		return ink2MetalErr("combine conv", err)
	}
	return ink2MetalErr("combine conv", d.CombineConv(r[0], r[1], k, r[2], r[3], r[4], r[5], n, kn, norm))
}

func (c *metalBackend) Ink2QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error {
	ink2MetalScope("Ink2QKVPost")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, k, v, qclip, rclip, kcache, vcache, meta)
	if err != nil {
		return ink2MetalErr("qkv post", err)
	}
	var mcv *metalgemm.Ink2KVConv
	if cv != nil {
		_, t, err := ink2MetalBind(cv.KW, cv.VW, cv.KDyn, cv.VDyn, cv.KHist, cv.VHist)
		if err != nil {
			return ink2MetalErr("qkv post conv", err)
		}
		mcv = &metalgemm.Ink2KVConv{KW: t[0], VW: t[1], KDyn: t[2], VDyn: t[3], KHist: t[4], VHist: t[5],
			Kn: cv.Kn, KNorm: cv.KNorm, VNorm: cv.VNorm}
	}
	return ink2MetalErr("qkv post", d.QKVPost(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], metalgemm.Ink2AttnParams(p), cap, mcv))
}

func (c *metalBackend) Ink2Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	ink2MetalScope("Ink2Attend")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, sinks, relBias, kcache, vcache, scores, out, meta)
	if err != nil {
		return ink2MetalErr("attend", err)
	}
	return ink2MetalErr("attend", d.Attend(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], metalgemm.Ink2AttnParams(p), extent, window, cap))
}

func (c *metalBackend) Ink2SigmoidMul(x, gate Ink2Ref, n int) error {
	ink2MetalScope("Ink2SigmoidMul")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, gate)
	if err != nil {
		return ink2MetalErr("sigmoid mul", err)
	}
	return ink2MetalErr("sigmoid mul", d.SigmoidMul(r[0], r[1], n))
}

func (c *metalBackend) Ink2Route(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32) error {
	ink2MetalScope("Ink2Route")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, bias, ids, wts)
	if err != nil {
		return ink2MetalErr("route", err)
	}
	return ink2MetalErr("route", d.Route(r[0], r[1], r[2], r[3], nexp, k, scale))
}

func (c *metalBackend) Ink2ExpertScale(y, ids, scale Ink2Ref, k, rows int) error {
	ink2MetalScope("Ink2ExpertScale")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(y, ids, scale)
	if err != nil {
		return ink2MetalErr("expert scale", err)
	}
	return ink2MetalErr("expert scale", d.ExpertScale(r[0], r[1], r[2], k, rows))
}

func (c *metalBackend) Ink2Situ(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	ink2MetalScope("Ink2Situ")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(gu, act)
	if err != nil {
		return ink2MetalErr("situ", err)
	}
	return ink2MetalErr("situ", d.Situ(r[0], r[1], k, width, actStride, beta, linearBeta))
}

func (c *metalBackend) Ink2Combine(down, wts, out Ink2Ref, k, n int) error {
	ink2MetalScope("Ink2Combine")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, out)
	if err != nil {
		return ink2MetalErr("moe combine", err)
	}
	return ink2MetalErr("moe combine", d.Combine(r[0], r[1], r[2], k, n))
}

func (c *metalBackend) Ink2Add(a, b, out Ink2Ref, n int) error {
	ink2MetalScope("Ink2Add")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(a, b, out)
	if err != nil {
		return ink2MetalErr("add", err)
	}
	return ink2MetalErr("add", d.Add(r[0], r[1], r[2], n))
}

func (c *metalBackend) Ink2SoftcapArgmax(logits Ink2Ref, n int, softcap float32, outIdx Ink2Ref) error {
	ink2MetalScope("Ink2SoftcapArgmax")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, outIdx)
	if err != nil {
		return ink2MetalErr("softcap argmax", err)
	}
	return ink2MetalErr("softcap argmax", d.SoftcapArgmax(r[0], n, softcap, r[1]))
}

// The resident driver treats a failed Ink2GraphBegin as "run eagerly", which
// on Metal is already one command buffer per token.
func (c *metalBackend) Ink2GraphBegin() error  { return metalgemm.ErrInk2GraphUnsupported }
func (c *metalBackend) Ink2GraphEnd() error    { return metalgemm.ErrInk2GraphUnsupported }
func (c *metalBackend) Ink2GraphLaunch() error { return metalgemm.ErrInk2GraphUnsupported }
func (c *metalBackend) Ink2GraphReset()        {}

// A verify chunk is already one command buffer, so there is no slot to capture.
func (c *metalBackend) Ink2GraphEndSlot(int) error    { return metalgemm.ErrInk2GraphUnsupported }
func (c *metalBackend) Ink2GraphLaunchSlot(int) error { return metalgemm.ErrInk2GraphUnsupported }
