//go:build linux && rocm && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakrocm -lamdhip64 -lstdc++ -lm
#include <stdlib.h>
#include "rocm_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

// rocmMu serializes the first ROCm backend's single HIP stream and device.
var rocmMu sync.Mutex

type rocmBackend struct {
	name      string
	tier      string
	totalMem  int64
	transient []*rocmBuf
}

type rocmBuf struct {
	ptr, scales unsafe.Pointer
	n, scalesN  int
	class       MemoryClass
	owned       bool
	alive       func() bool
	freed       uint32
}

func (b *rocmBuf) Ready() bool {
	if b == nil || b.ptr == nil || atomic.LoadUint32(&b.freed) != 0 {
		return false
	}
	return b.alive == nil || b.alive()
}

func init() {
	var name, gfx [256]C.char
	var total C.size_t
	if C.frocm_init(0, &name[0], C.int(len(name)), &gfx[0], C.int(len(gfx)), &total) != 0 {
		return
	}
	b := &rocmBackend{name: "rocm", tier: C.GoString(&gfx[0]), totalMem: uint64ToCapInt64(uint64(total))}
	if b.tier == "" {
		b.tier = C.GoString(&name[0])
	}
	Register(b)
}

func (r *rocmBackend) Name() string            { return r.name }
func (r *rocmBackend) Tier() string            { return r.tier }
func (r *rocmBackend) Class() CorrectnessClass { return Approx }
func (r *rocmBackend) Caps() Caps {
	return Caps{DeviceMemory: true, UploadDtype: true, CapacityProbe: r.totalMem > 0, FusedAttn: true}
}

func (r *rocmBackend) DeviceMemory() (total, free int64, known bool) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	var f, t C.size_t
	if C.frocm_mem_info(&f, &t) == 0 && t > 0 {
		return uint64ToCapInt64(uint64(t)), uint64ToCapInt64(uint64(f)), true
	}
	if r.totalMem > 0 {
		return r.totalMem, FreeUnknown, true
	}
	return 0, FreeUnknown, false
}

func rocmLastError(op string) error {
	p := C.frocm_last_error()
	msg := ""
	if p != nil {
		msg = C.GoString(p)
	}
	if msg == "" {
		msg = "native call failed"
	}
	return fmt.Errorf("rocm: %s: %s", op, msg)
}

func rocmCheck(code C.int, op string) {
	if code != 0 {
		panic(rocmLastError(op))
	}
}

func checkedTensorNumel(shape []int) (int, bool) {
	n := 1
	for _, d := range shape {
		if d < 0 || (d != 0 && n > int(^uint(0)>>1)/d) {
			return 0, false
		}
		n *= d
	}
	return n, true
}

func checkedROCmBytes(shape []int, width int) (int, bool) {
	n, ok := checkedTensorNumel(shape)
	if !ok || width <= 0 || (n != 0 && n > int(^uint(0)>>1)/width) {
		return 0, false
	}
	return n * width, true
}

func rocmDim(n int, name string) C.int {
	if n < 0 || uint64(n) > uint64(^uint32(0)>>1) {
		panic(fmt.Errorf("rocm: %s=%d exceeds native int range", name, n))
	}
	return C.int(n)
}

func (r *rocmBackend) alloc(n int, class MemoryClass, site string) *rocmBuf {
	if n <= 0 {
		n = 1
	}
	var p unsafe.Pointer
	if C.frocm_malloc(&p, C.size_t(n)) != 0 || p == nil {
		panic(&DeviceAllocError{Bytes: n, Site: site, Class: class})
	}
	return &rocmBuf{ptr: p, n: n, class: class, owned: true}
}

func (r *rocmBackend) tensor(shape []int, dt Dtype, class MemoryClass, transient bool) (Tensor, *rocmBuf) {
	nbytes, ok := checkedROCmBytes(shape, dt.Bytes())
	if !ok {
		panic(fmt.Errorf("rocm: invalid tensor shape %v", shape))
	}
	b := r.alloc(nbytes, class, "tensor")
	t := makeTensor(r, dt, RowMajor, append([]int(nil), shape...), nil, b)
	if transient {
		r.transient = append(r.transient, b)
	}
	return t, b
}

func (r *rocmBackend) deviceBuf(t Tensor, op string) *rocmBuf {
	b, ok := t.buf.(*rocmBuf)
	if !ok || b == nil || b.ptr == nil || t.be != r || atomic.LoadUint32(&b.freed) != 0 || (b.alive != nil && !b.alive()) {
		panic(fmt.Errorf("rocm: %s requires a live tensor owned by this backend", op))
	}
	want := 0
	var valid bool
	switch t.Dtype {
	case F32, F16, BF16:
		want, valid = checkedROCmBytes(t.Shape, t.Dtype.Bytes())
	case Q8_0:
		if len(t.Shape) == 2 && t.Quant != nil && t.Quant.Block > 0 && t.Shape[1] >= 0 && t.Shape[1]%t.Quant.Block == 0 {
			want, valid = checkedTensorNumel(t.Shape)
			scaleBytes, scaleOK := checkedROCmBytes([]int{t.Shape[0], t.Shape[1] / t.Quant.Block}, F32.Bytes())
			valid = valid && scaleOK && b.scales != nil && scaleBytes <= b.scalesN
		}
	case Q4_K, Q5_K, Q6_K:
		if len(t.Shape) == 2 && t.Shape[0] >= 0 && t.Shape[1] >= 0 && t.Shape[1]%256 == 0 {
			block := map[Dtype]int{Q4_K: 144, Q5_K: 176, Q6_K: 210}[t.Dtype]
			want, valid = checkedROCmBytes([]int{t.Shape[0], t.Shape[1] / 256}, block)
		}
	default:
		valid = false
	}
	if !valid || want > b.n {
		panic(fmt.Errorf("rocm: %s tensor metadata exceeds resident storage dtype=%s shape=%v bytes=%d resident=%d", op, t.Dtype, t.Shape, want, b.n))
	}
	return b
}

func requireF32Owned(r *rocmBackend, t Tensor, op string) *rocmBuf {
	if t.Dtype != F32 || t.Layout != RowMajor {
		panic(fmt.Errorf("rocm: %s requires row-major f32 tensor, got %s", op, t.Dtype))
	}
	return r.deviceBuf(t, op)
}

func (r *rocmBackend) Upload(t Tensor, as Dtype) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	h, ok := t.buf.(HostBuffer)
	if !ok || t.Layout != RowMajor {
		panic(fmt.Errorf("rocm: Upload requires row-major host tensor"))
	}
	numel, valid := checkedTensorNumel(t.Shape)
	if !valid {
		panic(fmt.Errorf("rocm: Upload invalid shape %v", t.Shape))
	}
	if as == Q8_0 && t.Dtype == F32 {
		if len(t.Shape) != 2 || t.Shape[1]%32 != 0 || len(h.F32()) != numel {
			panic(fmt.Errorf("rocm: Q8 upload requires [out,in] with in divisible by 32"))
		}
		q := QuantizeQ8(t.be, t.Shape, h.F32(), 32)
		t, h = q, q.buf.(HostBuffer)
	}
	switch t.Dtype {
	case F32:
		if as != F32 {
			panic(fmt.Errorf("rocm: unsupported upload conversion f32 -> %s", as))
		}
		f := h.F32()
		if len(f) != numel {
			panic(fmt.Errorf("rocm: Upload f32 payload=%d want=%d", len(f), numel))
		}
		out, b := r.tensor(t.Shape, F32, MemoryWeights, false)
		finishROCmUpload(b, func() {
			if len(f) > 0 {
				rocmCheck(C.frocm_h2d(b.ptr, unsafe.Pointer(&f[0]), C.size_t(b.n)), "upload f32")
			}
		}, r.freeBuf)
		return out
	case Q8_0:
		if as != Q8_0 || len(t.Shape) != 2 || t.Quant == nil || t.Quant.Block <= 0 || t.Shape[1]%t.Quant.Block != 0 {
			panic(fmt.Errorf("rocm: invalid Q8_0 upload metadata"))
		}
		codes, scales := h.I8(), t.Quant.Scale
		wantScales, scalesOK := checkedTensorNumel([]int{t.Shape[0], t.Shape[1] / t.Quant.Block})
		scaleBytes, bytesOK := checkedROCmBytes([]int{wantScales}, F32.Bytes())
		if !scalesOK || !bytesOK || len(codes) != numel || len(scales) != wantScales {
			panic(fmt.Errorf("rocm: invalid Q8_0 payload codes=%d scales=%d", len(codes), len(scales)))
		}
		b := uploadROCmQ8Storage(codes, scales, scaleBytes, r.alloc, func(dst, src unsafe.Pointer, bytes int, site string) {
			rocmCheck(C.frocm_h2d(dst, src, C.size_t(bytes)), site)
		}, r.freeBuf)
		q := *t.Quant
		q.Scale = nil
		return makeTensor(r, Q8_0, RowMajor, append([]int(nil), t.Shape...), &q, b)
	case Q4_K, Q5_K, Q6_K:
		if as != t.Dtype || len(t.Shape) != 2 || t.Shape[1]%256 != 0 || t.Quant == nil {
			panic(fmt.Errorf("rocm: invalid %s upload metadata", t.Dtype))
		}
		blockBytes := map[Dtype]int{Q4_K: 144, Q5_K: 176, Q6_K: 210}[t.Dtype]
		raw := h.I8()
		want, bytesOK := checkedROCmBytes([]int{t.Shape[0], t.Shape[1] / 256}, blockBytes)
		if !bytesOK || len(raw) != want {
			panic(fmt.Errorf("rocm: %s payload=%d want=%d", t.Dtype, len(raw), want))
		}
		b := r.alloc(want, MemoryWeights, "upload-"+t.Dtype.String())
		finishROCmUpload(b, func() {
			if want > 0 {
				rocmCheck(C.frocm_h2d(b.ptr, unsafe.Pointer(&raw[0]), C.size_t(want)), "upload "+t.Dtype.String())
			}
		}, r.freeBuf)
		q := *t.Quant
		return makeTensor(r, t.Dtype, RowMajor, append([]int(nil), t.Shape...), &q, b)
	default:
		panic(fmt.Errorf("rocm: Upload does not support %s", t.Dtype))
	}
}

func (r *rocmBackend) Host(t Tensor) ([]float32, bool) { return hostF32(t) }

// finishROCmUpload protects the single unpublished allocation in the F32 and
// packed K-quant upload arms. The caller holds rocmMu and retains ownership on
// success. A transfer failure attempts release before rethrowing the exact cause;
// a secondary native Free failure cannot replace the established upload failure.
func finishROCmUpload(owned *rocmBuf, transfer func(), release func(*rocmBuf)) {
	defer func() {
		if primary := recover(); primary != nil {
			func() {
				defer func() { _ = recover() }()
				release(owned)
			}()
			panic(primary)
		}
	}()
	transfer()
}

// uploadROCmQ8Storage owns the two unpublished Q8 allocations until both H2D
// operations succeed. Upload holds rocmMu and validates payload sizes first.
// Injected operations make failure ownership testable without global/native hooks.
func uploadROCmQ8Storage(codes []int8, scales []float32, scaleBytes int,
	allocate func(int, MemoryClass, string) *rocmBuf,
	upload func(unsafe.Pointer, unsafe.Pointer, int, string),
	release func(*rocmBuf),
) *rocmBuf {
	var data, scale *rocmBuf
	committed := false
	defer func() {
		primary := recover()
		if committed {
			return
		}
		var cleanupPanic any
		for _, owned := range []*rocmBuf{scale, data} {
			if owned == nil {
				continue
			}
			func() {
				defer func() {
					if failure := recover(); failure != nil && cleanupPanic == nil {
						cleanupPanic = failure
					}
				}()
				release(owned)
			}()
		}
		if primary != nil {
			panic(primary)
		}
		if cleanupPanic != nil {
			panic(cleanupPanic)
		}
	}()
	data = allocate(len(codes), MemoryWeights, "upload-q8")
	scale = allocate(scaleBytes, MemoryWeights, "upload-q8-scales")
	// Preserve the existing zero-element contract: both minimal allocations
	// exist, but neither empty host slice is indexed or transferred.
	if len(codes) > 0 {
		upload(data.ptr, unsafe.Pointer(&codes[0]), len(codes), "upload q8 codes")
		upload(scale.ptr, unsafe.Pointer(&scales[0]), scaleBytes, "upload q8 scales")
	}
	data.scales, data.scalesN = scale.ptr, scaleBytes
	committed = true
	return data
}

func (r *rocmBackend) Read(t Tensor) []float32 {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	if h, ok := t.buf.(HostBuffer); ok && t.Dtype == F32 {
		return h.F32()
	}
	b := requireF32Owned(r, t, "Read")
	n, _ := checkedTensorNumel(t.Shape) // deviceBuf already validated this metadata.
	nbytes, _ := checkedROCmBytes(t.Shape, F32.Bytes())
	out := make([]float32, n)
	if len(out) > 0 {
		rocmCheck(C.frocm_d2h(unsafe.Pointer(&out[0]), b.ptr, C.size_t(nbytes)), "read")
	}
	return out
}

func (r *rocmBackend) freeBuf(b *rocmBuf) {
	if b == nil || !b.owned || b.ptr == nil || !atomic.CompareAndSwapUint32(&b.freed, 0, 1) {
		return
	}
	rocmCheck(C.frocm_free(b.ptr), "free")
	b.ptr = nil
	if b.scales != nil {
		rocmCheck(C.frocm_free(b.scales), "free scales")
		b.scales = nil
	}
}

func (r *rocmBackend) Free(t Tensor) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	if b, ok := t.buf.(*rocmBuf); ok {
		r.freeBuf(b)
	}
}

func (r *rocmBackend) CloneTensor(t Tensor) (Tensor, error) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	b := r.deviceBuf(t, "CloneTensor")
	d, err := cloneROCmStorage(b, r.alloc, func(dst, src unsafe.Pointer, bytes int, site string) error {
		if C.frocm_d2d(dst, src, C.size_t(bytes)) != 0 {
			return rocmLastError(site)
		}
		return nil
	}, r.freeBuf)
	if err != nil {
		return Tensor{}, err
	}
	out := t
	out.Shape, out.buf = append([]int(nil), t.Shape...), d
	if t.Quant != nil {
		q := *t.Quant
		q.Scale = nil
		out.Quant = &q
	}
	return out, nil
}

// cloneROCmStorage is the exception-safe ownership portion of CloneTensor. Its
// only production caller holds rocmMu; the supplied operations are the existing
// native allocation/copy/free methods, not mutable process-global hooks.
// Auxiliary storage stays independently owned until both copies succeed, so a
// failed cleanup of either allocation cannot prevent attempting the other.
func cloneROCmStorage(source *rocmBuf,
	allocate func(int, MemoryClass, string) *rocmBuf,
	copyDevice func(unsafe.Pointer, unsafe.Pointer, int, string) error,
	release func(*rocmBuf),
) (result *rocmBuf, cause error) {
	var data, scales *rocmBuf
	committed := false
	defer func() {
		primary := recover()
		if committed {
			return
		}
		var cleanupPanic any
		for _, owned := range []*rocmBuf{scales, data} {
			if owned == nil {
				continue
			}
			func() {
				defer func() {
					if failure := recover(); failure != nil && cleanupPanic == nil {
						cleanupPanic = failure
					}
				}()
				release(owned)
			}()
		}
		if primary != nil {
			panic(primary)
		}
		if cause == nil && cleanupPanic != nil {
			panic(cleanupPanic)
		}
	}()
	data = allocate(source.n, source.class, "clone")
	if err := copyDevice(data.ptr, source.ptr, source.n, "clone"); err != nil {
		return nil, err
	}
	if source.scalesN > 0 {
		scales = allocate(source.scalesN, source.class, "clone-scales")
		if err := copyDevice(scales.ptr, source.scales, source.scalesN, "clone scales"); err != nil {
			return nil, err
		}
		data.scales, data.scalesN = scales.ptr, source.scalesN
	}
	committed = true
	return data, nil
}

func (r *rocmBackend) Recycle() {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	rocmCheck(C.frocm_sync(), "recycle fence")
	for _, b := range r.transient {
		r.freeBuf(b)
	}
	r.transient = r.transient[:0]
}

func (r *rocmBackend) matmul(w, x Tensor, rows int) Tensor {
	if w.Layout != RowMajor {
		panic(fmt.Errorf("rocm: matmul requires row-major weight tensor"))
	}
	wn, wok := checkedTensorNumel(w.Shape)
	xn, xok := checkedTensorNumel(x.Shape)
	if len(w.Shape) != 2 || !wok || !xok || rows <= 0 || w.Shape[1] <= 0 || wn == 0 || rows > int(^uint(0)>>1)/w.Shape[1] || xn != rows*w.Shape[1] || x.Dtype != F32 {
		panic(fmt.Errorf("rocm: matmul shape mismatch w=%v x=%v rows=%d", w.Shape, x.Shape, rows))
	}
	wb, xb := r.deviceBuf(w, "matmul weight"), requireF32Owned(r, x, "matmul input")
	out, in := w.Shape[0], w.Shape[1]
	cout, cin, crows := rocmDim(out, "matmul out"), rocmDim(in, "matmul in"), rocmDim(rows, "matmul rows")
	y, yb := r.tensor([]int{rows, out}, F32, MemoryScratchpad, true)
	var code C.int
	switch w.Dtype {
	case F32:
		code = C.frocm_matmul_f32((*C.float)(wb.ptr), (*C.float)(xb.ptr), (*C.float)(yb.ptr), cout, cin, crows)
	case Q8_0:
		if w.Quant == nil || w.Quant.Block <= 0 || wb.scales == nil {
			panic(fmt.Errorf("rocm: invalid resident Q8 weight"))
		}
		code = C.frocm_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), (*C.float)(xb.ptr), (*C.float)(yb.ptr), cout, cin, crows, rocmDim(w.Quant.Block, "q8 block"))
	case Q4_K:
		code = C.frocm_q4k_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(xb.ptr), (*C.float)(yb.ptr), cout, cin, crows)
	case Q5_K:
		code = C.frocm_q5k_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(xb.ptr), (*C.float)(yb.ptr), cout, cin, crows)
	case Q6_K:
		code = C.frocm_q6k_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(xb.ptr), (*C.float)(yb.ptr), cout, cin, crows)
	default:
		panic(fmt.Errorf("rocm: matmul unsupported weight dtype %s", w.Dtype))
	}
	rocmCheck(code, "matmul "+w.Dtype.String())
	if rows == 1 {
		y.Shape = []int{out}
	}
	return y
}

// SupportsDeviceWeightDtype reports the exact dtype set rocmBackend.matmul has a case for,
// matching that switch: F32, Q8_0, Q4_K, Q5_K and Q6_K. Every other dtype falls to the
// switch's panic default, so it is reported false here.
func (r *rocmBackend) SupportsDeviceWeightDtype(dt Dtype) bool {
	switch dt {
	case F32, Q8_0, Q4_K, Q5_K, Q6_K:
		return true
	default:
		return false
	}
}

func (r *rocmBackend) MatMul(w, x Tensor) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	return r.matmul(w, x, 1)
}
func (r *rocmBackend) BatchedMatMul(w, x Tensor, rows int) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	return r.matmul(w, x, rows)
}

func (r *rocmBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	xb, wb := requireF32Owned(r, x, "rmsnorm x"), requireF32Owned(r, weight, "rmsnorm weight")
	xn, xok := checkedTensorNumel(x.Shape)
	width, wok := checkedTensorNumel(weight.Shape)
	if !xok || !wok || width <= 0 || xn%width != 0 {
		panic(fmt.Errorf("rocm: rmsnorm shape mismatch"))
	}
	y, yb := r.tensor(x.Shape, F32, MemoryScratchpad, true)
	rocmCheck(C.frocm_rmsnorm_f32((*C.float)(xb.ptr), (*C.float)(wb.ptr), (*C.float)(yb.ptr), rocmDim(xn/width, "rmsnorm rows"), rocmDim(width, "rmsnorm width"), C.float(eps)), "rmsnorm")
	return y
}

func (r *rocmBackend) RoPE(x Tensor, pos, heads, headDim int, theta float64) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	xb := requireF32Owned(r, x, "rope")
	xn, ok := checkedTensorNumel(x.Shape)
	if !ok || pos < 0 || heads <= 0 || headDim <= 0 || heads > int(^uint(0)>>1)/headDim || xn != heads*headDim || theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
		panic(fmt.Errorf("rocm: rope shape mismatch"))
	}
	y, yb := r.tensor(x.Shape, F32, MemoryScratchpad, true)
	rocmCheck(C.frocm_rope_f32((*C.float)(xb.ptr), (*C.float)(yb.ptr), rocmDim(pos, "rope position"), rocmDim(heads, "rope heads"), rocmDim(headDim, "rope head dim"), C.double(theta)), "rope")
	return y
}

func (r *rocmBackend) SwiGLU(gate, up Tensor) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	gb, ub := requireF32Owned(r, gate, "swiglu gate"), requireF32Owned(r, up, "swiglu up")
	gn, gok := checkedTensorNumel(gate.Shape)
	un, uok := checkedTensorNumel(up.Shape)
	if !gok || !uok || gn != un {
		panic(fmt.Errorf("rocm: swiglu shape mismatch"))
	}
	y, yb := r.tensor(gate.Shape, F32, MemoryScratchpad, true)
	rocmCheck(C.frocm_swiglu_f32((*C.float)(gb.ptr), (*C.float)(ub.ptr), (*C.float)(yb.ptr), rocmDim(gn, "swiglu elements")), "swiglu")
	return y
}

// SwiGLUWithLimit exposes the optional asymmetric clamp operation to existing
// model consumers. Inputs remain resident and unchanged; output is owned F32.
func (r *rocmBackend) SwiGLUWithLimit(gate, up Tensor, limit float32) Tensor {
	if limit <= 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
		panic(fmt.Errorf("rocm: limited swiglu requires finite positive limit"))
	}
	rocmMu.Lock()
	defer rocmMu.Unlock()
	gb, ub := requireF32Owned(r, gate, "limited swiglu gate"), requireF32Owned(r, up, "limited swiglu up")
	gn, gok := checkedTensorNumel(gate.Shape)
	un, uok := checkedTensorNumel(up.Shape)
	if !gok || !uok || gn <= 0 || gn != un || len(gate.Shape) != len(up.Shape) || gate.Layout != RowMajor || up.Layout != RowMajor {
		panic(fmt.Errorf("rocm: limited swiglu requires matching nonempty row-major shapes"))
	}
	for i := range gate.Shape {
		if gate.Shape[i] != up.Shape[i] {
			panic(fmt.Errorf("rocm: limited swiglu shapes differ"))
		}
	}
	n := rocmDim(gn, "limited swiglu elements")
	y, yb := r.tensor(gate.Shape, F32, MemoryScratchpad, true)
	if C.frocm_swiglu_limit_f32((*C.float)(gb.ptr), (*C.float)(ub.ptr), (*C.float)(yb.ptr), n, C.float(limit)) != 0 {
		cause := rocmLastError("limited swiglu")
		// Capture the launch/stream error before another ABI call overwrites it.
		// Already under rocmMu: do not call Free, which would acquire it again.
		if C.frocm_free(yb.ptr) == 0 {
			atomic.StoreUint32(&yb.freed, 1)
			yb.ptr = nil
		} else {
			// Retain the transient allocation for normal session cleanup to retry.
			cause = fmt.Errorf("%w; output cleanup: %v", cause, rocmLastError("free limited swiglu output"))
		}
		panic(cause)
	}
	return y
}

func (r *rocmBackend) AddInPlace(dst, src Tensor) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	db, sb := requireF32Owned(r, dst, "add dst"), requireF32Owned(r, src, "add src")
	dn, dok := checkedTensorNumel(dst.Shape)
	sn, sok := checkedTensorNumel(src.Shape)
	if !dok || !sok || dn != sn {
		panic(fmt.Errorf("rocm: add shape mismatch"))
	}
	rocmCheck(C.frocm_add_f32((*C.float)(db.ptr), (*C.float)(sb.ptr), rocmDim(dn, "add elements")), "add")
}

func (r *rocmBackend) AddBias(dst, bias Tensor) {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	db, bb := requireF32Owned(r, dst, "bias dst"), requireF32Owned(r, bias, "bias")
	dn, dok := checkedTensorNumel(dst.Shape)
	width, wok := checkedTensorNumel(bias.Shape)
	if !dok || !wok || width <= 0 || dn%width != 0 {
		panic(fmt.Errorf("rocm: add-bias shape mismatch"))
	}
	rocmCheck(C.frocm_add_bias_f32((*C.float)(db.ptr), (*C.float)(bb.ptr), rocmDim(dn/width, "bias rows"), rocmDim(width, "bias width")), "add bias")
}

func (r *rocmBackend) Attention(q Tensor, store KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	qb := requireF32Owned(r, q, "attention q")
	kv, ok := store.(*rocmKV)
	if !ok || kv == nil || kv.be != r || layer < 0 || layer >= kv.cfg.NumLayers || grp <= 0 {
		panic(fmt.Errorf("rocm: invalid attention KV store or geometry"))
	}
	kv.ensureUsable()
	hd, nKV := kv.cfg.HeadDim, kv.cfg.NumKVHeads
	if grp > int(^uint(0)>>1)/nKV {
		panic(fmt.Errorf("rocm: attention head geometry overflows"))
	}
	nH := grp * nKV
	qn, qok := checkedTensorNumel(q.Shape)
	if !qok || nH > int(^uint(0)>>1)/hd || qn != nH*hd || kv.K[layer].rows != kv.V[layer].rows || kv.K[layer].rows <= 0 {
		panic(fmt.Errorf("rocm: attention shape mismatch"))
	}
	y, yb := r.tensor([]int{nH * hd}, F32, MemoryScratchpad, true)
	rocmCheck(C.frocm_attention_f32((*C.float)(qb.ptr), (*C.float)(kv.K[layer].ptr), (*C.float)(kv.V[layer].ptr), (*C.float)(yb.ptr), rocmDim(nH, "attention q heads"), rocmDim(nKV, "attention kv heads"), rocmDim(kv.K[layer].rows, "attention positions"), rocmDim(hd, "attention head dim"), rocmDim(grp, "attention group"), C.float(scale)), "attention")
	return y
}

func (r *rocmBackend) Argmax(logits Tensor) int {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	b := requireF32Owned(r, logits, "argmax")
	n, ok := checkedTensorNumel(logits.Shape)
	if !ok || n <= 0 {
		panic(fmt.Errorf("rocm: argmax empty tensor"))
	}
	var idx C.int
	rocmCheck(C.frocm_argmax_f32((*C.float)(b.ptr), rocmDim(n, "argmax elements"), &idx), "argmax")
	return int(idx)
}

// DSAIndexSelect runs GLM-MoE-DSA's learned-indexer score + top-k SELECTION for one query.
//
// This is a device-path WIRING, not a device kernel: the numerics delegate to the exact host
// reference dsaIndexSelectHost over the tensors read back from the device, so the selected key
// SET is bit-identical to the cpu-ref. That is deliberate, and it is the same boundary dsa.go
// documents at its "WHY the selection stays on the host" block: the indexer drives a DISCRETE
// top-k, so a single flipped entry would diverge the output far past any reduction-order cosine
// -- the selection is held reduction-faithful, not cosine-close. Even though rocmBackend.Class()
// is Approx, this method must NOT become an Approx path: unlike the dense GEMM and flash-attention
// lanes there is no cosine margin for a different key set. The scoring is f64 in dsaIndexSelectHost
// and the selected set must match the cpu-ref exactly, so this method runs on the HOST (it does
// not claim to run on the device). A device topk kernel is out of scope for this pass (see
// issue #1747's gold-plating boundary: no TMA/cluster on gfx1151).
//
// indexQ/indexK/weights are read via r.Read (host- or device-resident); only the small index
// list is returned. The returned length is min(topK, nValid) positions, score descending, ties
// by lower position (the DSAIndexBackend contract).
func (r *rocmBackend) DSAIndexSelect(indexQ, indexK, weights Tensor, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	rocmMu.Lock()
	defer rocmMu.Unlock()
	if nKeys <= 0 || topK <= 0 {
		return nil
	}
	return dsaIndexSelectHost(r.Read(indexQ), r.Read(indexK), r.Read(weights), nKeys, nH, indexDim, queryPos, topK, scale)
}
