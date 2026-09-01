//go:build darwin && arm64 && cgo

// q4k.go — Go side of the Metal q4_k dequant-GEMV/GEMM (q4k.m). This is the only resident
// route that fits a 27B model on the 36 GB unified pool (q4_k_m ≈ 16 GB; f16 ≈ 54 GB does
// not), and the only path to the llama.cpp-Metal bar: the CPU int8-SDOT kernel is
// compute-bound at ~23 GB/s and tops out ~1.4 tok/s decode, while the GPU has both the
// bandwidth and the parallel dequant FLOPs (which is how llama.cpp reaches 7.29/51.55 tok/s).
// The raw q4_k super-blocks stay resident on the GPU and each thread dequants its weight row
// on the fly, dotting against the f32 activation.

package metalgemm

/*
#include <stdint.h>
typedef struct {
    uintptr_t command_buffer;
    int committed;
    int completed_wait;
    int host_readback;
    int encoders;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
} mg_execution_event;
int  mg_q4k_upload(const unsigned char* raw, int out, int in);
int  mg_q4k_upload_nocopy(const unsigned char* raw, int out, int in);
int  mg_q4k_upload_span(const unsigned char* raw, size_t nbytes, size_t offset, int out, int in);
int  mg_q4k_gemv(int wid, const float* x, float* y, int vectorized_mode, mg_execution_event* event);
void mg_q4k_gemv_batch(int wid, const float* Xcat, int n, float* Ycat, mg_execution_event* event);
void mg_q4k_gemv_batch_multi(int wid, const float* Xcat, int n, float* Ycat, mg_execution_event* event);
void mg_q4k_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff, mg_execution_event* event);
int  mg_q4k_q8_gemv_group(const int* q4_wids, int nq4, const float* x, float* q4_y, const int* q4_yoff,
                           const int* q8_wids, int nq8, const signed char* xq, const float* xd,
                           float* q8_y, const int* q8_yoff, int inject_post_submit_failure,
                           mg_execution_event* event);
void mg_q4k_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y, mg_execution_event* event);
int  mg_q6k_upload(const unsigned char* raw, int out, int in);
void mg_q6k_gemv(int wid, const float* x, float* y, mg_execution_event* event);
void mg_q6k_gemm(int wid, const float* X, int P, float* Y, mg_execution_event* event);
void mg_q4k_mlp_q6down(int gate_wid, int up_wid, int down_wid, const float* x, float* y, mg_execution_event* event);
int  mg_q4k_mlp_q6down_batch(const int* gate_wids, const int* up_wids, const int* down_wids, int n, const float* x, float* Ycat, mg_execution_event* event);
int  mg_q4k_gemm(int wid, const float* X, int P, float* Y, int mm_mode, double* out_gpu_ms, mg_execution_event* event);
int  mg_q4k_gemm_group(const int* wids, int n, const float* X, int P, float* Ycat, const int* yoff, int mm_mode, double* out_gpu_ms, mg_execution_event* event);
void mg_q4k_release(int wid);
void mg_q4k_reset(void);
*/
import "C"

import (
	"errors"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// lastGEMMGPUMs holds the on-GPU execution window (in milliseconds) of the most recent q4_k prefill
// GEMM dispatch — a single GEMM (Q4KWeight.GEMM) or a whole group (GEMMGroup). It is the
// cb.GPUEndTime-cb.GPUStartTime window the Metal kernel reports after waitUntilCompleted, i.e. the
// true on-GPU compute time EXCLUDING the CPU-side encode/commit/sync/H2D round-trip. Stored as the
// float64 bits so it is lock-free; read it with LastGEMMGPUMs immediately after the call. The value
// is 0 until the first dispatch populates it. Package-level (not per-weight) because the model side
// times one call at a time under FAK_QPROFILE and reads the split right after — a single store/load
// is the smallest non-breaking shape that covers both the GEMM and the GEMMGroup path.
var lastGEMMGPUMs atomic.Uint64

// LastGEMMGPUMs returns the GPU-execute window (in ms) of the most recent Q4KWeight.GEMM or
// GEMMGroup dispatch — the on-GPU compute time only (cb.GPUEndTime-cb.GPUStartTime), excluding the
// CPU-side encode/commit/sync/H2D. Valid only immediately after the call returns; 0 if no dispatch
// has run yet. The model side reads it right after a wall-timed GEMM (under FAK_QPROFILE) to split
// its q4kTime into gpu_compute (this) vs roundtrip (wall - this). Reading Metal's already-completed
// command-buffer timestamps is cheap, so the profile path costs one extra out-param write per call
// and nothing when unused.
func LastGEMMGPUMs() float64 { return math.Float64frombits(lastGEMMGPUMs.Load()) }

// SetGEMVUseVectorized selects the experimental vectorized P=1 Q4_K kernel. The scalar
// q4k_gemv pipeline remains the default and is restored by passing false. Selection affects only
// Q4KWeight.GEMV; grouped, batched, fused-MLP, and prefill dispatch retain their existing kernels.
func SetGEMVUseVectorized(on bool) {
	q4kUseVectorized.Store(on)
}

var q4kUseVectorized atomic.Bool

func recordQ4KEvent(observation *ExecutionObservation, event *C.mg_execution_event) {
	if event == nil {
		return
	}
	observation.record(uintptr(event.command_buffer), event.committed != 0, event.completed_wait != 0,
		event.host_readback != 0, int(event.encoders), float64(event.gpu_milliseconds),
		float64(event.wait_milliseconds), event.timing_available != 0)
}

type q4kGEMVExecution int

const (
	q4kGEMVNotExecuted q4kGEMVExecution = iota
	q4kGEMVExecutedScalar
	q4kGEMVExecutedVectorized
)

type q4kGEMVMode int

const (
	q4kGEMVModeScalar q4kGEMVMode = iota
	q4kGEMVModeVectorized
	// q4kGEMVModeVectorizedUnavailable exercises the native fail-closed branch without
	// mutating the process-global Metal pipeline table. Production selection never emits it.
	q4kGEMVModeVectorizedUnavailable = -1
)

// Q4KGEMMExecution identifies the exact Q4_K prefill kernel that reached Metal dispatch.
// NotExecuted means selection declined before a command buffer was created or output was touched.
type Q4KGEMMExecution int

const (
	Q4KGEMMNotExecuted Q4KGEMMExecution = iota
	Q4KGEMMExecutedScalar
	Q4KGEMMExecutedMM32
	Q4KGEMMExecutedM5CooperativeSMEM
)

// Q4KGEMMIdentity binds the candidate selected for this shape to the kernel that actually
// reached Metal dispatch. Credit MM32 only when both fields are Q4KGEMMExecutedMM32; an
// unavailable optional pipeline is reported as requested MM32 / executed none.
type Q4KGEMMIdentity struct {
	Requested Q4KGEMMExecution
	Executed  Q4KGEMMExecution
}

// Q4KGEMMMode selects a Q4_K prefill kernel candidate. MM32 is shape-bounded: only exact P=32
// dispatches may execute it; every other prompt length executes the scalar kernel. The unavailable
// mode is a deterministic fail-closed witness and is never selected by production.
type Q4KGEMMMode int

const (
	Q4KGEMMModeScalar Q4KGEMMMode = iota
	Q4KGEMMModeMM32
	Q4KGEMMModeM5CooperativeSMEM
	Q4KGEMMModeMM32Unavailable              = -1
	Q4KGEMMModeM5CooperativeSMEMUnavailable = -2
)

var q4kUseMM atomic.Bool

func q4kGEMMModeForPrompt(P int) Q4KGEMMMode {
	if P == 32 && q4kUseMM.Load() {
		return Q4KGEMMModeMM32
	}
	return Q4KGEMMModeScalar
}

func q4kGEMMRequestedExecution(P int, mode Q4KGEMMMode) Q4KGEMMExecution {
	switch mode {
	case Q4KGEMMModeM5CooperativeSMEM, Q4KGEMMModeM5CooperativeSMEMUnavailable:
		return Q4KGEMMExecutedM5CooperativeSMEM
	case Q4KGEMMModeMM32, Q4KGEMMModeMM32Unavailable:
		if P == 32 {
			return Q4KGEMMExecutedMM32
		}
	}
	return Q4KGEMMExecutedScalar
}

func q4kGEMMIdentity(P int, mode Q4KGEMMMode, executed Q4KGEMMExecution) Q4KGEMMIdentity {
	return Q4KGEMMIdentity{
		Requested: q4kGEMMRequestedExecution(P, mode),
		Executed:  executed,
	}
}

// Q4KGEMMIdentityForMode returns the typed requested/executed identity for an explicit candidate.
// It is deterministic and does not inspect pipeline availability or create Metal work.
func Q4KGEMMIdentityForMode(P int, mode Q4KGEMMMode, executed Q4KGEMMExecution) Q4KGEMMIdentity {
	return q4kGEMMIdentity(P, mode, executed)
}

// Q4KGEMMRequestedExecution returns the shape-bounded kernel selected by the current opt-in.
// It is deterministic and does not inspect pipeline availability or create Metal work.
func Q4KGEMMRequestedExecution(P int) Q4KGEMMExecution {
	return q4kGEMMRequestedExecution(P, q4kGEMMModeForPrompt(P))
}

// Q4KWeight is a handle to a raw q4_k weight matrix [Out, In] resident on the GPU. In must be
// a multiple of 256 (the q4_k super-block size); the resident byte cost is Out*(In/256)*144.
type Q4KWeight struct {
	id      C.int
	Out, In int
	noCopy  bool
}

type q4kPinnedRaw struct {
	pin *runtime.Pinner
	raw []byte
}

var (
	q4kPinMu sync.Mutex
	q4kPins  = map[int]q4kPinnedRaw{}
)

// UploadQ4K makes a row-major q4_k payload (the verbatim GGUF super-block bytes, length
// out*(in/256)*144) resident for the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full.
// On Apple unified memory it first tries newBufferWithBytesNoCopy against the existing resident
// Go bytes, keeping their backing array pinned until ResetQ4K. If Metal rejects the no-copy
// buffer, it falls back to the older shared-buffer copy upload.
// UploadQ4KMappedSpan registers a Q4_K tensor inside a page-aligned, externally owned mapping.
// raw must remain mapped until Reset. Metal releases only its buffer handle; it never unmaps raw.
func UploadQ4KMappedSpan(raw []byte, offset, out, in int) *Q4KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 || len(raw) == 0 || offset < 0 || offset%32 != 0 {
		return nil
	}
	need := (in / 256) * out * 144
	if offset > len(raw) || need > len(raw)-offset {
		return nil
	}
	base := unsafe.Pointer(unsafe.SliceData(raw))
	pageSize := os.Getpagesize()
	if uintptr(base)%uintptr(pageSize) != 0 || len(raw)%pageSize != 0 {
		return nil
	}
	wid := int(C.mg_q4k_upload_span((*C.uchar)(base), C.size_t(len(raw)), C.size_t(offset), C.int(out), C.int(in)))
	if wid < 0 {
		return nil
	}
	return &Q4KWeight{id: C.int(wid), Out: out, In: in, noCopy: true}
}
func UploadQ4K(raw []byte, out, in int) *Q4KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	need := out * (in / 256) * 144
	if len(raw) < need {
		return nil
	}
	raw = raw[:need]
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()

	pin := new(runtime.Pinner)
	pin.Pin(&raw[0])
	id := C.mg_q4k_upload_nocopy((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id >= 0 {
		q4kPins[int(id)] = q4kPinnedRaw{pin: pin, raw: raw}
		return &Q4KWeight{id: id, Out: out, In: in, noCopy: true}
	}
	pin.Unpin()

	id = C.mg_q4k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	runtime.KeepAlive(raw)
	return &Q4KWeight{id: id, Out: out, In: in}
}

// GEMV computes y[Out] = W · x for one f32 activation row x (length In). y must have length
// >= Out. Both slices are accessed only during the call. This is the decode GEMV.
func (w *Q4KWeight) gemvWithEventsMode(x, y []float32, observation *ExecutionObservation, mode q4kGEMVMode) q4kGEMVExecution {
	if w == nil || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return q4kGEMVNotExecuted
	}
	var event C.mg_execution_event
	executed := C.mg_q4k_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])), C.int(mode), &event)
	recordQ4KEvent(observation, &event)
	return q4kGEMVExecution(executed)
}

func (w *Q4KWeight) gemvWithEvents(x, y []float32, observation *ExecutionObservation) q4kGEMVExecution {
	mode := q4kGEMVModeScalar
	if q4kUseVectorized.Load() {
		mode = q4kGEMVModeVectorized
	}
	return w.gemvWithEventsMode(x, y, observation, mode)
}

func (w *Q4KWeight) GEMVWithEvents(x, y []float32, observation *ExecutionObservation) {
	w.gemvWithEvents(x, y, observation)
}

const (
	q4kMultiVectorMin          = 4
	q4kMultiVectorMax          = 8
	q4kMultiVectorHidden       = 5120
	q4kMultiVectorIntermediate = 17408
)

func q4kUseMultiVector(out, in, n int) bool {
	if n < q4kMultiVectorMin || n > q4kMultiVectorMax || in != q4kMultiVectorHidden {
		return false
	}
	return out == q4kMultiVectorHidden || out == q4kMultiVectorIntermediate
}

func (w *Q4KWeight) gemvBatchRepeatedWithEvents(Xcat []float32, n int, Ycat []float32, observation *ExecutionObservation) {
	var event C.mg_execution_event
	C.mg_q4k_gemv_batch(w.id, (*C.float)(unsafe.Pointer(&Xcat[0])), C.int(n), (*C.float)(unsafe.Pointer(&Ycat[0])), &event)
	recordQ4KEvent(observation, &event)
}

// gemvBatchRepeated preserves the pre-observation helper for parity and benchmark tests.
func (w *Q4KWeight) gemvBatchRepeated(Xcat []float32, n int, Ycat []float32) {
	w.gemvBatchRepeatedWithEvents(Xcat, n, Ycat, nil)
}

// GEMVBatch runs n decode GEMVs of this same weight in ONE command buffer: Xcat is n contiguous
// activation rows (n*In floats), Ycat receives n result rows (n*Out floats). Batches of 4-8 on
// the measured Qwen projection shapes use the multi-vector kernel that dequantizes each Q4_K tile
// once for all rows. Every other batch and shape keeps the original repeated-GEMV encoder path.
func (w *Q4KWeight) GEMVBatchWithEvents(Xcat []float32, n int, Ycat []float32, observation *ExecutionObservation) {
	if w == nil || w.id < 0 || n <= 0 || len(Xcat) < n*w.In || len(Ycat) < n*w.Out {
		return
	}
	if q4kUseMultiVector(w.Out, w.In, n) {
		var event C.mg_execution_event
		C.mg_q4k_gemv_batch_multi(w.id, (*C.float)(unsafe.Pointer(&Xcat[0])), C.int(n), (*C.float)(unsafe.Pointer(&Ycat[0])), &event)
		recordQ4KEvent(observation, &event)
		return
	}
	w.gemvBatchRepeatedWithEvents(Xcat, n, Ycat, observation)
}

// GEMVGroup runs one decode GEMV per weight in ws — all reading the SAME activation x (length
// In, shared) — in a SINGLE Metal command buffer, and returns one result slice per weight (each
// length ws[i].Out). Every weight must share x's In. This is the live decode group pattern
// (q/k/v, gate/up, the GDN in_proj quad): it pays the per-command-buffer submit/sync once for the
// whole group and pipelines the dispatches. Returns nil on a shape mismatch or empty input.
func GEMVGroupWithEvents(ws []*Q4KWeight, x []float32, observation *ExecutionObservation) [][]float32 {
	n := len(ws)
	if n == 0 || ws[0] == nil || len(x) < ws[0].In {
		return nil
	}
	in := ws[0].In
	wids := make([]C.int, n)
	yoff := make([]C.int, n+1)
	off := 0
	for i, w := range ws {
		if w == nil || w.id < 0 || w.In != in {
			return nil
		}
		wids[i] = w.id
		yoff[i] = C.int(off)
		off += w.Out
	}
	yoff[n] = C.int(off)
	ycat := make([]float32, off)
	var event C.mg_execution_event
	C.mg_q4k_gemv_group(&wids[0], C.int(n), (*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0], &event)
	recordQ4KEvent(observation, &event)
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		out[i] = ycat[o : o+w.Out : o+w.Out]
		o += w.Out
	}
	return out
}

const ExecutionMixedQ4KQ8QKV ExecutionOperation = "mixed-q4_k-q8-qkv"

// MixedQ4KQ8PreflightError reports that mixed projection inputs were rejected before Metal
// created a command buffer. The caller may safely choose another whole-operation route.
type MixedQ4KQ8PreflightError struct{ Reason string }

func (e *MixedQ4KQ8PreflightError) Error() string { return "mixed Q4_K/Q8 preflight: " + e.Reason }

// MixedQ4KQ8PostSubmitError reports failure after the candidate command buffer was created.
// Callers must fail closed: retrying either projection separately could expose partial work.
type MixedQ4KQ8PostSubmitError struct{}

func (*MixedQ4KQ8PostSubmitError) Error() string {
	return "mixed Q4_K/Q8 Metal command buffer failed after creation"
}

// IsMixedQ4KQ8PostSubmit reports whether err requires fail-closed handling.
func IsMixedQ4KQ8PostSubmit(err error) bool {
	var target *MixedQ4KQ8PostSubmitError
	return errors.As(err, &target)
}

func mixedQ4KQ8StatusError(status int) error {
	if status < 0 {
		return &MixedQ4KQ8PostSubmitError{}
	}
	if status == 0 {
		return &MixedQ4KQ8PreflightError{Reason: "native resources unavailable"}
	}
	return nil
}

// GEMVGroupMixedQ4KQ8 applies at least one Q4_K and one Q8 weight sharing one activation in
// one caller-observed native command buffer. Every input is checked before native encoding.
func GEMVGroupMixedQ4KQ8(q4ws []*Q4KWeight, q8ws []*Q8Weight, x []float32, xq []int8, xd []float32, observation *ExecutionObservation) (q4out, q8out [][]float32, err error) {
	return gemvGroupMixedQ4KQ8(q4ws, q8ws, x, xq, xd, observation, false)
}

// gemvGroupMixedQ4KQ8 is the single production/native call path. The final flag exists only so
// the Darwin test can make the native boundary fail after a real commit and completed wait; the
// exported operation always passes false, and no package-global fault state can leak across calls.
func gemvGroupMixedQ4KQ8(q4ws []*Q4KWeight, q8ws []*Q8Weight, x []float32, xq []int8, xd []float32, observation *ExecutionObservation, injectPostSubmitFailure bool) (q4out, q8out [][]float32, err error) {
	if len(q4ws) == 0 || len(q8ws) == 0 {
		return nil, nil, &MixedQ4KQ8PreflightError{Reason: "both quantization groups are required"}
	}
	if q4ws[0] == nil || q8ws[0] == nil {
		return nil, nil, &MixedQ4KQ8PreflightError{Reason: "nil weight"}
	}
	in := q4ws[0].In
	if in <= 0 || q8ws[0].In != in || len(x) < in || len(xq) < in || len(xd) < q8ws[0].Nblk {
		return nil, nil, &MixedQ4KQ8PreflightError{Reason: "activation geometry mismatch"}
	}
	q4ids := make([]C.int, len(q4ws))
	q4off := make([]C.int, len(q4ws)+1)
	q4flatLen := 0
	for i, w := range q4ws {
		if w == nil || w.id < 0 || w.In != in || w.Out <= 0 {
			return nil, nil, &MixedQ4KQ8PreflightError{Reason: "invalid Q4_K weight"}
		}
		q4ids[i] = C.int(w.id)
		q4off[i] = C.int(q4flatLen)
		q4flatLen += w.Out
	}
	q4off[len(q4ws)] = C.int(q4flatLen)
	q8ids := make([]C.int, len(q8ws))
	q8off := make([]C.int, len(q8ws)+1)
	q8flatLen := 0
	for i, w := range q8ws {
		if w == nil || w.id < 0 || w.In != in || w.Nblk != q8ws[0].Nblk || w.Out <= 0 {
			return nil, nil, &MixedQ4KQ8PreflightError{Reason: "invalid Q8 weight"}
		}
		q8ids[i] = C.int(w.id)
		q8off[i] = C.int(q8flatLen)
		q8flatLen += w.Out
	}
	q8off[len(q8ws)] = C.int(q8flatLen)

	q4flat := make([]float32, q4flatLen)
	q8flat := make([]float32, q8flatLen)
	var event C.mg_execution_event
	injectFailure := C.int(0)
	if injectPostSubmitFailure {
		injectFailure = 1
	}
	status := int(C.mg_q4k_q8_gemv_group(
		(*C.int)(unsafe.Pointer(&q4ids[0])), C.int(len(q4ids)), (*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&q4flat[0])), (*C.int)(unsafe.Pointer(&q4off[0])),
		(*C.int)(unsafe.Pointer(&q8ids[0])), C.int(len(q8ids)), (*C.schar)(unsafe.Pointer(&xq[0])),
		(*C.float)(unsafe.Pointer(&xd[0])), (*C.float)(unsafe.Pointer(&q8flat[0])),
		(*C.int)(unsafe.Pointer(&q8off[0])), injectFailure, &event))
	observation.record(uintptr(event.command_buffer), event.committed != 0, event.completed_wait != 0,
		event.host_readback != 0, int(event.encoders), float64(event.gpu_milliseconds),
		float64(event.wait_milliseconds), event.timing_available != 0)
	if err := mixedQ4KQ8StatusError(status); err != nil {
		return nil, nil, err
	}
	q4out = make([][]float32, len(q4ws))
	for i := range q4ws {
		q4out[i] = q4flat[int(q4off[i]):int(q4off[i+1])]
	}
	q8out = make([][]float32, len(q8ws))
	for i := range q8ws {
		q8out[i] = q8flat[int(q8off[i]):int(q8off[i+1])]
	}
	return q4out, q8out, nil
}

// FusedMLP runs a whole dense SwiGLU MLP for one decode token — y = down( silu(gate·x) * (up·x) )
// — in ONE Metal command buffer, keeping the intermediate-wide gate/up/inter resident on the GPU
// (only x and y cross the boundary). Requires gate.In==up.In==down.Out (=H), gate.Out==up.Out==
// down.In (=I); len(x)>=H, len(y)>=H. Returns false on a shape mismatch (caller uses the per-matmul
// path). The activation is silu — the caller must gate on a non-GELU config.
func FusedMLPWithEvents(gate, up, down *Q4KWeight, x, y []float32, observation *ExecutionObservation) bool {
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || down.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || down.Out != gate.In {
		return false
	}
	if len(x) < gate.In || len(y) < down.Out {
		return false
	}
	var event C.mg_execution_event
	C.mg_q4k_mlp(gate.id, up.id, down.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])), &event)
	recordQ4KEvent(observation, &event)
	return true
}

func FusedMLP(gate, up, down *Q4KWeight, x, y []float32) bool {
	return FusedMLPWithEvents(gate, up, down, x, y, nil)
}

// Q6KWeight is a handle to a raw Q6_K weight matrix [Out, In] resident on the GPU (210-B
// super-blocks). In must be a multiple of 256; the resident byte cost is Out*(In/256)*210. It
// backs the fused MLP's down_proj when a q4_k_m GGUF quantizes down_proj to Q6_K. The id is offset
// by the C side's MG_Q6_BASE so it can never alias a Q4KWeight id.
type Q6KWeight struct {
	id      C.int
	Out, In int
}

// UploadQ6K makes a row-major Q6_K payload (verbatim GGUF super-block bytes, length
// out*(in/256)*210) resident for the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full.
func UploadQ6K(raw []byte, out, in int) *Q6KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	need := out * (in / 256) * 210
	if len(raw) < need {
		return nil
	}
	raw = raw[:need]
	id := C.mg_q6k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	runtime.KeepAlive(raw)
	return &Q6KWeight{id: id, Out: out, In: in}
}

// ID returns the backend handle for this matrix.
func (w *Q6KWeight) ID() int { return int(w.id) }

// GEMV computes y[Out] = W · x for one f32 activation row. It is the standalone decode/head
// twin of the Q6_K GEMV already used inside FusedMLPQ6Down.
func (w *Q6KWeight) GEMVWithEvents(x, y []float32, observation *ExecutionObservation) {
	if w == nil || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return
	}
	var event C.mg_execution_event
	C.mg_q6k_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])), &event)
	recordQ4KEvent(observation, &event)
}

func (w *Q6KWeight) GEMV(x, y []float32) { w.GEMVWithEvents(x, y, nil) }

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ for a resident Q6_K matrix. It is the prefill twin of
// the Q6_K GEMV used by the mixed Q4_K/Q6_K fused decode MLP. The kernel keeps the Q6_K bytes on
// the GPU and only moves the f32 activation panel/result, so q4_k_m dense down_proj no longer falls
// back to the CPU batched k-quant loop during hybrid Qwen prefill.
func (w *Q6KWeight) GEMMWithEvents(X []float32, P int, Y []float32, observation *ExecutionObservation) {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	var event C.mg_execution_event
	C.mg_q6k_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])), &event)
	recordQ4KEvent(observation, &event)
}

func (w *Q6KWeight) GEMM(X []float32, P int, Y []float32) { w.GEMMWithEvents(X, P, Y, nil) }

// FusedMLPQ6Down runs a whole dense SwiGLU MLP for one decode token — y = down( silu(gate·x) *
// (up·x) ) — in ONE Metal command buffer, exactly like FusedMLP, but with a Q6_K down_proj
// (gate/up stay Q4_K). The intermediate-wide gate/up/inter stays resident (only x and y cross the
// boundary). Requires gate.In==up.In==down.Out (=H), gate.Out==up.Out==down.In (=I); len(x)>=H,
// len(y)>=down.Out. Returns false on a shape mismatch. The activation is silu.
func FusedMLPQ6DownWithEvents(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32, observation *ExecutionObservation) bool {
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || down.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || down.Out != gate.In {
		return false
	}
	if len(x) < gate.In || len(y) < down.Out {
		return false
	}
	var event C.mg_execution_event
	C.mg_q4k_mlp_q6down(gate.id, up.id, down.id,
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])), &event)
	recordQ4KEvent(observation, &event)
	return true
}

func FusedMLPQ6Down(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	return FusedMLPQ6DownWithEvents(gate, up, down, x, y, nil)
}

// FusedMLPQ6DownBatch runs n experts' fused SwiGLU MLP (Q4_K gate/up, Q6_K down) — each y_e =
// down_e( silu(gate_e·x) * (up_e·x) ) over the SAME token activation x — into ONE Metal command
// buffer, so the top-k experts of a MoE layer pay the submit/sync once instead of n times (issue
// #1382, the mlp_decode decode lever). Ycat receives the n outputs concatenated (row e at e*down.Out);
// the caller applies the gate-weighted sum on the host so the reduction order matches the per-expert
// loop exactly. All experts must share one geometry (gate.In==up.In==down.Out=H, gate.Out==up.Out==
// down.In=I). Returns false if n<=0, len(x)<H, len(Ycat)<n*down.Out, any handle is invalid, or the
// backend declines a shape — the caller then runs the proven per-expert FusedMLPQ6Down loop.
func FusedMLPQ6DownBatchWithEvents(gate, up []*Q4KWeight, down []*Q6KWeight, x, Ycat []float32, observation *ExecutionObservation) bool {
	n := len(gate)
	if n == 0 || len(up) != n || len(down) != n {
		return false
	}
	H, I, Dout := gate[0].In, gate[0].Out, down[0].Out
	gw := make([]C.int, n)
	uw := make([]C.int, n)
	dw := make([]C.int, n)
	for e := 0; e < n; e++ {
		g, u, d := gate[e], up[e], down[e]
		if g == nil || u == nil || d == nil || g.id < 0 || u.id < 0 || d.id < 0 {
			return false
		}
		if g.In != u.In || g.Out != u.Out || d.In != g.Out || d.Out != g.In {
			return false
		}
		if g.In != H || g.Out != I || d.Out != Dout {
			return false // non-uniform batch geometry — decline (caller uses the per-expert loop)
		}
		gw[e], uw[e], dw[e] = C.int(g.id), C.int(u.id), C.int(d.id)
	}
	if len(x) < H || len(Ycat) < n*Dout {
		return false
	}
	var event C.mg_execution_event
	rc := C.mg_q4k_mlp_q6down_batch(&gw[0], &uw[0], &dw[0], C.int(n),
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&Ycat[0])), &event)
	recordQ4KEvent(observation, &event)
	return rc == 0
}

func FusedMLPQ6DownBatch(gate, up []*Q4KWeight, down []*Q6KWeight, x, Ycat []float32) bool {
	return FusedMLPQ6DownBatchWithEvents(gate, up, down, x, Ycat, nil)
}

// GEMMWithEventsMode computes Y[P, Out] = X[P, In] · Wᵀ with an explicit kernel selection and
// returns the typed requested/executed identity. MM32 applies only to exact P=32; P31/P33 and
// every other prompt length execute the scalar kernel. An unavailable requested candidate returns
// requested MM32 / executed none without a command buffer, execution event, timing update, or
// output mutation.
func (w *Q4KWeight) GEMMWithEventsMode(X []float32, P int, Y []float32, observation *ExecutionObservation, mode Q4KGEMMMode) Q4KGEMMIdentity {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return Q4KGEMMIdentity{}
	}
	var gpuMs C.double
	var event C.mg_execution_event
	executed := Q4KGEMMExecution(C.mg_q4k_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P),
		(*C.float)(unsafe.Pointer(&Y[0])), C.int(mode), &gpuMs, &event))
	recordQ4KEvent(observation, &event)
	if executed != Q4KGEMMNotExecuted {
		lastGEMMGPUMs.Store(math.Float64bits(float64(gpuMs)))
	}
	return q4kGEMMIdentity(P, mode, executed)
}

// GEMMWithEvents uses the process opt-in selected by SetGEMMUseMM. The default remains scalar and
// the returned identity reports the requested/executed pair for that shape.
func (w *Q4KWeight) GEMMWithEvents(X []float32, P int, Y []float32, observation *ExecutionObservation) Q4KGEMMIdentity {
	return w.GEMMWithEventsMode(X, P, Y, observation, q4kGEMMModeForPrompt(P))
}

// GEMMGroupIntoWithEventsMode runs one batched prefill GEMM per weight in ws — all reading the SAME
// activation panel X[P, In] — and places the concatenated results in ycat. Returned slices alias
// ycat and stay valid only until its owner reuses that backing. The Metal command buffer is
// committed and synchronously completed before any aliases are returned. It returns nil without
// dispatching when shapes are invalid, ycat is too small, or the requested candidate is unavailable.
// The second return binds the group to the exact kernel identity shared by every encoded weight.
func GEMMGroupIntoWithEventsMode(ws []*Q4KWeight, X []float32, P int, ycat []float32, observation *ExecutionObservation, mode Q4KGEMMMode) ([][]float32, Q4KGEMMIdentity) {
	n := len(ws)
	const maxCInt = int(^uint32(0) >> 1)
	if n == 0 || P <= 0 || P > maxCInt || ws[0] == nil || ws[0].In <= 0 || len(X)/P < ws[0].In {
		return nil, Q4KGEMMIdentity{}
	}
	in := ws[0].In
	wids := make([]C.int, n)
	yoff := make([]C.int, n+1) // yoff[i] = P*Σ_{j<i} out_j (element offset of weight i's [P,out_i] block)
	off := 0
	for i, w := range ws {
		if w == nil || w.id < 0 || w.In != in || w.Out <= 0 || w.Out > (maxCInt-off)/P {
			return nil, Q4KGEMMIdentity{}
		}
		wids[i] = w.id
		yoff[i] = C.int(off)
		off += P * w.Out
	}
	if len(ycat) < off {
		return nil, Q4KGEMMIdentity{}
	}
	ycat = ycat[:off]
	yoff[n] = C.int(off)
	var gpuMs C.double
	var event C.mg_execution_event
	executed := Q4KGEMMExecution(C.mg_q4k_gemm_group(&wids[0], C.int(n),
		(*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&ycat[0])),
		&yoff[0], C.int(mode), &gpuMs, &event))
	recordQ4KEvent(observation, &event)
	identity := q4kGEMMIdentity(P, mode, executed)
	if executed == Q4KGEMMNotExecuted {
		return nil, identity
	}
	lastGEMMGPUMs.Store(math.Float64bits(float64(gpuMs)))

	// Publish aliases only after the synchronous native call above has completed successfully.
	out := make([][]float32, n)
	for i := range ws {
		out[i] = ycat[int(yoff[i]):int(yoff[i+1]):int(yoff[i+1])]
	}
	return out, identity
}

// GEMMGroupIntoWithEventsIdentity uses the process opt-in selected by SetGEMMUseMM and returns
// both the aliases and their typed requested/executed kernel identity.
func GEMMGroupIntoWithEventsIdentity(ws []*Q4KWeight, X []float32, P int, ycat []float32, observation *ExecutionObservation) ([][]float32, Q4KGEMMIdentity) {
	return GEMMGroupIntoWithEventsMode(ws, X, P, ycat, observation, q4kGEMMModeForPrompt(P))
}

// GEMMGroupIntoWithEvents preserves the original aliases-only API.
func GEMMGroupIntoWithEvents(ws []*Q4KWeight, X []float32, P int, ycat []float32, observation *ExecutionObservation) [][]float32 {
	out, _ := GEMMGroupIntoWithEventsIdentity(ws, X, P, ycat, observation)
	return out
}

// GEMMGroupWithEventsMode runs one batched prefill GEMM per weight in ws — all reading the SAME
// X[P, In] (shared) — in a SINGLE Metal command buffer, returning one [P*Out_i] result slice per
// weight (token-major, Y[t*Out_i + o]). Every weight must share X's In. It is the prefill twin of
// GEMVGroup: the live prefill group pattern (a layer's q/k/v, gate/up, or the GDN in_proj quad all
// read the same post-norm panel), paying the per-command-buffer submit/sync once for the whole
// group instead of once per weight — the fix for the ~7-submits-per-layer prefill wall. Returns nil
// on a shape mismatch or empty input, so the caller falls back to per-weight GEMM.
func GEMMGroupWithEventsMode(ws []*Q4KWeight, X []float32, P int, observation *ExecutionObservation, mode Q4KGEMMMode) ([][]float32, Q4KGEMMIdentity) {
	n := len(ws)
	const maxCInt = int(^uint32(0) >> 1)
	if n == 0 || P <= 0 || P > maxCInt || ws[0] == nil || ws[0].In <= 0 || len(X)/P < ws[0].In {
		return nil, Q4KGEMMIdentity{}
	}
	in := ws[0].In
	off := 0
	for _, w := range ws {
		if w == nil || w.id < 0 || w.In != in || w.Out <= 0 || w.Out > (maxCInt-off)/P {
			return nil, Q4KGEMMIdentity{}
		}
		off += P * w.Out
	}
	ycat := make([]float32, off)
	return GEMMGroupIntoWithEventsMode(ws, X, P, ycat, observation, mode)
}

// GEMMGroupWithEventsIdentity uses the process opt-in selected by SetGEMMUseMM and returns the
// typed requested/executed kernel identity shared by the group.
func GEMMGroupWithEventsIdentity(ws []*Q4KWeight, X []float32, P int, observation *ExecutionObservation) ([][]float32, Q4KGEMMIdentity) {
	return GEMMGroupWithEventsMode(ws, X, P, observation, q4kGEMMModeForPrompt(P))
}

// GEMMGroupWithEvents preserves the original results-only API.
func GEMMGroupWithEvents(ws []*Q4KWeight, X []float32, P int, observation *ExecutionObservation) [][]float32 {
	out, _ := GEMMGroupWithEventsIdentity(ws, X, P, observation)
	return out
}

// ID returns the backend handle for this matrix.
func (w *Q4KWeight) ID() int { return int(w.id) }

// Release invalidates this handle and releases its resident Metal buffer. It is safe to call more than once, but callers must ensure no GEMM/GEMV is in flight.
func (w *Q4KWeight) Release() {
	if w == nil || w.id < 0 {
		return
	}
	id := int(w.id)
	q4kPinMu.Lock()
	C.mg_q4k_release(w.id)
	if pinned, ok := q4kPins[id]; ok {
		pinned.pin.Unpin()
		delete(q4kPins, id)
	}
	w.id = -1
	q4kPinMu.Unlock()
}

// NoCopy reports whether this handle aliases the caller's pinned raw q4_k bytes through
// newBufferWithBytesNoCopy instead of owning a copied Metal buffer.
func (w *Q4KWeight) NoCopy() bool { return w != nil && w.noCopy }

// SetGEMMUseMM selects the exact-P32 batched-GEMM candidate: true requests q4k_gemm_mm32 only
// when P==32, while P31/P33 and every other prompt length retain q4k_gemm. False is the default.
// The model layer flips this process-local opt-in from FAK_Q4K_MM.
func SetGEMMUseMM(on bool) {
	q4kUseMM.Store(on)
}

// ResetQ4K releases every resident q4_k weight buffer and the reused scratch (the q4_k twin of
// Reset). Call only when no Q4KWeight handle is still in use — every prior handle is invalidated.
func ResetQ4K() {
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()
	C.mg_q4k_reset()
	for id, pinned := range q4kPins {
		pinned.pin.Unpin()
		delete(q4kPins, id)
	}
}

func (w *Q4KWeight) GEMV(x, y []float32) { w.GEMVWithEvents(x, y, nil) }
func (w *Q4KWeight) GEMVBatch(Xcat []float32, n int, Ycat []float32) {
	w.GEMVBatchWithEvents(Xcat, n, Ycat, nil)
}
func GEMVGroup(ws []*Q4KWeight, x []float32) [][]float32 { return GEMVGroupWithEvents(ws, x, nil) }
func (w *Q4KWeight) GEMM(X []float32, P int, Y []float32) Q4KGEMMIdentity {
	return w.GEMMWithEvents(X, P, Y, nil)
}
func GEMMGroup(ws []*Q4KWeight, X []float32, P int) [][]float32 {
	return GEMMGroupWithEvents(ws, X, P, nil)
}

// GEMMGroupInto is GEMMGroupIntoWithEvents without execution observation.
func GEMMGroupInto(ws []*Q4KWeight, X []float32, P int, ycat []float32) [][]float32 {
	return GEMMGroupIntoWithEvents(ws, X, P, ycat, nil)
}
