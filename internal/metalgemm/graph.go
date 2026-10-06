//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
#include <stdlib.h>
typedef struct {
    int committed;
    int completed_wait;
    int encoders;
    int host_readbacks;
    int allocated_buffers;
    uint64_t retained_buffer_bytes;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
    int status_code;
    int error_code;
    int device_ok;
    char error_text[256];
} mg_graph_receipt;
void *mg_graph_begin(const float *xf, const signed char *xq, const float *xd, int P, int in);
void *mg_graph_encode_q4k(void *graph, int wid);
void *mg_graph_encode_q6k(void *graph, int wid);
void *mg_graph_encode_q8(void *graph, int wid);
void *mg_graph_encode_q4k_from(void *graph, int wid, void *input, int elems);
void *mg_graph_encode_q6k_from(void *graph, int wid, void *input, int elems);
void *mg_graph_quantize_q8(void *graph, void *input, int elems, void **scales);
void *mg_graph_encode_q8_from(void *graph, int wid, void *q, void *d, int elems);
int mg_graph_finish(void *graph, mg_graph_receipt *receipt, int inject_post_submit_failure);
int mg_graph_await_terminal(void *graph);
int mg_graph_read(void *graph, void *result, float *dst, int n);
int mg_graph_read_pack(void *graph, void **results, const int *sizes, int count, float *dst, int total);
void mg_graph_free(void *graph);
int mg_graph_live_owners(void);
int mg_graph_live_buffers(void);
void *mg_graph_test_hold_terminal(void *graph, int wait_limit_ms);
void mg_graph_test_release_terminal(void *gate);
void *mg_graph_xf_buffer(void *graph);
void *mg_graph_upload(void *graph, const float *src, int n);
int mg_graph_set_gemv_vectorized(void *graph, int mode);
int mg_graph_set_gemv_kernel(void *graph, int mode);
int mg_graph_gemv_executed(void *graph, int q6);
int mg_graph_set_gemv_p1(void *graph, int mode);
int mg_graph_set_mm_mode(void *graph, int mode);
int mg_graph_mm_mode(void *graph);
int mg_graph_set_q6k_mode(void *graph, int mode);
int mg_graph_q6k_mode(void *graph);
int mg_graph_set_buffer_pool(void *graph, int depth);
void mg_graph_recycle_result(void *graph, void *result);
void *mg_qwen35_graph_kv_alloc(int elems);
void mg_qwen35_graph_kv_free(void *kv);
int mg_qwen35_graph_kv_upload(void *kv, const float *src, int elems);
int mg_qwen35_graph_kv_download(void *kv, float *dst, int elems);
int mg_qwen35_graph_kv_upload_at(void *kv, long off, const float *src, long elems);
int mg_qwen35_graph_kv_download_at(void *kv, long off, float *dst, long elems);
int mg_qwen35_graph_attention_dkv(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    void *kv_kraw, void *kv_kpost, void *kv_v, int kv_off,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent);
int mg_qwen35_graph_kvq8_ready(void);
void *mg_qwen35_graph_kvq8_alloc(long bytes);
int mg_qwen35_graph_kvq8_write(void *buf, long off, const void *src, long bytes);
int mg_qwen35_graph_kvq8_read(void *buf, long off, void *dst, long bytes);
int mg_qwen35_graph_attention_q8(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    const signed char *prefix_kc, const float *prefix_ks, const signed char *prefix_vc, const float *prefix_vs,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent, void **kc, void **ks, void **vc, void **vs);
int mg_qwen35_graph_attention_dkv_q8(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    void *kv_kc, void *kv_ks, void *kv_vc, void *kv_vs, long elem_off,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent, void **kc, void **ks, void **vc, void **vs);
void *mg_qwen35_graph_norm(void *graph, void *input, const float *weight, int rows, int width, float eps, int gain1p, int last_only);
int mg_qwen35_graph_add(void *graph, void *x, void *y, int n);
int mg_qwen35_graph_bias(void *graph, void *x, const float *bias, int rows, int width);
int mg_qwen35_graph_swiglu(void *graph, void *gate, void *up, int n);
int mg_qwen35_graph_split(void *graph, void *src, int qwidth, int hd, void **q, void **gate);
int mg_qwen35_graph_attention(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
	const float *prefix_k, const float *prefix_v,
	int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
	int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent);
void *mg_gdn_graph_encode(void *graph, int owner, void *mixed, void *z, void *b, void *a,
    const float *conv, const float *alog, const float *dtbias, const float *norm,
    int tokens, int nk, int nv, int khd, int vhd, int kernel, float eps);
int mg_gdn_graph_checkpoint(void *graph, int live_owner, int backup_owner);
int mg_gdn_state_swap_buffers(int live_owner, int backup_owner);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"
)

var (
	errGraphTerminal = errors.New("metalgemm: graph is terminal")
	errGraphEmpty    = errors.New("metalgemm: graph has no encoded projections")
	// Once a timeout is observed, refuse new graph admissions until every
	// quarantined graph is terminal. Graphs already admitted before the first
	// timeout can still join the quarantine; their count is bounded by the
	// caller's pre-existing in-flight set rather than by this gate.
	projectionGraphQuarantines atomic.Int64
)

type GraphPostSubmitError struct{ Reason string }

func (e *GraphPostSubmitError) Error() string {
	return "metalgemm: graph failed after submit: " + e.Reason
}

type GraphReceipt struct {
	Committed, CompletedWait, TimingAvailable bool
	Encoders, IntermediateWaits               int
	IntermediateReadbacks, HostReadbacks      int
	HostUploadBytes, HostReadbackBytes        uint64
	GPUMilliseconds, WaitMilliseconds         float64
	// AllocatedBuffers and RetainedBufferBytes count distinct graph-tracked
	// Metal buffers, including input, intermediate and terminal results. All
	// remain retained until Free, so the bytes are their peak for this graph.
	// Untracked constants and attention temporaries, GDN state, persistent KV
	// and weights are outside this scope. The snapshot survives errors and teardown.
	AllocatedBuffers    int
	RetainedBufferBytes uint64
	// Q4KGEMVKernels and Q6KGEMVKernels are the P=1 GEMV kernels this graph actually encoded
	// on its SetGEMVDecode route (empty for the GEMM route), so a receipt shows whether the
	// mul_mv default or the legacy parity kernel ran.
	Q4KGEMVKernels, Q6KGEMVKernels GEMVKernelSet
}

type GraphResult struct {
	ptr      unsafe.Pointer
	out      int
	p        int
	graph    *ProjectionGraph
	released bool
}

// QuantizedGraphResult is a graph-owned Q8_0 activation panel. It can only be
// consumed by the graph that produced it and never crosses the host boundary.
type QuantizedGraphResult struct {
	q, d  unsafe.Pointer
	elems int
	graph *ProjectionGraph
}

type Qwen35GraphAttentionResult struct {
	Output, KRaw, KPost, V *GraphResult
}

// PromptPanelMaxTokens is the widest prompt panel the Qwen3.8 whole-forward graph
// admits in one command buffer. The ordered-panel kernels (qg_norm/add/swiglu/
// split/qk/attn/attn_online) and the fused GDN encoder loop generically over the
// graph's row count; the original witness set {1,2,3,4,32} was a conservative
// enumeration, not a hardware bound. Widening the admitted set up to this ceiling
// lets a long prompt ride one commit+terminal-wait per panel instead of one per
// 32 tokens (issue #13041). Kept as a small multiple of the decode token so the
// GDN recurrence and KV staging remain in the proven regime.
const PromptPanelMaxTokens = 128

// PromptPanelWitnessed reports whether the Qwen3.8 graph admits a prompt panel of
// rows tokens in one whole-forward pass. rows==0 is refused, matching the native
// qg_ordered_rows() guard; every positive row count up to PromptPanelMaxTokens is
// admitted.
func PromptPanelWitnessed(rows int) bool {
	return rows >= 1 && rows <= PromptPanelMaxTokens
}

type ProjectionGraph struct {
	ptr                     unsafe.Pointer
	p                       int
	encoders                int
	finished                bool
	freed                   bool
	readbacks               int
	hostUploadBytes         uint64
	injectPostSubmitFailure bool
	injectDeviceFault       bool
	gdnLeases               []gdnGraphLease
	gdnCheckpoints          []*GDNGraphCheckpoint
	testTerminalGate        unsafe.Pointer
	quarantineDone          chan struct{}
}

// InjectPostSubmitFailureForTest makes Finish return an accepted failure after
// the native command buffer has committed and completed. It exists solely for
// deterministic fail-closed lifecycle witnesses.
func (g *ProjectionGraph) InjectPostSubmitFailureForTest() {
	if g != nil {
		g.injectPostSubmitFailure = true
	}
}

// InjectDeviceFaultForTest makes the native receipt report a committed command
// buffer whose terminal status never reached Completed, without blocking on a
// real device fault. It exists solely to witness the stall classification on the
// real Finish path.
func (g *ProjectionGraph) InjectDeviceFaultForTest() {
	if g != nil {
		g.injectDeviceFault = true
	}
}

// holdTerminalForTest appends a real command-buffer wait behind the graph's
// encoded work and shortens only this graph's Finish timeout. Tests must install
// releaseTerminalForTest as cleanup immediately after this succeeds.
func (g *ProjectionGraph) holdTerminalForTest(wait time.Duration) error {
	if err := g.open(); err != nil {
		return err
	}
	if wait <= 0 || wait > time.Second {
		return errors.New("metalgemm: invalid test terminal wait")
	}
	gate := C.mg_graph_test_hold_terminal(g.ptr, C.int(wait.Milliseconds()))
	if gate == nil {
		return errors.New("metalgemm: install test terminal gate")
	}
	g.testTerminalGate = gate
	return nil
}

func (g *ProjectionGraph) releaseTerminalForTest() {
	if g != nil && g.testTerminalGate != nil {
		C.mg_graph_test_release_terminal(g.testTerminalGate)
		g.testTerminalGate = nil
	}
}

func (g *ProjectionGraph) awaitTerminalForTest() bool {
	return g != nil && g.ptr != nil && C.mg_graph_await_terminal(g.ptr) != 0
}

func graphLiveOwnerCount() int  { return int(C.mg_graph_live_owners()) }
func graphLiveBufferCount() int { return int(C.mg_graph_live_buffers()) }

// BeginProjectionGraph uploads one activation panel for all projections in the graph.
// xf is required by Q4_K/Q6_K; xq/xd are required by Q8. Supplying both permits mixed graphs.
func BeginProjectionGraph(xf []float32, xq []int8, xd []float32, P, in int) (*ProjectionGraph, error) {
	if projectionGraphQuarantines.Load() > 0 {
		return nil, errors.New("metalgemm: projection graph unavailable while a timed-out command buffer remains quarantined")
	}
	if P <= 0 || in <= 0 || len(xf) != 0 && len(xf) != P*in || len(xq) != 0 && len(xq) != P*in || len(xd) != 0 && len(xd) != P*(in/32) {
		return nil, fmt.Errorf("metalgemm: invalid projection graph panel P=%d in=%d xf=%d xq=%d xd=%d", P, in, len(xf), len(xq), len(xd))
	}
	var pf *C.float
	var pq *C.schar
	var pd *C.float
	if len(xf) > 0 {
		pf = (*C.float)(unsafe.Pointer(&xf[0]))
	}
	if len(xq) > 0 {
		pq = (*C.schar)(unsafe.Pointer(&xq[0]))
	}
	if len(xd) > 0 {
		pd = (*C.float)(unsafe.Pointer(&xd[0]))
	}
	p := C.mg_graph_begin(pf, pq, pd, C.int(P), C.int(in))
	if p == nil {
		return nil, errors.New("metalgemm: graph begin failed")
	}
	return &ProjectionGraph{
		ptr: p, p: P,
		hostUploadBytes: uint64(len(xf))*4 + uint64(len(xq)) + uint64(len(xd))*4,
	}, nil
}

func (g *ProjectionGraph) open() error {
	if g == nil || g.ptr == nil || g.finished || g.freed {
		return errGraphTerminal
	}
	return nil
}

// Input returns the graph-owned f32 activation uploaded at construction.
func (g *ProjectionGraph) Input(width int) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if width <= 0 || C.mg_graph_xf_buffer(g.ptr) == nil {
		return nil, errors.New("metalgemm: graph has no f32 input")
	}
	return &GraphResult{ptr: C.mg_graph_xf_buffer(g.ptr), out: width, p: g.p, graph: g}, nil
}

// Upload copies a second host f32 panel of P rows x width into the graph before commit and
// returns it as a graph-owned result, so one command buffer can consume two host activations
// (the dense prefill uploads the residual stream as the begin panel and the host attention
// output here). The copy is host memcpy into shared memory; it adds no encoder.
func (g *ProjectionGraph) Upload(src []float32, width int) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if width <= 0 || len(src) != g.p*width {
		return nil, fmt.Errorf("metalgemm: invalid graph upload P=%d width=%d len=%d", g.p, width, len(src))
	}
	ptr := C.mg_graph_upload(g.ptr, (*C.float)(unsafe.Pointer(&src[0])), C.int(len(src)))
	if ptr == nil {
		return nil, errors.New("metalgemm: graph upload failed")
	}
	g.hostUploadBytes += uint64(len(src)) * 4
	return &GraphResult{ptr: ptr, out: width, p: g.p, graph: g}, nil
}

// SetBufferPool enables the graph's per-shape idle-buffer recycle pool with the given
// maximum idle depth per shape (0 disables). Intermediate projection/norm outputs are
// recycled through Release instead of allocating a fresh Metal buffer per operation.
// It is opt-in and must be called before the first encode. Reuse is sound because every
// encoder shares one command buffer, so Metal's in-order execution guarantees a later
// reuse cannot overtake an earlier read. Terminal results (KV, final norm) are never
// released and are therefore never recycled.
func (g *ProjectionGraph) SetBufferPool(depth int) bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 || depth < 0 {
		return false
	}
	return C.mg_graph_set_buffer_pool(g.ptr, C.int(depth)) != 0
}

// Release returns a graph result whose last consumer has already been encoded to the
// graph's recycle pool. The result must not be read, AddInPlace'd, or otherwise consumed
// after Release. It is a no-op when the pool is disabled.
func (g *ProjectionGraph) Release(r *GraphResult) {
	if g == nil || g.ptr == nil || g.finished || g.freed || r == nil || r.ptr == nil || r.graph != g || r.released {
		return
	}
	C.mg_graph_recycle_result(g.ptr, r.ptr)
	r.released = true
}

// SetGEMVDecode opts this graph into the P=1 GEMV projection route: single-token Q4_K,
// Q6_K, and Q8 projections use the decode GEMV kernels (q4k_gemv/q4k_gemv_vectorized,
// q6k_gemv, q8_gemv) instead of the prefill GEMM pipeline, whose 64-wide token tile
// wastes 63/64 of its work at P=1. It must be called before the first encode and is
// inert for P!=1. Returns false when the route has already encoded work or the kernel
// is unavailable (fail-closed; the caller keeps the GEMM default).
func (g *ProjectionGraph) SetGEMVDecode() bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 {
		return false
	}
	return C.mg_graph_set_gemv_p1(g.ptr, 1) != 0
}

// SetGEMVVectorized selects the P=1 kernel variant used when SetGEMVDecode is active:
// true routes single-token Q4_K projections through q4k_gemv_vectorized (Q6_K, which has no
// vectorized kernel, keeps the default), false through the process default P=1 kernel
// (q4k_mul_mv, or the scalar q4k_gemv under FAK_Q4K_GEMV_KERNEL=legacy). It must be called
// before the first encode and is inert for P!=1. Returns false when the requested vectorized
// pipeline is unavailable (fail-closed).
func (g *ProjectionGraph) SetGEMVVectorized(mode bool) bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 {
		return false
	}
	m := C.int(0)
	if mode {
		m = 1
	}
	return C.mg_graph_set_gemv_vectorized(g.ptr, m) != 0
}

// SetGEMVKernel pins the P=1 kernel used when SetGEMVDecode is active: GEMVKernelNone is the
// process default, GEMVKernelScalar the legacy q4k_gemv/q6k_gemv parity reference,
// GEMVKernelVectorized q4k_gemv_vectorized (Q6_K keeps the default), and GEMVKernelMulMv the
// llama.cpp-shaped q4k_mul_mv/q6k_mul_mv. It must be called before the first encode and is inert
// for P!=1. An explicit kernel whose pipeline is unavailable returns false (fail-closed).
func (g *ProjectionGraph) SetGEMVKernel(k GEMVKernel) bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 {
		return false
	}
	mode, ok := q4kGEMVModeFor(k)
	if !ok {
		return false
	}
	return C.mg_graph_set_gemv_kernel(g.ptr, C.int(mode)) != 0
}

func (g *ProjectionGraph) add(ptr unsafe.Pointer, out int) (*GraphResult, error) {
	if ptr == nil {
		return nil, errors.New("metalgemm: graph projection encode failed")
	}
	g.encoders++
	return &GraphResult{ptr: ptr, out: out, p: g.p, graph: g}, nil
}

// SetQ4KGEMMMode sets the Q4_K projection candidate every EncodeQ4K / EncodeQ4KFrom in
// this graph dispatches: Q4KGEMMModeScalar (the historical default) or
// Q4KGEMMModeM5CooperativeSMEM, the wide-tile cooperative-SMEM candidate for the P>=64
// panel regime (fak#13041 / this leaf). It must be called before the first encode.
//
// It is fail-closed: requesting mode 2 for a graph whose P is below the kernel's 64-token
// eligibility, or when the optional pipeline is unavailable, returns false and leaves the
// graph on the scalar kernel — the caller must keep the scalar identity. This method does
// NOT consult the device/version crossover; the production selector
// (q4kGEMMModeForPrompt) only reaches mode 2 after that pin admits it, so an unwitnessed
// margin can never be requested here.
//
// Q4KGEMMModeSmallPGEMV (fak#13694) routes a 2<=P<=20 graph's Q4_K projections — and its Q6_K
// projections when the optional q6k_gemv_multiN pipelines exist — through the batched
// multi-token GEMV. It is likewise refused (false, scalar kept) for a P outside that band or
// when the Q4_K multi-token pipelines are unavailable.
func (g *ProjectionGraph) SetQ4KGEMMMode(mode Q4KGEMMMode) bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 {
		return false
	}
	var m C.int
	switch mode {
	case Q4KGEMMModeScalar:
		m = 0
	case Q4KGEMMModeM5CooperativeSMEM:
		m = 2
	case Q4KGEMMModeSmallPGEMV:
		m = 3
	case Q4KGEMMModeMulMM:
		m = 4
	default:
		return false
	}
	return C.mg_graph_set_mm_mode(g.ptr, m) != 0
}

// SetQ6KGEMMMode selects the Q6_K projection kernel every EncodeQ6K / EncodeQ6KFrom in this graph
// dispatches above the small-P band: Q4KGEMMModeScalar keeps the naive q6k_gemm (the default) and
// Q4KGEMMModeMulMM selects the fak#13692 q6k_mul_mm. Q4KGEMMModeSmallPGEMV is accepted only when
// the graph's Q4_K mode already routes the 2<=P<=20 band through the small-P GEMV, which then also
// carries its Q6_K projections (fak#13694). It must be called before the first encode and is
// fail-closed: an unavailable pipeline or any other mode returns false and keeps the naive kernel.
// P=1 graphs that opted into the GEMV route are unaffected.
func (g *ProjectionGraph) SetQ6KGEMMMode(mode Q4KGEMMMode) bool {
	if g == nil || g.ptr == nil || g.finished || g.freed || g.encoders != 0 {
		return false
	}
	switch mode {
	case Q4KGEMMModeScalar:
		return C.mg_graph_set_q6k_mode(g.ptr, 0) != 0
	case Q4KGEMMModeMulMM:
		return C.mg_graph_set_q6k_mode(g.ptr, 1) != 0
	case Q4KGEMMModeSmallPGEMV:
		return g.Q4KGEMMMode() == Q4KGEMMModeSmallPGEMV
	case Q4KGEMMModeMM32, Q4KGEMMModeM5CooperativeSMEM, Q4KGEMMModeMM32Unavailable,
		Q4KGEMMModeM5CooperativeSMEMUnavailable, Q4KGEMMModeSmallPGEMVUnavailable, Q4KGEMMModeMulMMUnavailable:
		return false
	default:
		return false
	}
}

// Q6KGEMMMode reports the Q6_K projection kernel this graph will encode above the small-P band
// (Q4KGEMMModeScalar for the naive q6k_gemm, or Q4KGEMMModeMulMM).
func (g *ProjectionGraph) Q6KGEMMMode() Q4KGEMMMode {
	if g == nil || g.ptr == nil || g.finished || g.freed {
		return Q4KGEMMModeScalar
	}
	if C.mg_graph_q6k_mode(g.ptr) == 1 {
		return Q4KGEMMModeMulMM
	}
	return Q4KGEMMModeScalar
}

// Q4KGEMMMode reports the Q4_K projection candidate this graph will request (scalar by
// default). It is the graph-side half of the typed requested/executed identity; the
// executed half is the mg_execution_event identity returned by the weight calls.
func (g *ProjectionGraph) Q4KGEMMMode() Q4KGEMMMode {
	if g == nil || g.ptr == nil || g.finished || g.freed {
		return Q4KGEMMModeScalar
	}
	switch intof := int(C.mg_graph_mm_mode(g.ptr)); intof {
	case 2:
		return Q4KGEMMModeM5CooperativeSMEM
	case 3:
		return Q4KGEMMModeSmallPGEMV
	case 4:
		return Q4KGEMMModeMulMM
	default:
		return Q4KGEMMModeScalar
	}
}
func (g *ProjectionGraph) EncodeQ4K(w *Q4KWeight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q4_K weight")
	}
	return g.add(C.mg_graph_encode_q4k(g.ptr, C.int(w.id)), w.Out)
}
func (g *ProjectionGraph) EncodeQ6K(w *Q6KWeight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q6_K weight")
	}
	q6kRegistryMu.RLock()
	defer q6kRegistryMu.RUnlock()
	if !q6kWeightValidLocked(w) {
		return nil, errors.New("metalgemm: released Q6_K weight")
	}
	return g.add(C.mg_graph_encode_q6k(g.ptr, C.int(w.id)), w.Out)
}

func (g *ProjectionGraph) EncodeQ4KFrom(w *Q4KWeight, input *GraphResult) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil || input == nil || input.graph != g || input.ptr == nil || input.p != g.p || input.out != w.In {
		return nil, errors.New("metalgemm: invalid graph Q4_K projection input")
	}
	return g.add(C.mg_graph_encode_q4k_from(g.ptr, C.int(w.id), input.ptr, C.int(input.p*input.out)), w.Out)
}

func (g *ProjectionGraph) EncodeQ6KFrom(w *Q6KWeight, input *GraphResult) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q6_K weight")
	}
	q6kRegistryMu.RLock()
	defer q6kRegistryMu.RUnlock()
	if !q6kWeightValidLocked(w) || input == nil || input.graph != g || input.ptr == nil || input.p != g.p || input.out != w.In {
		return nil, errors.New("metalgemm: invalid graph Q6_K projection input")
	}
	return g.add(C.mg_graph_encode_q6k_from(g.ptr, C.int(w.id), input.ptr, C.int(input.p*input.out)), w.Out)
}
func (g *ProjectionGraph) EncodeQ8(w *Q8Weight) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil {
		return nil, errors.New("metalgemm: nil Q8 weight")
	}
	return g.add(C.mg_graph_encode_q8(g.ptr, C.int(w.id)), w.Out)
}

// QuantizeQ8 encodes Q8_0 activation quantization for an f32 device result.
// The codes and scales remain graph-owned until Free.
func (g *ProjectionGraph) QuantizeQ8(input *GraphResult) (*QuantizedGraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if input == nil || input.graph != g || input.ptr == nil || input.p != g.p || input.out <= 0 || input.out%32 != 0 {
		return nil, errors.New("metalgemm: invalid graph quantization input")
	}
	var d unsafe.Pointer
	q := C.mg_graph_quantize_q8(g.ptr, input.ptr, C.int(input.p*input.out), &d)
	if q == nil || d == nil {
		return nil, errors.New("metalgemm: graph Q8 quantization encode failed")
	}
	g.encoders++
	return &QuantizedGraphResult{q: q, d: d, elems: input.p * input.out, graph: g}, nil
}

// EncodeQ8From encodes a Q8 projection from graph-owned activation codes and
// scales without committing, waiting, or reading the intermediate panel.
func (g *ProjectionGraph) EncodeQ8From(w *Q8Weight, input *QuantizedGraphResult) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if w == nil || input == nil || input.graph != g || input.q == nil || input.d == nil || input.elems != g.p*w.In {
		return nil, errors.New("metalgemm: invalid graph Q8 projection input")
	}
	return g.add(C.mg_graph_encode_q8_from(g.ptr, C.int(w.id), input.q, input.d, C.int(input.elems)), w.Out)
}

func (g *ProjectionGraph) qwenInput(input *GraphResult, rows, width int) error {
	if err := g.open(); err != nil {
		return err
	}
	if rows <= 0 || g.p != rows || input == nil || input.graph != g || input.ptr == nil || input.p != rows || input.out != width {
		return fmt.Errorf("metalgemm: Qwen graph requires owned P=%d width=%d input", rows, width)
	}
	return nil
}

func (g *ProjectionGraph) qwenP32Input(input *GraphResult, width int) error {
	return g.qwenInput(input, 32, width)
}

func qwenOrderedPanel(rows int) bool {
	return PromptPanelWitnessed(rows)
}

func (g *ProjectionGraph) RMSNorm(input *GraphResult, weight []float32, eps float32, gain1p bool) (*GraphResult, error) {
	if err := g.qwenInput(input, g.p, len(weight)); err != nil || eps <= 0 {
		if err == nil {
			err = errors.New("metalgemm: invalid Qwen RMSNorm epsilon")
		}
		return nil, err
	}
	gain := C.int(0)
	if gain1p {
		gain = 1
	}
	ptr := C.mg_qwen35_graph_norm(g.ptr, input.ptr, (*C.float)(unsafe.Pointer(&weight[0])), C.int(g.p), C.int(len(weight)), C.float(eps), gain, 0)
	result, err := g.add(ptr, len(weight))
	if err == nil {
		g.hostUploadBytes += uint64(len(weight)) * 4
	}
	return result, err
}

func (g *ProjectionGraph) LastRMSNorm(input *GraphResult, weight []float32, eps float32, gain1p bool) (*GraphResult, error) {
	if g == nil {
		return nil, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return nil, fmt.Errorf("metalgemm: Qwen final RMSNorm panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
	}
	if err := g.qwenInput(input, g.p, len(weight)); err != nil || eps <= 0 {
		if err == nil {
			err = errors.New("metalgemm: invalid Qwen final RMSNorm epsilon")
		}
		return nil, err
	}
	gain := C.int(0)
	if gain1p {
		gain = 1
	}
	ptr := C.mg_qwen35_graph_norm(g.ptr, input.ptr, (*C.float)(unsafe.Pointer(&weight[0])), C.int(g.p), C.int(len(weight)), C.float(eps), gain, 1)
	if ptr == nil {
		return nil, errors.New("metalgemm: Qwen final RMSNorm encode failed")
	}
	g.encoders++
	g.hostUploadBytes += uint64(len(weight)) * 4
	return &GraphResult{ptr: ptr, out: len(weight), p: 1, graph: g}, nil
}

func (g *ProjectionGraph) AddInPlace(dst, src *GraphResult) error {
	// A committed graph's command buffer can no longer take encoders; refuse instead
	// of tripping a Metal assertion on a finished MTLCommandBuffer.
	if err := g.open(); err != nil {
		return err
	}
	if dst == nil || src == nil || dst.graph != g || src.graph != g || dst.ptr == nil || src.ptr == nil || dst.p != src.p || dst.out != src.out || dst.p != g.p || dst.p <= 0 {
		return errors.New("metalgemm: invalid Qwen residual operands")
	}
	if C.mg_qwen35_graph_add(g.ptr, dst.ptr, src.ptr, C.int(dst.p*dst.out)) == 0 {
		return errors.New("metalgemm: Qwen residual encode failed")
	}
	g.encoders++
	return nil
}

func (g *ProjectionGraph) SwiGLUInPlace(gate, up *GraphResult) error {
	if err := g.open(); err != nil {
		return err
	}
	if gate == nil || up == nil || gate.graph != g || up.graph != g || gate.ptr == nil || up.ptr == nil || gate.p != g.p || up.p != g.p || gate.p <= 0 || gate.out != up.out {
		return errors.New("metalgemm: invalid Qwen SwiGLU operands")
	}
	if C.mg_qwen35_graph_swiglu(g.ptr, gate.ptr, up.ptr, C.int(g.p*gate.out)) == 0 {
		return errors.New("metalgemm: Qwen SwiGLU encode failed")
	}
	g.encoders++
	return nil
}

// AddBiasInPlace adds a host bias vector to every row of dst in place
// (dst[r*width+i] += bias[i]), the projection bias of dense Qwen2/Llama-family
// attention and MLP layers. The bias is staged once per call like the norm weights,
// so its bytes count toward HostUploadBytes; it encodes one compute encoder.
func (g *ProjectionGraph) AddBiasInPlace(dst *GraphResult, bias []float32) error {
	if err := g.open(); err != nil {
		return err
	}
	if dst == nil || dst.graph != g || dst.ptr == nil || dst.released || dst.p != g.p || dst.p <= 0 || dst.out <= 0 || len(bias) != dst.out {
		return errors.New("metalgemm: invalid graph bias operands")
	}
	if C.mg_qwen35_graph_bias(g.ptr, dst.ptr, (*C.float)(unsafe.Pointer(&bias[0])), C.int(dst.p), C.int(dst.out)) == 0 {
		return errors.New("metalgemm: graph bias encode failed")
	}
	g.encoders++
	g.hostUploadBytes += uint64(len(bias)) * 4
	return nil
}

func (g *ProjectionGraph) SplitGatedQ(input *GraphResult, qwidth, hd int) (q, gate *GraphResult, err error) {
	if g == nil {
		return nil, nil, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return nil, nil, fmt.Errorf("metalgemm: Qwen gated-Q panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
	}
	if err = g.qwenInput(input, g.p, 2*qwidth); err != nil || qwidth <= 0 || hd <= 0 || qwidth%hd != 0 {
		return nil, nil, err
	}
	var qp, gp unsafe.Pointer
	if C.mg_qwen35_graph_split(g.ptr, input.ptr, C.int(qwidth), C.int(hd), &qp, &gp) == 0 || qp == nil || gp == nil {
		return nil, nil, errors.New("metalgemm: Qwen gated-Q split encode failed")
	}
	g.encoders++
	return &GraphResult{ptr: qp, out: qwidth, p: g.p, graph: g}, &GraphResult{ptr: gp, out: qwidth, p: g.p, graph: g}, nil
}

// qwenAttentionInputs validates the attention operands. gate may be nil: that is the
// ungated epilogue (dense Llama/Qwen2 attention), where the output is the softmax-weighted
// V sum with no sigmoid gate. A non-nil gate must be an owned qwidth-wide panel and
// keeps the historical gated epilogue unchanged.
func (g *ProjectionGraph) qwenAttentionInputs(q, k, v, gate *GraphResult, qwidth, kvwidth int) error {
	for _, check := range []struct {
		r *GraphResult
		w int
	}{{q, qwidth}, {k, kvwidth}, {v, kvwidth}} {
		if err := g.qwenInput(check.r, g.p, check.w); err != nil {
			return err
		}
	}
	if gate != nil {
		return g.qwenInput(gate, g.p, qwidth)
	}
	return nil
}

// ptrOrNil is the native gate pointer: nil selects the ungated attention epilogue.
func (r *GraphResult) ptrOrNil() unsafe.Pointer {
	if r == nil {
		return nil
	}
	return r.ptr
}

// FullAttention encodes ordered-panel full attention against a host KV prefix. A nil
// gate runs the ungated epilogue; a non-nil gate applies out/(1+exp(-gate)).
func (g *ProjectionGraph) FullAttention(q, k, v, gate *GraphResult, qnorm, knorm, cosv, sinv, prefixK, prefixV []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionResult, error) {
	if g == nil {
		return Qwen35GraphAttentionResult{}, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return Qwen35GraphAttentionResult{}, fmt.Errorf("metalgemm: Qwen full-attention panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
	}
	qwidth, kvwidth := nH*hd, nKV*hd
	if err := g.qwenAttentionInputs(q, k, v, gate, qwidth, kvwidth); err != nil {
		return Qwen35GraphAttentionResult{}, err
	}
	qNormShapeOK := len(qnorm) == hd || len(qnorm) == qwidth
	kNormShapeOK := len(knorm) == hd || len(knorm) == kvwidth
	if !qNormShapeOK || !kNormShapeOK || hd < 2 || hd > 256 || rotary < 2 || rotary > hd || rotary%2 != 0 || len(cosv) != g.p*(rotary/2) || len(sinv) != len(cosv) || base < 0 || len(prefixK) != base*kvwidth || len(prefixV) != base*kvwidth || scale <= 0 || qkEps <= 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: invalid Qwen full-attention geometry")
	}
	var pk, pv *C.float
	if base > 0 {
		pk = (*C.float)(unsafe.Pointer(&prefixK[0]))
		pv = (*C.float)(unsafe.Pointer(&prefixV[0]))
	}
	gain := C.int(0)
	if gain1p {
		gain = 1
	}
	qkn := C.int(0)
	if qkNorm {
		qkn = 1
	}
	var outp, krawp, kpostp, vcurp unsafe.Pointer
	if C.mg_qwen35_graph_attention(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptrOrNil(),
		(*C.float)(unsafe.Pointer(&qnorm[0])), (*C.float)(unsafe.Pointer(&knorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])), pk, pv,
		C.int(base), C.int(nH), C.int(nKV), C.int(hd), C.int(rotary), C.float(scale), C.float(qkEps), gain, qkn, C.int(len(qnorm)), C.int(len(knorm)),
		&outp, &krawp, &kpostp, &vcurp) == 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: Qwen full-attention encode failed")
	}
	// Native full attention owns Q/K normalization, current-K/V append, and
	// attention as three encoders.
	g.encoders += 3
	g.hostUploadBytes += uint64(len(qnorm)+len(knorm)+len(cosv)+len(sinv)+len(prefixK)+len(prefixV)) * 4
	return Qwen35GraphAttentionResult{
		Output: &GraphResult{ptr: outp, out: qwidth, p: g.p, graph: g},
		KRaw:   &GraphResult{ptr: krawp, out: kvwidth, p: g.p, graph: g},
		KPost:  &GraphResult{ptr: kpostp, out: kvwidth, p: g.p, graph: g},
		V:      &GraphResult{ptr: vcurp, out: kvwidth, p: g.p, graph: g},
	}, nil
}

// FullAttentionDevice is FullAttention with the KV prefix held on the device in a
// persistent DeviceKV pair shared across a panel walk. `layer` selects the pair's
// layer slice (layer*layerStride floats) and `base` is the prefix row count already
// resident there. The panel's new K/V rows are appended on the device, so no host
// prefix memcpy and no per-panel host readback are paid; the caller reads the pair
// back ONCE at the end of the walk to satisfy the host cache contract.
//
// A nil or under-sized pair, or any native decline, returns an error and encodes
// nothing observable beyond its own graph; the caller falls back to the host walk.
// A nil gate runs the ungated epilogue (see FullAttention).
func (g *ProjectionGraph) FullAttentionDevice(q, k, v, gate *GraphResult, kv *DeviceKV, layer int, qnorm, knorm, cosv, sinv []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionResult, error) {
	if g == nil {
		return Qwen35GraphAttentionResult{}, errGraphTerminal
	}
	// kv.kraw may be nil (NewDeviceKVAttendOnly): the native entry then skips the KRaw
	// append, and the panel's KRaw stays readable as the returned graph result.
	if kv == nil || kv.kpost == nil || kv.v == nil || layer < 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: device KV is not allocated")
	}
	if !qwenOrderedPanel(g.p) {
		return Qwen35GraphAttentionResult{}, fmt.Errorf("metalgemm: Qwen full-attention panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
	}
	if nKV <= 0 || hd <= 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: invalid Qwen full-attention geometry")
	}
	kvwidth := nKV * hd
	kvOff := layer * kv.layerStride
	// The panel writes rows [base, base+P) of this layer's slice. Admitting
	// base+P > capacity would spill into the NEXT layer's row 0 (the native check is
	// against the whole buffer, not the slice), so refuse it here.
	if kvOff < 0 || kvOff+kv.layerStride > kv.elems || base < 0 || base+g.p > kv.layerStride/kvwidth {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: device KV layer slice out of range")
	}
	qwidth := nH * hd
	if err := g.qwenAttentionInputs(q, k, v, gate, qwidth, kvwidth); err != nil {
		return Qwen35GraphAttentionResult{}, err
	}
	qNormShapeOK := len(qnorm) == hd || len(qnorm) == qwidth
	kNormShapeOK := len(knorm) == hd || len(knorm) == kvwidth
	if !qNormShapeOK || !kNormShapeOK || hd < 2 || hd > 256 || rotary < 2 || rotary > hd || rotary%2 != 0 || len(cosv) != g.p*(rotary/2) || len(sinv) != len(cosv) || base < 0 || scale <= 0 || qkEps <= 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: invalid Qwen full-attention geometry")
	}
	gain := C.int(0)
	if gain1p {
		gain = 1
	}
	qkn := C.int(0)
	if qkNorm {
		qkn = 1
	}
	var outp, krawp, kpostp, vcurp unsafe.Pointer
	dkvOK := C.mg_qwen35_graph_attention_dkv(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptrOrNil(),
		(*C.float)(unsafe.Pointer(&qnorm[0])), (*C.float)(unsafe.Pointer(&knorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])), kv.kraw, kv.kpost, kv.v, C.int(kvOff),
		C.int(base), C.int(nH), C.int(nKV), C.int(hd), C.int(rotary), C.float(scale), C.float(qkEps), gain, qkn, C.int(len(qnorm)), C.int(len(knorm)),
		&outp, &krawp, &kpostp, &vcurp) != 0
	// kv's cleanup must not free the device pair while the native call binds it
	// (the committed command buffer retains it from then on).
	runtime.KeepAlive(kv)
	if !dkvOK {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: Qwen full-attention device-KV encode failed")
	}
	// Native full attention owns Q/K normalization, current-K/V append, and
	// attention as three encoders; the device append is the fourth.
	g.encoders += 3
	g.hostUploadBytes += uint64(len(qnorm)+len(knorm)+len(cosv)+len(sinv)) * 4
	return Qwen35GraphAttentionResult{
		Output: &GraphResult{ptr: outp, out: qwidth, p: g.p, graph: g},
		KRaw:   &GraphResult{ptr: krawp, out: kvwidth, p: g.p, graph: g},
		KPost:  &GraphResult{ptr: kpostp, out: kvwidth, p: g.p, graph: g},
		V:      &GraphResult{ptr: vcurp, out: kvwidth, p: g.p, graph: g},
	}, nil
}

// Qwen35GraphAttentionQ8Result is a full-attention result plus the panel's rows in
// the packed layout the attention consumed. KCodes/VCodes carry int8 codes four per
// float32 word (out = kvWidth/4), decoded from a terminal readback with KVQ8Codes;
// KScales/VScales carry kvWidth/KVQ8BlockSize scales per row.
type Qwen35GraphAttentionQ8Result struct {
	Qwen35GraphAttentionResult
	KCodes, KScales, VCodes, VScales *GraphResult
}

func cFlag(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// qwenQ8Attention validates the operands both packed-Q8 attention entries share.
// head_dim must be a whole number of Q8 blocks so a block never straddles two heads.
func (g *ProjectionGraph) qwenQ8Attention(q, k, v, gate *GraphResult, qnorm, knorm, cosv, sinv []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32) error {
	if g == nil {
		return errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return fmt.Errorf("metalgemm: Qwen full-attention panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
	}
	if nH <= 0 || nKV <= 0 || nH%nKV != 0 || hd < KVQ8BlockSize || hd > 256 || hd%KVQ8BlockSize != 0 {
		return errors.New("metalgemm: invalid Qwen Q8 full-attention geometry")
	}
	qwidth, kvwidth := nH*hd, nKV*hd
	for _, check := range []struct {
		r *GraphResult
		w int
	}{{q, qwidth}, {gate, qwidth}, {k, kvwidth}, {v, kvwidth}} {
		if err := g.qwenInput(check.r, g.p, check.w); err != nil {
			return err
		}
	}
	qNormShapeOK := len(qnorm) == hd || len(qnorm) == qwidth
	kNormShapeOK := len(knorm) == hd || len(knorm) == kvwidth
	if !qNormShapeOK || !kNormShapeOK || rotary < 2 || rotary > hd || rotary%2 != 0 || len(cosv) != g.p*(rotary/2) || len(sinv) != len(cosv) || base < 0 || scale <= 0 || qkEps <= 0 {
		return errors.New("metalgemm: invalid Qwen Q8 full-attention geometry")
	}
	return nil
}

func (g *ProjectionGraph) qwenQ8Result(qwidth, kvwidth int, outp, krawp, kpostp, vcurp, kcp, ksp, vcp, vsp unsafe.Pointer) Qwen35GraphAttentionQ8Result {
	res := func(ptr unsafe.Pointer, out int) *GraphResult {
		return &GraphResult{ptr: ptr, out: out, p: g.p, graph: g}
	}
	// Q/K norm, panel quantize, device append and attention.
	g.encoders += 4
	return Qwen35GraphAttentionQ8Result{
		Qwen35GraphAttentionResult: Qwen35GraphAttentionResult{Output: res(outp, qwidth), KRaw: res(krawp, kvwidth), KPost: res(kpostp, kvwidth), V: res(vcurp, kvwidth)},
		KCodes:                     res(kcp, kvwidth/4),
		KScales:                    res(ksp, kvwidth/KVQ8BlockSize),
		VCodes:                     res(vcp, kvwidth/4),
		VScales:                    res(vsp, kvwidth/KVQ8BlockSize),
	}
}

// FullAttentionQ8 is FullAttention over a host-owned PACKED Q8_0 prefix (#12981).
// prefix holds `base` rows in the host cache's realized layout, so the host keeps
// 1.125 B/element of token KV instead of 4 and the per-panel prefix upload shrinks
// by the same ratio. The panel's post-RoPE K and raw V rows are quantized on the
// device before it attends (the host q8 path's append-then-attend order) and come
// back packed, so the caller appends exactly the bytes the device attended. KRaw
// (the host's f32 pre-RoPE row), KPost and V are also returned f32.
func (g *ProjectionGraph) FullAttentionQ8(q, k, v, gate *GraphResult, qnorm, knorm, cosv, sinv []float32, prefix KVQ8Rows, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionQ8Result, error) {
	if err := g.qwenQ8Attention(q, k, v, gate, qnorm, knorm, cosv, sinv, base, nH, nKV, hd, rotary, scale, qkEps); err != nil {
		return Qwen35GraphAttentionQ8Result{}, err
	}
	qwidth, kvwidth := nH*hd, nKV*hd
	if prefix.Rows(kvwidth) != base {
		return Qwen35GraphAttentionQ8Result{}, errors.New("metalgemm: packed Q8 prefix does not hold base rows")
	}
	var pkc, pvc *C.schar
	var pks, pvs *C.float
	if base > 0 {
		pkc, pvc = (*C.schar)(unsafe.Pointer(&prefix.KCodes[0])), (*C.schar)(unsafe.Pointer(&prefix.VCodes[0]))
		pks, pvs = (*C.float)(unsafe.Pointer(&prefix.KScales[0])), (*C.float)(unsafe.Pointer(&prefix.VScales[0]))
	}
	var outp, krawp, kpostp, vcurp, kcp, ksp, vcp, vsp unsafe.Pointer
	if C.mg_qwen35_graph_attention_q8(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptr,
		(*C.float)(unsafe.Pointer(&qnorm[0])), (*C.float)(unsafe.Pointer(&knorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])), pkc, pks, pvc, pvs,
		C.int(base), C.int(nH), C.int(nKV), C.int(hd), C.int(rotary), C.float(scale), C.float(qkEps), cFlag(gain1p), cFlag(qkNorm), C.int(len(qnorm)), C.int(len(knorm)),
		&outp, &krawp, &kpostp, &vcurp, &kcp, &ksp, &vcp, &vsp) == 0 {
		return Qwen35GraphAttentionQ8Result{}, errors.New("metalgemm: Qwen Q8 full-attention encode failed")
	}
	g.hostUploadBytes += uint64(len(qnorm)+len(knorm)+len(cosv)+len(sinv)+len(prefix.KScales)+len(prefix.VScales))*4 + uint64(len(prefix.KCodes)+len(prefix.VCodes))
	return g.qwenQ8Result(qwidth, kvwidth, outp, krawp, kpostp, vcurp, kcp, ksp, vcp, vsp), nil
}

// FullAttentionDeviceQ8 is FullAttentionDevice over a packed DeviceKVQ8 store: layer
// `layer` must already hold `base` rows. The panel's packed K/V rows are appended at
// row base on the device and attention reads the packed prefix+panel in place, so a
// walk pays neither a host prefix copy nor an F32 mirror. The panel's packed rows
// (and f32 KRaw/KPost/V) are also returned as graph results, so a host cache can
// mirror the store with the exact bytes without a region download.
func (g *ProjectionGraph) FullAttentionDeviceQ8(q, k, v, gate *GraphResult, kv *DeviceKVQ8, layer int, qnorm, knorm, cosv, sinv []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionQ8Result, error) {
	if err := g.qwenQ8Attention(q, k, v, gate, qnorm, knorm, cosv, sinv, base, nH, nKV, hd, rotary, scale, qkEps); err != nil {
		return Qwen35GraphAttentionQ8Result{}, err
	}
	qwidth, kvwidth := nH*hd, nKV*hd
	if kv.width() != kvwidth {
		return Qwen35GraphAttentionQ8Result{}, errors.New("metalgemm: packed device KV width does not match the attention geometry")
	}
	elemOff, _, _, _, err := kv.region(layer, base, g.p)
	if err != nil {
		return Qwen35GraphAttentionQ8Result{}, err
	}
	elemOff -= base * kvwidth
	var outp, krawp, kpostp, vcurp, kcp, ksp, vcp, vsp unsafe.Pointer
	if C.mg_qwen35_graph_attention_dkv_q8(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptr,
		(*C.float)(unsafe.Pointer(&qnorm[0])), (*C.float)(unsafe.Pointer(&knorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])), kv.kc, kv.ks, kv.vc, kv.vs, C.long(elemOff),
		C.int(base), C.int(nH), C.int(nKV), C.int(hd), C.int(rotary), C.float(scale), C.float(qkEps), cFlag(gain1p), cFlag(qkNorm), C.int(len(qnorm)), C.int(len(knorm)),
		&outp, &krawp, &kpostp, &vcurp, &kcp, &ksp, &vcp, &vsp) == 0 {
		return Qwen35GraphAttentionQ8Result{}, errors.New("metalgemm: Qwen Q8 full-attention device-KV encode failed")
	}
	g.hostUploadBytes += uint64(len(qnorm)+len(knorm)+len(cosv)+len(sinv)) * 4
	return g.qwenQ8Result(qwidth, kvwidth, outp, krawp, kpostp, vcurp, kcp, ksp, vcp, vsp), nil
}

func (g *ProjectionGraph) GDN(state *GDNState, mixed, z, b, a *GraphResult, panel GDNPanel) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if !PromptPanelWitnessed(g.p) {
		return nil, &GDNDeclinedError{Reason: fmt.Sprintf("graph GDN panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)}
	}
	geometry, err := state.graphGeometry()
	if err != nil {
		return nil, err
	}
	var checkpoint *GDNGraphCheckpoint
	for _, candidate := range g.gdnCheckpoints {
		if candidate.backup == state {
			return nil, errors.New("metalgemm: checkpoint backup cannot be mutated by graph")
		}
		if candidate.live == state {
			checkpoint = candidate
		}
	}
	if checkpoint != nil && checkpoint.used {
		return nil, errors.New("metalgemm: checkpoint live owner already mutated by graph")
	}
	if checkpoint == nil {
		for _, lease := range g.gdnLeases {
			if lease.state == state {
				return nil, errors.New("metalgemm: GDN owner already retained by graph")
			}
		}
	}
	if err := geometry.validate(); err != nil {
		return nil, &GDNDeclinedError{Reason: err.Error()}
	}
	wants := []struct {
		r *GraphResult
		w int
	}{{mixed, geometry.convDim()}, {z, geometry.valueDim()}, {b, geometry.NumValueHeads}, {a, geometry.NumValueHeads}}
	for _, want := range wants {
		if err := g.qwenInput(want.r, g.p, want.w); err != nil {
			return nil, err
		}
	}
	shapes := []struct {
		name string
		got  int
		want int
	}{
		{"conv1d", len(panel.Conv1D), geometry.convDim() * geometry.ConvKernel},
		{"a_log", len(panel.ALog), geometry.NumValueHeads},
		{"dt_bias", len(panel.DTBias), geometry.NumValueHeads},
		{"norm", len(panel.Norm), geometry.ValueHeadDim},
	}
	for _, shape := range shapes {
		if shape.got != shape.want {
			return nil, &GDNDeclinedError{Reason: fmt.Sprintf("%s elements=%d, want %d", shape.name, shape.got, shape.want)}
		}
	}
	if panel.RMSNormEpsilon <= 0 {
		return nil, &GDNDeclinedError{Reason: "RMSNorm epsilon must be positive"}
	}
	var owner C.int
	var done chan struct{}
	if checkpoint != nil {
		owner = checkpoint.liveOwner
	} else {
		owner, done, err = state.retainGraph()
		if err != nil {
			return nil, err
		}
	}
	ptr := C.mg_gdn_graph_encode(g.ptr, owner, mixed.ptr, z.ptr, b.ptr, a.ptr,
		gdnF32(panel.Conv1D), gdnF32(panel.ALog), gdnF32(panel.DTBias), gdnF32(panel.Norm),
		C.int(g.p), C.int(geometry.NumKeyHeads), C.int(geometry.NumValueHeads), C.int(geometry.KeyHeadDim), C.int(geometry.ValueHeadDim), C.int(geometry.ConvKernel), C.float(panel.RMSNormEpsilon))
	if ptr == nil {
		if checkpoint == nil {
			state.releaseGraph(done)
		}
		return nil, errors.New("metalgemm: Qwen GDN graph encode failed")
	}
	if checkpoint != nil {
		checkpoint.used = true
	} else {
		g.gdnLeases = append(g.gdnLeases, gdnGraphLease{state: state, done: done})
	}
	g.encoders++
	g.hostUploadBytes += uint64(len(panel.Conv1D)+len(panel.ALog)+len(panel.DTBias)+len(panel.Norm)) * 4
	return &GraphResult{ptr: ptr, out: geometry.valueDim(), p: g.p, graph: g}, nil
}

func (g *ProjectionGraph) releaseGDNLeases(completed bool) {
	for _, lease := range g.gdnLeases {
		lease.state.completeGraph(lease.done, completed)
	}
	g.gdnLeases = nil
}

func (g *ProjectionGraph) finishGDNCheckpoints(receipt GraphReceipt) {
	completed := receipt.Committed && receipt.CompletedWait
	for _, checkpoint := range g.gdnCheckpoints {
		checkpoint.graphTerminal(completed)
	}
}

func (g *ProjectionGraph) abandonGDNCheckpoints() {
	for _, checkpoint := range g.gdnCheckpoints {
		checkpoint.graphTerminal(false)
	}
}

func (g *ProjectionGraph) Finish() (GraphReceipt, error) {
	if err := g.open(); err != nil {
		return GraphReceipt{}, err
	}
	if g.encoders == 0 {
		return GraphReceipt{}, errGraphEmpty
	}
	var r C.mg_graph_receipt
	// Submission consumes the owner even when the device reports an error; never permit a
	// second commit of the same native command buffer.
	g.finished = true
	inject := C.int(0)
	if g.injectPostSubmitFailure {
		inject = 1
	}
	if g.injectDeviceFault {
		inject = 2
	}
	q4kKernels := GEMVKernelSet(C.mg_graph_gemv_executed(g.ptr, 0))
	q6kKernels := GEMVKernelSet(C.mg_graph_gemv_executed(g.ptr, 1))
	ok := C.mg_graph_finish(g.ptr, &r, inject) != 0
	receipt := GraphReceipt{Committed: r.committed != 0, CompletedWait: r.completed_wait != 0, TimingAvailable: r.timing_available != 0, Encoders: int(r.encoders), HostReadbacks: int(r.host_readbacks), HostUploadBytes: g.hostUploadBytes, AllocatedBuffers: int(r.allocated_buffers), RetainedBufferBytes: uint64(r.retained_buffer_bytes), GPUMilliseconds: float64(r.gpu_milliseconds), WaitMilliseconds: float64(r.wait_milliseconds), Q4KGEMVKernels: q4kKernels, Q6KGEMVKernels: q6kKernels}
	if !ok && receipt.Committed && !receipt.CompletedWait {
		g.quarantineCommittedGraph()
		return receipt, commandBufferStallError(receipt.WaitMilliseconds, int(r.status_code), int(r.error_code), cString(&r.error_text[0]), "graph finish")
	}
	g.finishGDNCheckpoints(receipt)
	g.releaseGDNLeases(receipt.Committed && receipt.CompletedWait)
	if !ok {
		if receipt.Committed {
			// A committed buffer that did not reach MTLCommandBufferStatusCompleted is the
			// unbounded-wait failure metal_stall.go exists for. The native wait is not
			// interruptible, so classify the OBSERVED terminal state as the typed stall the
			// package already defines. The inject_post_submit_failure seam (inject=1) reports
			// completed=1 and so deliberately keeps its GraphPostSubmitError; only a
			// non-completed buffer becomes the typed stall.
			return receipt, &GraphPostSubmitError{Reason: "injected or device completion failure"}
		}
		return receipt, errors.New("metalgemm: graph submit failed")
	}
	return receipt, nil
}

// quarantineCommittedGraph transfers the sole native owner to a background
// terminal waiter. Metal may still reference every graph result and GDN owner,
// so caller Free becomes a no-op until the real command buffer is terminal.
// The failed request stays failed even if the device later reports Completed.
func (g *ProjectionGraph) quarantineCommittedGraph() {
	ptr := g.ptr
	g.ptr = nil
	g.freed = true
	g.quarantineDone = make(chan struct{})
	projectionGraphQuarantines.Add(1)
	go func() {
		defer close(g.quarantineDone)
		defer projectionGraphQuarantines.Add(-1)
		_ = C.mg_graph_await_terminal(ptr)
		g.finishGDNCheckpoints(GraphReceipt{Committed: true, CompletedWait: false})
		g.releaseGDNLeases(false)
		C.mg_graph_free(ptr)
	}()
}

// commandBufferStallError converts a non-completed native command buffer into the
// package's typed MetalCommandBufferStallError, folding in the device status and any
// bounded NSError text the native receipt captured. CheckCommandBufferWait owns the
// classification so the typed error is the same one every observer sees; when the
// wait itself is under the limit the buffer still never reached Completed (a
// device-side fault), and the same typed error is used so callers see one class.
func commandBufferStallError(waitMS float64, statusCode, errorCode int, deviceText, op string) error {
	err := CheckCommandBufferWait(op, waitMS, DefaultCommandBufferWaitLimit)
	if err == nil {
		err = MetalCommandBufferStallError{
			Operation:          op,
			WaitedMilliseconds: waitMS,
			LimitMilliseconds:  DefaultCommandBufferWaitLimit,
		}
	}
	if deviceText == "" {
		return err
	}
	return fmt.Errorf("%w (native status=%d error=%d: %s)", err, statusCode, errorCode, deviceText)
}

// cString copies a fixed-size C char buffer into a Go string.
func cString(p *C.char) string {
	if p == nil {
		return ""
	}
	var out []byte
	for i := 0; i < 256; i++ {
		c := *(*byte)(unsafe.Add(unsafe.Pointer(p), i))
		if c == 0 {
			break
		}
		out = append(out, c)
	}
	return string(out)
}

func (g *ProjectionGraph) Read(r *GraphResult) ([]float32, error) {
	if g == nil || !g.finished || g.freed {
		return nil, errGraphTerminal
	}
	if r == nil || r.graph != g || r.ptr == nil || r.released {
		return nil, errors.New("metalgemm: result does not belong to graph")
	}
	out := make([]float32, r.p*r.out)
	if len(out) > 0 && C.mg_graph_read(g.ptr, r.ptr, (*C.float)(unsafe.Pointer(&out[0])), C.int(len(out))) == 0 {
		return nil, errors.New("metalgemm: graph read failed")
	}
	g.readbacks++
	return out, nil
}

// FinishRead performs the graph's only commit/wait and then one packed host
// readback containing every requested result. Intermediate device results are
// never materialized as host slices.
func (g *ProjectionGraph) FinishRead(results ...*GraphResult) ([][]float32, GraphReceipt, error) {
	if len(results) == 0 {
		return nil, GraphReceipt{}, errors.New("metalgemm: graph terminal read requires a result")
	}
	total := 0
	for _, r := range results {
		if r == nil || r.graph != g || r.ptr == nil || r.p <= 0 || r.out <= 0 || r.released {
			return nil, GraphReceipt{}, errors.New("metalgemm: terminal result does not belong to graph")
		}
		total += r.p * r.out
	}
	receipt, err := g.Finish()
	if err != nil {
		return nil, receipt, err
	}
	packed := make([]float32, total)
	ptrBytes := C.size_t(len(results)) * C.size_t(unsafe.Sizeof(uintptr(0)))
	sizeBytes := C.size_t(len(results)) * C.size_t(unsafe.Sizeof(C.int(0)))
	cPtrs := C.malloc(ptrBytes)
	cSizes := C.malloc(sizeBytes)
	if cPtrs == nil || cSizes == nil {
		if cPtrs != nil {
			C.free(cPtrs)
		}
		if cSizes != nil {
			C.free(cSizes)
		}
		return nil, receipt, errors.New("metalgemm: allocate terminal result table")
	}
	defer C.free(cPtrs)
	defer C.free(cSizes)
	ptrs := unsafe.Slice((*unsafe.Pointer)(cPtrs), len(results))
	sizes := unsafe.Slice((*C.int)(cSizes), len(results))
	for i, r := range results {
		ptrs[i] = r.ptr
		sizes[i] = C.int(r.p * r.out)
	}
	if C.mg_graph_read_pack(g.ptr, (*unsafe.Pointer)(cPtrs), (*C.int)(cSizes), C.int(len(results)), (*C.float)(unsafe.Pointer(&packed[0])), C.int(total)) == 0 {
		return nil, receipt, errors.New("metalgemm: graph terminal read failed")
	}
	g.readbacks++
	receipt.HostReadbacks = 1
	receipt.HostReadbackBytes = uint64(total) * 4
	out := make([][]float32, len(results))
	off := 0
	for i, r := range results {
		n := r.p * r.out
		out[i] = packed[off : off+n]
		off += n
	}
	return out, receipt, nil
}
func (g *ProjectionGraph) Free() {
	if g != nil && g.ptr != nil && !g.freed {
		finished := g.finished
		C.mg_graph_free(g.ptr)
		if !finished {
			g.abandonGDNCheckpoints()
		}
		g.releaseGDNLeases(false)
		g.ptr = nil
		g.freed = true
	}
}
