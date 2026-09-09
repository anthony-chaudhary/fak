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
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
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
int mg_graph_read(void *graph, void *result, float *dst, int n);
int mg_graph_read_pack(void *graph, void **results, const int *sizes, int count, float *dst, int total);
void mg_graph_free(void *graph);
void *mg_graph_xf_buffer(void *graph);
void *mg_qwen35_graph_norm(void *graph, void *input, const float *weight, int rows, int width, float eps, int gain1p, int last_only);
int mg_qwen35_graph_add(void *graph, void *x, void *y, int n);
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
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

var (
	errGraphTerminal = errors.New("metalgemm: graph is terminal")
	errGraphEmpty    = errors.New("metalgemm: graph has no encoded projections")
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
}

type GraphResult struct {
	ptr   unsafe.Pointer
	out   int
	p     int
	graph *ProjectionGraph
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

type ProjectionGraph struct {
	ptr                     unsafe.Pointer
	p                       int
	encoders                int
	finished                bool
	freed                   bool
	readbacks               int
	hostUploadBytes         uint64
	injectPostSubmitFailure bool
	gdnLeases               []gdnGraphLease
	gdnCheckpoints          []*GDNGraphCheckpoint
}

type gdnGraphLease struct {
	state *GDNState
	done  chan struct{}
}

// GDNCheckpointReceipt describes persistent-state movement only.
type GDNCheckpointReceipt struct {
	EncodedDeviceCopies, DeviceCopies    int
	BufferSwaps                          int
	HostStateUploads, HostStateReadbacks int
}

// GDNGraphCheckpoint is a graph-bound rollback token for one live/backup pair.
type GDNGraphCheckpoint struct {
	mu                     sync.Mutex
	graph                  *ProjectionGraph
	live, backup           *GDNState
	liveOwner, backupOwner C.int
	liveDone, backupDone   chan struct{}
	used                   bool
	terminal, completed    bool
	closed, restored       bool
	leasesReleased         bool
	stateIdentitySHA256    string
	stateVersion           uint64
	checkpointGeneration   uint64
}

func gdnCheckpointIdentity(live, backup *GDNState, generation uint64) string {
	h := sha256.New()
	_, _ = h.Write([]byte("fak.metalgemm-gdn-checkpoint-owner/v1\x00"))
	var bits [8]byte
	write := func(value uint64) {
		binary.LittleEndian.PutUint64(bits[:], value)
		_, _ = h.Write(bits[:])
	}
	for _, value := range []uint64{
		live.ownerEpoch, uint64(live.conv), uint64(live.recurrent), live.version, generation,
		backup.ownerEpoch, uint64(backup.conv), uint64(backup.recurrent),
		uint64(live.geometry.NumKeyHeads), uint64(live.geometry.NumValueHeads),
		uint64(live.geometry.KeyHeadDim), uint64(live.geometry.ValueHeadDim), uint64(live.geometry.ConvKernel),
	} {
		write(value)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func lockGDNPair(a, b *GDNState) func() {
	first, second := a, b
	if uintptr(unsafe.Pointer(first)) > uintptr(unsafe.Pointer(second)) {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	return func() {
		second.mu.Unlock()
		first.mu.Unlock()
	}
}

// CheckpointGDN reserves live and backup until the checkpoint is restored or
// closed and encodes two private-to-private copies before later graph operations
// can mutate live. Pair locks always use pointer order; busy owners are declined
// without waiting.
func (g *ProjectionGraph) CheckpointGDN(live, backup *GDNState) (*GDNGraphCheckpoint, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if live == nil || backup == nil {
		return nil, &GDNDeclinedError{Reason: "checkpoint requires live and backup owners"}
	}
	if live == backup {
		return nil, &GDNDeclinedError{Reason: "checkpoint owners alias"}
	}
	unlock := lockGDNPair(live, backup)
	defer unlock()
	if live.closed || backup.closed {
		return nil, &GDNDeclinedError{Reason: "checkpoint owner is closed"}
	}
	if live.graphDone != nil || backup.graphDone != nil {
		return nil, &GDNDeclinedError{Reason: "checkpoint owner is busy"}
	}
	if live.geometry != backup.geometry {
		return nil, &GDNDeclinedError{Reason: "checkpoint owner geometry mismatch"}
	}
	if live.owner < 0 || backup.owner < 0 || live.owner == backup.owner {
		return nil, &GDNDeclinedError{Reason: "checkpoint native owners alias or are missing"}
	}
	liveDone, backupDone := make(chan struct{}), make(chan struct{})
	live.graphDone, backup.graphDone = liveDone, backupDone
	live.checkpointGen++
	checkpointGeneration := live.checkpointGen
	stateIdentity := gdnCheckpointIdentity(live, backup, checkpointGeneration)
	if C.mg_gdn_graph_checkpoint(g.ptr, live.owner, backup.owner) == 0 {
		live.graphDone, backup.graphDone = nil, nil
		close(liveDone)
		close(backupDone)
		return nil, errors.New("metalgemm: GDN graph checkpoint encode failed")
	}
	checkpoint := &GDNGraphCheckpoint{
		graph: g, live: live, backup: backup,
		liveOwner: live.owner, backupOwner: backup.owner,
		liveDone: liveDone, backupDone: backupDone,
		stateIdentitySHA256: stateIdentity, stateVersion: live.version,
		checkpointGeneration: checkpointGeneration,
	}
	g.gdnCheckpoints = append(g.gdnCheckpoints, checkpoint)
	g.encoders++
	return checkpoint, nil
}

// Receipt distinguishes copies merely encoded in an unsubmitted command buffer
// from copies proven complete by the graph's terminal fence. The checkpoint
// path performs no host state transfer.
func (c *GDNGraphCheckpoint) Receipt() GDNCheckpointReceipt {
	if c == nil {
		return GDNCheckpointReceipt{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	r := GDNCheckpointReceipt{EncodedDeviceCopies: 2}
	if c.completed {
		r.DeviceCopies = 2
	}
	if c.restored {
		r.BufferSwaps = 2
	}
	return r
}

// StateIdentity returns the immutable pre-mutation checkpoint owner/version
// identity only while the completed rollback token remains live. It never
// materializes convolution or recurrent buffers on the host.
func (c *GDNGraphCheckpoint) StateIdentity() (string, error) {
	if c == nil {
		return "", &GDNDeclinedError{Reason: "missing GDN checkpoint"}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.terminal || !c.completed {
		return "", &GDNDeclinedError{Reason: "GDN checkpoint identity unavailable before terminal completion"}
	}
	if c.closed || c.restored {
		return "", &GDNDeclinedError{Reason: "GDN checkpoint identity is stale"}
	}
	return c.stateIdentitySHA256, nil
}

func (c *GDNGraphCheckpoint) takeLeasesLocked() (chan struct{}, chan struct{}) {
	if c.leasesReleased {
		return nil, nil
	}
	c.leasesReleased = true
	return c.liveDone, c.backupDone
}

func (c *GDNGraphCheckpoint) releaseLeases(liveDone, backupDone chan struct{}) {
	if liveDone != nil {
		c.live.releaseGraph(liveDone)
	}
	if backupDone != nil {
		c.backup.releaseGraph(backupDone)
	}
}

func (c *GDNGraphCheckpoint) graphTerminal(completed bool) {
	c.mu.Lock()
	c.terminal = true
	c.completed = completed
	if completed {
		unlock := lockGDNPair(c.live, c.backup)
		if c.live.graphDone == c.liveDone && c.backup.graphDone == c.backupDone &&
			!c.live.closed && !c.backup.closed && c.live.owner == c.liveOwner && c.backup.owner == c.backupOwner {
			c.backup.version = c.stateVersion
			if c.used {
				c.live.version++
			}
		}
		unlock()
	}
	if !completed {
		c.closed = true
	}
	var liveDone, backupDone chan struct{}
	if c.closed {
		liveDone, backupDone = c.takeLeasesLocked()
	}
	c.mu.Unlock()
	c.releaseLeases(liveDone, backupDone)
}

// Close abandons rollback and releases the exclusive owner pair after the graph
// reaches a terminal fence. It is safe to call repeatedly. A pre-terminal Close
// records abandonment but cannot release buffers still referenced by Metal.
func (c *GDNGraphCheckpoint) Close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.closed = true
	var liveDone, backupDone chan struct{}
	if c.terminal {
		liveDone, backupDone = c.takeLeasesLocked()
	}
	c.mu.Unlock()
	c.releaseLeases(liveDone, backupDone)
}

// Restore atomically exchanges native private-buffer references after the
// graph's terminal wait, while the checkpoint still exclusively owns both
// states. It submits no Metal work and runs in O(1).
func (c *GDNGraphCheckpoint) Restore() error {
	if c == nil || c.graph == nil || c.live == nil || c.backup == nil {
		return &GDNDeclinedError{Reason: "missing GDN checkpoint"}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.restored {
		return &GDNDeclinedError{Reason: "GDN checkpoint already restored"}
	}
	if c.closed {
		return &GDNDeclinedError{Reason: "GDN checkpoint is closed"}
	}
	if !c.terminal || !c.completed {
		return &GDNDeclinedError{Reason: "GDN checkpoint graph has no completed terminal wait"}
	}
	unlock := lockGDNPair(c.live, c.backup)
	defer func() {
		unlock()
		if c.restored {
			liveDone, backupDone := c.takeLeasesLocked()
			c.releaseLeases(liveDone, backupDone)
		}
	}()
	if c.live.closed || c.backup.closed {
		return &GDNDeclinedError{Reason: "checkpoint owner is closed"}
	}
	if c.live.graphDone != c.liveDone || c.backup.graphDone != c.backupDone || c.leasesReleased {
		return &GDNDeclinedError{Reason: "checkpoint no longer exclusively owns state"}
	}
	if c.live.geometry != c.backup.geometry || c.live.owner != c.liveOwner || c.backup.owner != c.backupOwner {
		return &GDNDeclinedError{Reason: "checkpoint owner identity or geometry changed"}
	}
	if C.mg_gdn_state_swap_buffers(c.liveOwner, c.backupOwner) == 0 {
		return errors.New("metalgemm: GDN checkpoint restore failed")
	}
	c.live.version, c.backup.version = c.backup.version, c.live.version
	c.restored = true
	c.closed = true
	return nil
}

// InjectPostSubmitFailureForTest makes Finish return an accepted failure after
// the native command buffer has committed and completed. It exists solely for
// deterministic fail-closed lifecycle witnesses.
func (g *ProjectionGraph) InjectPostSubmitFailureForTest() {
	if g != nil {
		g.injectPostSubmitFailure = true
	}
}

// BeginProjectionGraph uploads one activation panel for all projections in the graph.
// xf is required by Q4_K/Q6_K; xq/xd are required by Q8. Supplying both permits mixed graphs.
func BeginProjectionGraph(xf []float32, xq []int8, xd []float32, P, in int) (*ProjectionGraph, error) {
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
func (g *ProjectionGraph) add(ptr unsafe.Pointer, out int) (*GraphResult, error) {
	if ptr == nil {
		return nil, errors.New("metalgemm: graph projection encode failed")
	}
	g.encoders++
	return &GraphResult{ptr: ptr, out: out, p: g.p, graph: g}, nil
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
	return rows == 2 || rows == 3 || rows == 4 || rows == 32
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
		return nil, fmt.Errorf("metalgemm: Qwen final RMSNorm panel P=%d outside witnessed set {2,3,4,32}", g.p)
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
	if gate == nil || up == nil || gate.graph != g || up.graph != g || gate.ptr == nil || up.ptr == nil || gate.p != g.p || up.p != g.p || gate.p <= 0 || gate.out != up.out {
		return errors.New("metalgemm: invalid Qwen SwiGLU operands")
	}
	if C.mg_qwen35_graph_swiglu(g.ptr, gate.ptr, up.ptr, C.int(g.p*gate.out)) == 0 {
		return errors.New("metalgemm: Qwen SwiGLU encode failed")
	}
	g.encoders++
	return nil
}

func (g *ProjectionGraph) SplitGatedQ(input *GraphResult, qwidth, hd int) (q, gate *GraphResult, err error) {
	if g == nil {
		return nil, nil, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return nil, nil, fmt.Errorf("metalgemm: Qwen gated-Q panel P=%d outside witnessed set {2,3,4,32}", g.p)
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

func (g *ProjectionGraph) FullAttention(q, k, v, gate *GraphResult, qnorm, knorm, cosv, sinv, prefixK, prefixV []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionResult, error) {
	if g == nil {
		return Qwen35GraphAttentionResult{}, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return Qwen35GraphAttentionResult{}, fmt.Errorf("metalgemm: Qwen full-attention panel P=%d outside witnessed set {2,3,4,32}", g.p)
	}
	qwidth, kvwidth := nH*hd, nKV*hd
	for _, check := range []struct {
		r *GraphResult
		w int
	}{{q, qwidth}, {gate, qwidth}, {k, kvwidth}, {v, kvwidth}} {
		if err := g.qwenInput(check.r, g.p, check.w); err != nil {
			return Qwen35GraphAttentionResult{}, err
		}
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
	if C.mg_qwen35_graph_attention(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptr,
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

func (g *ProjectionGraph) GDN(state *GDNState, mixed, z, b, a *GraphResult, panel GDNPanel) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	if (g.p < 1 || g.p > 4) && g.p != 32 {
		return nil, &GDNDeclinedError{Reason: fmt.Sprintf("graph GDN panel P=%d outside witnessed set {1,2,3,4,32}", g.p)}
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
	ok := C.mg_graph_finish(g.ptr, &r, inject) != 0
	receipt := GraphReceipt{Committed: r.committed != 0, CompletedWait: r.completed_wait != 0, TimingAvailable: r.timing_available != 0, Encoders: int(r.encoders), HostReadbacks: int(r.host_readbacks), HostUploadBytes: g.hostUploadBytes, GPUMilliseconds: float64(r.gpu_milliseconds), WaitMilliseconds: float64(r.wait_milliseconds)}
	g.finishGDNCheckpoints(receipt)
	g.releaseGDNLeases(receipt.Committed && receipt.CompletedWait)
	if !ok {
		if receipt.Committed {
			return receipt, &GraphPostSubmitError{Reason: "injected or device completion failure"}
		}
		return receipt, errors.New("metalgemm: graph submit failed")
	}
	return receipt, nil
}
func (g *ProjectionGraph) Read(r *GraphResult) ([]float32, error) {
	if g == nil || !g.finished || g.freed {
		return nil, errGraphTerminal
	}
	if r == nil || r.graph != g || r.ptr == nil {
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
		if r == nil || r.graph != g || r.ptr == nil || r.p <= 0 || r.out <= 0 {
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
