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
int mg_graph_read(void *graph, void *result, float *dst, int n);
int mg_graph_read_pack(void *graph, void **results, const int *sizes, int count, float *dst, int total);
void mg_graph_free(void *graph);
void *mg_graph_xf_buffer(void *graph);
int mg_graph_set_gemv_vectorized(void *graph, int mode);
int mg_graph_set_gemv_p1(void *graph, int mode);
int mg_graph_set_mm_mode(void *graph, int mode);
int mg_graph_mm_mode(void *graph);
int mg_graph_set_buffer_pool(void *graph, int depth);
void mg_graph_recycle_result(void *graph, void *result);
void *mg_qwen35_graph_kv_alloc(int elems);
void mg_qwen35_graph_kv_free(void *kv);
int mg_qwen35_graph_kv_upload(void *kv, const float *src, int elems);
int mg_qwen35_graph_kv_download(void *kv, float *dst, int elems);
int mg_qwen35_graph_attention_dkv(void *graph, void *q, void *k, void *v, void *gate,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    void *kv_kraw, void *kv_kpost, void *kv_v, int kv_off,
    int base, int nh, int nkv, int hd, int rotary, float scale, float qk_eps, int gain1p, int qknorm,
    int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost, void **vcurrent);
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

// DeviceKV is a caller-owned device-resident KV triple shared by every
// full-attention layer across a panel walk. Unlike a GraphResult it is NOT
// tracked on any ProjectionGraph, so a panel's graph Free() never reclaims it:
// one DeviceKV lives for the whole sequence and each panel's new KRaw/KPost/V
// rows are appended into it by mg_qwen35_graph_attention_dkv, so panel p+1 reads
// the KPost/V prefix on the device instead of paying a host readback + prefix
// re-upload.
//
// Each of the three sides reserves LayerStride = tokens*NumKVHeads*HeadDim floats
// per layer; layer l's rows begin at l*LayerStride. Close frees all three device
// buffers. Side 0 is KRaw (pre-norm key), side 1 is KPost (attention prefix) and
// side 2 is V (raw value).
type DeviceKV struct {
	kraw, kpost, v unsafe.Pointer
	elems          int // per-side capacity in floats
	layerStride    int // floats per layer per side
}

// NewDeviceKV allocates a persistent device KV triple sized for tokens tokens
// across layers full-attention layers, each layer holding kvWidth = nKV*hd
// floats per token. Returns nil when any device allocation or the geometry is
// unavailable, which the caller treats as a decline (fail-open to the host walk).
func NewDeviceKV(layers, tokens, kvWidth int) *DeviceKV {
	if layers <= 0 || tokens <= 0 || kvWidth <= 0 || tokens > int(^uint(0)>>1)/kvWidth {
		return nil
	}
	layerStride := tokens * kvWidth
	if layers > int(^uint(0)>>1)/layerStride {
		return nil
	}
	elems := layers * layerStride
	kraw := C.mg_qwen35_graph_kv_alloc(C.int(elems))
	if kraw == nil {
		return nil
	}
	kpost := C.mg_qwen35_graph_kv_alloc(C.int(elems))
	if kpost == nil {
		C.mg_qwen35_graph_kv_free(kraw)
		return nil
	}
	v := C.mg_qwen35_graph_kv_alloc(C.int(elems))
	if v == nil {
		C.mg_qwen35_graph_kv_free(kraw)
		C.mg_qwen35_graph_kv_free(kpost)
		return nil
	}
	return &DeviceKV{kraw: kraw, kpost: kpost, v: v, elems: elems, layerStride: layerStride}
}

// LayerStride reports the per-layer float width per side the pair was sized for,
// so a caller can validate its geometry before encoding.
func (d *DeviceKV) LayerStride() int {
	if d == nil {
		return 0
	}
	return d.layerStride
}

func (d *DeviceKV) side(side int) (unsafe.Pointer, error) {
	if d == nil || d.kraw == nil || d.kpost == nil || d.v == nil {
		return nil, errors.New("metalgemm: device KV is not allocated")
	}
	switch side {
	case 0:
		return d.kraw, nil
	case 1:
		return d.kpost, nil
	case 2:
		return d.v, nil
	default:
		return nil, errors.New("metalgemm: device KV side must be 0 (KRaw), 1 (KPost) or 2 (V)")
	}
}

// Upload copies host prefix rows into the device side at offset 0. It is used only
// to seed an already-resident prefix (e.g. a pre-existing cache); a fresh walk
// appends every row on the device and never uploads.
func (d *DeviceKV) Upload(side int, src []float32) error {
	return d.UploadRegion(side, 0, src)
}

// UploadRegion copies `src` into a device side starting at float offset `off`, so
// a layer's existing host prefix can be seeded at l*LayerStride() before a walk.
func (d *DeviceKV) UploadRegion(side, off int, src []float32) error {
	buf, err := d.side(side)
	if err != nil {
		return err
	}
	if len(src) == 0 {
		return nil
	}
	if off < 0 || off+len(src) > d.elems {
		return errors.New("metalgemm: device KV upload exceeds capacity")
	}
	// The native uploader copies to the buffer start, so upload the whole
	// [0, off+len) region: stage the existing device bytes below `off` first, then
	// the caller's rows, then write once. Device bytes below `off` are either a
	// previously seeded prefix or zero, so preserving them is required.
	staging := make([]float32, off+len(src))
	if off > 0 {
		if C.mg_qwen35_graph_kv_download(buf, (*C.float)(unsafe.Pointer(&staging[0])), C.int(off)) == 0 {
			return errors.New("metalgemm: device KV upload read-back failed")
		}
	}
	copy(staging[off:], src)
	if C.mg_qwen35_graph_kv_upload(buf, (*C.float)(unsafe.Pointer(&staging[0])), C.int(len(staging))) == 0 {
		return errors.New("metalgemm: device KV upload failed")
	}
	return nil
}

// Download copies a device side back to a host slice once, at the END of a
// device-resident walk, so the host cache contract the decode path depends on
// still holds. This is the single terminal readback that replaces the per-panel
// one.
func (d *DeviceKV) Download(side int, dst []float32) error {
	return d.DownloadRegion(side, 0, dst)
}

// DownloadRegion copies `dst` floats starting at float offset `off` from a device
// side. It lets one persistent triple serve every full-attention layer: layer l's
// rows live at l*LayerStride(), and the walk downloads each slice into the host
// cache row exactly once at the end.
func (d *DeviceKV) DownloadRegion(side, off int, dst []float32) error {
	buf, err := d.side(side)
	if err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	if off < 0 || off+len(dst) > d.elems {
		return errors.New("metalgemm: device KV download exceeds capacity")
	}
	// The native downloader copies from the buffer start, so read the layer slice
	// into a staging buffer sized to its end offset, then take the tail. This
	// allocates per layer but only once per walk.
	staging := make([]float32, off+len(dst))
	if C.mg_qwen35_graph_kv_download(buf, (*C.float)(unsafe.Pointer(&staging[0])), C.int(len(staging))) == 0 {
		return errors.New("metalgemm: device KV download failed")
	}
	copy(dst, staging[off:])
	return nil
}

// Close frees the device KV triple. Safe to call twice.
func (d *DeviceKV) Close() {
	if d == nil {
		return
	}
	if d.kraw != nil {
		C.mg_qwen35_graph_kv_free(d.kraw)
		d.kraw = nil
	}
	if d.kpost != nil {
		C.mg_qwen35_graph_kv_free(d.kpost)
		d.kpost = nil
	}
	if d.v != nil {
		C.mg_qwen35_graph_kv_free(d.v)
		d.v = nil
	}
	d.elems, d.layerStride = 0, 0
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

// InjectDeviceFaultForTest makes the native receipt report a committed command
// buffer whose terminal status never reached Completed, without blocking on a
// real device fault. It exists solely to witness the stall classification on the
// real Finish path.
func (g *ProjectionGraph) InjectDeviceFaultForTest() {
	if g != nil {
		g.injectDeviceFault = true
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
	if g == nil || g.ptr == nil || g.finished || g.freed || r == nil || r.ptr == nil || r.graph != g {
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
// true routes single-token Q4_K projections through q4k_gemv_vectorized, false through
// the scalar q4k_gemv. It must be called before the first encode and is inert for P!=1.
// Returns false when the requested vectorized pipeline is unavailable (fail-closed).
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
	default:
		return false
	}
	return C.mg_graph_set_mm_mode(g.ptr, m) != 0
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

func (g *ProjectionGraph) FullAttention(q, k, v, gate *GraphResult, qnorm, knorm, cosv, sinv, prefixK, prefixV []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionResult, error) {
	if g == nil {
		return Qwen35GraphAttentionResult{}, errGraphTerminal
	}
	if !qwenOrderedPanel(g.p) {
		return Qwen35GraphAttentionResult{}, fmt.Errorf("metalgemm: Qwen full-attention panel P=%d outside witnessed set [1,%d]", g.p, PromptPanelMaxTokens)
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

// FullAttentionDevice is FullAttention with the KV prefix held on the device in a
// persistent DeviceKV pair shared across a panel walk. `layer` selects the pair's
// layer slice (layer*layerStride floats) and `base` is the prefix row count already
// resident there. The panel's new K/V rows are appended on the device, so no host
// prefix memcpy and no per-panel host readback are paid; the caller reads the pair
// back ONCE at the end of the walk to satisfy the host cache contract.
//
// A nil or under-sized pair, or any native decline, returns an error and encodes
// nothing observable beyond its own graph; the caller falls back to the host walk.
func (g *ProjectionGraph) FullAttentionDevice(q, k, v, gate *GraphResult, kv *DeviceKV, layer int, qnorm, knorm, cosv, sinv []float32, base, nH, nKV, hd, rotary int, scale, qkEps float32, gain1p, qkNorm bool) (Qwen35GraphAttentionResult, error) {
	if g == nil {
		return Qwen35GraphAttentionResult{}, errGraphTerminal
	}
	if kv == nil || kv.kraw == nil || kv.kpost == nil || kv.v == nil || layer < 0 {
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
	if kvOff < 0 || kvOff+kv.layerStride > kv.elems || base > kv.layerStride/kvwidth {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: device KV layer slice out of range")
	}
	qwidth := nH * hd
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
	if C.mg_qwen35_graph_attention_dkv(g.ptr, q.ptr, k.ptr, v.ptr, gate.ptr,
		(*C.float)(unsafe.Pointer(&qnorm[0])), (*C.float)(unsafe.Pointer(&knorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])), kv.kraw, kv.kpost, kv.v, C.int(kvOff),
		C.int(base), C.int(nH), C.int(nKV), C.int(hd), C.int(rotary), C.float(scale), C.float(qkEps), gain, qkn, C.int(len(qnorm)), C.int(len(knorm)),
		&outp, &krawp, &kpostp, &vcurp) == 0 {
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
	ok := C.mg_graph_finish(g.ptr, &r, inject) != 0
	receipt := GraphReceipt{Committed: r.committed != 0, CompletedWait: r.completed_wait != 0, TimingAvailable: r.timing_available != 0, Encoders: int(r.encoders), HostReadbacks: int(r.host_readbacks), HostUploadBytes: g.hostUploadBytes, GPUMilliseconds: float64(r.gpu_milliseconds), WaitMilliseconds: float64(r.wait_milliseconds)}
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
			if !receipt.CompletedWait {
				return receipt, commandBufferStallError(receipt.WaitMilliseconds, int(r.status_code), int(r.error_code), cString(&r.error_text[0]), "graph finish")
			}
			return receipt, &GraphPostSubmitError{Reason: "injected or device completion failure"}
		}
		return receipt, errors.New("metalgemm: graph submit failed")
	}
	return receipt, nil
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
