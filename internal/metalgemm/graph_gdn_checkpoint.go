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
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"sync"
	"unsafe"
)

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
