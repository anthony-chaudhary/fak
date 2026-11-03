package compute

// CPU-guarded contract test for the NCCL tensor-parallel group lifecycle adapted from
// ashhart/TensorFold (Apache-2.0): zig/src/cuda/tp_group.zig, pin
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209. ADAPT decision — the Group lifecycle is
// re-expressed in idiomatic Go, no upstream code is copied: open(rank,world) validation,
// rank-order all-gather, matched-count send/recv, post-sync check, abort-once, and the
// destroy-after-abort refusal (an aborted communicator is never torn down twice).
//
// This file is the pure-Go reference model of that closed contract. The real NCCL-level
// state machine lives in cuda_nccl.cu (single-process ncclCommInitAll set:
// fcuda_nccl_abort / fcuda_nccl_destroy) and cuda_nccl_pg.cu (multi-process
// ncclCommInitRank comm: fcuda_nccl_pg_send / fcuda_nccl_pg_recv / fcuda_nccl_pg_check /
// fcuda_nccl_pg_abort plus the refusing fcuda_nccl_pg_destroy), compiled only under
// FAK_CUDA_NCCL; the Go wrappers are cuda_collective_pg.go. Because those layers build
// only with a CUDA+NCCL toolchain, the model here is what a CPU host can execute
// deterministically: the same flags, the same refusals, the same underlying call counts.

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

// Closed-contract sentinels the reference model reports; tests assert them with errors.Is,
// never on prose.
var (
	// ErrNCCLWorldMismatch: a rank/world disagreement at open or gather time (the C layer's
	// -2/-3 class: world <= 0, rank outside [0, world), shard count != world).
	ErrNCCLWorldMismatch = errors.New("nccl op-group: world/rank mismatch")
	// ErrNCCLCountMismatch: a send/recv pair whose element counts disagree, or unequal
	// gather bands (the matched-counts contract).
	ErrNCCLCountMismatch = errors.New("nccl op-group: unmatched send/recv count")
	// ErrNCCLNotInitialized: an operation on a group that was never opened (or is closed);
	// the C layer's -1 fail-closed refusal. Destroy alone is exempt — it is a safe no-op.
	ErrNCCLNotInitialized = errors.New("nccl op-group: not initialized")
	// ErrNCCLAborted: the post-sync check's report once the group has been aborted.
	ErrNCCLAborted = errors.New("nccl op-group: aborted")
)

// ncclOpGroup mirrors the C++ lifecycle state one-to-one:
//
//	opened       <-> g_pg_world != 0 (group active)
//	aborted      <-> g_pg_aborted latched
//	destroyed    <-> globals reset by destroy
//	abortCalls   <-> ncclCommAbort invocation count (abort-once pins this at 1)
//	destroyCalls <-> ncclCommDestroy invocation count (the double-destroy guard pins it)
type ncclOpGroup struct {
	rank         int
	world        int
	opened       bool
	aborted      bool
	destroyed    bool
	abortCalls   int
	destroyCalls int
	pending      map[int][][]float32 // peer -> queued sends awaiting a matched recv
	postSyncErr  error               // async error a preceding collective left on the stream
}

// openOpGroup is the open(rank,world) validation: world must be positive and rank must be a
// member of it, mirroring fcuda_nccl_pg_init's -2 refusal.
func openOpGroup(rank, world int) (*ncclOpGroup, error) {
	if world <= 0 || rank < 0 || rank >= world {
		return nil, fmt.Errorf("%w: rank=%d world=%d", ErrNCCLWorldMismatch, rank, world)
	}
	return &ncclOpGroup{rank: rank, world: world, opened: true, pending: make(map[int][][]float32)}, nil
}

// allGather places every rank's band in RANK ORDER (rank r's band at offset r*band) — the
// layout NCCL all-gather guarantees and both C bootstrap shapes document. Refuses a shard
// count that disagrees with world (world mismatch) and unequal bands (count mismatch).
func (g *ncclOpGroup) allGather(shards [][]float32) ([]float32, error) {
	if !g.opened || g.destroyed {
		return nil, fmt.Errorf("%w: group is not open", ErrNCCLNotInitialized)
	}
	if len(shards) != g.world {
		return nil, fmt.Errorf("%w: %d shards, world=%d", ErrNCCLWorldMismatch, len(shards), g.world)
	}
	band := len(shards[0])
	out := make([]float32, 0, band*g.world)
	for r, s := range shards {
		if len(s) != band {
			return nil, fmt.Errorf("%w: rank %d band %d, want %d", ErrNCCLCountMismatch, r, len(s), band)
		}
		out = append(out, s...)
	}
	return out, nil
}

// send queues `data` for rank peer, mirroring fcuda_nccl_pg_send's post (the C layer posts
// into NCCL instead of queueing). The peer must be a group member.
func (g *ncclOpGroup) send(peer int, data []float32) error {
	if !g.opened || g.destroyed {
		return fmt.Errorf("%w: group is not open", ErrNCCLNotInitialized)
	}
	if peer < 0 || peer >= g.world {
		return fmt.Errorf("%w: peer=%d world=%d", ErrNCCLWorldMismatch, peer, g.world)
	}
	g.pending[peer] = append(g.pending[peer], append([]float32(nil), data...))
	return nil
}

// recv takes the oldest outstanding send FROM peer and requires the SAME element count —
// the matched-counts contract. A missing post or a differently-sized one is refused with
// ErrNCCLCountMismatch, the model's stand-in for the deadlock/corruption NCCL would produce
// on a mismatched exchange.
func (g *ncclOpGroup) recv(peer int, want []float32) error {
	if !g.opened || g.destroyed {
		return fmt.Errorf("%w: group is not open", ErrNCCLNotInitialized)
	}
	if peer < 0 || peer >= g.world {
		return fmt.Errorf("%w: peer=%d world=%d", ErrNCCLWorldMismatch, peer, g.world)
	}
	q := g.pending[peer]
	have := 0
	if len(q) > 0 {
		have = len(q[0])
	}
	if have == 0 || have != len(want) {
		return fmt.Errorf("%w: peer=%d have=%d want=%d", ErrNCCLCountMismatch, peer, have, len(want))
	}
	copy(want, q[0])
	g.pending[peer] = q[1:]
	return nil
}

// check is the post-sync check: it surfaces the error a preceding collective left on the
// stream (the C layer materializes it via cudaStreamSynchronize) and the abort itself once
// latched. Mirrors fcuda_nccl_pg_check.
func (g *ncclOpGroup) check() error {
	if !g.opened || g.destroyed {
		return fmt.Errorf("%w: group is not open", ErrNCCLNotInitialized)
	}
	if g.aborted {
		return ErrNCCLAborted
	}
	return g.postSyncErr
}

// abort is abort-ONCE: the first call latches the group and records exactly one underlying
// ncclCommAbort; later calls are no-ops. Mirrors fcuda_nccl_pg_abort / fcuda_nccl_abort.
func (g *ncclOpGroup) abort() error {
	if !g.opened || g.destroyed {
		return fmt.Errorf("%w: group is not open", ErrNCCLNotInitialized)
	}
	if g.aborted {
		return nil
	}
	g.aborted = true
	g.abortCalls++
	return nil
}

// destroy is the ordered teardown with the destroy-after-abort refusal: after an abort it
// performs NO underlying ncclCommDestroy (abort-then-destroy is a no-op — the aborted
// communicator is never torn down twice) and only resets state. Idempotent when called
// again after a plain destroy, and a safe no-op when no group was ever opened (matching the
// C side's "safe to call when no group is active"). Mirrors fcuda_nccl_pg_destroy /
// fcuda_nccl_destroy.
func (g *ncclOpGroup) destroy() error {
	if !g.opened || g.destroyed {
		return nil
	}
	if !g.aborted {
		g.destroyCalls++
	}
	g.destroyed = true
	g.aborted = false
	g.pending = nil
	return nil
}

// TestNCCLOpGroupAbortOnceNoDestroy pins the TP-group lifecycle contract borrowed from
// ashhart/TensorFold zig/src/cuda/tp_group.zig@9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// (Apache-2.0, ADAPT): abort-once, no double destroy, abort-then-destroy no-op, world
// mismatch refusal, rank-order gather, and matched send/recv counts. CPU-guarded: the
// reference model runs on any host; the NCCL-level legs it mirrors need FAK_CUDA_NCCL.
//
// fak-test:runtime fast est=1ms lane=default
func TestNCCLOpGroupAbortOnceNoDestroy(t *testing.T) {
	t.Run("abort-once-no-double-destroy", func(t *testing.T) {
		g, err := openOpGroup(0, 2)
		if err != nil {
			t.Fatalf("openOpGroup(0,2) = %v, want ok", err)
		}
		if err := g.abort(); err != nil {
			t.Fatalf("first abort = %v, want nil", err)
		}
		if err := g.abort(); err != nil {
			t.Fatalf("second abort = %v, want nil (no-op)", err)
		}
		if g.abortCalls != 1 {
			t.Fatalf("abortCalls = %d, want 1 (abort-once)", g.abortCalls)
		}
		if err := g.destroy(); err != nil {
			t.Fatalf("destroy after abort = %v, want nil (no-op)", err)
		}
		if g.destroyCalls != 0 {
			t.Fatalf("destroyCalls = %d, want 0 (abort-then-destroy performs no ncclCommDestroy)", g.destroyCalls)
		}
	})

	t.Run("plain-destroy-tears-down-exactly-once", func(t *testing.T) {
		g, err := openOpGroup(1, 4)
		if err != nil {
			t.Fatalf("openOpGroup(1,4) = %v, want ok", err)
		}
		if err := g.destroy(); err != nil {
			t.Fatalf("destroy = %v, want nil", err)
		}
		if g.destroyCalls != 1 {
			t.Fatalf("destroyCalls = %d, want 1", g.destroyCalls)
		}
		if err := g.destroy(); err != nil {
			t.Fatalf("second destroy = %v, want nil (idempotent)", err)
		}
		if g.destroyCalls != 1 {
			t.Fatalf("destroyCalls = %d after idempotent retry, want 1", g.destroyCalls)
		}
	})

	t.Run("world-mismatch-refuses-open", func(t *testing.T) {
		for _, tc := range []struct{ rank, world int }{{2, 2}, {0, 0}, {-1, 2}, {5, 3}} {
			if _, err := openOpGroup(tc.rank, tc.world); !errors.Is(err, ErrNCCLWorldMismatch) {
				t.Fatalf("openOpGroup(%d,%d) err = %v, want ErrNCCLWorldMismatch", tc.rank, tc.world, err)
			}
		}
	})

	t.Run("rank-order-gather", func(t *testing.T) {
		g, err := openOpGroup(1, 3)
		if err != nil {
			t.Fatalf("openOpGroup(1,3) = %v, want ok", err)
		}
		out, err := g.allGather([][]float32{{1, 2}, {3, 4}, {5, 6}})
		if err != nil {
			t.Fatalf("allGather = %v, want ok", err)
		}
		// Rank 0's band first, rank 1's next, rank 2's last — the rank-order layout.
		if !slices.Equal(out, []float32{1, 2, 3, 4, 5, 6}) {
			t.Fatalf("allGather layout = %v, want [1 2 3 4 5 6] (rank order)", out)
		}
		if _, err := g.allGather([][]float32{{1, 2}, {3, 4}}); !errors.Is(err, ErrNCCLWorldMismatch) {
			t.Fatalf("short gather err = %v, want ErrNCCLWorldMismatch", err)
		}
		if _, err := g.allGather([][]float32{{1, 2}, {3}, {5, 6}}); !errors.Is(err, ErrNCCLCountMismatch) {
			t.Fatalf("unequal-band gather err = %v, want ErrNCCLCountMismatch", err)
		}
	})

	t.Run("matched-send-recv-counts", func(t *testing.T) {
		g, err := openOpGroup(0, 2)
		if err != nil {
			t.Fatalf("openOpGroup(0,2) = %v, want ok", err)
		}
		if err := g.send(1, []float32{9, 8}); err != nil {
			t.Fatalf("send = %v, want ok", err)
		}
		var got [2]float32
		if err := g.recv(1, got[:]); err != nil {
			t.Fatalf("matched recv = %v, want ok", err)
		}
		if got != ([2]float32{9, 8}) {
			t.Fatalf("recv payload = %v, want [9 8]", got)
		}
		// A differently-sized outstanding post is refused (matched counts).
		if err := g.send(1, []float32{1, 2, 3}); err != nil {
			t.Fatalf("send = %v, want ok", err)
		}
		var wide [4]float32
		if err := g.recv(1, wide[:]); !errors.Is(err, ErrNCCLCountMismatch) {
			t.Fatalf("mismatched recv err = %v, want ErrNCCLCountMismatch", err)
		}
		// With no matching outstanding post the recv is refused too.
		if err := g.recv(1, wide[:]); !errors.Is(err, ErrNCCLCountMismatch) {
			t.Fatalf("unbacked recv err = %v, want ErrNCCLCountMismatch", err)
		}
		// An out-of-range peer refuses at the protocol boundary.
		if err := g.send(2, []float32{1}); !errors.Is(err, ErrNCCLWorldMismatch) {
			t.Fatalf("send to peer 2 err = %v, want ErrNCCLWorldMismatch", err)
		}
	})

	t.Run("post-sync-check-surfaces-errors", func(t *testing.T) {
		g, err := openOpGroup(0, 2)
		if err != nil {
			t.Fatalf("openOpGroup(0,2) = %v, want ok", err)
		}
		if err := g.check(); err != nil {
			t.Fatalf("clean check = %v, want nil", err)
		}
		g.postSyncErr = errors.New("async kernel fault left on the default stream")
		if err := g.check(); err == nil {
			t.Fatal("check after a failed collective = nil, want the surfaced error")
		}
		if err := g.abort(); err != nil {
			t.Fatalf("abort = %v, want nil", err)
		}
		if err := g.check(); !errors.Is(err, ErrNCCLAborted) {
			t.Fatalf("check after abort = %v, want ErrNCCLAborted", err)
		}
	})

	t.Run("closed-group-fails-closed", func(t *testing.T) {
		var g ncclOpGroup // never opened: the zero value stands in for g_pg_world == 0
		if err := g.abort(); !errors.Is(err, ErrNCCLNotInitialized) {
			t.Fatalf("abort on closed group = %v, want ErrNCCLNotInitialized", err)
		}
		if err := g.check(); !errors.Is(err, ErrNCCLNotInitialized) {
			t.Fatalf("check on closed group = %v, want ErrNCCLNotInitialized", err)
		}
		if err := g.destroy(); err != nil {
			t.Fatalf("destroy on closed group = %v, want nil (safe no-op, like the C side)", err)
		}
	})
}
