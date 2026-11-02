// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Split-phase transfer ownership adapted from vllm/v1/kv_offload/tiering/manager.py
// at 5c8ae7c74aaa99d1f7fbeea942fd7282e3413810. Modified for synchronous Go
// SnapshotStore calls: prepare/pin under the tree lock, execute outside it,
// validate publication, and retain the job until actual callback retirement.
// Ordinary non-speculative planner lookup, scoped lookup, warm-disk and pressure
// callers use this transaction. Unscoped speculative planner lookup retains its
// existing serialized generic API; no blanket speculative unlock is claimed.
package radixkv

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	ErrRemoteTransferBusy    = errors.New("radixkv: remote snapshot transfer busy")
	ErrRemoteTransferStale   = errors.New("radixkv: remote snapshot transfer became stale")
	ErrRemoteTransferLock    = errors.New("radixkv: remote snapshot transfer requires the held tree lock")
	ErrRemoteConfigExhausted = errors.New("radixkv: remote snapshot configuration generation exhausted")
)

// RemoteSnapshotTransferStats is fixed-cardinality transaction accounting.
// These counters describe jobs, not byte/RSS bounds or measured serving gains.
// Like all Tree metadata, reading them requires the tree's external lock.
type RemoteSnapshotTransferStats struct {
	Active    int    `json:"active"`
	Started   uint64 `json:"started"`
	Completed uint64 `json:"completed"`
	Busy      uint64 `json:"busy"`
	Stale     uint64 `json:"stale"`
}

type remoteTransferState struct {
	active    *remoteTransferJob
	started   uint64
	completed uint64
	busy      uint64
	stale     uint64
}

// remoteTransferLock is created only at explicit Locked APIs or inside the
// ScopedTree method that already holds its own serialization lock. Ordinary
// Tree APIs never implicitly unlock a caller's mutex.
type remoteTransferLock struct {
	locker   sync.Locker
	released bool
}

type remoteTransferJob struct {
	tree         *Tree
	state        *remoteTransferState
	node         *node
	record       RecordHandle
	generation   uint64
	ref          *remoteSnapshotRef
	namespace    string
	digest       string
	scope        string
	tokens       int
	store        SnapshotStore
	backend      compute.Backend
	config       model.Config
	lock         *remoteTransferLock
	finished     bool
	staleCounted bool
}

// RemoteSnapshotTransfers returns transaction counters under the caller's lock.
func (t *Tree) RemoteSnapshotTransfers() RemoteSnapshotTransferStats {
	if t == nil || t.remoteTransfers == nil {
		return RemoteSnapshotTransferStats{}
	}
	s := t.remoteTransfers
	out := RemoteSnapshotTransferStats{Started: s.started, Completed: s.completed, Busy: s.busy, Stale: s.stale}
	if s.active != nil {
		out.Active = 1
	}
	return out
}

func (t *Tree) beginRemoteTransfer(ctx context.Context, ns string, n *node, lock *remoteTransferLock) (*remoteTransferJob, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if lock != nil && lock.locker == nil {
		return nil, ErrRemoteTransferLock
	}
	handle, ok := n.RecordHandle()
	if !ok || !t.ValidateRecord(handle) {
		return nil, ErrRemoteTransferStale
	}
	if t.remoteTransfers == nil {
		t.remoteTransfers = &remoteTransferState{}
	}
	state := t.remoteTransfers
	// No queue or background goroutine: admission is bounded before payload
	// allocation and before taking a breaker probe. Reentrant callbacks cannot
	// wait for their own job; a lookup truthfully falls back to a cold miss.
	if state.active != nil {
		state.busy++
		return nil, ErrRemoteTransferBusy
	}
	job := &remoteTransferJob{
		tree: t, state: state, node: n, record: handle, generation: t.remoteConfigGeneration,
		ref: n.remoteSnapshot, namespace: ns, digest: t.snapshotDigest(ns, n),
		scope: t.snapshotScope(ns), tokens: n.plen, store: t.remoteSnapshotStore,
		backend: t.remoteBackend, config: t.remoteConfig, lock: lock,
	}
	n.refs++
	state.active = job
	state.started++
	return job, nil
}

// execute returns only after the synchronous callback actually retires.
// Cancellation does not release the job while a store may still read/write.
// Reacquisition is deferred even through a callback panic, before any outer
// caller defer can run. CloneToHost/Marshal remain in the locked prepare phase.
func (j *remoteTransferJob) execute(work func() error) (err error) {
	if j.lock != nil {
		j.lock.released = true
		j.lock.locker.Unlock()
		defer j.lock.locker.Lock()
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("radixkv: remote snapshot callback panic: %v", recovered)
		}
	}()
	return work()
}

func (j *remoteTransferJob) validate(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	t := j.tree
	if t.remoteConfigGeneration != j.generation || !t.RemoteSnapshotEnabled() ||
		!t.ValidateRecord(j.record) || j.node.remoteSnapshot != j.ref ||
		j.node.plen != j.tokens || t.snapshotScope(j.namespace) != j.scope ||
		t.snapshotDigest(j.namespace, j.node) != j.digest {
		if !j.staleCounted {
			j.state.stale++
			j.staleCounted = true
		}
		return ErrRemoteTransferStale
	}
	return nil
}

// discard retires only the independently acquired destination owner. The
// source tree owner is never transferred to or freed by the I/O callback.
func (j *remoteTransferJob) discard(snap *model.PrefixSnapshot) {
	if snap != nil {
		_ = j.execute(func() error { snap.Close(); return nil })
	}
}

func (j *remoteTransferJob) finish() {
	if j == nil || j.finished {
		return
	}
	j.finished = true
	j.tree.Done(j.node)
	if j.state.active == j {
		j.state.active = nil
	}
	j.state.completed++
}

// LookupSnapshotTieredContextLocked is the explicit split-phase counterpart of
// LookupSnapshotTieredContext. The caller MUST already hold lock, and every
// access to this tree must use that SAME external lock. The method returns with
// it held, including errors and recovered store panics. It releases it only for
// remote transport and independently owned restore/retirement, never for local
// or SSD work. The returned boundary still requires exactly one Done under lock.
// Passing an unlocked/wrong mutex violates this API; use the ordinary method
// when no shared external mutex is owned.
func (t *Tree) LookupSnapshotTieredContextLocked(ctx context.Context, tokens []int, lock sync.Locker) (*Node, *model.PrefixSnapshot, int, SnapshotTier, error) {
	if lock == nil {
		return nil, nil, 0, SnapshotTierMiss, ErrRemoteTransferLock
	}
	return t.lookupSnapshotTieredContext(ctx, tokens, &remoteTransferLock{locker: lock})
}

// StageSnapshotToRemoteLocked has the same held-lock contract as the locked
// lookup. A stale successful Put may leave an unreferenced object in the
// external store: SnapshotStore has no Delete. It NEVER publishes a stale
// local reference or authorizes the caller to discard the source on refusal.
func (t *Tree) StageSnapshotToRemoteLocked(ctx context.Context, digest string, lock sync.Locker) SnapshotTransfer {
	if lock == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Reason: ErrRemoteTransferLock.Error()}
	}
	return t.stageSnapshotToRemote(ctx, digest, &remoteTransferLock{locker: lock})
}

// RestoreSnapshotFromRemoteLocked has the same held-lock contract. A late
// stale/canceled destination is closed before the transaction is released.
func (t *Tree) RestoreSnapshotFromRemoteLocked(ctx context.Context, digest string, lock sync.Locker) SnapshotTransfer {
	if lock == nil {
		return SnapshotTransfer{Outcome: SnapshotTransferFault, Digest: digest, Reason: ErrRemoteTransferLock.Error()}
	}
	return t.restoreSnapshotFromRemoteIntoTree(ctx, digest, &remoteTransferLock{locker: lock})
}
