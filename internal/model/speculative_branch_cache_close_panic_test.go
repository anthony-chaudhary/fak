package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type branchCacheClosePanicBackend struct {
	compute.Backend
	owner   sync.Mutex
	fault   error
	flushes int
}

func (b *branchCacheClosePanicBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *branchCacheClosePanicBackend) BeginBatch()                        {}
func (b *branchCacheClosePanicBackend) FlushBatch() {
	b.flushes++
	panic(b.fault)
}

// Retirement must precede fallible teardown: sync.Once does not retry a panic.
// fak-test:runtime fast est=1ms lane=default
func TestSessionCloseDetachesBranchCacheBeforeCleanupPanic(t *testing.T) {
	fault := errors.New("branch-cache close cleanup")
	be := &branchCacheClosePanicBackend{Backend: compute.Default(), fault: fault}
	// No model pin or device allocations: this isolates the registry lifetime.
	s := &Session{Backend: be}
	t.Cleanup(func() { DetachBranchCache(s) })
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{MaxEntries: 1})
	NewSpeculativeEngine(s, nil, SpeculativeEngineConfig{BranchCache: cache})
	if GetSessionBranchCache(s) != cache {
		t.Fatal("engine did not attach the caller-owned cache")
	}
	var got any
	func() { defer func() { got = recover() }(); s.Close() }()
	if got != fault || be.flushes != 1 {
		t.Fatalf("cleanup panic=%v, flushes=%d; want original panic and one flush", got, be.flushes)
	}
	if GetSessionBranchCache(s) != nil {
		t.Error("interrupted close retained its branch-cache registration")
	}
	if !be.owner.TryLock() {
		t.Fatal("interrupted close retained backend execution ownership")
	}
	be.owner.Unlock()
	if s.backendExecutionScope != nil {
		t.Fatal("interrupted close retained the session execution scope")
	}
	s.Close()
	if be.flushes != 1 || GetSessionBranchCache(s) != nil {
		t.Error("repeated interrupted close was not idempotent")
	}
}
