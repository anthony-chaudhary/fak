package ctxmmu

import "testing"

// TestMMUCloseReleasesGlobalRegistries pins that (*MMU).Close removes the
// instance from activeMMUs and all three lazy per-MMU global tables, and is
// idempotent and nil-safe.
func TestMMUCloseReleasesGlobalRegistries(t *testing.T) {
	activeMMUsMu.RLock()
	baseActive := len(activeMMUs)
	activeMMUsMu.RUnlock()
	mmuCOWMu.Lock()
	baseCOW := len(mmuCOWPageTables)
	mmuCOWMu.Unlock()
	mmuForkMu.Lock()
	baseFork := len(mmuForkManagers)
	mmuForkMu.Unlock()
	mmuKVPoolMu.Lock()
	baseKV := len(mmuKVPools)
	mmuKVPoolMu.Unlock()

	var nilMMU *MMU
	nilMMU.Close()

	const n = 64
	mmus := make([]*MMU, 0, n)
	for i := 0; i < n; i++ {
		m := NewWithLimit(4)
		_ = m.COWPageTable()
		_ = m.ForkManager()
		_ = m.KVPool()
		mmus = append(mmus, m)
	}

	for _, m := range mmus {
		m.Close()
	}
	// Idempotent second close must not panic or corrupt slices.
	mmus[0].Close()

	activeMMUsMu.RLock()
	gotActive := len(activeMMUs)
	activeMMUsMu.RUnlock()
	if gotActive != baseActive {
		t.Fatalf("activeMMUs len = %d, want %d after Close (LEAK)", gotActive, baseActive)
	}

	mmuCOWMu.Lock()
	gotCOW := len(mmuCOWPageTables)
	mmuCOWMu.Unlock()
	if gotCOW != baseCOW {
		t.Fatalf("mmuCOWPageTables len = %d, want %d after Close (LEAK)", gotCOW, baseCOW)
	}

	mmuForkMu.Lock()
	gotFork := len(mmuForkManagers)
	mmuForkMu.Unlock()
	if gotFork != baseFork {
		t.Fatalf("mmuForkManagers len = %d, want %d after Close (LEAK)", gotFork, baseFork)
	}

	mmuKVPoolMu.Lock()
	gotKV := len(mmuKVPools)
	mmuKVPoolMu.Unlock()
	if gotKV != baseKV {
		t.Fatalf("mmuKVPools len = %d, want %d after Close (LEAK)", gotKV, baseKV)
	}
}
