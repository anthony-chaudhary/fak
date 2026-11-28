package ctxmmu

import (
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=15ms lane=default
func TestProofLeakL1_CtxMMURegistryStateMachine(t *testing.T) {
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

	type mmuGhost struct {
		cow  bool
		fork bool
		kv   bool
	}

	live := make(map[*MMU]*mmuGhost)
	var liveList []*MMU
	var closedList []*MMU

	t.Cleanup(func() {
		for _, m := range liveList {
			m.Close()
		}
	})

	rng := rand.New(rand.NewSource(2026112801))
	const steps = 1000

	for step := 0; step < steps; step++ {
		op := rng.Intn(6)
		switch op {
		case 0: // New
			if len(liveList) < 64 {
				m := NewWithLimit(4)
				live[m] = &mmuGhost{}
				liveList = append(liveList, m)
			}
		case 1: // TouchCOW
			if len(liveList) > 0 {
				m := liveList[rng.Intn(len(liveList))]
				_ = m.COWPageTable()
				live[m].cow = true
			}
		case 2: // TouchFork
			if len(liveList) > 0 {
				m := liveList[rng.Intn(len(liveList))]
				_ = m.ForkManager()
				live[m].fork = true
			}
		case 3: // TouchKVPool
			if len(liveList) > 0 {
				m := liveList[rng.Intn(len(liveList))]
				_ = m.KVPool()
				live[m].kv = true
			}
		case 4: // Close
			if len(liveList) > 0 {
				idx := rng.Intn(len(liveList))
				m := liveList[idx]
				m.Close()
				delete(live, m)
				liveList[idx] = liveList[len(liveList)-1]
				liveList = liveList[:len(liveList)-1]
				closedList = append(closedList, m)
			}
		case 5: // Duplicate Close (including nil)
			if len(closedList) > 0 && rng.Intn(4) != 0 {
				m := closedList[rng.Intn(len(closedList))]
				m.Close()
			} else {
				var nilMMU *MMU
				nilMMU.Close()
			}
		}

		var wantCOW, wantFork, wantKV int
		for _, g := range live {
			if g.cow {
				wantCOW++
			}
			if g.fork {
				wantFork++
			}
			if g.kv {
				wantKV++
			}
		}
		wantActive := len(live)

		activeMMUsMu.RLock()
		gotActive := len(activeMMUs)
		activeMMUsMu.RUnlock()
		if gotActive != baseActive+wantActive {
			t.Fatalf("step %d (op=%d): len(activeMMUs) = %d, want %d (base=%d + live=%d)",
				step, op, gotActive, baseActive+wantActive, baseActive, wantActive)
		}

		mmuCOWMu.Lock()
		gotCOW := len(mmuCOWPageTables)
		mmuCOWMu.Unlock()
		if gotCOW != baseCOW+wantCOW {
			t.Fatalf("step %d (op=%d): len(mmuCOWPageTables) = %d, want %d (base=%d + liveWithCOW=%d)",
				step, op, gotCOW, baseCOW+wantCOW, baseCOW, wantCOW)
		}

		mmuForkMu.Lock()
		gotFork := len(mmuForkManagers)
		mmuForkMu.Unlock()
		if gotFork != baseFork+wantFork {
			t.Fatalf("step %d (op=%d): len(mmuForkManagers) = %d, want %d (base=%d + liveWithFork=%d)",
				step, op, gotFork, baseFork+wantFork, baseFork, wantFork)
		}

		mmuKVPoolMu.Lock()
		gotKV := len(mmuKVPools)
		mmuKVPoolMu.Unlock()
		if gotKV != baseKV+wantKV {
			t.Fatalf("step %d (op=%d): len(mmuKVPools) = %d, want %d (base=%d + liveWithKV=%d)",
				step, op, gotKV, baseKV+wantKV, baseKV, wantKV)
		}
	}
}
