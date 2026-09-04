package session

import (
	"fmt"
	"sync"
	"testing"
)

// TestTableTrimEvictsSecondaryMaps verifies that when traces are evicted from the Table LRU,
// secondary maps (relayArmed, resumeWaiters, terminateSignals) retain zero dangling pointers
// across 100,000 synthetic trace evictions.
func TestTableTrimEvictsSecondaryMaps(t *testing.T) {
	const limit = 50
	const totalTraces = 100000

	tbl := NewTableWithLimit(limit)

	for i := 0; i < totalTraces; i++ {
		trace := fmt.Sprintf("trace-%d", i)
		tbl.SetPriority(trace, 1)

		tbl.mu.Lock()
		if tbl.relayArmed == nil {
			tbl.relayArmed = map[string]bool{}
		}
		tbl.relayArmed[trace] = true

		if tbl.resumeWaiters == nil {
			tbl.resumeWaiters = map[string][]chan struct{}{}
		}
		tbl.resumeWaiters[trace] = []chan struct{}{make(chan struct{})}

		if tbl.terminateSignals == nil {
			tbl.terminateSignals = map[string]chan struct{}{}
		}
		tbl.terminateSignals[trace] = make(chan struct{})
		tbl.mu.Unlock()
	}

	tbl.mu.RLock()
	defer tbl.mu.RUnlock()

	// 1. Verify retained table state is bounded by limit
	if len(tbl.state) > limit {
		t.Fatalf("len(tbl.state) = %d, want <= %d", len(tbl.state), limit)
	}
	if len(tbl.index) > limit {
		t.Fatalf("len(tbl.index) = %d, want <= %d", len(tbl.index), limit)
	}

	// 2. Verify auxiliary maps are bounded by limit
	if len(tbl.relayArmed) > limit {
		t.Fatalf("len(tbl.relayArmed) = %d, want <= %d (leaked auxiliary entries)", len(tbl.relayArmed), limit)
	}
	if len(tbl.resumeWaiters) > limit {
		t.Fatalf("len(tbl.resumeWaiters) = %d, want <= %d (leaked auxiliary entries)", len(tbl.resumeWaiters), limit)
	}
	if len(tbl.terminateSignals) > limit {
		t.Fatalf("len(tbl.terminateSignals) = %d, want <= %d (leaked auxiliary entries)", len(tbl.terminateSignals), limit)
	}

	// 3. Verify zero dangling pointers: every entry in secondary maps must correspond to an active trace in tbl.state
	for k := range tbl.relayArmed {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("relayArmed contains dangling trace %q not in tbl.state", k)
		}
	}
	for k := range tbl.resumeWaiters {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("resumeWaiters contains dangling trace %q not in tbl.state", k)
		}
	}
	for k := range tbl.terminateSignals {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("terminateSignals contains dangling trace %q not in tbl.state", k)
		}
	}

	// 4. Specifically verify that evicted traces are absent from all secondary maps
	for i := 0; i < totalTraces-limit; i++ {
		evicted := fmt.Sprintf("trace-%d", i)
		if tbl.relayArmed[evicted] {
			t.Fatalf("relayArmed retains evicted trace %q", evicted)
		}
		if _, ok := tbl.resumeWaiters[evicted]; ok {
			t.Fatalf("resumeWaiters retains evicted trace %q", evicted)
		}
		if _, ok := tbl.terminateSignals[evicted]; ok {
			t.Fatalf("terminateSignals retains evicted trace %q", evicted)
		}
	}
}

// TestTableTrimEvictsSecondaryMapsConcurrent verifies race safety and bounds under concurrent eviction.
func TestTableTrimEvictsSecondaryMapsConcurrent(t *testing.T) {
	const limit = 20
	const goroutines = 8
	const perGoroutine = 5000

	tbl := NewTableWithLimit(limit)
	var wg sync.WaitGroup

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				trace := fmt.Sprintf("trace-g%d-%d", gid, i)

				tbl.mu.Lock()
				st := tbl.getLocked(trace)
				st.Priority = 1

				if tbl.relayArmed == nil {
					tbl.relayArmed = map[string]bool{}
				}
				tbl.relayArmed[trace] = true

				if tbl.resumeWaiters == nil {
					tbl.resumeWaiters = map[string][]chan struct{}{}
				}
				tbl.resumeWaiters[trace] = []chan struct{}{make(chan struct{})}

				if tbl.terminateSignals == nil {
					tbl.terminateSignals = map[string]chan struct{}{}
				}
				tbl.terminateSignals[trace] = make(chan struct{})

				tbl.putLocked(st)
				tbl.mu.Unlock()
			}
		}(g)
	}
	wg.Wait()

	tbl.mu.RLock()
	defer tbl.mu.RUnlock()

	if len(tbl.state) > limit {
		t.Fatalf("len(tbl.state) = %d, want <= %d", len(tbl.state), limit)
	}
	if len(tbl.relayArmed) > limit {
		t.Fatalf("len(tbl.relayArmed) = %d, want <= %d", len(tbl.relayArmed), limit)
	}
	if len(tbl.resumeWaiters) > limit {
		t.Fatalf("len(tbl.resumeWaiters) = %d, want <= %d", len(tbl.resumeWaiters), limit)
	}
	if len(tbl.terminateSignals) > limit {
		t.Fatalf("len(tbl.terminateSignals) = %d, want <= %d", len(tbl.terminateSignals), limit)
	}

	for k := range tbl.relayArmed {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("relayArmed contains dangling trace %q not in tbl.state", k)
		}
	}
	for k := range tbl.resumeWaiters {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("resumeWaiters contains dangling trace %q not in tbl.state", k)
		}
	}
	for k := range tbl.terminateSignals {
		if _, ok := tbl.state[k]; !ok {
			t.Fatalf("terminateSignals contains dangling trace %q not in tbl.state", k)
		}
	}
}
