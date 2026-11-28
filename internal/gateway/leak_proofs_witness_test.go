package gateway

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// fak-test:runtime fast est=40ms lane=default
func TestProofLeakL2_A2APushSecondaryConservationStateMachine(t *testing.T) {
	store := getA2AStore()
	store.mu.Lock()
	savedTasks := store.tasks
	store.tasks = make(map[string]*a2aTask)
	store.mu.Unlock()

	a2aPushMu.Lock()
	savedPush := a2aPushSingleton
	a2aPushSingleton = newA2APushStore()
	ps := a2aPushSingleton
	a2aPushMu.Unlock()

	t.Cleanup(func() {
		store.mu.Lock()
		store.tasks = savedTasks
		store.mu.Unlock()
		a2aPushMu.Lock()
		a2aPushSingleton = savedPush
		a2aPushMu.Unlock()
	})

	rng := rand.New(rand.NewSource(2026112803))
	baseTime := time.Date(2026, 11, 28, 0, 0, 0, 0, time.UTC)
	seq := 0

	nextTS := func() time.Time {
		seq++
		return baseTime.Add(time.Duration(seq) * time.Millisecond)
	}

	var activeIDs []string
	// Pre-seed near capacity so the PRNG trace exercises eviction immediately.
	const preseed = a2aMaxTasks - 16
	for i := 0; i < preseed; i++ {
		id := fmt.Sprintf("pre-%05d", i)
		ts := nextTS()
		store.mu.Lock()
		store.insertLocked(id, &a2aTask{
			TaskID:    id,
			State:     "completed",
			CreatedAt: ts,
			UpdatedAt: ts,
		})
		store.mu.Unlock()
		activeIDs = append(activeIDs, id)
		if i%2 == 0 {
			ps.setConfig(id, a2aPushConfig{
				URL:       "https://example.com/hook",
				CallerID:  "caller-pre",
				CreatedAt: ts,
			})
		}
	}

	// Keep activeIDs synced with store.tasks after evictions.
	syncActiveIDs := func() {
		store.mu.RLock()
		activeIDs = activeIDs[:0]
		for id := range store.tasks {
			activeIDs = append(activeIDs, id)
		}
		store.mu.RUnlock()
	}

	const steps = 1200
	newIdx := 0
	for step := 0; step < steps; step++ {
		op := rng.Intn(4)
		switch op {
		case 0: // Insert new task (+ optional push config)
			id := fmt.Sprintf("task-%05d", newIdx)
			newIdx++
			ts := nextTS()
			state := "completed"
			if rng.Intn(4) == 0 {
				state = "working"
			}
			store.mu.Lock()
			store.insertLocked(id, &a2aTask{
				TaskID:    id,
				State:     state,
				CreatedAt: ts,
				UpdatedAt: ts,
			})
			store.mu.Unlock()
			if rng.Intn(2) == 0 {
				ps.setConfig(id, a2aPushConfig{
					URL:       "https://example.com/hook",
					CallerID:  "caller-new",
					CreatedAt: ts,
				})
			}
			syncActiveIDs()

		case 1: // Overwrite existing task
			if len(activeIDs) > 0 {
				id := activeIDs[rng.Intn(len(activeIDs))]
				ts := nextTS()
				store.mu.Lock()
				store.insertLocked(id, &a2aTask{
					TaskID:    id,
					State:     "completed",
					CreatedAt: ts,
					UpdatedAt: ts,
				})
				store.mu.Unlock()
			}

		case 2: // Register push config for an existing task
			if len(activeIDs) > 0 {
				id := activeIDs[rng.Intn(len(activeIDs))]
				ts := nextTS()
				ps.setConfig(id, a2aPushConfig{
					URL:       "https://example.com/hook-reg",
					CallerID:  "caller-reg",
					CreatedAt: ts,
				})
			}

		case 3: // Overwrite existing push config
			ps.mu.RLock()
			var existingCfgID string
			for id := range ps.configs {
				existingCfgID = id
				break
			}
			ps.mu.RUnlock()
			if existingCfgID != "" {
				ts := nextTS()
				ps.setConfig(existingCfgID, a2aPushConfig{
					URL:       "https://example.com/hook-upd",
					CallerID:  "caller-upd",
					CreatedAt: ts,
				})
			}
		}

		// Assert conservation and capacity invariants after EVERY transition.
		store.mu.RLock()
		taskCount := len(store.tasks)
		ps.mu.RLock()
		cfgCount := len(ps.configs)
		for cfgID := range ps.configs {
			if _, ok := store.tasks[cfgID]; !ok {
				ps.mu.RUnlock()
				store.mu.RUnlock()
				t.Fatalf("step %d (op=%d): orphaned push config for evicted/missing task %q", step, op, cfgID)
			}
		}
		ps.mu.RUnlock()
		store.mu.RUnlock()

		if taskCount > a2aMaxTasks {
			t.Fatalf("step %d (op=%d): len(tasks) = %d, want <= %d", step, op, taskCount, a2aMaxTasks)
		}
		if cfgCount > a2aMaxTasks {
			t.Fatalf("step %d (op=%d): len(configs) = %d, want <= %d", step, op, cfgCount, a2aMaxTasks)
		}
	}
}

// fak-test:runtime fast est=40ms lane=default
func TestProofLeakL1_ServerCloseIdempotentStateMachine(t *testing.T) {
	type srvGhost struct {
		srv     *Server
		armed   bool
		control bool
		session bool
		evictor *IdleGatedEvictor
		closed  bool
	}

	var live []*srvGhost
	var closed []*srvGhost

	t.Cleanup(func() {
		for _, g := range live {
			g.srv.Close()
		}
	})

	rng := rand.New(rand.NewSource(2026112804))
	const steps = 600

	for step := 0; step < steps; step++ {
		op := rng.Intn(7)
		switch op {
		case 0: // New Server
			if len(live) < 12 {
				srv := newTestServer(t)
				live = append(live, &srvGhost{srv: srv})
			}
		case 1: // Arm toolproc reuse
			if len(live) > 0 {
				g := live[rng.Intn(len(live))]
				g.srv.SetToolprocReuse(toolproc.ArmedConfig{}, nil)
				g.armed = true
			}
		case 2: // ControlManager
			if len(live) > 0 {
				g := live[rng.Intn(len(live))]
				_ = g.srv.ControlManager()
				g.control = true
			}
		case 3: // Session clientRuntime
			if len(live) > 0 {
				g := live[rng.Intn(len(live))]
				_ = g.srv.clientRuntime()
				g.session = true
			}
		case 4: // IdleGatedEvictor
			if len(live) > 0 {
				g := live[rng.Intn(len(live))]
				if g.evictor == nil {
					ev := NewIdleGatedEvictor(nil, nil)
					g.srv.SetIdleGatedEvictor(ev)
					ev.Start(context.Background())
					g.evictor = ev
				}
			}
		case 5: // Close a live server
			if len(live) > 0 {
				idx := rng.Intn(len(live))
				g := live[idx]
				g.srv.Close()
				g.armed = false
				g.control = false
				g.session = false
				g.closed = true
				live[idx] = live[len(live)-1]
				live = live[:len(live)-1]
				closed = append(closed, g)
			}
		case 6: // Re-Close an already closed server (idempotent)
			if len(closed) > 0 {
				g := closed[rng.Intn(len(closed))]
				g.srv.Close()
			}
		}

		// Assert exact global sync.Map membership and evictor state after EVERY transition.
		for _, g := range live {
			if _, ok := reuseSeams.Load(g.srv); ok != g.armed {
				t.Fatalf("step %d (op=%d): live reuseSeams.Load = %v, want %v", step, op, ok, g.armed)
			}
			if _, ok := serverControlManagers.Load(g.srv); ok != g.control {
				t.Fatalf("step %d (op=%d): live serverControlManagers.Load = %v, want %v", step, op, ok, g.control)
			}
			if _, ok := sessionClientRuntimes.Load(g.srv); ok != g.session {
				t.Fatalf("step %d (op=%d): live sessionClientRuntimes.Load = %v, want %v", step, op, ok, g.session)
			}
			if g.evictor != nil && !g.evictor.IsRunning() {
				t.Fatalf("step %d (op=%d): live evictor stopped prematurely", step, op)
			}
		}
		for _, g := range closed {
			if _, ok := reuseSeams.Load(g.srv); ok {
				t.Fatalf("step %d (op=%d): closed server still in reuseSeams", step, op)
			}
			if _, ok := serverControlManagers.Load(g.srv); ok {
				t.Fatalf("step %d (op=%d): closed server still in serverControlManagers", step, op)
			}
			if _, ok := sessionClientRuntimes.Load(g.srv); ok {
				t.Fatalf("step %d (op=%d): closed server still in sessionClientRuntimes", step, op)
			}
			if g.evictor != nil && g.evictor.IsRunning() {
				t.Fatalf("step %d (op=%d): closed server evictor still running", step, op)
			}
		}
	}
}
