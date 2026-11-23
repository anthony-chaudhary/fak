package compute

import (
	"sync"
	"testing"
)

// TestGraphEvictGuard pins the llama.cpp CUDA graph-pool pending-use guard
// adapted in graph_evict_guard.go (ggml/src/ggml-cuda/common.cuh
// @00f113eceb2758158582d56f98bf476b2fe3ed32, MIT): a captured graph queued for
// replay is non-evictable until its replay completes, so an eviction pass under
// memory pressure cannot free a graph that is still in flight.
func TestGraphEvictGuard(t *testing.T) {
	t.Run("queued graph is not evictable", func(t *testing.T) {
		g := NewGraphEvictGuard()
		if !g.Evictable("a") {
			t.Fatal("a fresh graph must be evictable")
		}
		g.QueueReplay("a")
		if g.Evictable("a") {
			t.Fatal("a graph queued for replay must NOT be evictable (in-flight graph freed)")
		}
		if g.CanEvict("a") {
			t.Fatal("CanEvict must agree with Evictable: queued graph is not evictable")
		}
		if g.InFlight("a") != 1 || g.Pending() != 1 {
			t.Fatalf("InFlight=%d Pending=%d, want 1/1", g.InFlight("a"), g.Pending())
		}
	})

	t.Run("evictable only after replay completes", func(t *testing.T) {
		g := NewGraphEvictGuard()
		g.QueueReplay("a")
		if g.Evictable("a") {
			t.Fatal("queued graph must not be evictable before completion")
		}
		g.CompleteReplay("a")
		if !g.Evictable("a") {
			t.Fatal("graph must return to the eviction pool after its replay completes")
		}
		if g.InFlight("a") != 0 || g.Pending() != 0 {
			t.Fatalf("InFlight=%d Pending=%d, want 0/0 after completion", g.InFlight("a"), g.Pending())
		}
	})

	t.Run("concurrent replays keep it non-evictable until the last finishes", func(t *testing.T) {
		g := NewGraphEvictGuard()
		g.QueueReplay("a")
		g.QueueReplay("a")
		g.QueueReplay("a")
		if g.InFlight("a") != 3 {
			t.Fatalf("InFlight=%d, want 3", g.InFlight("a"))
		}
		g.CompleteReplay("a")
		if g.Evictable("a") {
			t.Fatal("graph must stay non-evictable while a replay is still outstanding")
		}
		g.CompleteReplay("a")
		if g.Evictable("a") {
			t.Fatal("graph must stay non-evictable until the LAST replay completes")
		}
		g.CompleteReplay("a")
		if !g.Evictable("a") {
			t.Fatal("graph must be evictable once the last replay completes")
		}
	})

	t.Run("unrelated graph stays evictable throughout", func(t *testing.T) {
		g := NewGraphEvictGuard()
		g.QueueReplay("a")
		if !g.Evictable("b") {
			t.Fatal("an unrelated graph must stay evictable while a is in flight")
		}
		g.CompleteReplay("a")
		if !g.Evictable("b") {
			t.Fatal("an unrelated graph must stay evictable after a completes")
		}
	})

	t.Run("stray completion cannot free an in-flight graph", func(t *testing.T) {
		g := NewGraphEvictGuard()
		g.CompleteReplay("a") // no outstanding replay: must be a no-op
		if !g.Evictable("a") {
			t.Fatal("a stray completion must not make a never-queued graph non-evictable")
		}
		g.QueueReplay("a")
		g.CompleteReplay("a")
		g.CompleteReplay("a") // extra completion must not underflow the count
		if !g.Evictable("a") {
			t.Fatal("graph must be evictable after its one replay completes")
		}
		if g.InFlight("a") != 0 {
			t.Fatalf("InFlight=%d, want 0 (no underflow)", g.InFlight("a"))
		}
	})

	t.Run("eviction pass skips queued entries and reclaims after completion", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		for _, k := range []string{"a", "b", "c"} {
			if out, _ := p.Run(k, 1, func() error { return nil }); out != GraphRunCaptured {
				t.Fatalf("insert %s = %s, want captured", k, out)
			}
		}
		g := NewGraphEvictGuard()
		// a is the LRU victim, but it is queued for replay: the pass must skip it
		// and pick the next evictable entry (b).
		g.QueueReplay("a")
		victim, ok := g.EvictableVictim(p)
		if !ok || victim != "b" {
			t.Fatalf("victim = (%q, %v), want (b, true): the queued LRU entry must be skipped", victim, ok)
		}
		// With every entry in flight there is no valid victim at all.
		g.QueueReplay("b")
		g.QueueReplay("c")
		if victim, ok := g.EvictableVictim(p); ok {
			t.Fatalf("victim = (%q, true), want no victim while every entry is in flight", victim)
		}
		// Once a's replay completes it returns to the pool and is the LRU victim.
		g.CompleteReplay("a")
		if victim, ok := g.EvictableVictim(p); !ok || victim != "a" {
			t.Fatalf("victim = (%q, %v), want (a, true) after a's replay completes", victim, ok)
		}
	})

	t.Run("victim selection is safe beside a running policy", func(t *testing.T) {
		eng := newFakeGraphCacheEngine()
		p, err := NewGraphCachePolicy(eng, GraphCachePolicyOptions{On: true, Cap: 4, MinKeep: 1})
		if err != nil {
			t.Fatalf("NewGraphCachePolicy: %v", err)
		}
		g := NewGraphEvictGuard()
		g.QueueReplay("a")
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				_, _ = p.Run([]string{"a", "b", "c"}[i%3], 1, func() error { return nil })
			}
		}()
		for i := 0; i < 200; i++ {
			if victim, ok := g.EvictableVictim(p); ok && victim == "a" {
				t.Error("a queued graph was offered as a victim while the policy ran")
				break
			}
		}
		wg.Wait()
	})

	t.Run("concurrent queue and complete are race-free", func(t *testing.T) {
		g := NewGraphEvictGuard()
		const workers = 8
		var wg sync.WaitGroup
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				g.QueueReplay("a")
				_ = g.Evictable("a")
				g.CompleteReplay("a")
			}()
		}
		wg.Wait()
		if !g.Evictable("a") || g.Pending() != 0 {
			t.Fatalf("after balanced concurrent replays: Evictable=%v Pending=%d, want true/0", g.Evictable("a"), g.Pending())
		}
	})
}
