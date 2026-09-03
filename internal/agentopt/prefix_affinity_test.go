package agentopt

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func TestPrefixAffinityRouting(t *testing.T) {
	ctx := context.Background()

	t.Run("biases routing toward warm cache instance", func(t *testing.T) {
		router := NewPrefixAffinityRouter()

		inst1 := NewServingInstance("engine-1", 4)
		inst2 := NewServingInstance("engine-2", 4)
		_ = router.RegisterInstance(inst1)
		_ = router.RegisterInstance(inst2)

		systemPrompt := "You are an autonomous engineering agent for fak."
		repoContext := "Repo: fak; Architecture: Agent Kernel; Packages: agentopt, gateway, session."

		// Warm engine-1 with the shared system prompt and repo context.
		warmReq := WorkerRequest{
			ID:             "warmup-1",
			TaskKey:        "init-session",
			SystemPrompt:   systemPrompt,
			WorkspaceScope: repoContext,
		}
		if err := router.WarmInstance("engine-1", warmReq); err != nil {
			t.Fatalf("failed to warm engine-1: %v", err)
		}

		// Send request from Worker A sharing the same system prompt and repo context.
		workerReq := WorkerRequest{
			ID:             "req-worker-A",
			WorkerID:       "worker-A",
			TaskKey:        "session-A",
			SystemPrompt:   systemPrompt,
			WorkspaceScope: repoContext,
			Prompt:         "Implement prefix affinity routing feature.",
		}

		routeRes, err := router.Route(ctx, workerReq)
		if err != nil {
			t.Fatalf("route failed: %v", err)
		}

		if routeRes.InstanceID != "engine-1" {
			t.Fatalf("expected routing to warm engine-1, got %q", routeRes.InstanceID)
		}
		if !routeRes.WarmHit {
			t.Fatalf("expected cache hit on warm instance, got hit=false")
		}
		if routeRes.MatchedBlocks <= 0 {
			t.Fatalf("expected matched blocks > 0, got %d", routeRes.MatchedBlocks)
		}
		if routeRes.AffinityScore <= 0 {
			t.Fatalf("expected positive affinity score, got %f", routeRes.AffinityScore)
		}
	})

	t.Run("multiple concurrent workers sharing repo context route to same instance", func(t *testing.T) {
		router := NewPrefixAffinityRouter()

		inst1 := NewServingInstance("engine-1", 8)
		inst2 := NewServingInstance("engine-2", 8)
		_ = router.RegisterInstance(inst1)
		_ = router.RegisterInstance(inst2)

		systemPrompt := "You are an autonomous coding subagent."
		repoContext := "Package: internal/agentopt; Files: prefix_affinity.go, early_exit.go."

		// Warm engine-2
		_ = router.WarmInstance("engine-2", WorkerRequest{
			SystemPrompt:   systemPrompt,
			WorkspaceScope: repoContext,
		})

		// 5 different workers with unique session IDs and distinct prompts
		workers := []string{"worker-alpha", "worker-beta", "worker-gamma", "worker-delta", "worker-epsilon"}
		for _, w := range workers {
			req := WorkerRequest{
				ID:             "req-" + w,
				WorkerID:       w,
				TaskKey:        "session-" + w,
				SystemPrompt:   systemPrompt,
				WorkspaceScope: repoContext,
				Prompt:         fmt.Sprintf("Task for %s: optimize serving", w),
			}
			routeRes, err := router.Route(ctx, req)
			if err != nil {
				t.Fatalf("failed to route worker %s: %v", w, err)
			}
			if routeRes.InstanceID != "engine-2" {
				t.Fatalf("worker %s expected route to engine-2, got %s", w, routeRes.InstanceID)
			}
			if !routeRes.WarmHit {
				t.Fatalf("worker %s expected cache hit", w)
			}
		}

		stats := router.Stats()
		if stats.TotalRouted != 5 {
			t.Fatalf("expected 5 routed requests, got %d", stats.TotalRouted)
		}
		if stats.WarmHits != 5 {
			t.Fatalf("expected 5 cache hits, got %d", stats.WarmHits)
		}
		if stats.WarmHitRatio != 1.0 {
			t.Fatalf("expected cache hit ratio 1.0, got %f", stats.WarmHitRatio)
		}
	})

	t.Run("spills over to cold instance when warm instance reaches capacity", func(t *testing.T) {
		router := NewPrefixAffinityRouter(RouterConfig{
			PrefixWeight:      1.0,
			LoadWeight:        0.5,
			SaturationPenalty: 2.0,
			DefaultCapacity:   4,
		})

		inst1 := NewServingInstance("engine-warm", 4)
		inst2 := NewServingInstance("engine-cold", 4)
		_ = router.RegisterInstance(inst1)
		_ = router.RegisterInstance(inst2)

		systemPrompt := "Standard System Prompt"
		repoContext := "Standard Repo Context"

		_ = router.WarmInstance("engine-warm", WorkerRequest{
			SystemPrompt:   systemPrompt,
			WorkspaceScope: repoContext,
		})

		// Saturate engine-warm with 4 active requests (100% capacity)
		inst1.SetActiveRequests(4)
		inst2.SetActiveRequests(0)

		req := WorkerRequest{
			ID:             "overflow-req",
			SystemPrompt:   systemPrompt,
			WorkspaceScope: repoContext,
			Prompt:         "Work on new issue",
		}

		routeRes, err := router.Route(ctx, req)
		if err != nil {
			t.Fatalf("routing failed: %v", err)
		}

		// Even though engine-warm has cached prefix, it is saturated, so request spills over to engine-cold.
		if routeRes.InstanceID != "engine-cold" {
			t.Fatalf("expected spillover to engine-cold due to saturation, got %s", routeRes.InstanceID)
		}
	})

	t.Run("differentiates prefix affinity across divergent contexts", func(t *testing.T) {
		router := NewPrefixAffinityRouter()

		instA := NewServingInstance("engine-repo-A", 4)
		instB := NewServingInstance("engine-repo-B", 4)
		_ = router.RegisterInstance(instA)
		_ = router.RegisterInstance(instB)

		sys := "Common System Prompt"
		repoA := "Repository A: web frontend codebase"
		repoB := "Repository B: kernel runtime codebase"

		_ = router.WarmInstance("engine-repo-A", WorkerRequest{SystemPrompt: sys, WorkspaceScope: repoA})
		_ = router.WarmInstance("engine-repo-B", WorkerRequest{SystemPrompt: sys, WorkspaceScope: repoB})

		// Request for repo A
		reqA := WorkerRequest{
			ID:             "req-A",
			SystemPrompt:   sys,
			WorkspaceScope: repoA,
			Prompt:         "Fix CSS button styling",
		}
		decA, err := router.Route(ctx, reqA)
		if err != nil {
			t.Fatalf("route A failed: %v", err)
		}
		if decA.InstanceID != "engine-repo-A" {
			t.Fatalf("expected reqA to route to engine-repo-A, got %s", decA.InstanceID)
		}

		// Request for repo B
		reqB := WorkerRequest{
			ID:             "req-B",
			SystemPrompt:   sys,
			WorkspaceScope: repoB,
			Prompt:         "Fix memory allocation lock",
		}
		decB, err := router.Route(ctx, reqB)
		if err != nil {
			t.Fatalf("route B failed: %v", err)
		}
		if decB.InstanceID != "engine-repo-B" {
			t.Fatalf("expected reqB to route to engine-repo-B, got %s", decB.InstanceID)
		}
	})

	t.Run("route and acquire release lifecycle", func(t *testing.T) {
		router := NewPrefixAffinityRouter()
		inst := NewServingInstance("engine-live", 2)
		_ = router.RegisterInstance(inst)

		req := WorkerRequest{
			ID:             "lifecycle-req",
			SystemPrompt:   "System",
			WorkspaceScope: "Context",
			Prompt:         "Query",
		}

		if inst.GetActiveRequests() != 0 {
			t.Fatalf("expected 0 active requests initially, got %d", inst.GetActiveRequests())
		}

		routeRes, release, err := router.RouteAndAcquire(ctx, req)
		if err != nil {
			t.Fatalf("RouteAndAcquire failed: %v", err)
		}
		if routeRes.InstanceID != "engine-live" {
			t.Fatalf("expected engine-live, got %s", routeRes.InstanceID)
		}
		if inst.GetActiveRequests() != 1 {
			t.Fatalf("expected active requests 1 after acquire, got %d", inst.GetActiveRequests())
		}

		// Calling release decrements active count
		release()
		if inst.GetActiveRequests() != 0 {
			t.Fatalf("expected active requests 0 after release, got %d", inst.GetActiveRequests())
		}

		// Multiple release calls are idempotent
		release()
		if inst.GetActiveRequests() != 0 {
			t.Fatalf("expected active requests 0 after idempotent release, got %d", inst.GetActiveRequests())
		}
	})

	t.Run("concurrent route and acquire stress test", func(t *testing.T) {
		router := NewPrefixAffinityRouter()
		for i := 1; i <= 3; i++ {
			_ = router.RegisterInstance(NewServingInstance(fmt.Sprintf("inst-%d", i), 10))
		}

		var wg sync.WaitGroup
		concurrency := 30

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				r := WorkerRequest{
					ID:             fmt.Sprintf("conc-%d", idx),
					SystemPrompt:   "Shared system prompt for all workers",
					WorkspaceScope: fmt.Sprintf("Repo Context %d", idx%3),
					Prompt:         "Inspect code",
				}
				dec, rel, err := router.RouteAndAcquire(ctx, r)
				if err != nil {
					t.Errorf("concurrent route error: %v", err)
					return
				}
				if dec.InstanceID == "" {
					t.Errorf("empty instance ID")
				}
				rel()
			}(i)
		}
		wg.Wait()

		stats := router.Stats()
		if stats.TotalRouted != int64(concurrency) {
			t.Fatalf("expected total routed %d, got %d", concurrency, stats.TotalRouted)
		}
	})
}

func TestPrefixTrie(t *testing.T) {
	trie := NewPrefixTrie()
	if trie.NodeCount() != 0 {
		t.Fatalf("expected initial node count 0, got %d", trie.NodeCount())
	}

	blocks := []string{"b1", "b2", "b3"}
	inserted := trie.Insert(blocks)
	if inserted != 3 {
		t.Fatalf("expected 3 inserted, got %d", inserted)
	}
	if trie.NodeCount() != 3 {
		t.Fatalf("expected 3 nodes, got %d", trie.NodeCount())
	}

	// Exact match
	if l := trie.MatchLength(blocks); l != 3 {
		t.Fatalf("expected match length 3, got %d", l)
	}

	// Partial match
	if l := trie.MatchLength([]string{"b1", "b2", "different"}); l != 2 {
		t.Fatalf("expected match length 2, got %d", l)
	}

	// No match
	if l := trie.MatchLength([]string{"other"}); l != 0 {
		t.Fatalf("expected match length 0, got %d", l)
	}

	// Re-inserting existing path
	reinserted := trie.Insert([]string{"b1", "b2"})
	if reinserted != 0 {
		t.Fatalf("expected 0 new nodes on re-insert, got %d", reinserted)
	}

	trie.Clear()
	if trie.NodeCount() != 0 {
		t.Fatalf("expected 0 nodes after clear, got %d", trie.NodeCount())
	}
}

func TestExtractBlocks(t *testing.T) {
	router := NewPrefixAffinityRouter()

	// Explicit tokens take precedence
	reqExplicit := WorkerRequest{
		Tokens: []string{"custom-1", "custom-2"},
	}
	blocks := router.ExtractBlocks(reqExplicit)
	if len(blocks) != 2 || blocks[0] != "custom-1" || blocks[1] != "custom-2" {
		t.Fatalf("expected explicit tokens, got %v", blocks)
	}

	// Automatic chunking of system and repo prompts
	reqText := WorkerRequest{
		SystemPrompt:   "System prompt instructions here",
		WorkspaceScope: "Repository facts and definitions here",
		Prompt:         "User question",
	}
	blocksText := router.ExtractBlocks(reqText)
	if len(blocksText) != 3 {
		t.Fatalf("expected 3 blocks (sys, repo, prompt), got %d: %v", len(blocksText), blocksText)
	}
}

func TestRouterInstanceManagement(t *testing.T) {
	router := NewPrefixAffinityRouter()

	// Register errors on nil or empty
	if err := router.RegisterInstance(nil); err == nil {
		t.Fatal("expected error on registering nil instance")
	}
	if err := router.RegisterInstance(&ServingInstance{ID: ""}); err == nil {
		t.Fatal("expected error on registering empty instance ID")
	}

	instA := NewServingInstance("inst-a", 5)
	instB := NewServingInstance("inst-b", 5)
	_ = router.RegisterInstance(instA)
	_ = router.RegisterInstance(instB)

	instances := router.ListInstances()
	if len(instances) != 2 {
		t.Fatalf("expected 2 instances, got %d", len(instances))
	}
	if instances[0].ID != "inst-a" || instances[1].ID != "inst-b" {
		t.Fatalf("expected ordered instances [inst-a, inst-b], got [%s, %s]", instances[0].ID, instances[1].ID)
	}

	if _, exists := router.GetInstance("inst-a"); !exists {
		t.Fatal("expected inst-a to exist")
	}

	if err := router.UnregisterInstance("inst-a"); err != nil {
		t.Fatalf("failed to unregister inst-a: %v", err)
	}
	if _, exists := router.GetInstance("inst-a"); exists {
		t.Fatal("expected inst-a to be unregistered")
	}
	if err := router.UnregisterInstance("non-existent"); err == nil {
		t.Fatal("expected error unregistering non-existent instance")
	}
}
