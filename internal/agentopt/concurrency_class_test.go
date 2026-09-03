package agentopt

import (
	"fmt"
	"sync"
	"testing"
)

func TestConcurrencyClassArbiter(t *testing.T) {
	t.Run("StrictBudgetEnforcement", func(t *testing.T) {
		arbiter := NewConcurrencyClassArbiter(map[string]int{
			"global":  1,
			"cluster": 2,
		})

		// 1. Acquire single allowed global lease
		res1 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "global",
			LaneName:     "global-trunk",
			TreePatterns: []string{"cmd/fak/**"},
			WorkerID:     "worker-global-1",
		})
		if !res1.Granted || !res1.Acquired || res1.Lease == nil {
			t.Fatalf("expected global lease granted, got %+v", res1)
		}
		if res1.Reason != "" {
			t.Fatalf("expected empty reason on grant, got %s", res1.Reason)
		}

		// 2. Second global lease must be refused due to budget exhausted
		res2 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "global",
			LaneName:     "global-secondary",
			TreePatterns: []string{"internal/agentopt/**"},
			WorkerID:     "worker-global-2",
		})
		if res2.Granted || res2.Acquired {
			t.Fatalf("expected second global lease refused, got %+v", res2)
		}
		if res2.Reason != RefuseBudgetExhausted {
			t.Fatalf("expected %s, got %s", RefuseBudgetExhausted, res2.Reason)
		}

		// 3. Cluster leases up to limit (2)
		resC1 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "cluster-a",
			TreePatterns: []string{"internal/agentopt/**"},
			WorkerID:     "worker-cluster-1",
		})
		if !resC1.Granted {
			t.Fatalf("expected cluster-a granted, got %+v", resC1)
		}

		resC2 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "cluster-b",
			TreePatterns: []string{"internal/model/**"},
			WorkerID:     "worker-cluster-2",
		})
		if !resC2.Granted {
			t.Fatalf("expected cluster-b granted, got %+v", resC2)
		}

		// 4. Third cluster lease must be refused
		resC3 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "cluster-c",
			TreePatterns: []string{"internal/vdso/**"},
			WorkerID:     "worker-cluster-3",
		})
		if resC3.Granted {
			t.Fatalf("expected cluster-c refused by budget, got %+v", resC3)
		}
		if resC3.Reason != RefuseBudgetExhausted {
			t.Fatalf("expected %s, got %s", RefuseBudgetExhausted, resC3.Reason)
		}

		// 5. Release one cluster lease and re-attempt
		if !arbiter.ReleaseLease("worker-cluster-1", "cluster-a") {
			t.Fatalf("failed to release cluster-a")
		}

		resC3Retry := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "cluster-c",
			TreePatterns: []string{"internal/vdso/**"},
			WorkerID:     "worker-cluster-3",
		})
		if !resC3Retry.Granted {
			t.Fatalf("expected cluster-c granted after release, got %+v", resC3Retry)
		}

		// Release global trunk
		if !arbiter.ReleaseLease("worker-global-1", "global-trunk") {
			t.Fatalf("failed to release global-trunk")
		}

		// Now global secondary should be granted
		res2Retry := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "global",
			LaneName:     "global-secondary",
			TreePatterns: []string{"internal/agentopt/**"},
			WorkerID:     "worker-global-2",
		})
		if !res2Retry.Granted {
			t.Fatalf("expected global secondary granted after release, got %+v", res2Retry)
		}
	})

	t.Run("CollisionPrevention", func(t *testing.T) {
		arbiter := NewConcurrencyClassArbiter(map[string]int{
			"cluster": 10,
		})

		// 1. Acquire root path
		res1 := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-parent",
			TreePatterns: []string{"internal/agentopt/**"},
			WorkerID:     "worker-1",
		})
		if !res1.Granted {
			t.Fatalf("expected lane-parent granted, got %+v", res1)
		}

		// 2. Subtree collision
		resSub := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-child",
			TreePatterns: []string{"internal/agentopt/sub/**"},
			WorkerID:     "worker-2",
		})
		if resSub.Granted {
			t.Fatalf("expected lane-child refused by tree collision, got %+v", resSub)
		}
		if resSub.Reason != RefuseTreeCollision {
			t.Fatalf("expected %s, got %s", RefuseTreeCollision, resSub.Reason)
		}

		// 3. Exact file collision
		resFile := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-file",
			TreePatterns: []string{"internal/agentopt/concurrency_class.go"},
			WorkerID:     "worker-3",
		})
		if resFile.Granted {
			t.Fatalf("expected lane-file refused by tree collision, got %+v", resFile)
		}
		if resFile.Reason != RefuseTreeCollision {
			t.Fatalf("expected %s, got %s", RefuseTreeCollision, resFile.Reason)
		}

		// 4. Same lane name collision
		resSameName := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-parent",
			TreePatterns: []string{"cmd/fak/**"},
			WorkerID:     "worker-4",
		})
		if resSameName.Granted {
			t.Fatalf("expected same lane name refused by collision, got %+v", resSameName)
		}
		if resSameName.Reason != RefuseTreeCollision {
			t.Fatalf("expected %s, got %s", RefuseTreeCollision, resSameName.Reason)
		}

		// 5. Universal wildcard collision
		resWildcard := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-all",
			TreePatterns: []string{"**"},
			WorkerID:     "worker-5",
		})
		if resWildcard.Granted {
			t.Fatalf("expected universal wildcard refused by collision, got %+v", resWildcard)
		}
		if resWildcard.Reason != RefuseTreeCollision {
			t.Fatalf("expected %s, got %s", RefuseTreeCollision, resWildcard.Reason)
		}

		// 6. Disjoint path succeeds
		resDisjoint := arbiter.AcquireLease(LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-disjoint",
			TreePatterns: []string{"internal/vdso/**"},
			WorkerID:     "worker-6",
		})
		if !resDisjoint.Granted {
			t.Fatalf("expected lane-disjoint granted, got %+v", resDisjoint)
		}
	})

	t.Run("ReleaseValidationAndListing", func(t *testing.T) {
		arbiter := NewConcurrencyClassArbiter(map[string]int{
			"cluster": 5,
		})

		req := LaneLeaseRequest{
			LaneKind:     "cluster",
			LaneName:     "lane-alpha",
			TreePatterns: []string{"internal/agentopt/**"},
			WorkerID:     "worker-alpha",
		}
		res := arbiter.AcquireLease(req)
		if !res.Granted {
			t.Fatalf("expected lease granted")
		}

		// Wrong worker ID cannot release
		if arbiter.ReleaseLease("worker-imposter", "lane-alpha") {
			t.Fatalf("expected imposter release to return false")
		}

		// Non-existent lane cannot release
		if arbiter.ReleaseLease("worker-alpha", "lane-nonexistent") {
			t.Fatalf("expected nonexistent lane release to return false")
		}

		// List leases verification
		leases := arbiter.ListLeases()
		if len(leases) != 1 {
			t.Fatalf("expected 1 active lease, got %d", len(leases))
		}
		if leases[0].LaneName != "lane-alpha" || leases[0].WorkerID != "worker-alpha" {
			t.Fatalf("unexpected lease data: %+v", leases[0])
		}

		// Mutation of returned slice must not affect arbiter
		leases[0].LaneName = "corrupted"
		leases2 := arbiter.ListLeases()
		if leases2[0].LaneName != "lane-alpha" {
			t.Fatalf("internal lease was modified via returned slice")
		}

		// Successful release
		if !arbiter.ReleaseLease("worker-alpha", "lane-alpha") {
			t.Fatalf("expected valid release to return true")
		}
		if len(arbiter.ListLeases()) != 0 {
			t.Fatalf("expected 0 active leases after release")
		}
	})

	t.Run("ConcurrentStressArbitration", func(t *testing.T) {
		const workers = 20
		const iterations = 50
		const maxClusterBudget = 4

		arbiter := NewConcurrencyClassArbiter(map[string]int{
			"cluster": maxClusterBudget,
		})

		var wg sync.WaitGroup
		wg.Add(workers)

		for i := 0; i < workers; i++ {
			workerNum := i
			go func() {
				defer wg.Done()
				workerID := fmt.Sprintf("worker-%d", workerNum)
				laneName := fmt.Sprintf("lane-%d", workerNum)
				tree := []string{fmt.Sprintf("internal/module_%d/**", workerNum)}

				for iter := 0; iter < iterations; iter++ {
					res := arbiter.AcquireLease(LaneLeaseRequest{
						LaneKind:     "cluster",
						LaneName:     laneName,
						TreePatterns: tree,
						WorkerID:     workerID,
					})

					if res.Granted {
						// Verify budget invariant is strictly held
						active := arbiter.ActiveCount("cluster")
						if active > maxClusterBudget {
							t.Errorf("budget invariant violated: active %d > max %d", active, maxClusterBudget)
						}
						// Release lease
						if !arbiter.ReleaseLease(workerID, laneName) {
							t.Errorf("worker %s failed to release %s", workerID, laneName)
						}
					} else {
						if res.Reason != RefuseBudgetExhausted && res.Reason != RefuseTreeCollision {
							t.Errorf("unexpected refusal reason: %s", res.Reason)
						}
					}
				}
			}()
		}

		wg.Wait()

		if len(arbiter.ListLeases()) != 0 {
			t.Fatalf("expected all leases released, got %d", len(arbiter.ListLeases()))
		}
	})
}
