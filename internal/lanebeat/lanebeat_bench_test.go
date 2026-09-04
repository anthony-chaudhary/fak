package lanebeat

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchDecisionSink Decision
	benchLeaseSink    Lease
)

func makeSyntheticLeases(count int, lanesCount int) []Lease {
	baseTime := time.Date(2026, 8, 7, 10, 0, 0, 0, time.UTC)
	leases := make([]Lease, count)
	for i := 0; i < count; i++ {
		laneName := fmt.Sprintf("lane-%03d", i%lanesCount)
		leases[i] = Lease{
			Lane:       laneName,
			Holder:     fmt.Sprintf("worker-%03d", i),
			HostID:     "HOST-PROD-01",
			LoopTS:     fmt.Sprintf("2026-08-07T10:%02d:%02dZ", (i/60)%60, i%60),
			AcquiredAt: baseTime.Add(time.Duration(i) * time.Second),
		}
	}
	return leases
}

func TestBenchmarkFixtures(t *testing.T) {
	leases := makeSyntheticLeases(10, 2)
	if len(leases) != 10 {
		t.Fatalf("expected 10 leases, got %d", len(leases))
	}
}

func BenchmarkDecide_AdmitBeat(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawnTime := now.Add(-20 * time.Minute)
	h := Holder{
		Lane:         "gateway",
		HostID:       "HOST-PROD-01",
		PID:          12345,
		Alive:        true,
		StartedAt:    spawnTime,
		LastOutputAt: now.Add(-30 * time.Second),
		MaxHold:      45 * time.Minute,
		QuietAfter:   DefaultQuietAfter,
	}
	live := []Lease{
		{
			Lane:       "gateway",
			Holder:     "agent-worker-1",
			HostID:     "HOST-PROD-01",
			LoopTS:     "2026-08-07T11:45:00Z",
			AcquiredAt: spawnTime.Add(2 * time.Minute),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Decide(h, live, now)
		if !res.Beat {
			b.Fatal("expected beat to be admitted")
		}
		benchDecisionSink = res
	}
}

func BenchmarkDecide_RefusalRungs(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawnTime := now.Add(-20 * time.Minute)

	baseHolder := Holder{
		Lane:         "gateway",
		HostID:       "HOST-PROD-01",
		PID:          12345,
		Alive:        true,
		StartedAt:    spawnTime,
		LastOutputAt: now.Add(-30 * time.Second),
		MaxHold:      45 * time.Minute,
	}

	validLease := Lease{
		Lane:       "gateway",
		Holder:     "agent-worker-1",
		HostID:     "HOST-PROD-01",
		LoopTS:     "2026-08-07T11:45:00Z",
		AcquiredAt: spawnTime.Add(2 * time.Minute),
	}

	cases := []struct {
		name       string
		holder     Holder
		live       []Lease
		wantReason string
	}{
		{
			name: "HolderDead",
			holder: func() Holder {
				h := baseHolder
				h.Alive = false
				return h
			}(),
			live:       []Lease{validLease},
			wantReason: ReasonHolderDead,
		},
		{
			name: "HolderPastDeadline",
			holder: func() Holder {
				h := baseHolder
				h.StartedAt = now.Add(-50 * time.Minute)
				return h
			}(),
			live:       []Lease{validLease},
			wantReason: ReasonHolderPastDeadline,
		},
		{
			name: "HolderQuiet",
			holder: func() Holder {
				h := baseHolder
				h.LastOutputAt = now.Add(-20 * time.Minute)
				return h
			}(),
			live:       []Lease{validLease},
			wantReason: ReasonHolderQuiet,
		},
		{
			name:       "NoLeaseOnLane",
			holder:     baseHolder,
			live:       []Lease{},
			wantReason: ReasonNoLeaseOnLane,
		},
		{
			name:   "ForeignHost",
			holder: baseHolder,
			live: []Lease{
				{
					Lane:       "gateway",
					Holder:     "agent-worker-1",
					HostID:     "HOST-PROD-OTHER",
					LoopTS:     "2026-08-07T11:45:00Z",
					AcquiredAt: spawnTime.Add(2 * time.Minute),
				},
			},
			wantReason: ReasonForeignHost,
		},
		{
			name:   "LeasePredatesHolder",
			holder: baseHolder,
			live: []Lease{
				{
					Lane:       "gateway",
					Holder:     "agent-worker-1",
					HostID:     "HOST-PROD-01",
					LoopTS:     "2026-08-07T11:00:00Z",
					AcquiredAt: spawnTime.Add(-10 * time.Minute),
				},
			},
			wantReason: ReasonLeasePredatesHolder,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := Decide(tc.holder, tc.live, now)
				if res.Beat || res.Reason != tc.wantReason {
					b.Fatalf("expected reason %q, got %q", tc.wantReason, res.Reason)
				}
				benchDecisionSink = res
			}
		})
	}
}

func BenchmarkDecide_LeaseScaling(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawnTime := now.Add(-20 * time.Minute)

	holder := Holder{
		Lane:         "lane-005",
		HostID:       "HOST-PROD-01",
		PID:          12345,
		Alive:        true,
		StartedAt:    spawnTime,
		LastOutputAt: now.Add(-30 * time.Second),
		MaxHold:      45 * time.Minute,
	}

	sizes := []int{10, 50, 200, 1000}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%d_leases", size), func(b *testing.B) {
			live := makeSyntheticLeases(size, 20)
			live = append(live, Lease{
				Lane:       "lane-005",
				Holder:     "active-worker",
				HostID:     "HOST-PROD-01",
				LoopTS:     "2026-08-07T11:55:00Z",
				AcquiredAt: spawnTime.Add(5 * time.Minute),
			})

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := Decide(holder, live, now)
				if !res.Beat {
					b.Fatalf("expected beat to be admitted in scaling test")
				}
				benchDecisionSink = res
			}
		})
	}
}

func BenchmarkNewestOnLane_SiblingConflict(b *testing.B) {
	siblingCounts := []int{2, 5, 20, 100}
	baseTime := time.Date(2026, 8, 7, 10, 0, 0, 0, time.UTC)

	for _, count := range siblingCounts {
		b.Run(fmt.Sprintf("%d_siblings", count), func(b *testing.B) {
			leases := make([]Lease, count)
			for i := 0; i < count; i++ {
				leases[i] = Lease{
					Lane:       "contested-lane",
					Holder:     fmt.Sprintf("sibling-%03d", i),
					HostID:     "HOST-PROD-01",
					LoopTS:     fmt.Sprintf("2026-08-07T10:%02d:00Z", i),
					AcquiredAt: baseTime.Add(time.Duration(i) * time.Minute),
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				lease, ok := newestOnLane(leases, "contested-lane")
				if !ok {
					b.Fatal("expected newest lease to be found")
				}
				benchLeaseSink = lease
			}
		})
	}
}

func BenchmarkSupervisorFleetTick(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawnTime := now.Add(-15 * time.Minute)

	const workerCount = 20
	const laneCount = 10

	workers := make([]Holder, workerCount)
	for i := 0; i < workerCount; i++ {
		workers[i] = Holder{
			Lane:         fmt.Sprintf("lane-%03d", i%laneCount),
			HostID:       "HOST-PROD-01",
			PID:          20000 + i,
			Alive:        i != 3,
			StartedAt:    spawnTime,
			LastOutputAt: now.Add(-time.Duration(i*30) * time.Second),
			MaxHold:      45 * time.Minute,
		}
	}

	liveLeases := makeSyntheticLeases(100, laneCount)
	for l := 0; l < laneCount; l++ {
		liveLeases = append(liveLeases, Lease{
			Lane:       fmt.Sprintf("lane-%03d", l),
			Holder:     fmt.Sprintf("worker-on-%03d", l),
			HostID:     "HOST-PROD-01",
			LoopTS:     fmt.Sprintf("2026-08-07T11:%02d:00Z", 40+l),
			AcquiredAt: spawnTime.Add(time.Duration(l+1) * time.Minute),
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		beaten := 0
		for w := 0; w < workerCount; w++ {
			dec := Decide(workers[w], liveLeases, now)
			if dec.Beat {
				beaten++
			}
			benchDecisionSink = dec
		}
		if beaten == 0 {
			b.Fatal("expected at least some workers to be beaten")
		}
	}
}

func BenchmarkDecide_Parallel(b *testing.B) {
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawnTime := now.Add(-20 * time.Minute)

	live := makeSyntheticLeases(50, 10)
	for l := 0; l < 10; l++ {
		live = append(live, Lease{
			Lane:       fmt.Sprintf("lane-%03d", l),
			Holder:     fmt.Sprintf("worker-%d", l),
			HostID:     "HOST-PROD-01",
			LoopTS:     fmt.Sprintf("2026-08-07T11:%02d:00Z", 50+l),
			AcquiredAt: spawnTime.Add(time.Duration(l+1) * time.Minute),
		})
	}

	holders := make([]Holder, 10)
	for l := 0; l < 10; l++ {
		holders[l] = Holder{
			Lane:         fmt.Sprintf("lane-%03d", l),
			HostID:       "HOST-PROD-01",
			PID:          1000 + l,
			Alive:        true,
			StartedAt:    spawnTime,
			LastOutputAt: now.Add(-time.Duration(l) * time.Minute),
			MaxHold:      45 * time.Minute,
		}
	}

	b.Run("ParallelAdmission", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			idx := 0
			for pb.Next() {
				h := holders[idx%len(holders)]
				res := Decide(h, live, now)
				benchDecisionSink = res
				idx++
			}
		})
	})
}
