package hostplacement

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchSinkDecision   Decision
	benchSinkBool       bool
	benchSinkHeartbeats []Heartbeat
)

func makeCluster(n int, now time.Time, ttl time.Duration, mode string) []Heartbeat {
	hosts := make([]Heartbeat, n)
	for i := 0; i < n; i++ {
		hostname := fmt.Sprintf("host-%04d", i)
		switch mode {
		case "all_eligible":
			hosts[i] = Heartbeat{
				Hostname:     hostname,
				LiveHeadroom: float64(100 - (i % 50)),
				Saturation:   0.10 + float64(i%50)*0.01, // 0.10 to 0.59
				TS:           now,
			}
		case "half_saturated":
			sat := 0.20
			if i%2 == 1 {
				sat = 0.95 // saturated over 0.90 threshold
			}
			hosts[i] = Heartbeat{
				Hostname:     hostname,
				LiveHeadroom: float64(50 - (i % 20)),
				Saturation:   sat,
				TS:           now,
			}
		case "none_eligible":
			// Alternating between saturated and stale
			ts := now
			sat := 0.98
			if i%2 == 0 {
				ts = now.Add(-ttl - time.Second)
				sat = 0.10
			}
			hosts[i] = Heartbeat{
				Hostname:     hostname,
				LiveHeadroom: 10,
				Saturation:   sat,
				TS:           ts,
			}
		case "ties_headroom":
			hosts[i] = Heartbeat{
				Hostname:     hostname,
				LiveHeadroom: float64(i),
				Saturation:   0.40, // all identical saturation
				TS:           now,
			}
		default:
			hosts[i] = Heartbeat{
				Hostname:     hostname,
				LiveHeadroom: float64(n - i),
				Saturation:   float64(i) / float64(n),
				TS:           now,
			}
		}
	}
	return hosts
}

func BenchmarkHeartbeat_Fresh(b *testing.B) {
	now := base
	hb := Heartbeat{Hostname: "host-bench", TS: now}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = hb.Fresh(now, ttl)
	}
}

func BenchmarkHeartbeat_Eligible(b *testing.B) {
	now := base
	hb := Heartbeat{Hostname: "host-bench", Saturation: 0.35, TS: now}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = hb.Eligible(0.90, now, ttl)
	}
}

func BenchmarkPlace_Scale(b *testing.B) {
	now := base
	sizes := []int{2, 8, 32, 128, 512}

	for _, n := range sizes {
		hosts := makeCluster(n, now, ttl, "all_eligible")
		b.Run(fmt.Sprintf("%d_hosts", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkDecision = Place(hosts, 0.90, now, ttl)
			}
		})
	}
}

func BenchmarkPlace_Distributions(b *testing.B) {
	now := base
	modes := []string{"all_eligible", "half_saturated", "none_eligible", "ties_headroom"}

	for _, mode := range modes {
		hosts := makeCluster(64, now, ttl, mode)
		b.Run(mode, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkDecision = Place(hosts, 0.90, now, ttl)
			}
		})
	}
}

func BenchmarkRegistry_Observe(b *testing.B) {
	now := base

	b.Run("new_hosts", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r := NewRegistry()
			for j := 0; j < 32; j++ {
				r.Observe(Heartbeat{
					Hostname:     fmt.Sprintf("host-%d", j),
					LiveHeadroom: 10,
					Saturation:   0.5,
					TS:           now,
				})
			}
		}
	})

	b.Run("update_fresher", func(b *testing.B) {
		r := NewRegistry()
		for j := 0; j < 32; j++ {
			r.Observe(Heartbeat{
				Hostname:     fmt.Sprintf("host-%d", j),
				LiveHeadroom: 10,
				Saturation:   0.5,
				TS:           now,
			})
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r.Observe(Heartbeat{
				Hostname:     "host-15",
				LiveHeadroom: 12,
				Saturation:   0.45,
				TS:           now.Add(time.Duration(i) * time.Millisecond),
			})
		}
	})

	b.Run("stale_ignored", func(b *testing.B) {
		r := NewRegistry()
		r.Observe(Heartbeat{
			Hostname:     "host-steady",
			LiveHeadroom: 10,
			Saturation:   0.5,
			TS:           now,
		})
		staleHB := Heartbeat{
			Hostname:     "host-steady",
			LiveHeadroom: 99,
			Saturation:   0.01,
			TS:           now.Add(-time.Minute),
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r.Observe(staleHB)
		}
	})
}

func BenchmarkRegistry_Hosts(b *testing.B) {
	now := base
	sizes := []int{8, 32, 128}

	for _, n := range sizes {
		r := NewRegistry()
		for j := 0; j < n; j++ {
			r.Observe(Heartbeat{
				Hostname:     fmt.Sprintf("host-%04d", (j*37)%n),
				LiveHeadroom: float64(j),
				Saturation:   0.25,
				TS:           now,
			})
		}
		b.Run(fmt.Sprintf("%d_hosts", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkHeartbeats = r.Hosts()
			}
		})
	}
}

func BenchmarkRegistry_Place(b *testing.B) {
	now := base
	sizes := []int{8, 32, 128}

	for _, n := range sizes {
		r := NewRegistry()
		for j := 0; j < n; j++ {
			r.Observe(Heartbeat{
				Hostname:     fmt.Sprintf("host-%04d", j),
				LiveHeadroom: float64(100 - (j % 50)),
				Saturation:   0.10 + float64(j%50)*0.01,
				TS:           now,
			})
		}
		b.Run(fmt.Sprintf("%d_hosts", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkDecision = r.Place(0.90, now, ttl)
			}
		})
	}
}

func TestPlacementAllocationBudget(t *testing.T) {
	now := base
	hosts := makeCluster(32, now, ttl, "all_eligible")

	// Pre-warm
	for i := 0; i < 5; i++ {
		_ = Place(hosts, 0.90, now, ttl)
	}

	allocs := testing.AllocsPerRun(100, func() {
		_ = Place(hosts, 0.90, now, ttl)
	})

	// Place allocates the eligible slice plus internal sort interface wrapping.
	// We budget at most 5 allocations per decision even for 32 hosts.
	const maxAllocs = 5.0
	if allocs > maxAllocs {
		t.Fatalf("Place(32 hosts) allocs per run = %.1f, exceeds budget of %.1f", allocs, maxAllocs)
	}
}
