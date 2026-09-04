package fleetbus

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkFleetBus exercises directive publish, roster discovery, and ack folding in a loop.
func BenchmarkFleetBus(b *testing.B) {
	now := time.Date(2026, 8, 5, 12, 0, 0, 0, time.UTC)
	bus, err := OpenDir(b.TempDir())
	if err != nil {
		b.Fatalf("OpenDir: %v", err)
	}

	for i := 0; i < 8; i++ {
		inst, ref := NewInstance(fmt.Sprintf("node-%d", i), "box-1", "worker", 1000+i, "127.0.0.1:8080", []Op{"steer", "pause"}, now)
		if ref != nil {
			b.Fatalf("NewInstance: %v", ref)
		}
		if err := bus.Announce(inst); err != nil {
			b.Fatalf("Announce: %v", err)
		}
	}

	d, ref := NewDirective("benchmark-issuer", "steer", "payload", Selector{All: true}, time.Hour, "bench", now)
	if ref != nil {
		b.Fatalf("NewDirective: %v", ref)
	}
	roster, err := bus.Instances(now, DefaultInstanceTTL)
	if err != nil {
		b.Fatalf("Instances: %v", err)
	}
	acks := make([]Ack, len(roster))
	for i, inst := range roster {
		acks[i] = Ack{
			Schema:    AckSchema,
			Directive: d.ID,
			Instance:  inst.ID,
			Status:    AckApplied,
			Witness:   "benchmark-applied",
			Affected:  1,
			AckedUTC:  utc(now),
		}
	}

	b.Run("AckFolding", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep := Fold(d, roster, acks, now)
			if !rep.Complete {
				b.Fatalf("expected Complete")
			}
		}
	})

	b.Run("RosterDiscovery", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r, err := bus.Instances(now, DefaultInstanceTTL)
			if err != nil {
				b.Fatalf("Instances: %v", err)
			}
			if len(r) != len(roster) {
				b.Fatalf("expected %d instances, got %d", len(roster), len(r))
			}
		}
	})

	b.Run("DirectivePublish", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDir, r := NewDirective("issuer", "steer", "payload", Selector{All: true}, time.Minute, "", now.Add(time.Duration(i)*time.Millisecond))
			if r != nil {
				b.Fatalf("NewDirective: %v", r)
			}
			if err := bus.Publish(benchDir); err != nil {
				b.Fatalf("Publish: %v", err)
			}
		}
	})
}
