package eviction

import (
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: four tiny policies, with no clocks, goroutines or I/O.
func TestSIEVEReportsZeroHashVictims(t *testing.T) {
	t.Parallel()
	type event struct {
		hash   uint64
		keyLen uint16
	}
	for _, hash := range []uint64{0, 17} {
		for _, admission := range []bool{false, true} {
			t.Run(fmt.Sprintf("hash=%d/admission=%t", hash, admission), func(t *testing.T) {
				var events []event
				p := NewPolicy("sieve", 2, func(hash uint64, keyLen uint16) {
					events = append(events, event{hash, keyLen})
				})
				if got := p.Admit(hash, 7); len(got) != 0 || p.Size() != 1 || len(events) != 0 {
					t.Fatalf("first admission: victims=%v size=%d events=%v", got, p.Size(), events)
				}
				if admission {
					if got := p.Admit(1, 11); len(got) != 0 {
						t.Fatalf("second admission: victims=%v", got)
					}
					if got := p.Admit(2, 13); len(got) != 1 || got[0] != hash {
						t.Fatalf("capacity admission: victims=%v, want [%d]", got, hash)
					}
					if p.Size() != 2 || p.Access(hash) || !p.Access(1) || !p.Access(2) {
						t.Fatal("capacity admission changed surviving membership")
					}
				} else {
					if !p.Access(hash) || !p.EvictOne() || p.Size() != 0 || p.Access(hash) {
						t.Fatal("forced eviction did not report and remove the visited victim")
					}
				}
				if len(events) != 1 || events[0] != (event{hash, 7}) {
					t.Fatalf("victim callback=%v, want [%v]", events, event{hash, 7})
				}
				if admission {
					for i, want := range []event{{1, 11}, {2, 13}} {
						if !p.EvictOne() || p.Size() != uint64(1-i) || len(events) != i+2 || events[i+1] != want {
							t.Fatalf("drain %d: size=%d events=%v, want victim %v", i, p.Size(), events, want)
						}
					}
				}
				n := len(events)
				if p.EvictOne() || p.Size() != 0 || len(events) != n {
					t.Fatal("empty policy reported or emitted another victim")
				}
			})
		}
	}
}
