package fleetfreeze

import (
	"testing"
)

// BenchmarkFleetFreeze exercises freeze state evaluation and decision checking in a loop.
func BenchmarkFleetFreeze(b *testing.B) {
	st := Freeze("benchmark freeze", testFreezeUnix)
	ops := []OpClass{OpSpawn, OpWitnessClose, OpStatusRefresh, OpComment}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		op := ops[i%len(ops)]
		d := Allowed(st, op)
		if op == OpSpawn && d.Allow {
			b.Fatalf("expected OpSpawn to be held")
		}
	}
}

func TestBenchmarkFleetFreezeSanity(t *testing.T) {
	st := Freeze("benchmark freeze sanity", testFreezeUnix)
	d := Allowed(st, OpSpawn)
	if d.Allow {
		t.Fatalf("expected OpSpawn to be held during sanity check")
	}
}
