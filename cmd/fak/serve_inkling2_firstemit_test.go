package main

import (
	"testing"
	"time"
)

// first_emit_ms times request receipt to the first emitted token, so it must
// cover the prefill that precedes the first emit. The emit callback sleeps so the
// stamp clears the host timer tick, which can read a sub-millisecond span as zero.
func TestInkling2LoopStampsFirstEmitAfterPrefill(t *testing.T) {
	eng, _ := inkling2FakeEngine([]int{201, 204, 'h', 'i', 209})
	var got []int
	res, err := eng.Generate(t.Context(), []int{200, 'q'}, 2, nil, func(id int) bool {
		if len(got) == 0 {
			time.Sleep(20 * time.Millisecond)
		}
		got = append(got, id)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("no token emitted")
	}
	if res.FirstEmitSeconds < 0.02 {
		t.Fatalf("FirstEmitSeconds = %v, want the first emit stamped after its 20ms callback", res.FirstEmitSeconds)
	}
	if res.FirstEmitSeconds < res.PrefillSeconds {
		t.Fatalf("first emit %.6fs precedes prefill end %.6fs", res.FirstEmitSeconds, res.PrefillSeconds)
	}
}
