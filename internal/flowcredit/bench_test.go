package flowcredit

import "testing"

// BenchmarkFlowCredit exercises credit acquisition (TryReserve) and window
// advancement (Grant) in a continuous loop.
func BenchmarkFlowCredit(b *testing.B) {
	g := NewLedger()
	l := Lane{
		Receiver: "recv-1",
		Sender:   "send-1",
		Class:    "bench",
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		seq := uint64(i + 1)
		cum := uint64((i + 1) * 10)
		g.Grant(l, seq, cum)
		if !g.TryReserve(l, 10) {
			b.Fatal("TryReserve failed to acquire granted credit")
		}
	}
}
