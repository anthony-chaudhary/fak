package computetrace

import (
	"testing"
	"time"
)

func BenchmarkComputeTrace(b *testing.B) {
	r := New(b.N)
	e := Event{
		RunID:            "bench-run",
		RequestID:        "bench-req",
		Operation:        "matmul",
		Phase:            "kernel",
		Backend:          "cuda",
		Device:           "cuda:0",
		Kernel:           "sgemm",
		StartedAt:        time.Unix(1, 2).UTC(),
		DurationNS:       100,
		TimerDomain:      "cuda_event",
		Bytes:            4096,
		Shapes:           [][]int{{128, 128}},
		ProvenanceDigest: Digest("cuda", "sgemm"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Record(e)
	}
}
