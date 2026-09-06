package observer

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func BenchmarkObserveSyncBarrierContention(b *testing.B) {
	p := NewPool(Config{
		WorkerCount:    8,
		QueueSize:      4096,
		BarrierTimeout: 100 * time.Millisecond,
	})
	if err := p.Start(); err != nil {
		b.Fatalf("failed to start pool: %v", err)
	}
	defer p.Close()

	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var i int
		for pb.Next() {
			i++
			sessionID := fmt.Sprintf("bench-contention-%d", i%4)
			if i%4 == 0 {
				obs := StepObservation{
					SessionID: sessionID,
					Tool:      "Edit",
					Diff:      "@@ -1 +1 @@\n-old\n+new",
				}
				_, _ = p.ObserveSyncBarrier(ctx, obs)
			} else {
				obs := StepObservation{
					SessionID: sessionID,
					Tool:      "Read",
					Args:      "file.go",
				}
				ch := p.ObserveAsync(ctx, obs)
				<-ch
			}
		}
	})
}
