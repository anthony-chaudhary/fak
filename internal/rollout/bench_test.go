package rollout

import (
	"strconv"
	"testing"
)

func BenchmarkRollout(b *testing.B) {
	state, err := baseState().Stage(generationB, 5_000, 10_000, "bench-salt")
	if err != nil {
		b.Fatalf("Stage() error = %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionID := "session-" + strconv.Itoa(i%1000)
		if _, err := state.Select(sessionID, 0); err != nil {
			b.Fatalf("Select() error = %v", err)
		}
	}
}
