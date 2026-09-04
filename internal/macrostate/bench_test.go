package macrostate

import (
	"strconv"
	"testing"
	"time"
)

// BenchmarkMacroState exercises state transitions and compaction in a loop.
func BenchmarkMacroState(b *testing.B) {
	now := time.Unix(1000, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := &Store{}
		id1 := strconv.Itoa(i*3 + 1)
		id2 := strconv.Itoa(i*3 + 2)
		id3 := strconv.Itoa(i*3 + 3)

		// State transition: Promote
		_, err := s.Apply(Event{
			Schema:     Schema,
			ID:         id1,
			At:         now,
			Kind:       Promote,
			Key:        "lane",
			Value:      "macrostate",
			Provenance: "bench:r1",
		})
		if err != nil {
			b.Fatalf("promote failed: %v", err)
		}

		// State transition: Correct
		_, err = s.Apply(Event{
			Schema:     Schema,
			ID:         id2,
			At:         now,
			Kind:       Correct,
			Key:        "lane",
			Value:      "macrostate-v2",
			Provenance: "bench:r2",
			Replaces:   id1,
		})
		if err != nil {
			b.Fatalf("correct failed: %v", err)
		}

		// Compaction
		res := s.Compact(now)
		if len(res) == 0 {
			b.Fatalf("compact empty")
		}

		// State transition: Delete
		_, err = s.Apply(Event{
			Schema:     Schema,
			ID:         id3,
			At:         now,
			Kind:       Delete,
			Key:        "lane",
			Provenance: "bench:r3",
		})
		if err != nil {
			b.Fatalf("delete failed: %v", err)
		}

		_ = s.Compact(now)
	}
}
