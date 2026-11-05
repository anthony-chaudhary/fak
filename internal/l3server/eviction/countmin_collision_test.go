package eviction

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCountMinSketchCollisionSeparation(t *testing.T) {
	t.Parallel()
	t.Run("full hash and distinct rows", func(t *testing.T) {
		for _, tc := range []struct {
			width uint64
			cold  uint64
		}{
			{64, 65},
			{1024, 1025},
			{64, 1<<63 | 1},
			{64, 36}, // shares row zero with hash 1, but not the other rows
		} {
			s := NewCountMinSketch(tc.width)
			for i := 0; i < 20; i++ {
				s.Increment(1)
			}
			if hot, cold := s.Estimate(1), s.Estimate(tc.cold); hot != 20 || cold != 0 {
				t.Fatalf("width=%d cold=%d: estimates=(%d,%d), want (20,0)", tc.width, tc.cold, hot, cold)
			}
			s.Increment(tc.cold)
			if hot, cold := s.Estimate(1), s.Estimate(tc.cold); hot != 20 || cold != 1 {
				t.Fatalf("width=%d cold=%d after increment: estimates=(%d,%d), want (20,1)", tc.width, tc.cold, hot, cold)
			}
		}
	})
	t.Run("hot candidate wins admission", func(t *testing.T) {
		var callbacks []uint64
		p := NewPolicy("wtinylfu", 4, func(hash uint64, keyLen uint16) {
			if keyLen != 1 {
				t.Errorf("eviction key length=%d, want 1", keyLen)
			}
			callbacks = append(callbacks, hash)
		})
		// Capacity four makes the sketch width 64. Hashes 1 and 65 used to
		// collide in all rows, falsely tying the cold victim and hot candidate.
		for _, hash := range []uint64{1, 2, 3, 65} {
			if evicted := p.Admit(hash, 1); len(evicted) != 0 {
				t.Fatalf("early eviction for %d: %v", hash, evicted)
			}
		}
		for i := 0; i < 10; i++ {
			if !p.Access(65) {
				t.Fatal("hot window candidate is not tracked")
			}
		}
		if evicted := p.Admit(4, 1); !reflect.DeepEqual(evicted, []uint64{1}) {
			t.Fatalf("admission evicted %v, want cold victim [1]", evicted)
		}
		if !reflect.DeepEqual(callbacks, []uint64{1}) || p.Size() != 4 {
			t.Fatalf("callbacks=%v size=%d, want [1] and 4", callbacks, p.Size())
		}
		if !p.Access(65) || p.Access(1) {
			t.Fatal("hot candidate lost or cold victim retained")
		}
	})
}
