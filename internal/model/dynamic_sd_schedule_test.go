package model

import (
	"errors"
	"testing"
)

// TestDynamicSDScheduleLookup pins the borrowed vLLM dynamic-SD-schedule
// contract (ADAPT of vllm/v1/spec_decode/dynamic/utils.py@ae065918640a198a56f6f17eb45e251c3229fe41):
// sparse (batch_size, K) points normalize into a dense, monotonic table;
// duplicates and out-of-range K reject with the named sentinel errors; and batch
// sizes above the largest point clamp to the last entry.
func TestDynamicSDScheduleLookup(t *testing.T) {
	t.Run("sparse points normalize to a dense monotonic table", func(t *testing.T) {
		s, err := NewDynamicSDSchedule([]SDSchedulePoint{
			{BatchSize: 1, K: 4},
			{BatchSize: 8, K: 2},
			{BatchSize: 16, K: 0},
		}, 32, 4)
		if err != nil {
			t.Fatalf("NewDynamicSDSchedule: %v", err)
		}
		// Exact dense table: points hold their K, gaps carry forward the previous
		// K (step interpolation), and the tail clamps to the last entry.
		want := map[int]int{
			1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4,
			8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2,
			16: 0, 17: 0, 31: 0, 32: 0,
		}
		for b, k := range want {
			got, ok := s.K(b)
			if !ok {
				t.Fatalf("K(%d): ok = false, want true", b)
			}
			if got != k {
				t.Errorf("K(%d) = %d, want %d", b, got, k)
			}
		}
		// Monotonicity enforced: the whole covered range is non-increasing.
		prev, _ := s.K(1)
		for b := 2; b <= 32; b++ {
			k, _ := s.K(b)
			if k > prev {
				t.Fatalf("table not non-increasing at batch %d: %d > %d", b, k, prev)
			}
			prev = k
		}
	})

	t.Run("duplicate batch size rejects with the named error", func(t *testing.T) {
		_, err := NewDynamicSDSchedule([]SDSchedulePoint{
			{BatchSize: 1, K: 4},
			{BatchSize: 1, K: 2},
		}, 32, 4)
		if !errors.Is(err, ErrDuplicateBatchSize) {
			t.Errorf("err = %v, want ErrDuplicateBatchSize", err)
		}
	})

	t.Run("out-of-range K rejects with the named error", func(t *testing.T) {
		// Above the configured maximum.
		if _, err := NewDynamicSDSchedule([]SDSchedulePoint{{BatchSize: 1, K: 5}}, 32, 4); !errors.Is(err, ErrKOutOfRange) {
			t.Errorf("K above max: err = %v, want ErrKOutOfRange", err)
		}
		// Below the lower bound.
		if _, err := NewDynamicSDSchedule([]SDSchedulePoint{{BatchSize: 1, K: -1}}, 32, 4); !errors.Is(err, ErrNegativeK) {
			t.Errorf("negative K: err = %v, want ErrNegativeK", err)
		}
	})

	t.Run("non-monotonic K rejects with the named error", func(t *testing.T) {
		_, err := NewDynamicSDSchedule([]SDSchedulePoint{
			{BatchSize: 1, K: 2},
			{BatchSize: 8, K: 4}, // K increases with batch size
		}, 32, 4)
		if !errors.Is(err, ErrNonMonotonicK) {
			t.Errorf("err = %v, want ErrNonMonotonicK", err)
		}
	})

	t.Run("batch size above the largest point clamps to the last entry", func(t *testing.T) {
		s, err := NewDynamicSDSchedule([]SDSchedulePoint{
			{BatchSize: 1, K: 4},
			{BatchSize: 8, K: 2},
			{BatchSize: 16, K: 0},
		}, 32, 4)
		if err != nil {
			t.Fatalf("NewDynamicSDSchedule: %v", err)
		}
		for _, b := range []int{17, 32, 33, 100} {
			got, ok := s.K(b)
			if !ok {
				t.Fatalf("K(%d): ok = false, want true", b)
			}
			if got != 0 {
				t.Errorf("K(%d) = %d, want 0 (clamp to last entry)", b, got)
			}
		}
	})

	t.Run("effective K is the min of the three axes", func(t *testing.T) {
		s, err := NewDynamicSDSchedule([]SDSchedulePoint{
			{BatchSize: 1, K: 4},
			{BatchSize: 8, K: 2},
			{BatchSize: 16, K: 0},
		}, 32, 4)
		if err != nil {
			t.Fatalf("NewDynamicSDSchedule: %v", err)
		}
		// schedule K(4)=4, adaptive=2, governor=3 -> min = 2 (adaptive governs).
		if got := s.EffectiveK(4, 2, 3); got != 2 {
			t.Errorf("EffectiveK(4,2,3) = %d, want 2", got)
		}
		// schedule K(4)=4, adaptive=5, governor=3 -> min = 3 (governor governs).
		if got := s.EffectiveK(4, 5, 3); got != 3 {
			t.Errorf("EffectiveK(4,5,3) = %d, want 3", got)
		}
		// schedule K(8)=2, adaptive=5, governor=3 -> min = 2 (schedule governs).
		if got := s.EffectiveK(8, 5, 3); got != 2 {
			t.Errorf("EffectiveK(8,5,3) = %d, want 2", got)
		}
		// schedule K(16)=0 -> min = 0 regardless of the other axes.
		if got := s.EffectiveK(16, 5, 3); got != 0 {
			t.Errorf("EffectiveK(16,5,3) = %d, want 0", got)
		}
	})

	t.Run("unconfigured schedule imposes no operator bound", func(t *testing.T) {
		var s DynamicSDSchedule // zero value: default-off
		if got := s.EffectiveK(8, 5, 3); got != 3 {
			t.Errorf("EffectiveK(8,5,3) on zero schedule = %d, want 3 (min of the other axes)", got)
		}
		if got := s.EffectiveK(8, 2, 3); got != 2 {
			t.Errorf("EffectiveK(8,2,3) on zero schedule = %d, want 2 (min of the other axes)", got)
		}
		if _, ok := s.K(8); ok {
			t.Errorf("K(8) on zero schedule: ok = true, want false (no operator bound)")
		}
	})
}
