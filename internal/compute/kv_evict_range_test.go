package compute

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s
func TestCPUKVEvictRangeClipsBeforeAddition(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                 string
		from, count, bounded int
	}{
		{"overflow", 1, maxInt, 2}, {"oversized", 1, 9, 2},
		{"exact_tail", 1, 2, 2}, {"middle", 1, 1, 1},
		{"negative_from", -1, maxInt, 0}, {"past_end", 3, maxInt, 0},
		{"zero_count", 1, 0, 0}, {"negative_count", 1, -1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := cpu()
			cfg := KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000}
			k := c.NewKV(cfg)
			defer k.Free()
			for p := 0; p < 3; p++ {
				for l := 0; l < cfg.NumLayers; l++ {
					row := []float32{float32(p + l + 1), float32(p - l + 2)}
					x := NewF32(c, []int{cfg.HeadDim}, row)
					k.AppendKV(l, x, x, x, p)
				}
			}
			want := k.Clone()
			defer want.Free()
			wantRemoved := want.Evict(tc.from, tc.bounded)
			if got := k.Evict(tc.from, tc.count); got != wantRemoved {
				t.Fatalf("removed=%d want=%d", got, wantRemoved)
			}
			a, b := k.(*cpuKV), want.(*cpuKV)
			if !reflect.DeepEqual(a.pos, b.pos) || !reflect.DeepEqual(a.K, b.K) || !reflect.DeepEqual(a.Kraw, b.Kraw) || !reflect.DeepEqual(a.V, b.V) {
				t.Fatal("eviction differs from bounded range oracle")
			}
		})
	}
}
