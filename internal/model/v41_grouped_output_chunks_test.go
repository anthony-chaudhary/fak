package model

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41GroupedPanelChunks(t *testing.T) {
	for _, tc := range []struct {
		name  string
		rows  int
		allow map[int]bool
		want  []int
	}{
		{"qualified_256", 256, map[int]bool{64: true}, []int{64, 64, 64, 64}},
		{"qualified_tail", 127, map[int]bool{64: true, 32: true, 16: true}, []int{64, 32, 16, 8, 4, 2, 1}},
		{"limited_32", 80, map[int]bool{32: true, 16: true}, []int{32, 32, 16}},
		{"nonmonotone", 96, map[int]bool{64: true}, []int{64, 8, 8, 8, 8}},
		{"legacy", 31, nil, []int{8, 8, 8, 4, 2, 1}},
		{"scalar_tail", 1, nil, []int{1}},
		{"empty", 0, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := map[int]int{}
			got := v41GroupedPanelChunks(tc.rows, func(n int) bool {
				calls[n]++
				if n <= 8 || n > 64 || calls[n] > 1 {
					t.Fatalf("unexpected admission count: rows=%d calls=%d", n, calls[n])
				}
				return tc.allow[n]
			})
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("chunks=%v want=%v", got, tc.want)
			}
		})
	}
}

// Integration witness: requires the model package fixtures.
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputWidenedChunkParity(t *testing.T) {
	for _, tc := range []struct{ rows, chunks int }{{16, 1}, {32, 1}, {64, 1}, {127, 7}, {256, 4}} {
		t.Run(itoa(tc.rows), func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			defer s.Close()
			input := v41GroupedPanelInput(tc.rows, m.Cfg.NumHeads*m.Cfg.HeadDim)
			want := v41GroupedPanelCacheOracle(t, m, input, tc.rows)
			got, outcome, err := v41GroupedPanelCacheCall(s, input, tc.rows)
			if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
				t.Fatalf("outcome=%d err=%v parity=%t", outcome, err, reflect.DeepEqual(got, want))
			}
			if b.rowReads != tc.chunks || b.gathers != tc.chunks || b.matmuls != tc.chunks*(m.Cfg.OGroups+1) {
				t.Fatalf("reads/gathers/matmuls=%d/%d/%d want=%d/%d/%d", b.rowReads, b.gathers, b.matmuls, tc.chunks, tc.chunks, tc.chunks*(m.Cfg.OGroups+1))
			}
		})
	}
}
