package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestKimiK31TrioPagedTopK(t *testing.T) {
	cases := []struct {
		name         string
		scores       []float32
		seq, k, size int
		pages, want  []int
	}{
		{"nonidentity_pages", []float32{1, 9, 3, 8, 7, 99}, 5, 3, 2, []int{4, 1, 7}, []int{9, 3, 14}},
		{"padding", []float32{2}, 1, 3, 2, []int{3}, []int{6, -1, -1}},
		{"empty", nil, 0, 2, 2, nil, []int{-1, -1}},
		{"ties", []float32{1, 1, 1}, 3, 2, 2, []int{5, 2}, []int{10, 11}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := KimiK31TrioPagedTopK(tc.scores, tc.seq, tc.k, tc.size, tc.pages)
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, %v; want %v", got, err, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestKimiK31TrioPagedTopKRefuses(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name         string
		seq, k, size int
		pages        []int
	}{
		{"negative_sequence", -1, 1, 2, nil}, {"sequence_past_scores", 3, 1, 2, nil},
		{"zero_k", 1, 0, 2, []int{0}}, {"zero_page_size", 1, 1, 0, nil},
		{"missing_page", 1, 1, 2, nil}, {"negative_page", 1, 1, 2, []int{-1}},
		{"page_overflow", 2, 1, 2, []int{maxInt}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := KimiK31TrioPagedTopK([]float32{2, 1}, tc.seq, tc.k, tc.size, tc.pages)
			if got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
				t.Fatalf("got %v, %v; want nil and geometry refusal", got, err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestKimiK31TrioPagedTopKNaNDomain(t *testing.T) {
	nan := float32(math.NaN())
	if got, err := KimiK31TrioPagedTopK([]float32{nan, 1}, 2, 1, 2, []int{0}); got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
		t.Fatalf("visible NaN: got %v, %v; want refusal", got, err)
	}
	got, err := KimiK31TrioPagedTopK([]float32{1, nan}, 1, 1, 2, []int{3})
	if err != nil || !reflect.DeepEqual(got, []int{6}) {
		t.Fatalf("masked NaN: got %v, %v; want [6]", got, err)
	}
}

// fak-test:runtime fast est=1ms
func TestKimiK31TrioBackendRequirements(t *testing.T) {
	expected := KimiK31TrioRequirements{Architecture: "kimik31", SparseMLA: true, TopKIndexer: true}
	if got := KimiK31TrioDescriptor(); got != expected {
		t.Fatalf("descriptor %v, want %v", got, expected)
	}
	for _, sparse := range []bool{false, true} {
		for _, indexer := range []bool{false, true} {
			got, err := SelectKimiK31TrioBackend("kimik31", sparse, indexer)
			if sparse && indexer {
				if err != nil || got != expected {
					t.Fatalf("qualified pair: %v, %v", got, err)
				}
			} else if !errors.Is(err, ErrKimiK31TrioBackend) || got != (KimiK31TrioRequirements{}) {
				t.Fatalf("missing capability (%v,%v): %v, %v", sparse, indexer, got, err)
			}
		}
	}
	for _, arch := range []string{"", "deepseek2", "kimi-k31", "kimik3"} {
		got, err := SelectKimiK31TrioBackend(arch, true, true)
		if !errors.Is(err, ErrKimiK31TrioBackend) || got != (KimiK31TrioRequirements{}) {
			t.Fatalf("wrong canonical architecture %q: %v, %v", arch, got, err)
		}
	}
}
