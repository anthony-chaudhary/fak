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
func TestKimiK31TrioRawTopKMode(t *testing.T) {
	// INDICES mode of the upstream paged indexer: page_tables omitted, so the
	// output carries the selected logical indices, -1 padded past min(k, seq).
	cases := []struct {
		name        string
		scores      []float32
		seq, k      int
		want        []int
		wantRefused bool
	}{
		{"distinct", []float32{0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4}, 8, 4, []int{1, 3, 4, 6}, false},
		{"padding", []float32{2}, 1, 3, []int{0, -1, -1}, false},
		{"empty", nil, 0, 2, []int{-1, -1}, false},
		{"ties", []float32{1, 1, 1}, 3, 2, []int{0, 1}, false},
		{"negative_sequence", []float32{1}, -1, 1, nil, true},
		{"sequence_past_scores", []float32{1}, 2, 1, nil, true},
		{"zero_k", []float32{1}, 1, 0, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := KimiK31TrioRawTopK(tc.scores, tc.seq, tc.k)
			if tc.wantRefused {
				if got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
					t.Fatalf("got %v, %v; want geometry refusal", got, err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, %v; want %v", got, err, tc.want)
			}
		})
	}
	nan := float32(math.NaN())
	if got, err := KimiK31TrioRawTopK([]float32{nan, 1}, 2, 1); got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
		t.Fatalf("visible NaN: got %v, %v; want refusal", got, err)
	}
}

// fak-test:runtime fast est=1ms
func TestKimiK31TrioPagedTopKRowsBatch(t *testing.T) {
	// Batched paged contract over a fixed page table: row 0 selects across a
	// full page boundary, row 1 is the empty-sequence all--1 row, row 2 takes
	// the trivial short-sequence path. pageSize 4, topK 4.
	scores := [][]float32{
		{0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4},
		{0.5},
		{0.2, 0.1, 0.3, 0.9, 0.4},
	}
	seqLens := []int{8, 0, 3}
	pageTables := [][]int{{10, 11, 12, 13}, {0, 0, 0, 0}, {5, 6, 7, 8}}
	want := [][]int{
		{41, 43, 44, 46}, // 1->10*4+1, 3->10*4+3, 4->11*4+0, 6->11*4+2
		{-1, -1, -1, -1},
		{20, 21, 22, -1}, // trivial path: positions 0,1,2 in order
	}
	got, err := KimiK31TrioPagedTopKRows(scores, seqLens, pageTables, 4, 4)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("batch oracle: got %v, %v; want %v", got, err, want)
	}

	// A zero-row batch returns an empty non-nil result.
	empty, err := KimiK31TrioPagedTopKRows(nil, nil, nil, 4, 4)
	if err != nil || len(empty) != 0 || empty == nil {
		t.Fatalf("zero-row batch: got %v, %v; want empty non-nil", empty, err)
	}

	// Disagreement between the three row slices refuses the whole call.
	if got, err := KimiK31TrioPagedTopKRows(scores, seqLens[:2], pageTables, 4, 4); got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
		t.Fatalf("shape mismatch: got %v, %v; want geometry refusal", got, err)
	}

	// One bad row refuses the whole call with no partial output.
	badSeq := []int{8, -1, 3}
	if got, err := KimiK31TrioPagedTopKRows(scores, badSeq, pageTables, 4, 4); got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
		t.Fatalf("bad row: got %v, %v; want geometry refusal", got, err)
	}
	if got, err := KimiK31TrioPagedTopKRows(scores, seqLens, pageTables, 0, 4); got != nil || !errors.Is(err, ErrKimiK31TrioGeometry) {
		t.Fatalf("zero page size: got %v, %v; want geometry refusal", got, err)
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
