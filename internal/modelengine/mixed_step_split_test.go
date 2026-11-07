package modelengine

import (
	"errors"
	"testing"
)

// TestMixedStepSplitRecord pins the closed contract of the mixed-step split
// record: exact uint32[9] slot values for known width/length layouts, the
// all-zero (unsplit) fail-open cases, the min(1+K, 8) decode-width cap, and
// the view-slicing semantics a split-aware op applies to the record. The
// expected records mirror the upstream modular test suite at
// 88ab8913af6b9298fe914e0f20d87a49e4172d0d, extended with the two batch
// max-length slots (slots 0-1) this fak port carries in the same array.
func TestMixedStepSplitRecord(t *testing.T) {
	const k = 3 // draft width used by most cases below
	w := MixedStepDecodeWidth(k)

	t.Run("decode-width-cap", func(t *testing.T) {
		cases := []struct {
			drafts int
			want   int
		}{
			{drafts: 0, want: 1},
			{drafts: 3, want: 4},
			{drafts: 7, want: 8},
			{drafts: 15, want: 8},
		}
		for _, tc := range cases {
			if got := MixedStepDecodeWidth(tc.drafts); got != tc.want {
				t.Fatalf("MixedStepDecodeWidth(%d) = %d, want %d", tc.drafts, got, tc.want)
			}
		}
	})

	cases := []struct {
		name     string
		widths   []int
		valid    []int
		decodeW  int
		want     MixedStepSplit
		unsplit  bool
		wantDec  []int // rows the op routes to the decode route
		wantPref []int // rows the op routes to the prefill route
	}{
		{
			// Decode rows first make the head view.
			name:     "decode-rows-first-make-head-view",
			widths:   []int{4, 4, 604},
			valid:    []int{100, 200, 1700},
			decodeW:  w,
			want:     MixedStepSplit{604, 1700, 2, 8, 4, 200, 604, 1700, 1700},
			wantDec:  []int{0, 1},
			wantPref: []int{2},
		},
		{
			// Decode rows last make the tail view.
			name:     "decode-rows-last-make-tail-view",
			widths:   []int{604, 9, 4, 4},
			valid:    []int{1700, 50, 100, 200},
			decodeW:  w,
			want:     MixedStepSplit{604, 1700, 2, 613, 604, 1700, 4, 200, 1750},
			wantDec:  []int{2, 3},
			wantPref: []int{0, 1},
		},
		{
			// The longer end run takes the decode view.
			name:     "longer-end-run-takes-decode-view",
			widths:   []int{4, 604, 4, 4},
			valid:    []int{10, 20, 30, 40},
			decodeW:  w,
			want:     MixedStepSplit{604, 40, 2, 608, 604, 20, 4, 40, 30},
			wantDec:  []int{2, 3},
			wantPref: []int{0, 1},
		},
		{
			// A decode row past the first prefill row stays in the prefill
			// view: correct, only slower.
			name:     "decode-row-after-prompt-row-stays-in-prefill-view",
			widths:   []int{4, 604, 4, 700},
			valid:    []int{10, 20, 30, 40},
			decodeW:  w,
			want:     MixedStepSplit{700, 40, 1, 4, 4, 10, 700, 40, 90},
			wantDec:  []int{0},
			wantPref: []int{1, 2, 3},
		},
		{
			// Equal end runs: the tie goes to the head.
			name:     "equal-end-runs-tie-goes-to-head",
			widths:   []int{4, 604, 4},
			valid:    []int{10, 20, 30},
			decodeW:  w,
			want:     MixedStepSplit{604, 30, 1, 4, 4, 10, 604, 30, 50},
			wantDec:  []int{0},
			wantPref: []int{1, 2},
		},
		{
			// A one-token prompt chunk is as wide as a verify row, so it
			// rides the decode view.
			name:     "one-token-chunk-rides-decode-view",
			widths:   []int{4, 4, 300},
			valid:    []int{1, 2, 3},
			decodeW:  w,
			want:     MixedStepSplit{300, 3, 2, 8, 4, 2, 300, 3, 3},
			wantDec:  []int{0, 1},
			wantPref: []int{2},
		},
		{
			// Short prompt rows wider than the verify width stay in the
			// prefill view.
			name:     "short-prompt-rows-wider-than-verify-stay-in-prefill-view",
			widths:   []int{4, 6, 300},
			valid:    []int{1, 2, 3},
			decodeW:  w,
			want:     MixedStepSplit{300, 3, 1, 4, 4, 1, 300, 3, 5},
			wantDec:  []int{0},
			wantPref: []int{1, 2},
		},
		{
			// No prefill row: all-decode batch, unsplit.
			name:     "all-decode-no-split",
			widths:   []int{4, 4, 4},
			valid:    []int{1, 1, 1},
			decodeW:  w,
			want:     MixedStepSplit{4, 1, 0, 0, 0, 0, 0, 0, 0},
			unsplit:  true,
			wantPref: []int{0, 1, 2},
		},
		{
			// No decode row at either end: all-prefill batch, unsplit.
			name:     "all-prefill-no-split",
			widths:   []int{600, 700},
			valid:    []int{1, 1},
			decodeW:  w,
			want:     MixedStepSplit{700, 1, 0, 0, 0, 0, 0, 0, 0},
			unsplit:  true,
			wantPref: []int{0, 1},
		},
		{
			// Widest row at exactly the decode cap: still no prefill row,
			// unsplit even though rows exceed the K=3 decode width.
			name:     "all-spec-width-no-split",
			widths:   []int{4, 8},
			valid:    []int{1, 1},
			decodeW:  w,
			want:     MixedStepSplit{8, 1, 0, 0, 0, 0, 0, 0, 0},
			unsplit:  true,
			wantPref: []int{0, 1},
		},
		{
			// Decode row only in the middle: no end run, unsplit.
			name:     "decode-in-middle-no-end-run",
			widths:   []int{600, 4, 700},
			valid:    []int{1, 1, 1},
			decodeW:  w,
			want:     MixedStepSplit{700, 1, 0, 0, 0, 0, 0, 0, 0},
			unsplit:  true,
			wantPref: []int{0, 1, 2},
		},
		{
			// Empty batch: fully zero record.
			name:    "empty-batch",
			widths:  nil,
			valid:   nil,
			decodeW: w,
			want:    MixedStepSplit{},
			unsplit: true,
		},
		{
			// Single-token decode rows split without drafts (K = 0).
			name:     "single-token-decode-rows-split-without-drafts",
			widths:   []int{1, 1, 500},
			valid:    []int{5, 6, 7},
			decodeW:  MixedStepDecodeWidth(0),
			want:     MixedStepSplit{500, 7, 2, 2, 1, 6, 500, 7, 7},
			wantDec:  []int{0, 1},
			wantPref: []int{2},
		},
		{
			// Seven drafts put eight-token rows in the decode view; a chunk
			// just past the decode width stays in the prefill view.
			name:     "seven-drafts-put-eight-token-rows-in-decode-view",
			widths:   []int{607, 8, 8},
			valid:    []int{700, 40, 50},
			decodeW:  MixedStepDecodeWidth(7),
			want:     MixedStepSplit{607, 700, 1, 607, 607, 700, 8, 50, 700},
			wantDec:  []int{1, 2},
			wantPref: []int{0},
		},
		{
			name:     "eight-token-rows-then-wider-row",
			widths:   []int{8, 8, 9},
			valid:    []int{1, 2, 3},
			decodeW:  MixedStepDecodeWidth(7),
			want:     MixedStepSplit{9, 3, 2, 16, 8, 2, 9, 3, 3},
			wantDec:  []int{0, 1},
			wantPref: []int{2},
		},
		{
			// All rows at exactly the decode cap: no prefill row, unsplit.
			name:     "all-eight-token-rows-no-split",
			widths:   []int{8, 8, 8},
			valid:    []int{1, 2, 3},
			decodeW:  MixedStepDecodeWidth(7),
			want:     MixedStepSplit{8, 3, 0, 0, 0, 0, 0, 0, 0},
			unsplit:  true,
			wantPref: []int{0, 1, 2},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MixedStepSplitRecord(tc.widths, tc.valid, tc.decodeW)
			if err != nil {
				t.Fatalf("MixedStepSplitRecord(%v, %v, %d) unexpected error: %v", tc.widths, tc.valid, tc.decodeW, err)
			}
			if got != tc.want {
				t.Fatalf("MixedStepSplitRecord(%v, %v, %d) = %v, want %v", tc.widths, tc.valid, tc.decodeW, got, tc.want)
			}
			if got.IsUnsplit() != tc.unsplit {
				t.Fatalf("IsUnsplit() = %v, want %v (record %v)", got.IsUnsplit(), tc.unsplit, got)
			}
			if got.HeadRows() != int(got[MixedStepSlotHeadRows]) {
				t.Fatalf("HeadRows() = %d disagrees with slot %d = %d", got.HeadRows(), MixedStepSlotHeadRows, got[MixedStepSlotHeadRows])
			}

			// Fail-open routing: a split-aware op takes the view whose max
			// query length fits its decode width to the decode route and
			// the rest to the prefill route; an unsplit record sends every
			// row down the prefill route, which is correct, only slower.
			dec, pref := opRoute(got, len(tc.widths), tc.decodeW)
			if !equalInts(dec, tc.wantDec) || !equalInts(pref, tc.wantPref) {
				t.Fatalf("opRoute(%v, %d) = decode %v / prefill %v, want decode %v / prefill %v", got, len(tc.widths), dec, pref, tc.wantDec, tc.wantPref)
			}
		})
	}

	t.Run("mismatched-lengths-refused", func(t *testing.T) {
		got, err := MixedStepSplitRecord([]int{4, 500}, []int{1}, w)
		if !errors.Is(err, ErrMixedStepSplitMismatch) {
			t.Fatalf("MixedStepSplitRecord mismatch: err = %v, want errors.Is %v", err, ErrMixedStepSplitMismatch)
		}
		var zero MixedStepSplit
		if got != zero {
			t.Fatalf("refused call returned record %v, want zero value", got)
		}
	})
}

// opRoute mirrors how a split-aware attention op consumes the record: the
// view whose max query length fits the op's decode width takes the decode
// route, the other view takes the prefill route, and an unsplit (or
// inconsistent) record keeps the unsplit route for the whole batch. In a
// valid split exactly one view fits: the winning end run is all decode-width
// rows while the other view holds at least one row wider than the cap.
func opRoute(rec MixedStepSplit, n, decodeWidth int) (decode, prefill []int) {
	if rec.IsUnsplit() {
		for i := range n {
			prefill = append(prefill, i)
		}
		return nil, prefill
	}
	r := rec.HeadRows()
	headFits := int(rec[MixedStepSlotHeadMaxQuery]) <= decodeWidth
	tailFits := int(rec[MixedStepSlotTailMaxQuery]) <= decodeWidth
	switch {
	case headFits && !tailFits:
		for i := 0; i < r; i++ {
			decode = append(decode, i)
		}
		for i := r; i < n; i++ {
			prefill = append(prefill, i)
		}
	case tailFits && !headFits:
		for i := 0; i < r; i++ {
			prefill = append(prefill, i)
		}
		for i := r; i < n; i++ {
			decode = append(decode, i)
		}
	default:
		// Inconsistent record: fail open to the unsplit route.
		for i := range n {
			prefill = append(prefill, i)
		}
	}
	return decode, prefill
}
