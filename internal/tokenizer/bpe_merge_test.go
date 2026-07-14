package tokenizer

import (
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// bpeNaive is the reference O(n^2) merge loop bpe replaced (#4263): every pass re-scans
// all adjacent pairs for the single lowest-rank merge, applies every non-overlapping
// occurrence left-to-right, and rebuilds the slice. It is the exact pre-change body,
// kept here as the oracle the incremental bpe must match bit-for-bit.
func bpeNaive(mergeRank map[tokenPair]int, encoded string) []string {
	var syms []string
	for _, r := range encoded {
		syms = append(syms, string(r))
	}
	if len(syms) == 0 {
		return nil
	}
	for {
		bestRank := len(mergeRank)
		best := tokenPair{}
		found := false
		for i := 0; i+1 < len(syms); i++ {
			pair := tokenPair{left: syms[i], right: syms[i+1]}
			rank, ok := mergeRank[pair]
			if ok && rank < bestRank {
				bestRank = rank
				best = pair
				found = true
			}
		}
		if !found {
			return syms
		}
		next := syms[:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == best.left && syms[i+1] == best.right {
				next = append(next, best.left+best.right)
				i++
				continue
			}
			next = append(next, syms[i])
		}
		syms = next
	}
}

// doublingMergeRank is a monotone merge table (each rule's output is only consumed by a
// strictly later rule, exactly as a real HF/GGML table) that progressively pairs the
// metaspace marker: ▁+▁ -> ▁▁, ▁▁+▁▁ -> ▁▁▁▁, and so on, plus a few letter merges. A run
// of k markers therefore collapses through ~log(k) rank levels with many occurrences per
// level — the metaspace-indentation shape #4263 targets, and the naive path's worst case.
func doublingMergeRank() map[tokenPair]int {
	mr := map[tokenPair]int{}
	rank := 0
	unit := "▁" // ▁
	for i := 0; i < 10; i++ {
		mr[tokenPair{left: unit, right: unit}] = rank
		rank++
		unit += unit
	}
	// A short chain over letters so the table is not spaces-only.
	for _, m := range []tokenPair{{"d", "e"}, {"de", "f"}, {"f", "o"}, {"fo", "o"}} {
		mr[m] = rank
		rank++
	}
	return mr
}

// TestBPEIncrementalMatchesNaive asserts the incremental merge frontier reproduces the
// naive loop's symbols bit-for-bit across the real fixtures and a long metaspace run —
// the byte-exact acceptance bar of #4263.
func TestBPEIncrementalMatchesNaive(t *testing.T) {
	small, err := ParseJSON([]byte(`{
	  "model": {
	    "type": "BPE",
	    "vocab": {
	      "H": 2, "e": 3, "l": 4, "o": 5, "He": 6, "Hel": 7, "Hell": 8, "Hello": 9,
	      "Ġ": 10, "w": 11, "r": 12, "d": 13, "wo": 14, "wor": 15, "worl": 16, "world": 17, "Ġworld": 18
	    },
	    "merges": ["H e", "He l", "Hel l", "Hell o", "w o", "wo r", "wor l", "worl d", "Ġ world"]
	  },
	  "decoder": {"type": "ByteLevel"}
	}`))
	if err != nil {
		t.Fatalf("ParseJSON: %v", err)
	}
	gemma, err := FromGGML([]string{"a", "b", "▁", "▁b", "<0x0A>"}, []string{"▁ b"}, nil, "gemma4")
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	doubling := &Tokenizer{mergeRank: doublingMergeRank()}

	cases := []struct {
		name    string
		mr      map[tokenPair]int
		encoded string
	}{
		{"empty", small.mergeRank, ""},
		{"single", small.mergeRank, "H"},
		{"hello", small.mergeRank, byteLevelEncode("Hello")},
		{"hello-world", small.mergeRank, byteLevelEncode("Hello world")},
		{"gemma-space-b", gemma.mergeRank, metaspaceEncode(" b b b")},
		{"doubling-1", doubling.mergeRank, strings.Repeat("▁", 1)},
		{"doubling-3", doubling.mergeRank, strings.Repeat("▁", 3)},
		{"doubling-7", doubling.mergeRank, strings.Repeat("▁", 7)},
		{"doubling-64", doubling.mergeRank, strings.Repeat("▁", 64)},
		// A 4-space (metaspace) indented code block repeated — the quadratic-cliff shape.
		{"indent-block", doubling.mergeRank, strings.Repeat(strings.Repeat("▁", 4)+"defoo", 200)},
	}
	for _, tc := range cases {
		tok := &Tokenizer{mergeRank: tc.mr}
		got, err := tok.bpe(tc.encoded)
		if err != nil {
			t.Fatalf("%s: bpe: %v", tc.name, err)
		}
		want := bpeNaive(tc.mr, tc.encoded)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: incremental bpe = %v, naive = %v", tc.name, got, want)
		}
	}
}

// benchInput is a metaspace-indented code block repeated to reach the requested marker
// length, the long single-chunk input Encode hands bpe for a Gemma-style tokenizer.
func benchInput(markers int) string {
	block := strings.Repeat("▁", 4) + "defoo"
	var b strings.Builder
	for b.Len() < markers*3 { // ▁ is 3 bytes in UTF-8
		b.WriteString(block)
	}
	return b.String()
}

// BenchmarkBPEIncremental vs BenchmarkBPENaive show the O(n^2) -> ~O(n log n) improvement:
// the naive ns/op grows ~quadratically with input length while the incremental frontier
// grows near-linearly. Run: go test ./internal/tokenizer -run=X -bench=BPE -benchmem.
func BenchmarkBPEIncremental(b *testing.B) {
	tok := &Tokenizer{mergeRank: doublingMergeRank()}
	for _, n := range []int{64, 256, 1024, 4096} {
		in := benchInput(n)
		b.Run(sizeLabel(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := tok.bpe(in); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBPENaive(b *testing.B) {
	mr := doublingMergeRank()
	for _, n := range []int{64, 256, 1024, 4096} {
		in := benchInput(n)
		b.Run(sizeLabel(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				bpeNaive(mr, in)
			}
		})
	}
}

func sizeLabel(n int) string {
	if n >= 1024 {
		return "n" + strconv.Itoa(n/1024) + "k"
	}
	return "n" + strconv.Itoa(n)
}
