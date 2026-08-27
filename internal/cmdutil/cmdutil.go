// Package cmdutil holds small, behavior-identical helpers that were copy-pasted
// across the cmd/* demo and bench mains (argmax over logits, the LCG token-id
// generator, duration medians, the HTTP JSON writer). Extracting the one shared
// copy each keeps the binaries byte-for-byte equivalent while removing the
// duplicated bodies the slop scorecard flags.
package cmdutil

import (
	"encoding/json"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchids"
)

// Argmax returns the index of the first maximal element of v, or 0 when v is
// empty. Every cmd/* copy computed the first-argmax; this version additionally
// guards the empty slice (the few copies that indexed v[0] would have panicked).
func Argmax(v []float32) int {
	if len(v) == 0 {
		return 0
	}
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// Ms converts a duration to fractional milliseconds.
func Ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// MedianMS returns the median of ds in fractional milliseconds. It copies before
// sorting so the caller's slice is left untouched. An empty ds returns 0 — the
// same degenerate-input guard Argmax applies, so a caller that forgets the
// non-empty precondition gets 0 instead of an index-out-of-range panic.
func MedianMS(ds []time.Duration) float64 {
	if len(ds) == 0 {
		return 0
	}
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[len(cp)/2].Nanoseconds()) / 1e6
}

// WriteJSON writes v as a JSON response with the application/json content type.
func WriteJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// LCGIDs generates n pseudo-random token ids in [0,vocab) from a 32-bit LCG
// seeded by seed (added to the classic xorshift constant). It returns nil for
// n <= 0 or vocab <= 0 — an empty id range has no valid ids, and a vocab of 0
// would divide by zero in the modulo below. Pass seed 0 to reproduce the
// unseeded copies.
func LCGIDs(n, vocab int, seed uint64) []int {
	if n <= 0 || vocab <= 0 {
		return nil
	}
	return benchids.LCG(n, vocab, seed)
}

// CapPositive caps n when cap is positive and otherwise returns at least one.
func CapPositive(n, cap int) int {
	if cap > 0 && n > cap {
		return cap
	}
	if n < 1 {
		return 1
	}
	return n
}

// MaxAbsDiffF32 returns the largest absolute difference over the shorter input.
func MaxAbsDiffF32(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var maxDiff float64
	for i := 0; i < n; i++ {
		diff := float64(a[i] - b[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

// MarkdownCell escapes table separators and flattens line breaks while
// preserving all other bytes for Markdown table cells.
func MarkdownCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}
