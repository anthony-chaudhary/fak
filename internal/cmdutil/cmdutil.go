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
	"time"
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
// sorting so the caller's slice is left untouched. ds must be non-empty.
func MedianMS(ds []time.Duration) float64 {
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
// n <= 0. Pass seed 0 to reproduce the unseeded copies.
func LCGIDs(n, vocab int, seed uint64) []int {
	if n <= 0 {
		return nil
	}
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}
