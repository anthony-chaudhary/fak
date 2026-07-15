package radixkv

import (
	"slices"
	"testing"
)

// gallopRunLen is SGLang's RadixKey.match run-compare — gallop in doubling windows
// to bracket the first divergence, then binary-search the single straddling window
// (issue #3891). It returns EXACTLY refRunLen's matched length; it lives in test
// scope, not walk(), because BenchmarkCommonRunLen below shows the borrow does not
// win in Go (see the walk() comment). This is the retirement witness: correct
// (TestCommonRunLenMatchesReference) but not faster, kept so the measurement is
// reproducible and the dead end is not silently re-attempted.
func gallopRunLen(key, toks []int) int {
	n := len(key)
	if len(toks) < n {
		n = len(toks)
	}
	lo, step := 0, 1
	for lo < n {
		hi := lo + step
		if hi > n {
			hi = n
		}
		if !slices.Equal(key[lo:hi], toks[lo:hi]) {
			// First mismatch is inside [lo, hi); key[:lo] already matches toks[:lo].
			l, h := lo, hi
			for l < h {
				mid := int(uint(l+h) >> 1)
				if slices.Equal(key[l:mid+1], toks[l:mid+1]) {
					l = mid + 1
				} else {
					h = mid
				}
			}
			return l
		}
		lo = hi
		step *= 2
	}
	return n
}

// BenchmarkCommonRunLen is the #3891 measurement: matching an identical edge run
// (the ~2000-token shared system-prompt case) and an early-divergence case, gallop
// vs the token-by-token reference. On Go []int the gallop is a wash on a full match
// (the doubling windows sum to the same element compares) and ~20% slower on early
// divergence (extra slices.Equal call overhead with no vectorization payoff), so
// walk() keeps the token-by-token loop.
func BenchmarkCommonRunLen(b *testing.B) {
	key := make([]int, 2048)
	for i := range key {
		key[i] = i
	}
	full := append([]int(nil), key...) // whole edge matches (identical-prefix case)
	div := append([]int(nil), key...)  // diverges 1/4 in
	div[512] = -1

	cases := []struct {
		name string
		toks []int
	}{{"full", full}, {"diverge25", div}}
	for _, c := range cases {
		b.Run("gallop/"+c.name, func(b *testing.B) {
			var s int
			for i := 0; i < b.N; i++ {
				s += gallopRunLen(key, c.toks)
			}
			_ = s
		})
		b.Run("reference/"+c.name, func(b *testing.B) {
			var s int
			for i := 0; i < b.N; i++ {
				s += refRunLen(key, c.toks)
			}
			_ = s
		})
	}
}
