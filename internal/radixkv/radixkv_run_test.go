package radixkv

import (
	"math/rand"
	"testing"
)

// refRunLen is the token-by-token reference the issue #3891 gallop/binary-search
// path must match exactly (the pre-optimization inner loop of walk).
func refRunLen(key, toks []int) int {
	j := 0
	for j < len(key) && j < len(toks) && toks[j] == key[j] {
		j++
	}
	return j
}

// TestCommonRunLenMatchesReference is the no-behavior-change witness for #3891:
// the gallop-then-binary-search commonRunLen must return the SAME matched length
// as the token-by-token reference on every input, so walk is byte-identical.
func TestCommonRunLenMatchesReference(t *testing.T) {
	// Exhaustive small cases over a tiny alphabet so many prefixes collide and
	// diverge at every possible position (0..len), plus length mismatches.
	for keyLen := 0; keyLen <= 6; keyLen++ {
		for tokLen := 0; tokLen <= 6; tokLen++ {
			key := make([]int, keyLen)
			toks := make([]int, tokLen)
			// enumerate all binary (alphabet {0,1}) fillings of key and toks
			for kmask := 0; kmask < 1<<keyLen; kmask++ {
				for b := 0; b < keyLen; b++ {
					key[b] = (kmask >> b) & 1
				}
				for tmask := 0; tmask < 1<<tokLen; tmask++ {
					for b := 0; b < tokLen; b++ {
						toks[b] = (tmask >> b) & 1
					}
					if got, want := commonRunLen(key, toks), refRunLen(key, toks); got != want {
						t.Fatalf("commonRunLen(%v,%v)=%d, ref=%d", key, toks, got, want)
					}
				}
			}
		}
	}

	// Randomized larger cases, including long identical prefixes with a single
	// planted divergence at a random position — the hot path #3891 optimizes.
	rng := rand.New(rand.NewSource(3891))
	for iter := 0; iter < 20000; iter++ {
		n := rng.Intn(300)
		shared := rng.Intn(n + 1) // length of the guaranteed-common prefix
		alpha := 1 + rng.Intn(4)  // small alphabet => frequent coincidental matches
		key := make([]int, n)
		toks := make([]int, rng.Intn(300))
		for i := range key {
			key[i] = rng.Intn(alpha)
		}
		for i := range toks {
			if i < shared && i < len(key) {
				toks[i] = key[i] // force a common prefix of length ~shared
			} else {
				toks[i] = rng.Intn(alpha)
			}
		}
		if got, want := commonRunLen(key, toks), refRunLen(key, toks); got != want {
			t.Fatalf("iter %d: commonRunLen=%d ref=%d\nkey=%v\ntoks=%v", iter, got, want, key, toks)
		}
	}

	// A fully-identical long run (the ~2000-token system-prompt case): the whole
	// edge matches, so no binary search runs and the result is min(len,len).
	long := make([]int, 2048)
	for i := range long {
		long[i] = i % 131
	}
	cp := append([]int(nil), long...)
	if got := commonRunLen(long, cp); got != len(long) {
		t.Fatalf("full-run commonRunLen=%d, want %d", got, len(long))
	}
	if got := commonRunLen(long, cp[:1000]); got != 1000 {
		t.Fatalf("truncated-run commonRunLen=%d, want 1000", got)
	}
}
