package main

import (
	"errors"
	"slices"
	"testing"
)

// hashIDTestCorpus is 0..n-1, so a block's first token is its corpus offset.
func hashIDTestCorpus(n int) []int {
	corpus := make([]int, n)
	for i := range corpus {
		corpus[i] = i
	}
	return corpus
}

func newHashIDTestCache(t *testing.T, cfg HashIDBlockCacheConfig) *HashIDBlockCache {
	t.Helper()
	if cfg.Corpus == nil {
		cfg.Corpus = hashIDTestCorpus(10007)
	}
	c, err := NewHashIDBlockCache(cfg)
	if err != nil {
		t.Fatalf("NewHashIDBlockCache: %v", err)
	}
	return c
}

func mustBuild(t *testing.T, c *HashIDBlockCache, numTokens int, ids []int64, blockSize int) ([]int, HashIDLayout) {
	t.Helper()
	tokens, layout, err := c.Build(numTokens, ids, blockSize)
	if err != nil {
		t.Fatalf("Build(%d, %v, %d): %v", numTokens, ids, blockSize, err)
	}
	if len(tokens) != numTokens {
		t.Fatalf("Build(%d, %v, %d) returned %d tokens", numTokens, ids, blockSize, len(tokens))
	}
	return tokens, layout
}

func TestHashIDBlockCache(t *testing.T) {
	t.Run("three layouts", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7})
		c.SetTrace("trace-a")

		exact, layout := mustBuild(t, c, 30, []int64{1, 2, 3}, 10)
		if layout != HashIDLayoutExactTile {
			t.Fatalf("exact tile layout = %v", layout)
		}

		partial, layout := mustBuild(t, c, 24, []int64{1, 2, 9}, 10)
		if layout != HashIDLayoutPartialLastBlock {
			t.Fatalf("partial layout = %v", layout)
		}
		if !slices.Equal(partial[:20], exact[:20]) {
			t.Fatal("partial layout did not reuse the two full blocks")
		}
		if c.Len() != 4 {
			t.Fatalf("resident blocks = %d, want 4 (ids 1,2,3,9)", c.Len())
		}

		prefix, layout := mustBuild(t, c, 37, []int64{1, 2, 3}, 10)
		if layout != HashIDLayoutPrefixOnly {
			t.Fatalf("prefix-only layout = %v", layout)
		}
		if !slices.Equal(prefix[:30], exact) {
			t.Fatal("prefix-only layout did not reuse the hashed prefix")
		}
		if c.Len() != 4 {
			t.Fatalf("tail pad was cached: resident blocks = %d, want 4", c.Len())
		}
		again, _ := mustBuild(t, c, 37, []int64{1, 2, 3}, 10)
		if slices.Equal(again[30:], prefix[30:]) {
			t.Fatal("tail pad repeated across requests; it must be a fresh draw")
		}

		unhashed, layout := mustBuild(t, c, 12, nil, 10)
		if layout != HashIDLayoutUnhashed || len(unhashed) != 12 {
			t.Fatalf("unhashed layout = %v len = %d", layout, len(unhashed))
		}
	})

	t.Run("per trace reseed", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7})
		c.SetTrace("trace-a")
		a, _ := mustBuild(t, c, 20, []int64{41, 42}, 10)

		c.SetTrace("trace-b")
		if c.Len() != 0 {
			t.Fatalf("SetTrace kept %d blocks", c.Len())
		}
		b, _ := mustBuild(t, c, 20, []int64{41, 42}, 10)
		if slices.Equal(a, b) {
			t.Fatal("same hash ids under two trace ids produced identical tokens")
		}

		c.SetTrace("trace-a")
		back, _ := mustBuild(t, c, 20, []int64{41, 42}, 10)
		if !slices.Equal(a, back) {
			t.Fatal("re-entering a trace did not reproduce its blocks")
		}

		// A second cache stands in for another process replaying the trace.
		other := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7})
		other.SetTrace("trace-a")
		replay, _ := mustBuild(t, other, 20, []int64{41, 42}, 10)
		if !slices.Equal(a, replay) {
			t.Fatal("two caches with one seed and trace id disagreed")
		}

		reseeded := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 8})
		reseeded.SetTrace("trace-a")
		diff, _ := mustBuild(t, reseeded, 20, []int64{41, 42}, 10)
		if slices.Equal(a, diff) {
			t.Fatal("a different seed produced identical tokens")
		}
	})

	t.Run("entry cap evicts least recently used", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7, MaxEntries: 2})
		if c.Cap() != 2 {
			t.Fatalf("Cap = %d, want 2", c.Cap())
		}
		c.SetTrace("trace-a")
		one, _ := mustBuild(t, c, 10, []int64{1}, 10)
		mustBuild(t, c, 10, []int64{2}, 10)
		mustBuild(t, c, 10, []int64{1}, 10) // touch 1: id 2 is now the oldest
		mustBuild(t, c, 10, []int64{3}, 10)

		if c.Len() != 2 {
			t.Fatalf("resident blocks = %d, want the cap of 2", c.Len())
		}
		if !c.Contains(1) || c.Contains(2) || !c.Contains(3) {
			t.Fatalf("resident = {1:%v 2:%v 3:%v}, want id 2 evicted",
				c.Contains(1), c.Contains(2), c.Contains(3))
		}

		mustBuild(t, c, 10, []int64{2}, 10) // evicts 1
		if c.Contains(1) {
			t.Fatal("id 1 survived an insert past the cap")
		}
		regenerated, _ := mustBuild(t, c, 10, []int64{1}, 10)
		if !slices.Equal(one, regenerated) {
			t.Fatal("an evicted block regenerated to different tokens")
		}

		// More ids in one request than the cap still assembles every block.
		wide, _ := mustBuild(t, c, 50, []int64{10, 11, 12, 13, 14}, 10)
		if c.Len() != 2 {
			t.Fatalf("resident blocks after a wide request = %d, want 2", c.Len())
		}
		eleven, _ := mustBuild(t, c, 10, []int64{11}, 10)
		if !slices.Equal(wide[10:20], eleven) {
			t.Fatal("a block evicted mid-request was not the one emitted")
		}
	})

	t.Run("layout names", func(t *testing.T) {
		for layout, want := range map[HashIDLayout]string{
			HashIDLayoutUnhashed:         "unhashed",
			HashIDLayoutExactTile:        "exact_tile",
			HashIDLayoutPartialLastBlock: "partial_last_block",
			HashIDLayoutPrefixOnly:       "prefix_only",
			HashIDLayout(99):             "HashIDLayout(99)",
		} {
			if got := layout.String(); got != want {
				t.Errorf("HashIDLayout(%d).String() = %q, want %q", int(layout), got, want)
			}
		}
	})

	t.Run("default entry cap", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{})
		if c.Cap() != DefaultHashIDBlockCacheEntries {
			t.Fatalf("Cap = %d, want %d", c.Cap(), DefaultHashIDBlockCacheEntries)
		}
	})

	t.Run("size conflict", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7})
		mustBuild(t, c, 14, []int64{1, 2}, 10) // id 2 cached at 4 tokens
		_, _, err := c.Build(20, []int64{1, 2}, 10)
		if !errors.Is(err, ErrHashIDBlockSizeConflict) {
			t.Fatalf("err = %v, want ErrHashIDBlockSizeConflict", err)
		}
	})

	t.Run("size conflict covers resident blocks only", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7, MaxEntries: 1})
		short, _ := mustBuild(t, c, 4, []int64{2}, 10) // id 2 cached at 4 tokens
		mustBuild(t, c, 10, []int64{3}, 10)            // evicts id 2
		full, _ := mustBuild(t, c, 10, []int64{2}, 10)
		if !slices.Equal(full[:4], short) {
			t.Fatal("an evicted block regenerated at a new size from a different offset")
		}
	})

	t.Run("separator", func(t *testing.T) {
		sep := -1
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Seed: 7, SeparatorToken: &sep})
		tokens, _ := mustBuild(t, c, 23, []int64{1, 2, 3}, 10)
		for _, at := range []int{0, 10, 20} {
			if tokens[at] != sep {
				t.Fatalf("tokens[%d] = %d, want the separator", at, tokens[at])
			}
		}
		if n := countOf(tokens, sep); n != 3 {
			t.Fatalf("separator appears %d times, want once per block", n)
		}
		if _, _, err := c.Build(3, []int64{1, 2, 3}, 1); !errors.Is(err, ErrHashIDBlockLayout) {
			t.Fatalf("block size 1 with a separator: err = %v", err)
		}
	})

	t.Run("block longer than corpus wraps", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{Corpus: []int{5, 6, 7}})
		tokens, _ := mustBuild(t, c, 8, []int64{1}, 8)
		for i := 1; i < len(tokens); i++ {
			if want := 5 + (tokens[i-1]-5+1)%3; tokens[i] != want {
				t.Fatalf("tokens = %v: not a wrapped corpus walk at %d", tokens, i)
			}
		}
	})

	t.Run("rejects incompatible input", func(t *testing.T) {
		c := newHashIDTestCache(t, HashIDBlockCacheConfig{})
		for name, call := range map[string]func() error{
			"zero tokens":     func() error { _, _, err := c.Build(0, []int64{1}, 10); return err },
			"zero block size": func() error { _, _, err := c.Build(10, []int64{1}, 0); return err },
			"block size that would wrap": func() error {
				_, _, err := c.Build(10, []int64{1, 2, 3}, 1<<62)
				return err
			},
			"overshoot by a whole block": func() error {
				_, _, err := c.Build(10, []int64{1, 2, 3}, 10)
				return err
			},
			"empty corpus": func() error {
				_, err := NewHashIDBlockCache(HashIDBlockCacheConfig{})
				return err
			},
			"negative cap": func() error {
				_, err := NewHashIDBlockCache(HashIDBlockCacheConfig{Corpus: []int{1}, MaxEntries: -1})
				return err
			},
		} {
			if err := call(); !errors.Is(err, ErrHashIDBlockLayout) {
				t.Errorf("%s: err = %v, want ErrHashIDBlockLayout", name, err)
			}
		}
	})
}

func countOf(tokens []int, want int) int {
	n := 0
	for _, tok := range tokens {
		if tok == want {
			n++
		}
	}
	return n
}
