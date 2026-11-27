//go:build darwin && arm64 && cgo

package main

import (
	"context"
	"os"
	"testing"
)

// TestInkling2MetalServePrefixReuse runs a growing multi-turn conversation, then a
// branch that shares only part of the earlier prompt, through a reuse-on engine and
// a reuse-off engine on one resident Metal model. Every turn must emit the same tokens,
// and the reuse-on turns after the first must report cached prompt tokens.
func TestInkling2MetalServePrefixReuse(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to an Inkling-2-Mini GGUF")
	}
	m, be, err := openInkling2MetalResident(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	filler := func(seed, n int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = 1000 + (seed+i*7919)%50000
		}
		return out
	}
	const maxNew = 8
	var turns [][]int
	prompt := filler(1, 1024)
	turns = append(turns, prompt)
	for k := 2; k <= 4; k++ {
		prompt = append(append([]int(nil), prompt...), filler(k, 256)...)
		turns = append(turns, prompt)
	}
	// a branch that shares 600 tokens, which is not a chunk boundary
	turns = append(turns, append(append([]int(nil), turns[3][:600]...), filler(9, 300)...))

	run := func(reuse string) ([][]int, []int) {
		t.Setenv("FAK_INKLING2_PREFIX_REUSE", reuse)
		eng := newInkling2LoopEngine(m, be)
		defer eng.Close()
		var outs [][]int
		var cached []int
		for _, p := range turns {
			var got []int
			res, err := eng.Generate(context.Background(), p, maxNew, nil, func(id int) bool {
				got = append(got, id)
				return true
			})
			if err != nil {
				t.Fatalf("turn of %d tokens with reuse %s: %v", len(p), reuse, err)
			}
			outs = append(outs, got)
			cached = append(cached, res.CachedPromptTokens)
		}
		return outs, cached
	}

	on, cachedOn := run("on")
	off, cachedOff := run("off")
	for i := range turns {
		if !inkling2EqualInts(on[i], off[i]) {
			t.Fatalf("turn %d tokens differ with reuse: on %v, off %v", i, on[i], off[i])
		}
		if cachedOff[i] != 0 {
			t.Fatalf("turn %d reused %d prompt tokens with reuse off", i, cachedOff[i])
		}
	}
	if cachedOn[0] != 0 {
		t.Fatalf("first turn reused %d prompt tokens on a fresh engine", cachedOn[0])
	}
	for i := 1; i < 4; i++ {
		if cachedOn[i] == 0 {
			t.Fatalf("turn %d reused no prompt tokens with reuse on", i)
		}
	}
	t.Logf("cached prompt tokens with reuse on: %v", cachedOn)
}
