//go:build cuda && cgo

package ggufload

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// ink2AppendTokens is a deterministic token stream over the vocabulary.
func ink2AppendTokens(n, vocab int, seed uint64) []int {
	out := make([]int, n)
	x := seed*2654435761 + 1
	for i := range out {
		x = x*6364136223846793005 + 1442695040888963407
		out[i] = int((x >> 33) % uint64(vocab))
	}
	return out
}

// ink2Top2 returns the two largest logits; their gap is the greedy margin.
func ink2Top2(lg []float32) (top1, top2 float32) {
	top1, top2 = float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range lg {
		if v > top1 {
			top1, top2 = v, top1
		} else if v > top2 {
			top2 = v
		}
	}
	return top1, top2
}

// ink2Diff counts bit-unequal logits and the largest absolute gap.
func ink2Diff(a, b []float32) (differ int, maxAbs float64) {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			differ++
		}
		maxAbs = math.Max(maxAbs, math.Abs(float64(a[i])-float64(b[i])))
	}
	return differ, maxAbs
}

// ink2AppendArm is the greedy decode of one arm: the token fed at each step and
// the logits that predicted it.
type ink2AppendArm struct {
	name  string
	toks  []int
	lgs   [][]float32
	first []float32 // logits after the appended prompt, before decode
}

// ink2AppendDecode decodes gen greedy tokens the way serve does: a step runs only
// between emitted tokens, so the decode after a checkpoint takes gen-1 steps.
func ink2AppendDecode(t *testing.T, r *inkling2ResidentSession, next, gen int) (toks []int, lgs [][]float32) {
	t.Helper()
	for i := 0; i < gen; i++ {
		lg, err := r.s.lastLogits()
		if err != nil {
			t.Fatal(err)
		}
		lgs = append(lgs, lg)
		toks = append(toks, next)
		if i < gen-1 {
			if next, err = r.Step(next); err != nil {
				t.Fatal(err)
			}
		}
	}
	return toks, lgs
}

// ink2AppendPrefill runs the serve split for a prompt on a session: everything up
// to the last token as one span, then the last token alone.
func ink2AppendPrefill(t *testing.T, r *inkling2ResidentSession, prompt []int) int {
	t.Helper()
	next, err := r.Prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	return next
}

// TestInkling2CUDAPrefixAppendParity separates numerics from state for the prefix
// reuse path, where an appended turn must decode as a fresh prefill would.
//
//	F  fresh session, prompt P3 as serve runs it with reuse off
//	X  fresh session, P1[:K] as its own span, then the rest of P3 in the reuse
//	   split, with no checkpoint and no decode (the reuse arm minus the restore)
//	S  reuse arm: P1 then a checkpoint at K, a decode, Restore to K, then the rest
//	   of P3
//
// X and S run the same kernels on the same spans, so any bit difference between
// them is restore state (a bug). F against X is the split numerics: a different
// span boundary, which can flip a near tie. Logits are compared bit for bit at
// every decode step, and the greedy margin is logged at the first token that
// differs.
func TestInkling2CUDAPrefixAppendParity(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFIX_APPEND_PARITY")
	defer m.Close()
	if !inkling2ResidentEligible(m, be) {
		t.Skip("resident forward not eligible on this backend")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	const gen = 64 // serve's maxNew
	var p1, p3 []int
	if dir := os.Getenv("FAK_INKLING2_PREFIX_APPEND_TEXTS"); dir != "" {
		p1, p3 = ink2AppendTextPrompts(t, m, dir)
	} else {
		vocab := m.HP.VocabUnpadded
		p1 = ink2AppendTokens(1500, vocab, 1)
		p3 = append(append([]int{}, p1...), ink2AppendTokens(200, vocab, 2)...)
	}
	n1 := len(p1)
	K, L := n1-1, len(p3)
	if !ink2IsPrefix(p1[:K], p3) {
		t.Fatalf("step-1 prompt minus its last token is not a prefix of step 3 (lcp %d of %d)", ink2LCP(p1[:K], p3), K)
	}

	open := func() *inkling2ResidentSession {
		s, err := NewInkling2LoopSession(m, be)
		if err != nil {
			t.Fatal(err)
		}
		return s.(*inkling2ResidentSession)
	}
	run := func(name string, body func(r *inkling2ResidentSession) int) ink2AppendArm {
		r := open()
		defer r.Close()
		next := body(r)
		first, err := r.s.lastLogits()
		if err != nil {
			t.Fatal(err)
		}
		toks, lgs := ink2AppendDecode(t, r, next, gen)
		return ink2AppendArm{name: name, toks: toks, lgs: lgs, first: first}
	}

	F := run("F fresh-serve-split", func(r *inkling2ResidentSession) int {
		ink2AppendPrefill(t, r, p3[:L-1])
		return ink2AppendPrefill(t, r, p3[L-1:])
	})
	X := run("X fresh-split-no-restore", func(r *inkling2ResidentSession) int {
		ink2AppendPrefill(t, r, p3[:K])
		ink2AppendPrefill(t, r, p3[K:L-1])
		return ink2AppendPrefill(t, r, p3[L-1:])
	})
	S := run("S reuse-checkpoint-restore", func(r *inkling2ResidentSession) int {
		ink2AppendPrefill(t, r, p1[:K])
		if err := r.Checkpoint(gen); err != nil {
			t.Fatal(err)
		}
		next1 := ink2AppendPrefill(t, r, p1[K:])
		ink2AppendDecode(t, r, next1, gen) // the first conversation's decode, which the restore must undo
		if err := r.Restore(); err != nil {
			t.Fatalf("restore: %v", err)
		}
		if r.Pos() != K {
			t.Fatalf("restore pos=%d want %d", r.Pos(), K)
		}
		ink2AppendPrefill(t, r, p3[K:L-1])
		return ink2AppendPrefill(t, r, p3[L-1:])
	})

	report := func(a, b ink2AppendArm) {
		differ, maxAbs := ink2Diff(a.first, b.first)
		t.Logf("%s vs %s: first logits differ=%d/%d maxAbs=%g argmax-equal=%v",
			a.name, b.name, differ, len(a.first), maxAbs, ink2ArgMax(a.first) == ink2ArgMax(b.first))
		for i := range a.toks {
			if a.toks[i] == b.toks[i] {
				continue
			}
			a1, a2 := ink2Top2(a.lgs[i])
			b1, b2 := ink2Top2(b.lgs[i])
			t.Logf("%s vs %s: first token diff at step %d; margin %s=%.4f nat, %s=%.4f nat",
				a.name, b.name, i, a.name, a1-a2, b.name, b1-b2)
			return
		}
		t.Logf("%s vs %s: %d decode tokens equal", a.name, b.name, len(a.toks))
	}
	report(F, X)
	report(X, S)
	report(F, S)

	// the restore path must equal the split-matched fresh path bit for bit
	for i := range X.lgs {
		if X.toks[i] != S.toks[i] {
			t.Fatalf("state bug: S token %d=%d, X=%d", i, S.toks[i], X.toks[i])
		}
		if differ, maxAbs := ink2Diff(X.lgs[i], S.lgs[i]); differ != 0 {
			t.Fatalf("state bug: decode step %d logits differ in %d/%d (maxAbs %g)", i, differ, len(X.lgs[i]), maxAbs)
		}
	}
	if _, maxAbs := ink2Diff(X.first, S.first); maxAbs != 0 {
		t.Fatalf("state bug: appended-prompt logits differ (maxAbs %g)", maxAbs)
	}
}

func ink2ArgMax(lg []float32) int {
	best := 0
	for i, v := range lg {
		if v > lg[best] {
			best = i
		}
	}
	return best
}

// ink2IsPrefix reports whether a is a leading run of b.
func ink2IsPrefix(a, b []int) bool { return len(a) <= len(b) && ink2LCP(a, b) == len(a) }

// ink2AppendTextPrompts tokenizes the protocol's step-1 and step-3 texts (dir
// holds step1.txt and step3.txt) with the GGUF's embedded tokenizer. No chat
// template is applied, so the token IDs can differ from what the server renders.
func ink2AppendTextPrompts(t *testing.T, m *Inkling2LoopModel, dir string) (p1, p3 []int) {
	t.Helper()
	read := func(name string) string {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	f, err := Open(os.Getenv("FAK_INKLING2_GGUF"))
	if err != nil {
		t.Fatal(err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("no embedded tokenizer in the GGUF")
	}
	tok, err := tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatal(err)
	}
	if p1, err = tok.Encode(read("step1.txt")); err != nil {
		t.Fatal(err)
	}
	if p3, err = tok.Encode(read("step3.txt")); err != nil {
		t.Fatal(err)
	}
	t.Logf("protocol prompts: step1=%d tokens, step3=%d tokens", len(p1), len(p3))
	return p1, p3
}
