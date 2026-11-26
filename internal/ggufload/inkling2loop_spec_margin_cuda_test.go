//go:build cuda && cgo

package ggufload

import (
	"context"
	"runtime"
	"testing"
)

// FAK_INKLING2_SPEC_MARGIN=1 feeds the plain greedy continuation through the depth-4
// batched verify (every draft is the true token, so every verify row sits on the
// plain prefix) and logs, for the watched output positions and for every row that
// disagrees with plain greedy, the verify row's top-1, top-2 and logit gap next to
// the plain decode's gap at the same position. The gap is in logprob nats, since the
// argmax kernel softcaps the logits in place before they are read back.
func TestInkling2CUDASpecMarginDump(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_SPEC_MARGIN")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	prompt := inkling2OracleCases[0].prompt
	const n, depth = 128, 4
	ref, plainRU, plainMargin, err := inkling2ResidentGreedyMargins(m, be, prompt, n)
	if err != nil {
		t.Fatal(err)
	}
	watch := map[int]bool{104: true, 111: true}
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := s.enableSpec(depth); err != nil {
		t.Fatal(err)
	}
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	if next != ref[0] {
		t.Fatalf("prefill next %d, plain greedy %d", next, ref[0])
	}
	T := depth + 1
	c, mismatches, first := 0, 0, -1
	for c+T+1 <= len(ref) {
		g, err := s.verify(ref[c : c+T])
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < T; i++ {
			k := c + i + 1 // output position this row predicts
			logits, err := s.dev.Ink2Read(at(s.verifyLogits, i*s.output.rows), s.hp.VocabUnpadded)
			if err != nil {
				t.Fatal(err)
			}
			top, second := inkling2TopTwo(logits)
			gap := float64(logits[top] - logits[second])
			if top != ref[k] {
				mismatches++
				if first < 0 {
					first = k
				}
			}
			if watch[k] || top != ref[k] {
				t.Logf("spec margin pos %d: verify argmax %d (greedy %d, row argmax %d) top2 %d gap %.4f nats; plain step top2 %d gap %.4f nats; verify argmax==plain %v",
					k, g[i], ref[k], top, second, gap, plainRU[k], plainMargin[k], top == ref[k])
			}
		}
		if err := s.commitVerify(T, T); err != nil {
			t.Fatal(err)
		}
		c += T
	}
	t.Logf("spec margin dump: depth %d, %d verify rows, %d row(s) disagree with plain greedy, first disagreeing position %d",
		depth, c, mismatches, first)
}
