//go:build cuda && cgo

package ggufload

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"
)

// inkling2LogProbs is log-softmax over the unpadded vocab of a head output the
// argmax kernel already softcapped in place.
func inkling2LogProbs(capped []float32, vocab int) []float64 {
	lp := make([]float64, vocab)
	mx := math.Inf(-1)
	for i := range lp {
		lp[i] = float64(capped[i])
		mx = math.Max(mx, lp[i])
	}
	var z float64
	for _, v := range lp {
		z += math.Exp(v - mx)
	}
	lz := mx + math.Log(z)
	for i := range lp {
		lp[i] -= lz
	}
	return lp
}

// inkling2KL is KL(p || q) over two log-probability rows, and whether their argmax agrees.
func inkling2KL(p, q []float64) (float64, bool) {
	var kl float64
	ap, aq := 0, 0
	for i := range p {
		if p[i] > p[ap] {
			ap = i
		}
		if q[i] > q[aq] {
			aq = i
		}
		if pi := math.Exp(p[i]); pi > 0 {
			kl += pi * (p[i] - q[i])
		}
	}
	return kl, ap == aq
}

// inkling2TeacherForcedLogProbs prefills prompt on the selected arm, then steps the
// given continuation, returning the next-token distribution at the prompt end and after
// every forced token.
func inkling2TeacherForcedLogProbs(t *testing.T, m *Inkling2LoopModel, s *inkling2Resident, prompt, forced []int) [][]float64 {
	t.Helper()
	read := func() []float64 {
		raw, err := s.lastLogits()
		if err != nil {
			t.Fatal(err)
		}
		return inkling2LogProbs(raw, m.HP.VocabUnpadded)
	}
	var err error
	if s.batched {
		_, err = s.prefill(context.Background(), prompt)
	} else {
		for _, tok := range prompt {
			if _, err = s.step(tok); err != nil {
				break
			}
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float64{read()}
	for _, tok := range forced {
		if _, err := s.step(tok); err != nil {
			t.Fatal(err)
		}
		out = append(out, read())
	}
	return out
}

// FAK_INKLING2_PREFILL_KL=<steps> measures KL(per-token || batched) of the next-token
// distribution at the end of each oracle prompt (and a 1300-token prompt that crosses
// chunks and wraps the local window), then over <steps> decode positions teacher-forced
// on the per-token greedy continuation, so a cache hand-off error shows up after the
// prompt as well as at its last position.
func TestInkling2CUDABatchedPrefillKL(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFILL_KL")
	defer m.Close()
	steps, err := strconv.Atoi(os.Getenv("FAK_INKLING2_PREFILL_KL"))
	if err != nil || steps < 0 {
		t.Fatalf("FAK_INKLING2_PREFILL_KL=%q: want a decode step count", os.Getenv("FAK_INKLING2_PREFILL_KL"))
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	prompts := []struct {
		name   string
		prompt []int
	}{{"pp1300", inkling2PrefillPrompt(1300)}}
	for _, c := range inkling2OracleCases {
		prompts = append(prompts, struct {
			name   string
			prompt []int
		}{c.name, c.prompt})
	}
	arm := func(batched bool) *inkling2Resident {
		s, err := newInkling2Resident(m, be)
		if err != nil {
			t.Fatal(err)
		}
		s.batched = batched
		return s
	}
	var sum, worst float64
	var n, agree int
	for _, c := range prompts {
		ref := arm(false)
		forced, _, err := inkling2ResidentGreedyOn(ref, c.prompt, steps+1)
		ref.close()
		if err != nil {
			t.Fatal(err)
		}
		forced = forced[:steps]
		ref = arm(false)
		want := inkling2TeacherForcedLogProbs(t, m, ref, c.prompt, forced)
		ref.close()
		cand := arm(true)
		got := inkling2TeacherForcedLogProbs(t, m, cand, c.prompt, forced)
		cand.close()
		var csum, cmax float64
		cagree := 0
		end, _ := inkling2KL(want[0], got[0])
		for i := range want {
			kl, same := inkling2KL(want[i], got[i])
			csum += kl
			cmax = math.Max(cmax, kl)
			if same {
				cagree++
			}
		}
		t.Logf("inkling2 prefill KL: %s pp%d positions %d mean %.6f max %.6f (at prompt end %.6f) same-top %d/%d",
			c.name, len(c.prompt), len(want), csum/float64(len(want)), cmax, end, cagree, len(want))
		sum += csum
		worst = math.Max(worst, cmax)
		n += len(want)
		agree += cagree
	}
	t.Logf("inkling2 prefill KL: all positions %d mean %.6f max %.6f same-top %d/%d (%.2f%%)",
		n, sum/float64(n), worst, agree, n, 100*float64(agree)/float64(n))
	// the per-token path itself moves by ~0.01 nats when only the reduction order
	// changes (llama.cpp's own batch-size sensitivity on this file is KLD 0.0095)
	if mean := sum / float64(n); mean > 0.01 {
		t.Fatalf("mean KL %.6f exceeds 0.01 nats", mean)
	}
}

// inkling2ResidentGreedyOn is the greedy loop of inkling2ResidentGreedy on an open
// session whose prefill arm the caller chose.
func inkling2ResidentGreedyOn(s *inkling2Resident, prompt []int, n int) ([]int, int, error) {
	next, err := 0, error(nil)
	if s.batched {
		next, err = s.prefill(context.Background(), prompt)
	} else {
		for _, tok := range prompt {
			if next, err = s.step(tok); err != nil {
				break
			}
		}
	}
	if err != nil {
		return nil, 0, err
	}
	out := []int{next}
	for len(out) < n {
		if next, err = s.step(next); err != nil {
			return nil, 0, err
		}
		out = append(out, next)
	}
	return out, len(out), nil
}

// inkling2Top2 is the argmax of a log-probability row and its margin over the runner-up.
func inkling2Top2(lp []float64) (int, float64) {
	a, b := 0, 1
	if lp[b] > lp[a] {
		a, b = b, a
	}
	for i := 2; i < len(lp); i++ {
		if lp[i] > lp[a] {
			a, b = i, a
		} else if lp[i] > lp[b] {
			b = i
		}
	}
	return a, lp[a] - lp[b]
}

// FAK_INKLING2_PREFILL_ORACLE_FORCED=<case> teacher-forces the llama.cpp greedy
// continuation of one oracle case on both prefill arms and logs, per position, each
// arm's top token and margin and how far llama.cpp's token sits below it, so an oracle
// divergence can be split into what the batched prompt KV changes and what both arms share.
func TestInkling2CUDAPrefillOracleForced(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_PREFILL_ORACLE_FORCED")
	defer m.Close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var c *inkling2OracleCase
	for i := range inkling2OracleCases {
		if inkling2OracleCases[i].name == os.Getenv("FAK_INKLING2_PREFILL_ORACLE_FORCED") {
			c = &inkling2OracleCases[i]
		}
	}
	if c == nil {
		t.Fatalf("no oracle case %q", os.Getenv("FAK_INKLING2_PREFILL_ORACLE_FORCED"))
	}
	rows := map[bool][][]float64{}
	for _, batched := range []bool{false, true} {
		s, err := newInkling2Resident(m, be)
		if err != nil {
			t.Fatal(err)
		}
		s.batched = batched
		rows[batched] = inkling2TeacherForcedLogProbs(t, m, s, c.prompt, c.greedy[:len(c.greedy)-1])
		s.close()
	}
	arm := func(lp []float64, want int) string {
		top, margin := inkling2Top2(lp)
		return fmt.Sprintf("top %d margin %.3f llama-gap %.3f", top, margin, lp[top]-lp[want])
	}
	for i, want := range c.greedy {
		kl, _ := inkling2KL(rows[false][i], rows[true][i])
		t.Logf("forced %s step %2d llama %d (margin %.3f) | per-token %s | batched %s | KL %.6f",
			c.name, i, want, c.margin[i], arm(rows[false][i], want), arm(rows[true][i], want), kl)
	}
}
