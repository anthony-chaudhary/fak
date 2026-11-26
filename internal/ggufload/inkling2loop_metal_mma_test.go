//go:build darwin && arm64 && cgo

package ggufload

import (
	"context"
	"runtime"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2MetalNoDraft never proposes, so every step is the spec loop's plain step.
type inkling2MetalNoDraft struct{}

func (inkling2MetalNoDraft) Draft([]int, int) []int                      { return nil }
func (inkling2MetalNoDraft) Observe(*inkling2Resident, []int, int) error { return nil }

// inkling2MetalTopTwo returns the largest and second-largest entries of v,
// ties to the lower index as the argmax kernel breaks them.
func inkling2MetalTopTwo(v []float32) (top, second int) {
	top, second = 0, -1
	for i := 1; i < len(v); i++ {
		switch {
		case v[i] > v[top]:
			top, second = i, top
		case second < 0 || v[i] > v[second]:
			second = i
		}
	}
	return top, second
}

// inkling2MetalPlain is the plain greedy a spec run on sbe must reproduce, with
// its timing: the resident decode, or in MMA mode the spec loop's plain steps
// (width-1 MMA verifies), which must follow the decode GEMV's greedy up to an
// admissible near tie.
func inkling2MetalPlain(t *testing.T, m *Inkling2LoopModel, be, sbe compute.Backend, prompt []int, n int) ([]int, Inkling2LoopGreedyStats) {
	t.Helper()
	if !inkling2VerifyMMA(sbe) {
		ids, st, err := inkling2ResidentGreedy(m, be, prompt, n)
		if err != nil {
			t.Fatalf("plain: %v", err)
		}
		return ids, st
	}
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "0")
	gemv, runnerUp, margin, gst := inkling2MetalGreedyMargins(t, m, be, prompt, n)
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "1")
	mma, gap := inkling2MetalMMAGreedyGaps(t, m, sbe, prompt, gemv)
	ids, st, err := inkling2ResidentSpecGreedy(m, sbe, prompt, n, 1, func(*inkling2Resident) (inkling2Drafter, error) {
		return inkling2MetalNoDraft{}, nil
	})
	if err != nil {
		t.Fatalf("MMA plain: %v", err)
	}
	if !slices.Equal(ids, mma) {
		t.Fatalf("the spec loop's plain steps decoded %v, width-1 MMA verifies %v", ids, mma)
	}
	t.Logf("MMA plain %.2f tok/s (%d plain steps), GEMV plain %.2f tok/s, %s",
		st.DecodeTokensPerSecond(), st.PlainSteps, gst.DecodeTokensPerSecond(), inkling2MetalSpecLoad())
	for i := range ids {
		if ids[i] == gemv[i] {
			continue
		}
		tie := inkling2NearTie(margin[i], ids[i] == runnerUp[i], gap[i])
		t.Logf("MMA_VS_GEMV token %d: MMA %d, GEMV %d (runner-up %d, GEMV margin %.4f, MMA gap %.4f nats): near tie %v",
			i, ids[i], gemv[i], runnerUp[i], margin[i], gap[i], tie)
		if !tie {
			t.Errorf("MMA plain greedy leaves the GEMV's at token %d outside a near tie", i)
		}
		return ids, st.Inkling2LoopGreedyStats
	}
	t.Logf("MMA_VS_GEMV: %d tokens identical", len(ids))
	return ids, st.Inkling2LoopGreedyStats
}

// inkling2MetalGreedyMargins is resident greedy with, per step, the runner-up
// and the top-1 minus top-2 softcapped logit gap.
func inkling2MetalGreedyMargins(t *testing.T, m *Inkling2LoopModel, be compute.Backend, prompt []int, n int) (toks, runnerUp []int, margin []float64, st Inkling2LoopGreedyStats) {
	t.Helper()
	toks, st, err := inkling2ResidentGreedy(m, be, prompt, n)
	if err != nil {
		t.Fatalf("GEMV plain: %v", err)
	}
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	next, err := s.prefill(context.Background(), prompt)
	for i := 0; i < n && err == nil; i++ {
		var logits []float32
		if logits, err = s.lastLogits(); err != nil {
			break
		}
		top, second := inkling2MetalTopTwo(logits[:s.hp.VocabUnpadded])
		if top != next || next != toks[i] {
			t.Fatalf("GEMV step %d: greedy %d, logits' top-1 %d, timed run %d", i, next, top, toks[i])
		}
		runnerUp = append(runnerUp, second)
		margin = append(margin, float64(logits[top]-logits[second]))
		if i+1 < n {
			next, err = s.step(toks[i])
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	return toks, runnerUp, margin, st
}

// inkling2MetalMMAGreedyGaps is greedy decode by width-1 verifies, and per step
// how far ref's token sits below that step's top-1, in softcapped nats.
func inkling2MetalMMAGreedyGaps(t *testing.T, m *Inkling2LoopModel, be compute.Backend, prompt, ref []int) (toks []int, gap []float64) {
	t.Helper()
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if err := s.enableSpec(1); err != nil {
		t.Fatal(err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := s.lastLogits()
	for i := range ref {
		if err != nil {
			t.Fatal(err)
		}
		v := logits[:s.hp.VocabUnpadded]
		if top, _ := inkling2MetalTopTwo(v); top != next {
			t.Fatalf("MMA step %d: greedy %d, logits' top-1 %d", i, next, top)
		}
		toks, gap = append(toks, next), append(gap, float64(v[next]-v[ref[i]]))
		if i+1 == len(ref) {
			break
		}
		var got []int
		if got, err = s.verify([]int{next}); err != nil {
			t.Fatal(err)
		}
		if err = s.commitVerify(1, 1); err != nil {
			t.Fatal(err)
		}
		next = got[0]
		logits, err = s.dev.Ink2Read(s.verifyLogits, s.hp.VocabUnpadded)
	}
	return toks, gap
}
