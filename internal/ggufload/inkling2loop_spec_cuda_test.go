//go:build cuda && cgo

package ggufload

import (
	"context"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

// FAK_INKLING2_VERIFY_COST=1 times one batched forward over T tokens (T=1..8) against
// the graph-replayed single-token step at the same depth. The batched forward is the
// candidate verify pass of a speculative step, so its cost per T bounds the speedup.
func TestInkling2CUDAVerifyCost(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_VERIFY_COST")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if !s.batched {
		t.Skip("backend has no batched forward")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	prompt := inkling2OracleCases[0].prompt
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	const reps = 16
	for i := 0; i < 8; i++ { // eager warm token, then graph capture
		if next, err = s.step(next); err != nil {
			t.Fatal(err)
		}
	}
	base := s.pos
	start := time.Now()
	for i := 0; i < reps; i++ {
		if next, err = s.step(next); err != nil {
			t.Fatal(err)
		}
	}
	step := time.Since(start) / reps
	t.Logf("inkling2 verify cost: graph step %.3f ms (%.1f tok/s) at pos %d", ms(step), 1/step.Seconds(), base)
	toks := []int{next, 13, 149917, 151451, 149879, 149876, 151465, 151480, 149876}
	if err := s.enableSpec(8); err != nil {
		t.Fatal(err)
	}
	lo, hi := 1, 9
	if v, err := strconv.Atoi(os.Getenv("FAK_INKLING2_VERIFY_COST_T")); err == nil && v >= 1 && v <= 9 {
		lo, hi = v, v
	}
	for T := lo; T <= hi; T++ {
		// each width gets its own captured verify: the first run is eager and warms
		// the scratch, the second captures, the rest replay
		s.specMax, s.verifyWarm, s.verifyReady, s.keepReady = T, false, false, false
		var eager, graphed time.Duration
		for r := 0; r < reps+2; r++ {
			s.pos = base
			t0 := time.Now()
			if _, err := s.verify(toks[:T]); err != nil {
				t.Fatal(err)
			}
			if err := s.commitVerify(T, 1); err != nil {
				t.Fatal(err)
			}
			switch {
			case r == 0:
				eager = time.Since(t0)
			case r >= 2:
				graphed += time.Since(t0)
			}
		}
		per := graphed / reps
		t.Logf("inkling2 verify cost: T=%d eager %.3f ms; graph %.3f ms = %.2fx step (graph ready %v)",
			T, ms(eager), ms(per), per.Seconds()/step.Seconds(), s.verifyReady)
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// FAK_INKLING2_SPEC_ORACLE=1: speculative greedy fed the plain greedy continuation
// (some tokens corrupted) must reproduce plain greedy exactly, which exercises
// accept, reject and the conv-history rewind; the uncorrupted runs log the
// throughput ceiling of each draft depth.
func TestInkling2CUDASpecOracle(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_SPEC_ORACLE")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	n := 128
	prompt := inkling2OracleCases[0].prompt
	ref, st, err := inkling2ResidentGreedy(m, be, prompt, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("inkling2 spec oracle: plain %.2f tok/s", st.DecodeTokensPerSecond())
	check := func(t *testing.T, name string, depth int, corrupt func(int) bool) {
		got, sst, err := inkling2ResidentSpecGreedy(m, be, prompt, n, depth, func(*inkling2Resident) (inkling2Drafter, error) {
			return &inkling2OracleDrafter{ref: ref, corrupt: corrupt}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("inkling2 spec oracle: %s depth %d: %.2f tok/s; %d verifies, accepted %d/%d, plain steps %d, verify %.3f ms/call, adaptive widths %v",
			name, depth, sst.DecodeTokensPerSecond(), sst.Verifies, sst.Accepted, sst.Drafted, sst.PlainSteps,
			ms(sst.Verify)/float64(max(1, sst.Verifies)), sst.AdaptWidths[1:depth+1])
		for i := range got {
			if got[i] != ref[i] {
				t.Errorf("%s depth %d: token %d = %d, plain greedy %d", name, depth, i, got[i], ref[i])
				return
			}
		}
	}
	check(t, "corrupt", 4, func(i int) bool { return i%5 == 2 })
	check(t, "corrupt", 7, func(i int) bool { return i%9 == 4 || i%13 == 0 })
	for _, depth := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
		check(t, "perfect", depth, nil)
	}
	t.Run("adaptive width", func(t *testing.T) {
		t.Setenv("FAK_INKLING2_SPEC_ADAPT", "1")
		check(t, "adaptive corrupt", 4, func(i int) bool { return i%5 == 2 })
		check(t, "adaptive corrupt", 8, func(i int) bool { return i%3 != 0 })
		check(t, "adaptive perfect", 8, nil)
	})
}

// FAK_INKLING2_SPEC_DIAG=1 logs, for the first verifies of a perfect depth-2 draft,
// what was fed, what verify returned and the plain greedy reference.
func TestInkling2CUDASpecDiag(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_SPEC_DIAG")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	prompt := inkling2OracleCases[0].prompt
	ref, _, err := inkling2ResidentGreedy(m, be, prompt, 24)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ref %v", ref)
	for _, mode := range []string{"0", "1", "1w", "1p", "0p"} {
		graph := mode[:1]
		warm, partial := len(mode) > 1 && mode[1] == 'w', len(mode) > 1 && mode[1] == 'p'
		t.Setenv("FAK_INKLING2_GRAPH", graph)
		s, err := newInkling2Resident(m, be)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.enableSpec(2); err != nil {
			t.Fatal(err)
		}
		next, err := s.prefill(context.Background(), prompt)
		if err != nil {
			t.Fatal(err)
		}
		if warm {
			if _, err := s.verify([]int{next, next, next}); err != nil {
				t.Fatal(err)
			}
			if err := s.commitVerify(3, 0); err != nil {
				t.Fatal(err)
			}
		}
		out := []int{next}
		for v := 0; v < 6 && len(out)+3 < len(ref); v++ {
			fed := []int{out[len(out)-1], ref[len(out)], ref[len(out)+1]}
			if partial && v%2 == 0 {
				fed[2] = 7
			}
			g, err := s.verify(fed)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("mode=%s pos %d fed %v -> g %v; ref next %v", mode, s.pos, fed, g, ref[len(out):len(out)+3])
			a := 0
			for a < 2 && fed[a+1] == g[a] {
				a++
			}
			if err := s.commitVerify(3, a+1); err != nil {
				t.Fatal(err)
			}
			out = append(out, fed[1:a+1]...)
			out = append(out, g[a])
		}
		s.close()
	}
}
