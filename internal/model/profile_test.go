package model

import "testing"

// TestProfileMatchesProven pins the instrumented profiler twin (profToken) to the
// proven decode path (Prefill + Step) bit-for-bit. If the twin ever drifts from the
// code it claims to measure, this goes red — so the observability numbers are always
// about the real forward pass, never a stale copy.
func TestProfileMatchesProven(t *testing.T) {
	m, _ := loadFixture(t)
	vocab := m.Cfg.VocabSize

	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 13) % vocab
	}

	// proven path
	sP := m.NewSession()
	sP.Prefill(prompt)
	// twin path: identical prefill via profToken (head only on last, like Prefill)
	sT := m.NewSession()
	p := newProfiler()
	for i, id := range prompt {
		sT.profToken(p, id, sT.Cache.Len(), i == len(prompt)-1)
	}

	// decode the same id sequence on both; assert the twin tracks the proven path. The
	// argmax must match EXACTLY every step on every arch (token-level correctness); the raw
	// logits are byte-exact on amd64 and within FMA noise (fmaCrossPathTol) on arches where
	// gc auto-fuses multiply-add — see fmatol_other_test.go for why two equivalent Go paths
	// can't be byte-identical there.
	id := 7
	for step := 0; step < 24; step++ {
		want := sP.Step(id)
		got := sT.profToken(p, id, sT.Cache.Len(), true)
		if len(want) != len(got) {
			t.Fatalf("step %d: len %d != %d", step, len(want), len(got))
		}
		if argmax(want) != argmax(got) {
			t.Fatalf("step %d: twin argmax %d != proven argmax %d (profiler diverged at the token level)",
				step, argmax(got), argmax(want))
		}
		for j := range want {
			d := want[j] - got[j]
			if d < 0 {
				d = -d
			}
			if float64(d) > fmaCrossPathTol {
				t.Fatalf("step %d pos %d: proven %v != twin %v (|Δ|=%.3e > %.0e — profiler drifted from proven path)",
					step, j, want[j], got[j], d, fmaCrossPathTol)
			}
		}
		id = (id*48271 + 1) % vocab
	}
}

func TestMeasureCleanDecodeReportsSessionStepLatency(t *testing.T) {
	m, _ := loadFixture(t)
	c := m.MeasureCleanDecode(4, 2)
	if c.Mode != "clean_decode" {
		t.Fatalf("mode = %q, want clean_decode", c.Mode)
	}
	if c.PromptTokens != 4 || c.Steps != 2 {
		t.Fatalf("shape = prompt %d steps %d, want prompt 4 steps 2", c.PromptTokens, c.Steps)
	}
	if c.TotalNanos <= 0 || c.PerTokenMS <= 0 {
		t.Fatalf("expected positive timing, got total_nanos=%d per_token_ms=%f", c.TotalNanos, c.PerTokenMS)
	}
}

// TestProfileDecodeAgreesWithCleanDecode is the regression guard for issue #31: the
// instrumented per-op profiler (ProfileDecode) and the uninstrumented Session.Step
// measurement (MeasureCleanDecode) must agree on the SAME workload, because both now
// drive the SAME parallel matmul kernel (parMatRows). Before the fix the profiler used
// the serial matRows and reported ~2.7x the clean per-token latency — a number readers
// mistook for achievable decode throughput.
//
// We assert the profiler is not WILDLY slower than clean decode. The per-op time.Now()
// instrumentation still adds a little overhead (the profiler must be >= clean), so we
// allow a generous ceiling — but a return to the serial-kernel regime would blow past it.
// The 2.7x serial gap sat at ~270%; the profiler+clean both being parallel pulls it well
// under this bar. The bound is deliberately loose so the test is robust to a loaded shared
// box; it exists to catch a re-serialization, not to pin an exact ratio.
func TestProfileDecodeAgreesWithCleanDecode(t *testing.T) {
	m, _ := loadFixture(t)
	const prompt, steps = 16, 24

	// Take the best (smallest) of a few reps for each to blunt scheduler/contention noise
	// on this shared box — we are comparing two means-of-measuring the same kernel, not
	// chasing a tight number.
	best := func(f func() float64) float64 {
		b := f()
		for i := 0; i < 2; i++ {
			if v := f(); v < b {
				b = v
			}
		}
		return b
	}
	clean := best(func() float64 { return m.MeasureCleanDecode(prompt, steps).PerTokenMS })
	prof := best(func() float64 { return m.ProfileDecode(prompt, steps).PerTokenMS })

	if clean <= 0 || prof <= 0 {
		t.Fatalf("non-positive timing: clean=%.3f prof=%.3f ms/tok", clean, prof)
	}
	// The instrumented profiler may not be FASTER than clean decode beyond a noise
	// margin (it does strictly more work — the per-op timing). Full-package runs on
	// shared Windows hosts can swing enough between the two best-of samples that 15%
	// produced false failures; 30% still catches a structurally different fast path.
	if prof < 0.70*clean {
		t.Fatalf("instrumented profiler (%.2f ms/tok) implausibly faster than clean decode (%.2f ms/tok)", prof, clean)
	}
	// The headline invariant: the profiler must agree with clean decode to within ~2x. The
	// pre-#31 serial profiler sat at ~2.7x; both-parallel pulls it under 2x with comfortable
	// margin on a quiet box. This is the assertion that goes red if profToken ever reverts
	// to the serial matRows kernel.
	if prof > 2.0*clean {
		t.Fatalf("instrumented profiler (%.2f ms/tok) disagrees >2x with clean Session.Step decode (%.2f ms/tok); "+
			"profToken likely regressed to the serial matmul kernel (issue #31)", prof, clean)
	}
}
