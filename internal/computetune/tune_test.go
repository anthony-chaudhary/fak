package computetune

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeCandidate struct {
	id     string
	result func(Profile) ([]float32, error)
}

func (c fakeCandidate) ID() string                                          { return c.id }
func (c fakeCandidate) Run(_ context.Context, p Profile) ([]float32, error) { return c.result(p) }

func profile(m, n, k int) Profile {
	return Profile{Operation: OpMatMul, M: m, N: n, K: k, DType: "f32", Compat: Compatibility{Backend: "cpu", Device: "test-cpu", SoftwareRevision: "fak-r1", KernelRevision: "kernels-r2"}, Frequency: 100, Weight: 1}
}

func exact(got, want []float32) error {
	if !reflect.DeepEqual(got, want) {
		return errors.New("mismatch")
	}
	return nil
}

func TestTuneRejectsIncorrectAndSelectsPerProfileWinners(t *testing.T) {
	profiles := []Profile{profile(1, 2, 3), profile(2, 2, 3)}
	ref := fakeCandidate{id: "reference", result: func(p Profile) ([]float32, error) { return []float32{float32(p.M), float32(p.N), float32(p.K)}, nil }}
	fallback := fakeCandidate{id: "fallback", result: ref.result}
	fastNarrow := fakeCandidate{id: "fast-narrow", result: ref.result}
	fastWide := fakeCandidate{id: "fast-wide", result: ref.result}
	wrong := fakeCandidate{id: "wrong", result: func(Profile) ([]float32, error) { return []float32{-1}, nil }}
	candidates := []Candidate{fallback, fastNarrow, fastWide, wrong}
	timings := map[string]map[int]time.Duration{
		"fallback":    {1: 10 * time.Microsecond, 2: 10 * time.Microsecond},
		"fast-narrow": {1: 4 * time.Microsecond, 2: 12 * time.Microsecond},
		"fast-wide":   {1: 13 * time.Microsecond, 2: 3 * time.Microsecond},
	}
	measured := map[string]int{}
	measure := func(_ context.Context, c Candidate, p Profile) (time.Duration, error) {
		measured[c.ID()]++
		return timings[c.ID()][p.M], nil
	}
	policy := Policy{Warmup: 1, Repeats: 3, Statistic: "median", TimerDomain: "host-monotonic", FallbackCandidate: "fallback", SelectionOverhead: 12 * time.Microsecond}

	manifest, report, err := Tune(context.Background(), profiles, candidates, ref, exact, measure, policy)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := manifest.Lookup(profiles[0]); got != "fast-narrow" {
		t.Fatalf("profile 0 selected %q", got)
	}
	if got, _ := manifest.Lookup(profiles[1]); got != "fast-wide" {
		t.Fatalf("profile 1 selected %q", got)
	}
	if measured["wrong"] != 0 {
		t.Fatalf("incorrect candidate was timed %d times", measured["wrong"])
	}
	for _, pr := range report.Profiles {
		wrongResult := resultByID(pr.Candidates, "wrong")
		if wrongResult == nil || wrongResult.Correct || !strings.Contains(wrongResult.RefusalReason, "incorrect result") {
			t.Fatalf("missing correctness refusal: %+v", wrongResult)
		}
		if pr.BreakEvenReuse == 0 {
			t.Fatalf("missing break-even accounting: %+v", pr)
		}
	}
	if report.ManifestDigest == "" || report.Policy.TimerDomain != "host-monotonic" {
		t.Fatalf("incomplete report: %+v", report)
	}
}

func TestManifestFailsClosedAndDispatchFallsBack(t *testing.T) {
	p := profile(1, 2, 3)
	manifest, err := NewManifest([]Entry{{Profile: p, CandidateID: "fast"}})
	if err != nil {
		t.Fatal(err)
	}
	fast := fakeCandidate{id: "fast", result: func(Profile) ([]float32, error) { return []float32{1}, nil }}
	fallback := fakeCandidate{id: "fallback", result: func(Profile) ([]float32, error) { return []float32{2}, nil }}
	candidates := map[string]Candidate{"fast": fast}

	got, id, err := DispatchMatMul(context.Background(), p, manifest, candidates, fallback)
	if err != nil || id != "fast" || !reflect.DeepEqual(got, []float32{1}) {
		t.Fatalf("exact dispatch got=%v id=%q err=%v", got, id, err)
	}

	incompatible := p
	incompatible.Compat.Device = "different-device"
	got, id, err = DispatchMatMul(context.Background(), incompatible, manifest, candidates, fallback)
	if err != nil || id != "fallback" || !reflect.DeepEqual(got, []float32{2}) {
		t.Fatalf("fallback dispatch got=%v id=%q err=%v", got, id, err)
	}

	entries := manifest.Entries()
	entries[0].CandidateID = "mutated"
	if got, _ := manifest.Lookup(p); got != "fast" {
		t.Fatalf("manifest was mutable: %q", got)
	}
	first, _ := manifest.Digest()
	second, _ := manifest.Digest()
	if first != second {
		t.Fatalf("digest not deterministic: %q != %q", first, second)
	}
}

func TestTuneRecordsExecutionAndTimingFailures(t *testing.T) {
	p := profile(1, 1, 1)
	ref := fakeCandidate{id: "reference", result: func(Profile) ([]float32, error) { return []float32{1}, nil }}
	fallback := fakeCandidate{id: "fallback", result: ref.result}
	runFail := fakeCandidate{id: "run-fail", result: func(Profile) ([]float32, error) { return nil, errors.New("unsupported shape") }}
	timeFail := fakeCandidate{id: "time-fail", result: ref.result}
	measure := func(_ context.Context, c Candidate, _ Profile) (time.Duration, error) {
		if c.ID() == "time-fail" {
			return 0, errors.New("device event unavailable")
		}
		return time.Microsecond, nil
	}
	_, report, err := Tune(context.Background(), []Profile{p}, []Candidate{fallback, runFail, timeFail}, ref, exact, measure, Policy{Repeats: 1, Statistic: "median", TimerDomain: "device-event", FallbackCandidate: "fallback"})
	if err != nil {
		t.Fatal(err)
	}
	results := report.Profiles[0].Candidates
	if got := resultByID(results, "run-fail"); got == nil || !strings.Contains(got.RefusalReason, "unsupported shape") {
		t.Fatalf("run refusal missing: %+v", got)
	}
	if got := resultByID(results, "time-fail"); got == nil || !strings.Contains(got.RefusalReason, "device event unavailable") {
		t.Fatalf("timing refusal missing: %+v", got)
	}
}

// signTestFixture times each candidate from a per-candidate script of durations,
// so a test can place a slow or failing round at an exact paired round.
type signTestFixture struct {
	calls  map[string]int
	script func(id string, call int) (time.Duration, error)
}

func (f *signTestFixture) measure(_ context.Context, c Candidate, _ Profile) (time.Duration, error) {
	f.calls[c.ID()]++
	return f.script(c.ID(), f.calls[c.ID()])
}

func TestTuneRecordsExecutionAndTimingFailuresUnderSignTest(t *testing.T) {
	const alpha = 0.05
	const medianRepeats = 3
	p := profile(1, 1, 1)
	ref := fakeCandidate{id: "reference", result: func(Profile) ([]float32, error) { return []float32{1}, nil }}
	fallback := fakeCandidate{id: "fallback", result: ref.result}
	fast := fakeCandidate{id: "fast", result: ref.result}
	policy := Policy{Repeats: medianRepeats, Statistic: "median", TimerDomain: "device-event", FallbackCandidate: "fallback", SignTestAlpha: alpha}
	tune := func(t *testing.T, candidates []Candidate, pol Policy, script func(string, int) (time.Duration, error)) (Manifest, Report, *signTestFixture, error) {
		t.Helper()
		f := &signTestFixture{calls: map[string]int{}, script: script}
		m, r, err := Tune(context.Background(), []Profile{p}, candidates, ref, exact, f.measure, pol)
		return m, r, f, err
	}

	t.Run("mixture wealth is the closed form of its integral", func(t *testing.T) {
		for _, tc := range []struct{ rounds, favourable int }{{1, 0}, {1, 1}, {7, 7}, {9, 4}, {30, 22}, {200, 120}} {
			const steps = 200000
			integral := 0.0
			for step := 0; step < steps; step++ {
				q := 0.5 + (float64(step)+0.5)/float64(2*steps)
				integral += 2 * math.Pow(2*q, float64(tc.favourable)) * math.Pow(2*(1-q), float64(tc.rounds-tc.favourable))
			}
			integral /= float64(2 * steps)
			closed := signTestWealth(tc.rounds, tc.favourable)
			if math.Abs(closed-integral) > 1e-6*math.Max(integral, 1) {
				t.Fatalf("M_%d(%d) = %v, quadrature %v", tc.rounds, tc.favourable, closed, integral)
			}
		}
		// All rounds ahead has its own closed form, (2^(n+1) - 1) / (n + 1).
		for n := 0; n <= 40; n++ {
			want := (math.Pow(2, float64(n+1)) - 1) / float64(n+1)
			if got := signTestWealth(n, n); math.Abs(got-want) > 1e-9*want {
				t.Fatalf("M_%d(%d) = %v, want %v", n, n, got, want)
			}
		}
		if even := signTestWealth(100000, 50000); math.IsNaN(even) || even <= 0 || even > 1 {
			t.Fatalf("M_100000(50000) = %v, want a finite value at most one", even)
		}
		if got := signTestWealth(0, 0); got != 1 {
			t.Fatalf("wealth before any round = %v, want 1", got)
		}
		if six, seven := signTestWealth(6, 6), signTestWealth(7, 7); !(six < 1/alpha && seven >= 1/alpha) {
			t.Fatalf("M_6(6)=%v M_7(7)=%v: want the lead to resolve at exactly seven rounds all ahead", six, seven)
		}
	})

	t.Run("either side is timed first about as often as the other", func(t *testing.T) {
		const draws = 40000
		order := newRoundOrder()
		first := 0
		for i := 0; i < draws; i++ {
			if order.challengerFirst() {
				first++
			}
		}
		if first < draws*19/40 || first > draws*21/40 {
			t.Fatalf("challenger timed first in %d of %d rounds, want about half", first, draws)
		}
	})

	t.Run("a challenger ahead in every round leads in the fewest rounds", func(t *testing.T) {
		m, report, f, err := tune(t, []Candidate{fallback, fast}, policy, func(id string, _ int) (time.Duration, error) {
			if id == "fast" {
				return 99 * time.Microsecond, nil
			}
			return 100 * time.Microsecond, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := m.Lookup(p); got != "fast" {
			t.Fatalf("selected %q, want fast", got)
		}
		st := resultByID(report.Profiles[0].Candidates, "fast").SignTest
		if st == nil || !st.Leads || st.Rounds != 7 || st.Ahead != 7 || len(st.Ratios) != st.Rounds || st.Wealth*alpha < 1 {
			t.Fatalf("sign test receipt = %+v, want a lead after seven rounds all ahead", st)
		}
		for i, ratio := range st.Ratios {
			if ratio != 0.99 {
				t.Fatalf("round %d ratio = %v, want the in-round challenger/fallback ratio 0.99", i+1, ratio)
			}
		}
		if f.calls["fast"] != medianRepeats+7 || f.calls["fallback"] != medianRepeats+7 {
			t.Fatalf("calls = %v, want each side timed once per paired round", f.calls)
		}
		if resultByID(report.Profiles[0].Candidates, "fallback").SignTest != nil {
			t.Fatal("the fallback is the incumbent and carries no sign test of its own")
		}

		// The rounds a lead takes follow from the error rate alone: 31/5 >= 4 > 15/4.
		loose := policy
		loose.SignTestAlpha = 0.25
		_, report, _, err = tune(t, []Candidate{fallback, fast}, loose, func(id string, _ int) (time.Duration, error) {
			if id == "fast" {
				return 99 * time.Microsecond, nil
			}
			return 100 * time.Microsecond, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if st := resultByID(report.Profiles[0].Candidates, "fast").SignTest; st == nil || !st.Leads || st.Rounds != 4 {
			t.Fatalf("sign test receipt at alpha 0.25 = %+v, want a lead after four rounds", st)
		}
	})

	t.Run("the comparison stops at the first round not ahead", func(t *testing.T) {
		const slowRound = 3
		m, report, f, err := tune(t, []Candidate{fallback, fast}, policy, func(id string, call int) (time.Duration, error) {
			if id != "fast" {
				return 100 * time.Microsecond, nil
			}
			if call == medianRepeats+slowRound {
				return 100 * time.Microsecond, nil
			}
			return 50 * time.Microsecond, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := m.Lookup(p); got != "fallback" {
			t.Fatalf("selected %q, want the fallback kept: a lower median alone must not admit", got)
		}
		fastResult := resultByID(report.Profiles[0].Candidates, "fast")
		st := fastResult.SignTest
		if st == nil || st.Leads || st.Rounds != slowRound || st.Ahead != slowRound-1 {
			t.Fatalf("sign test receipt = %+v, want it stopped unresolved at round %d", st, slowRound)
		}
		if !fastResult.Correct || fastResult.RefusalReason != "" {
			t.Fatalf("an unresolved challenger is not a failure: %+v", fastResult)
		}
		if f.calls["fast"] != medianRepeats+slowRound {
			t.Fatalf("fast timed %d times, want no round after the first not ahead", f.calls["fast"])
		}
		if report.Profiles[0].BreakEvenReuse != 0 {
			t.Fatalf("break-even = %d for a kept fallback", report.Profiles[0].BreakEvenReuse)
		}
	})

	t.Run("a paired-round timing failure is recorded and the next challenger is tried", func(t *testing.T) {
		flaky := fakeCandidate{id: "flaky", result: ref.result}
		runFail := fakeCandidate{id: "run-fail", result: func(Profile) ([]float32, error) { return nil, errors.New("unsupported shape") }}
		m, report, _, err := tune(t, []Candidate{fallback, flaky, fast, runFail}, policy, func(id string, call int) (time.Duration, error) {
			switch id {
			case "flaky":
				if call > medianRepeats+1 {
					return 0, errors.New("device event unavailable")
				}
				return 10 * time.Microsecond, nil
			case "fast":
				return 60 * time.Microsecond, nil
			}
			return 100 * time.Microsecond, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		results := report.Profiles[0].Candidates
		got := resultByID(results, "flaky")
		if got == nil || got.Correct || !strings.Contains(got.RefusalReason, "sign test timing: device event unavailable") {
			t.Fatalf("paired timing refusal missing: %+v", got)
		}
		if got.SignTest == nil || got.SignTest.Rounds != 1 || got.SignTest.Leads || got.Statistic != 0 {
			t.Fatalf("failed challenger = %+v receipt %+v, want the one completed round kept and no statistic", got, got.SignTest)
		}
		// Two challengers share the error rate, so each must reach 2/alpha:
		// 511/9 >= 40 > 255/8 takes eight rounds where one challenger takes seven.
		if st := resultByID(results, "fast").SignTest; st == nil || !st.Leads || st.Alpha != alpha/2 || st.Rounds != 8 {
			t.Fatalf("second challenger receipt = %+v, want a lead at half the error rate after eight rounds", st)
		}
		if rf := resultByID(results, "run-fail"); rf == nil || !strings.Contains(rf.RefusalReason, "unsupported shape") || rf.SignTest != nil {
			t.Fatalf("run refusal missing or sign-tested: %+v", rf)
		}
		if sel, _ := m.Lookup(p); sel != "fast" {
			t.Fatalf("selected %q, want the next challenger that leads", sel)
		}
	})

	t.Run("a fallback that cannot be timed in a paired round fails the tuning", func(t *testing.T) {
		_, _, _, err := tune(t, []Candidate{fallback, fast}, policy, func(id string, call int) (time.Duration, error) {
			if id == "fallback" && call > medianRepeats {
				return 0, nil
			}
			if id == "fast" {
				return 50 * time.Microsecond, nil
			}
			return 100 * time.Microsecond, nil
		})
		if err == nil || !strings.Contains(err.Error(), "failed a paired round") || !strings.Contains(err.Error(), "non-positive duration") {
			t.Fatalf("err = %v, want the fallback paired-round failure", err)
		}
	})

	t.Run("off by default and bounded to a valid error rate", func(t *testing.T) {
		off := policy
		off.SignTestAlpha = 0
		m, report, f, err := tune(t, []Candidate{fallback, fast}, off, func(id string, _ int) (time.Duration, error) {
			if id == "fast" {
				return 99 * time.Microsecond, nil
			}
			return 100 * time.Microsecond, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := m.Lookup(p); got != "fast" {
			t.Fatalf("median-only selection chose %q", got)
		}
		if resultByID(report.Profiles[0].Candidates, "fast").SignTest != nil || f.calls["fast"] != medianRepeats {
			t.Fatalf("sign test ran with the policy off: calls=%v", f.calls)
		}
		for _, bad := range []float64{-0.05, 1, 1.5, math.NaN()} {
			invalid := policy
			invalid.SignTestAlpha = bad
			if _, _, _, err := tune(t, []Candidate{fallback, fast}, invalid, func(string, int) (time.Duration, error) { return time.Microsecond, nil }); err == nil {
				t.Fatalf("alpha %v accepted", bad)
			}
		}
	})
}
