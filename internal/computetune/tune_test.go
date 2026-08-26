package computetune

import (
	"context"
	"errors"
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
