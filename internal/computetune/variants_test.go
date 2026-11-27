package computetune

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

// variantFake is a CPU-reference stand-in: out has M entries, each scale*(row+1).
type variantFake struct {
	id    string
	shape func(p Profile, scale float64) []float32
}

func (f variantFake) ID() string                                          { return f.id }
func (f variantFake) Run(_ context.Context, p Profile) ([]float32, error) { return f.shape(p, 1), nil }
func (f variantFake) RunScaled(_ context.Context, p Profile, s float64) ([]float32, error) {
	return f.shape(p, s), nil
}

func refRows(p Profile, s float64) []float32 {
	out := make([]float32, p.M)
	for i := range out {
		out[i] = float32(s) * float32(i+1)
	}
	return out
}

func variantTune(t *testing.T, cand Candidate, policy Policy) ProfileResult {
	t.Helper()
	ref := variantFake{id: "ref", shape: refRows}
	exact := func(got, want []float32) error {
		if len(got) != len(want) {
			return errLen
		}
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				return errVal
			}
		}
		return nil
	}
	measure := func(_ context.Context, c Candidate, _ Profile) (time.Duration, error) {
		if c.ID() == "ref" {
			return 2 * time.Millisecond, nil
		}
		return time.Millisecond, nil
	}
	policy.Warmup, policy.Repeats, policy.Statistic = 0, 3, "median"
	policy.TimerDomain, policy.FallbackCandidate = "host", "ref"
	_, rep, err := Tune(context.Background(), []Profile{profile(64, 8, 8)}, []Candidate{ref, cand}, ref, exact, measure, policy)
	if err != nil {
		t.Fatal(err)
	}
	return rep.Profiles[0]
}

type varErr string

func (e varErr) Error() string { return string(e) }

const (
	errLen = varErr("length mismatch")
	errVal = varErr("value mismatch")
)

func TestTuneRefusesScaleShortcutCandidate(t *testing.T) {
	abs := variantFake{id: "abs", shape: func(p Profile, s float64) []float32 {
		out := refRows(p, s)
		for i := range out {
			out[i] = float32(math.Abs(float64(out[i])))
		}
		return out
	}}
	// Without scales the shortcut is admitted (today's behaviour) and wins on time.
	if pr := variantTune(t, abs, Policy{}); pr.SelectedID != "abs" {
		t.Fatalf("scale 1 only: selected %q", pr.SelectedID)
	}
	pr := variantTune(t, abs, Policy{InputScales: []float64{1, 0.01, -1}})
	if pr.SelectedID != "ref" {
		t.Fatalf("selected %q, want fallback", pr.SelectedID)
	}
	if r := resultByID(pr.Candidates, "abs"); r.Correct || !strings.Contains(r.RefusalReason, "input scale -1") {
		t.Fatalf("abs result %+v", r)
	}
}

func TestTuneRaggedShapeVariants(t *testing.T) {
	dropTail := variantFake{id: "tail", shape: func(p Profile, s float64) []float32 {
		out := refRows(p, s)
		if p.M == 63 {
			out[len(out)-1] = 0
		}
		return out
	}}
	if pr := variantTune(t, dropTail, Policy{}); pr.SelectedID != "tail" {
		t.Fatalf("variants off: selected %q", pr.SelectedID)
	}
	pr := variantTune(t, dropTail, Policy{RaggedVariants: true})
	if r := resultByID(pr.Candidates, "tail"); pr.SelectedID != "ref" || r.Correct || !strings.Contains(r.RefusalReason, "incorrect at ragged variant") {
		t.Fatalf("selected %q result %+v", pr.SelectedID, r)
	}
	good := variantFake{id: "good", shape: refRows}
	if pr := variantTune(t, good, Policy{RaggedVariants: true, InputScales: []float64{1, 0.01, -1}}); pr.SelectedID != "good" {
		t.Fatalf("correct variant not admitted: %q", pr.SelectedID)
	}
}

func TestVariantProfiles(t *testing.T) {
	if got := len(variantProfiles(profile(64, 8, 8))); got != 5 {
		t.Fatalf("got %d variants", got)
	}
	if got := len(variantProfiles(profile(1, 1, 1))); got != 1 {
		t.Fatalf("got %d variants for 1x1x1", got)
	}
}
