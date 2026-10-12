package computetrace

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"testing"
)

func TestActivationSampleIndices(t *testing.T) {
	if got, want := ActivationSampleIndices(512, 8), []int{0, 64, 128, 192, 256, 320, 384, 448}; !slices.Equal(got, want) {
		t.Fatalf("width 512: %v, want %v", got, want)
	}
	if got, want := ActivationSampleIndices(5, 8), []int{0, 1, 2, 3, 4}; !slices.Equal(got, want) {
		t.Fatalf("width 5: %v, want %v", got, want)
	}
	if got := ActivationSampleIndices(0, 8); got != nil {
		t.Fatalf("width 0: %v, want nil", got)
	}
}

func TestActivationSampleBudget(t *testing.T) {
	fill := func(r *Recorder, events int) Artifact {
		for i := 0; i < events; i++ {
			r.Record(indexedSample(i, 0, "s", make([]float32, 8)))
		}
		return r.Artifact()
	}
	if a := fill(New(1000), 1000); a.RetainedSampleValues != MaxActivationSamplesTotal || a.DroppedSampleValues != 8000-MaxActivationSamplesTotal {
		t.Fatalf("default budget retained=%d dropped=%d, want %d/%d", a.RetainedSampleValues, a.DroppedSampleValues, MaxActivationSamplesTotal, 8000-MaxActivationSamplesTotal)
	}
	if a := fill(NewWithSampleBudget(1000, 8000), 1000); a.RetainedSampleValues != 8000 || a.DroppedSampleValues != 0 {
		t.Fatalf("raised budget retained=%d dropped=%d, want 8000/0", a.RetainedSampleValues, a.DroppedSampleValues)
	}
	r, disable := EnableWithSampleBudget(10, 16, "run", "req")
	Record(indexedSample(0, 0, "s", make([]float32, 8)))
	Record(indexedSample(1, 0, "s", make([]float32, 8)))
	Record(indexedSample(2, 0, "s", make([]float32, 8)))
	disable()
	if a := r.Artifact(); a.RetainedSampleValues != 16 || a.DroppedSampleValues != 8 {
		t.Fatalf("enabled budget retained=%d dropped=%d, want 16/8", a.RetainedSampleValues, a.DroppedSampleValues)
	}
}

func TestActivationErrorProfile(t *testing.T) {
	tol := ActivationTolerance{Atol: 1e-3}
	ref := collect(
		indexedSample(0, 0, "a", []float32{1, 2}),
		indexedSample(0, 1, "a", []float32{1, 2}),
		indexedSample(0, 0, "b", []float32{4, 0}),
		indexedSample(1, 0, "a", []float32{1, 2}),
		indexedSample(1, 0, "b", []float32{4, 8}),
	)

	t.Run("equal", func(t *testing.T) {
		p := ActivationErrorProfile(ref, ref, tol)
		if p.Verdict != TraceEqual || len(p.Rows) != 4 || p.FirstDivergent != nil || p.ComparedSamples != 10 {
			t.Fatalf("self profile verdict=%s rows=%d first=%v compared=%d", p.Verdict, len(p.Rows), p.FirstDivergent, p.ComparedSamples)
		}
	})

	t.Run("every row measured past the first divergence", func(t *testing.T) {
		got := collect(
			indexedSample(0, 0, "a", []float32{1, 2}),
			indexedSample(0, 1, "a", []float32{1, 2.0005}),
			indexedSample(0, 0, "b", []float32{4.5, 0}),
			indexedSample(1, 0, "a", []float32{1, 2}),
			indexedSample(1, 0, "b", []float32{4, 10}),
		)
		p := ActivationErrorProfile(ref, got, tol)
		if p.Verdict != TraceDivergent {
			t.Fatalf("verdict=%s, want DIVERGENT", p.Verdict)
		}
		if p.FirstDivergent == nil || *p.FirstDivergent != (ActivationStage{Layer: 0, Stage: "b"}) {
			t.Fatalf("first divergent=%v, want layer 0 stage b", p.FirstDivergent)
		}
		want := []struct {
			layer     int
			stage     string
			tokens    int
			abs, rel  float64
			match     bool
			mismatchN int
		}{
			{0, "a", 2, 0.0005, 0.0005 / 2.0005, true, 0},
			{0, "b", 1, 0.5, 0.5 / 4.5, false, 1},
			{1, "a", 1, 0, 0, true, 0},
			{1, "b", 1, 2, 0.2, false, 1},
		}
		if len(p.Rows) != len(want) {
			t.Fatalf("rows=%d, want %d", len(p.Rows), len(want))
		}
		for i, w := range want {
			r := p.Rows[i]
			if r.Layer != w.layer || r.Stage != w.stage || r.Tokens != w.tokens || r.Match != w.match || r.Mismatches != w.mismatchN || !r.Comparable {
				t.Fatalf("row %d = %+v, want %+v", i, r, w)
			}
			if math.Abs(r.MaxAbsError-w.abs) > 1e-6 || math.Abs(r.MaxRelError-w.rel) > 1e-6 {
				t.Fatalf("row %d abs=%g rel=%g, want %g/%g", i, r.MaxAbsError, r.MaxRelError, w.abs, w.rel)
			}
		}
	})

	t.Run("refused and excluded samples", func(t *testing.T) {
		got := collect(
			indexedSample(0, 0, "a", []float32{1, 2}),
			indexedSample(0, 1, "a", []float32{1, 2}),
			indexedSample(0, 0, "b", []float32{4, 99}),
			indexedSample(1, 0, "a", []float32{1, 2}),
			indexedSample(1, 0, "b", []float32{4, 8}),
		)
		got.Events[2].ExcludedSampleIndices = []int{1}
		got.Events[3].WeightDigest = Digest("other")
		p := ActivationErrorProfile(ref, got, tol)
		if p.Verdict != TraceIncomparable {
			t.Fatalf("verdict=%s, want INCOMPARABLE for a refused row", p.Verdict)
		}
		b0, a1 := p.Rows[1], p.Rows[2]
		if !b0.Match || b0.ExcludedSamples != 1 || b0.ComparedSamples != 1 {
			t.Fatalf("excluded row = %+v, want 1 compared match and 1 excluded", b0)
		}
		if a1.Comparable || a1.Refusals != 1 || a1.Reason == "" {
			t.Fatalf("provenance row = %+v, want refused", a1)
		}
	})

	t.Run("non-finite is a mismatch and stays encodable", func(t *testing.T) {
		got := collect(
			indexedSample(0, 0, "a", []float32{float32(math.NaN()), 2}),
			indexedSample(0, 1, "a", []float32{1, 2}),
			indexedSample(0, 0, "b", []float32{4, 0}),
			indexedSample(1, 0, "a", []float32{1, 2}),
			indexedSample(1, 0, "b", []float32{4, 8}),
		)
		p := ActivationErrorProfile(ref, got, tol)
		if p.Verdict != TraceDivergent || p.Rows[0].NonFinite != 1 || p.Rows[0].Match {
			t.Fatalf("NaN profile verdict=%s row=%+v", p.Verdict, p.Rows[0])
		}
		if _, err := json.Marshal(p); err != nil {
			t.Fatalf("profile not JSON-encodable: %v", err)
		}
		if n := got.ClampNonFinite(); n != 1 || got.Events[0].Status != NonFiniteClampedStatus {
			t.Fatalf("clamped=%d status=%q", n, got.Events[0].Status)
		}
		if _, err := json.Marshal(got); err != nil {
			t.Fatalf("clamped trace not JSON-encodable: %v", err)
		}
	})

	t.Run("missing identity and dropped samples are unproven", func(t *testing.T) {
		got := collect(ref.Events[:4]...)
		got.DroppedSampleValues = 2
		p := ActivationErrorProfile(ref, got, tol)
		if p.Verdict != TraceIncomparable || !p.Incomplete || p.MissingIdentities != 1 {
			t.Fatalf("verdict=%s incomplete=%v missing=%d", p.Verdict, p.Incomplete, p.MissingIdentities)
		}
	})
}

func TestActivationAdoptDigests(t *testing.T) {
	src := collect(indexedSample(0, 0, "a", []float32{1}))
	src.InputTokens = []int{1, 2, 3}
	ref := collect(indexedSample(0, 0, "a", []float32{1}))
	ref.Events[0].InputDigest, ref.Events[0].WeightDigest = "ext", ""
	ref.InputTokens = []int{1, 2, 4}

	if cmp := CompareActivationTraces(ref, src, ActivationTolerance{}); cmp.Verdict != TraceIncomparable {
		t.Fatalf("pre-adoption verdict=%s, want INCOMPARABLE", cmp.Verdict)
	}
	if err := AdoptActivationDigests(&ref, src); !errors.Is(err, ErrAdoptTokensMismatch) {
		t.Fatalf("token mismatch err=%v", err)
	}
	ref.InputTokens = nil
	if err := AdoptActivationDigests(&ref, src); !errors.Is(err, ErrAdoptTokensMissing) {
		t.Fatalf("missing tokens err=%v", err)
	}
	ref.InputTokens = []int{1, 2, 3}
	if err := AdoptActivationDigests(&ref, src); err != nil {
		t.Fatalf("adopt: %v", err)
	}
	if cmp := CompareActivationTraces(ref, src, ActivationTolerance{}); cmp.Verdict != TraceEqual {
		t.Fatalf("post-adoption verdict=%s reason=%s, want EQUAL", cmp.Verdict, cmp.Reason)
	}
	mixed := collect(indexedSample(0, 0, "a", []float32{1}), indexedSample(0, 1, "a", []float32{1}))
	mixed.Events[1].InputDigest = Digest("other")
	mixed.InputTokens = ref.InputTokens
	if err := AdoptActivationDigests(&ref, mixed); !errors.Is(err, ErrAdoptDigestsMixed) {
		t.Fatalf("mixed digests err=%v", err)
	}
}

func TestActivationRecorderOperationAllowlist(t *testing.T) {
	const layers, tokens = 41, 7
	r, disable := EnableWithOptions(layers*tokens, RecorderOptions{MaxSamples: layers * tokens * 8, Operations: []string{"v41.activation"}}, "run", "req")
	for l := 0; l < layers; l++ {
		for tok := 0; tok < tokens; tok++ {
			for k := 0; k < 10; k++ {
				Record(Event{Operation: "matmul", Phase: "kernel"})
			}
			e := indexedSample(l, tok, "residual_streams", make([]float32, 8))
			e.Operation = "v41.activation"
			Record(e)
		}
	}
	disable()
	a := r.Artifact()
	if len(a.Events) != layers*tokens || a.Dropped != 0 || a.DroppedSampleValues != 0 {
		t.Fatalf("events=%d dropped=%d dropped_samples=%d, want %d/0/0", len(a.Events), a.Dropped, a.DroppedSampleValues, layers*tokens)
	}
	for _, e := range a.Events {
		if e.Operation != "v41.activation" {
			t.Fatalf("allowlisted recorder retained %q", e.Operation)
		}
	}
	if last := a.Events[len(a.Events)-1]; last.Layer != layers-1 || last.Token != tokens-1 {
		t.Fatalf("last event layer=%d token=%d, want %d/%d", last.Layer, last.Token, layers-1, tokens-1)
	}
	if plain := New(1); func() bool { plain.Record(Event{Operation: "matmul"}); return len(plain.Artifact().Events) != 1 }() {
		t.Fatal("a recorder without an allowlist must keep every operation")
	}
}
