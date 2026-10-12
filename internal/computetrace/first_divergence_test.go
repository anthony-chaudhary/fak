package computetrace

import (
	"math"
	"slices"
	"testing"
)

// qwenPair converts the qwen4exp fixture and returns it as an adopted reference
// plus an identical fak-shaped actual trace.
func qwenPair(t *testing.T) (Artifact, Artifact) {
	t.Helper()
	ref, err := ConvertLlamaCppDumpArch(writeQwen4ExpDump(t, []int{11, 12, 13}), LlamaCppArchQwen4Exp)
	if err != nil {
		t.Fatal(err)
	}
	got := Artifact{Schema: Schema, InputTokens: slices.Clone(ref.InputTokens)}
	for _, e := range ref.Events {
		e.InputDigest, e.WeightDigest = Digest("fak-input"), Digest("fak-weights")
		e.SampleValues = slices.Clone(e.SampleValues)
		e.ExcludedSampleIndices = nil
		got.Events = append(got.Events, e)
	}
	if err := AdoptActivationDigests(&ref, got); err != nil {
		t.Fatal(err)
	}
	return ref, got
}

func perturb(a *Artifact, layer int, stage string, scale float32) {
	for i := range a.Events {
		e := &a.Events[i]
		if e.Layer == layer && e.Phase == stage {
			for j := range e.SampleValues {
				e.SampleValues[j] = e.SampleValues[j]*scale + 0.5
			}
		}
	}
}

var testDivTol = DivergenceTolerance{RelL2: 1e-3}

func TestFirstDivergenceReportsInjectedLayer(t *testing.T) {
	ref, got := qwenPair(t)
	if r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{}); r.Verdict != TraceEqual || r.ParityLayers != qwenFixtureLayers+1 || r.FirstDivergent != nil {
		t.Fatalf("identical traces: verdict=%s parity=%d first=%v reason=%s", r.Verdict, r.ParityLayers, r.FirstDivergent, r.Reason)
	}

	perturb(&got, 30, "residual_streams", 1.5)
	perturb(&got, 17, "moe_sum", 1.5)
	r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
	if r.Verdict != TraceDivergent || r.FirstDivergent == nil || *r.FirstDivergent != (ActivationStage{Layer: 17, Stage: "moe_sum"}) {
		t.Fatalf("verdict=%s first=%v", r.Verdict, r.FirstDivergent)
	}
	if r.ParityLayers != 17 {
		t.Fatalf("parity layers=%d, want 17", r.ParityLayers)
	}
	verdicts := map[int]string{}
	for _, l := range r.Layers {
		verdicts[l.Layer] = l.Verdict
	}
	// No short-circuit: every layer is measured, and the later injection shows.
	if len(r.Layers) != qwenFixtureLayers+2 || verdicts[16] != RowMatch || verdicts[17] != RowDivergent || verdicts[18] != RowMatch || verdicts[30] != RowDivergent || verdicts[47] != RowMatch {
		t.Fatalf("layer verdicts=%v (%d layers)", verdicts, len(r.Layers))
	}
	for _, row := range r.Rows {
		if row.Layer == 17 && row.Stage == "moe_sum" && (row.RelL2 <= testDivTol.RelL2 || row.Tokens != 3) {
			t.Fatalf("injected row=%+v", row)
		}
	}
	if r.Strict.Verdict != TraceDivergent || r.Strict.Divergent.Layer != 17 {
		t.Fatalf("strict=%+v", r.Strict)
	}
}

func TestFirstDivergenceCosineGate(t *testing.T) {
	ref, got := qwenPair(t)
	// Scaling keeps the direction (cosine 1) but moves the norm.
	for i := range got.Events {
		e := &got.Events[i]
		if e.Layer == 5 && e.Phase == "moe_sum" {
			for j := range e.SampleValues {
				e.SampleValues[j] *= -1
			}
		}
	}
	r := FirstDivergence(ref, got, DivergenceTolerance{RelL2: 3, MinCosine: 0.9}, ActivationTolerance{})
	if r.FirstDivergent == nil || r.FirstDivergent.Layer != 5 {
		t.Fatalf("cosine gate missed the sign flip: %v", r.FirstDivergent)
	}
	if r := FirstDivergence(ref, got, DivergenceTolerance{RelL2: 3}, ActivationTolerance{}); r.Verdict != TraceEqual {
		t.Fatalf("rel_l2 3 accepts a sign flip without a cosine gate: %s", r.Verdict)
	}
}

func TestFirstDivergenceFailsClosed(t *testing.T) {
	ref, got := qwenPair(t)
	if r := FirstDivergence(ref, got, DivergenceTolerance{RelL2: math.NaN()}, ActivationTolerance{}); r.Verdict != TraceIncomparable {
		t.Fatalf("NaN tolerance verdict=%s", r.Verdict)
	}
	missing := got
	missing.Events = slices.DeleteFunc(slices.Clone(got.Events), func(e Event) bool { return e.Layer == 9 && e.Phase == "moe_sum" && e.Token == 1 })
	r := FirstDivergence(ref, missing, testDivTol, ActivationTolerance{})
	if r.Verdict != TraceIncomparable || r.ParityLayers != 9 || r.Missing != 1 {
		t.Fatalf("missing identity: verdict=%s parity=%d missing=%d", r.Verdict, r.ParityLayers, r.Missing)
	}
	unadopted, _ := ConvertLlamaCppDumpArch(writeQwen4ExpDump(t, []int{11, 12, 13}), LlamaCppArchQwen4Exp)
	if r := FirstDivergence(unadopted, got, testDivTol, ActivationTolerance{}); r.Verdict != TraceIncomparable || r.ParityLayers != 0 {
		t.Fatalf("unadopted digests: verdict=%s parity=%d", r.Verdict, r.ParityLayers)
	}
	nonFinite := got
	nonFinite.Events = slices.Clone(got.Events)
	for i := range nonFinite.Events {
		if e := &nonFinite.Events[i]; e.Layer == 2 && e.Phase == "gdn_z" && e.Token == 0 {
			e.SampleValues = slices.Clone(e.SampleValues)
			e.SampleValues[0] = float32(math.Inf(1))
		}
	}
	if r := FirstDivergence(ref, nonFinite, testDivTol, ActivationTolerance{}); r.FirstDivergent == nil || *r.FirstDivergent != (ActivationStage{Layer: 2, Stage: "gdn_z"}) {
		t.Fatalf("non-finite sample: %v", r.FirstDivergent)
	}
}

// The deepseek41 kv RoPE tail is excluded: a difference there is not a divergence.
func TestFirstDivergenceHonoursExcludedSamples(t *testing.T) {
	ref, err := ConvertLlamaCppDump("testdata/llamacpp-deepseek41-dump")
	if err != nil {
		t.Fatal(err)
	}
	got := Artifact{Schema: Schema, InputTokens: slices.Clone(ref.InputTokens)}
	for _, e := range ref.Events {
		e.InputDigest, e.WeightDigest = Digest("in"), Digest("w")
		e.SampleValues = slices.Clone(e.SampleValues)
		if e.Phase == "kv_latent" {
			e.SampleValues[len(e.SampleValues)-1] += 100
		}
		got.Events = append(got.Events, e)
	}
	if err := AdoptActivationDigests(&ref, got); err != nil {
		t.Fatal(err)
	}
	r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
	if r.Verdict != TraceEqual || r.ParityLayers != 2 {
		t.Fatalf("verdict=%s parity=%d first=%v", r.Verdict, r.ParityLayers, r.FirstDivergent)
	}
	if r.Strict.Verdict != TraceDivergent {
		t.Fatalf("strict comparison ignores exclusions and must still flag the tail: %s", r.Strict.Verdict)
	}
}

// Within a layer the first divergence follows that layer's execution order,
// not the order a stage name first appeared anywhere in the trace.
func TestFirstDivergencePerLayerStageOrder(t *testing.T) {
	for _, tc := range []struct {
		layer        int
		early, later string
	}{
		{17, "moe_sum", "residual_streams"},
		{3, "attn_core", "moe_sum"},
	} {
		ref, got := qwenPair(t)
		perturb(&got, tc.layer, tc.later, 1.5)
		perturb(&got, tc.layer, tc.early, 1.5)
		r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
		if r.FirstDivergent == nil || *r.FirstDivergent != (ActivationStage{Layer: tc.layer, Stage: tc.early}) {
			t.Fatalf("layer %d: first=%v, want %s", tc.layer, r.FirstDivergent, tc.early)
		}
	}
}

func TestFirstDivergenceRefusalDoesNotHideDivergence(t *testing.T) {
	ref, got := qwenPair(t)
	perturb(&got, 6, "moe_sum", 1.5)
	for i := range got.Events {
		if e := &got.Events[i]; e.Layer == 6 && e.Phase == "moe_sum" && e.Token == 0 {
			e.InputDigest = Digest("other-input")
		}
	}
	r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
	if r.FirstDivergent == nil || r.FirstDivergent.Layer != 6 {
		t.Fatalf("first=%v verdict=%s", r.FirstDivergent, r.Verdict)
	}
	for _, row := range r.Rows {
		if row.Layer == 6 && row.Stage == "moe_sum" && (row.Refusals != 1 || row.Verdict != RowDivergent) {
			t.Fatalf("row=%+v", row)
		}
	}
}

func TestFirstDivergenceParityStopsAtLayerGap(t *testing.T) {
	ref, got := qwenPair(t)
	drop := func(e Event) bool { return e.Layer == 4 }
	ref.Events = slices.DeleteFunc(ref.Events, drop)
	got.Events = slices.DeleteFunc(got.Events, drop)
	if r := FirstDivergence(ref, got, testDivTol, ActivationTolerance{}); r.ParityLayers != 4 {
		t.Fatalf("parity=%d across a missing layer 4, want 4", r.ParityLayers)
	}
}

func TestFirstDivergenceClampedAndExtraEvidence(t *testing.T) {
	ref, got := qwenPair(t)
	clamped := got
	clamped.Events = slices.Clone(got.Events)
	for i := range clamped.Events {
		if e := &clamped.Events[i]; e.Layer == 8 && e.Phase == "moe_sum" && e.Token == 2 {
			e.SampleValues = slices.Clone(e.SampleValues)
			e.SampleValues[0] = float32(math.NaN())
		}
	}
	if n := clamped.ClampNonFinite(); n != 1 {
		t.Fatalf("clamped %d", n)
	}
	// Clamp the same sample on both sides: equal sentinels must not read as parity.
	refClamped := ref
	refClamped.Events = slices.Clone(ref.Events)
	for i := range refClamped.Events {
		if e := &refClamped.Events[i]; e.Layer == 8 && e.Phase == "moe_sum" && e.Token == 2 {
			e.SampleValues = slices.Clone(e.SampleValues)
			e.SampleValues[0] = float32(math.NaN())
		}
	}
	refClamped.ClampNonFinite()
	if r := FirstDivergence(refClamped, clamped, testDivTol, ActivationTolerance{}); r.FirstDivergent == nil || r.FirstDivergent.Layer != 8 {
		t.Fatalf("clamped samples: first=%v verdict=%s", r.FirstDivergent, r.Verdict)
	}
	extra := got
	extra.Events = append(slices.Clone(got.Events), Event{Phase: "moe_sum", Layer: 99, SampleValues: []float32{1}, InputDigest: got.Events[0].InputDigest, WeightDigest: got.Events[0].WeightDigest})
	if r := FirstDivergence(ref, extra, testDivTol, ActivationTolerance{}); r.Verdict != TraceIncomparable || r.Extra != 1 {
		t.Fatalf("extra identity: verdict=%s extra=%d", r.Verdict, r.Extra)
	}
	dropped := got
	dropped.Dropped = 1
	if r := FirstDivergence(ref, dropped, testDivTol, ActivationTolerance{}); r.Verdict != TraceIncomparable || !r.Incomplete {
		t.Fatalf("dropped events: verdict=%s incomplete=%t", r.Verdict, r.Incomplete)
	}
}
