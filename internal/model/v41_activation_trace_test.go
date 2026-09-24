package model

// v41_activation_trace_test.go -- the #13325 witness for the bounded V4.1
// activation-checkpoint producer.
//
// The test is deliberately built on PRE-EXISTING API only (computetrace's
// Enable/Artifact/CompareActivationTraces plus the reduced model fixture), so it
// compiles against the parent commit and fails at RUNTIME there: no V4.1 forward
// emits activation samples on the parent, so the enabled trace is empty and every
// stage assertion below fails closed. That red-then-green runtime symptom is the
// land gate's required witness, not a compile error.
//
// It drives the real Model.Forward path (forwardV41 -> v41Layer) and asserts the
// four boundary identities the producer must emit -- q_latent, kv_latent,
// attention_projected, moe_sum -- with their expected widths, a stable
// deterministic sample-index pattern, and bound provenance. An independently
// perturbed copy of one expected attention checkpoint must be reported as the
// FIRST sampled divergence, and a cap-truncated trace must be INCOMPARABLE rather
// than read as agreement.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func TestV41ActivationTrace(t *testing.T) {
	m := v41ReducedModel(t)
	ids := []int{1, 3, 5}
	cfg := m.Cfg

	wantStages := []string{"q_latent", "kv_latent", "attention_projected", "moe_sum"}
	wantWidth := map[string]int{
		"q_latent":            cfg.QLoraRank,
		"kv_latent":           cfg.HeadDim,
		"attention_projected": cfg.HiddenSize,
		"moe_sum":             cfg.HiddenSize,
	}
	tol := computetrace.ActivationTolerance{Atol: 1e-6, Rtol: 1e-6}

	t.Run("disabled records nothing and preserves logits", func(t *testing.T) {
		base := m.Forward(ids)
		baseAllocs := testing.AllocsPerRun(3, func() { _ = m.Forward(ids) })

		rec, disable := computetrace.Enable(0, "run-disabled", "req-disabled")
		defer disable()
		if computetrace.Enabled() {
			t.Fatal("a zero-limit recorder must report disabled")
		}
		if got := m.Forward(ids); !v41LogitsEqual(base, got) {
			t.Fatal("tracing disabled changed the forward logits")
		}
		if n := len(rec.Artifact().Events); n != 0 {
			t.Fatalf("a disabled recorder retained %d events, want 0", n)
		}
		disabledAllocs := testing.AllocsPerRun(3, func() { _ = m.Forward(ids) })
		if disabledAllocs != baseAllocs {
			t.Fatalf("disabled tracing added forward allocations: %v vs untraced %v", disabledAllocs, baseAllocs)
		}
	})

	var full computetrace.Artifact
	t.Run("enabled emits ordered stage records with provenance and shape", func(t *testing.T) {
		rec, disable := computetrace.Enable(4096, "run-full", "req-full")
		defer disable()
		_ = m.Forward(ids)
		full = rec.Artifact()

		if want := len(wantStages) * len(ids); len(full.Events) != want {
			t.Fatalf("emitted %d activation events, want %d (stages x tokens)", len(full.Events), want)
		}
		var inputDigest, weightDigest string
		for si, stage := range wantStages {
			width := wantWidth[stage]
			for ti := 0; ti < len(ids); ti++ {
				e := full.Events[si*len(ids)+ti]
				if e.Layer != 0 || e.Token != ti {
					t.Fatalf("event[%d] identity layer=%d token=%d, want layer=0 token=%d", si*len(ids)+ti, e.Layer, e.Token, ti)
				}
				if e.Phase != stage {
					t.Fatalf("event[%d] stage=%q, want %q", si*len(ids)+ti, e.Phase, stage)
				}
				if e.Operation != "v41.activation" || e.Backend != "host" || e.Device != "host" {
					t.Fatalf("event[%d] provenance identity op=%q backend=%q device=%q", si*len(ids)+ti, e.Operation, e.Backend, e.Device)
				}
				if len(e.Shapes) != 1 || len(e.Shapes[0]) != 1 || e.Shapes[0][0] != width {
					t.Fatalf("event[%d] stage=%s shape=%v, want [[%d]]", si*len(ids)+ti, stage, e.Shapes, width)
				}
				if len(e.SampleValues) == 0 || len(e.SampleIndices) != len(e.SampleValues) {
					t.Fatalf("event[%d] stage=%s samples=%d indices=%d, want non-empty parallel slices", si*len(ids)+ti, stage, len(e.SampleValues), len(e.SampleIndices))
				}
				prev := -1
				for _, idx := range e.SampleIndices {
					if idx <= prev || idx >= width {
						t.Fatalf("event[%d] stage=%s sample index %d not strictly increasing in [0,%d)", si*len(ids)+ti, stage, idx, width)
					}
					prev = idx
				}
				if e.InputDigest == "" || e.WeightDigest == "" || e.ProvenanceDigest == "" {
					t.Fatalf("event[%d] stage=%s has incomplete provenance input=%q weight=%q provenance=%q", si*len(ids)+ti, stage, e.InputDigest, e.WeightDigest, e.ProvenanceDigest)
				}
				if inputDigest == "" {
					inputDigest, weightDigest = e.InputDigest, e.WeightDigest
				} else if e.InputDigest != inputDigest || e.WeightDigest != weightDigest {
					t.Fatalf("event[%d] stage=%s provenance drifted within one forward", si*len(ids)+ti, stage)
				}
			}
		}
	})

	t.Run("a repeat forward is EQUAL to the captured reference", func(t *testing.T) {
		if full.Schema != computetrace.Schema {
			t.Fatalf("captured schema=%q, want %q", full.Schema, computetrace.Schema)
		}
		rec, disable := computetrace.Enable(4096, "run-repeat", "req-repeat")
		defer disable()
		_ = m.Forward(ids)
		if cmp := computetrace.CompareActivationTraces(full, rec.Artifact(), tol); cmp.Verdict != computetrace.TraceEqual {
			t.Fatalf("repeat forward verdict=%q reason=%q, want EQUAL", cmp.Verdict, cmp.Reason)
		}
	})

	t.Run("an independently perturbed attention checkpoint is the first divergence", func(t *testing.T) {
		got := full
		got.Events = append([]computetrace.Event(nil), full.Events...)
		at := -1
		for i := range got.Events {
			if got.Events[i].Phase == "attention_projected" && got.Events[i].Token == 0 {
				at = i
				break
			}
		}
		if at < 0 {
			t.Fatal("no attention_projected event for token 0 to perturb")
		}
		got.Events[at].SampleValues = append([]float32(nil), got.Events[at].SampleValues...)
		last := len(got.Events[at].SampleValues) - 1
		got.Events[at].SampleValues[last] += 1.0

		cmp := computetrace.CompareActivationTraces(full, got, tol)
		if cmp.Verdict != computetrace.TraceDivergent {
			t.Fatalf("perturbed attention verdict=%q reason=%q, want DIVERGENT", cmp.Verdict, cmp.Reason)
		}
		if cmp.Divergent.Stage != "attention_projected" || cmp.Divergent.Token != 0 {
			t.Fatalf("first divergence=%+v, want stage=attention_projected token=0", cmp.Divergent)
		}
	})

	t.Run("cap truncation is incomplete evidence, never agreement", func(t *testing.T) {
		rec, disable := computetrace.Enable(2, "run-truncated", "req-truncated")
		defer disable()
		_ = m.Forward(ids)
		trunc := rec.Artifact()
		if trunc.Dropped == 0 {
			t.Fatalf("a 2-event cap retained %d events with no dropped-event counter", len(trunc.Events))
		}
		if cmp := computetrace.CompareActivationTraces(full, trunc, tol); cmp.Verdict != computetrace.TraceIncomparable {
			t.Fatalf("truncated-trace verdict=%q, want INCOMPARABLE", cmp.Verdict)
		}
	})
}
