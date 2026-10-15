package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// One bounded CPU-storage witness: retention changes only, with the original
// device-seam collapse count, trace samples, and failure boundary preserved.
// fak-test:runtime medium est=6s lane=default
func TestV41PrefillHiddenRetention(t *testing.T) {
	ids := []int{1, 2}
	build := func() *Model { return v41MHCProjVariant(t, "F32", false, false, 2) }
	var reference *Activations
	var referenceTrace computetrace.Artifact
	for _, lastHeadOnly := range []bool{false, true} {
		m := build()
		b := newV41MHCTestBackend(true)
		s := v41DenseTestSession(t, m, b)
		rec, disable := computetrace.Enable(4096, "hidden-retention", "hidden-retention")
		act, err := m.forwardV41WithHeadSelection(ids, s.v41State(), lastHeadOnly)
		trace := rec.Artifact()
		disable()
		if err != nil {
			t.Fatal(err)
		}
		wantPanels := m.Cfg.NumLayers + 1
		if lastHeadOnly {
			wantPanels = 1
		}
		if len(act.Hidden) != wantPanels {
			t.Fatalf("lastHeadOnly=%t retained %d hidden panels, want %d", lastHeadOnly, len(act.Hidden), wantPanels)
		}
		for _, hidden := range act.Hidden {
			if len(hidden) != len(ids)*m.Cfg.HiddenSize {
				t.Fatalf("hidden width=%d", len(hidden))
			}
		}
		// Every layer still performs attention, FFN and final hidden collapse.
		// The final collapse's checked readback and finite validation are retained.
		if b.calls["pre"] != 3*m.Cfg.NumLayers || len(b.outputs) != 0 {
			t.Fatalf("lastHeadOnly=%t collapse calls=%d live outputs=%d", lastHeadOnly, b.calls["pre"], len(b.outputs))
		}
		if !lastHeadOnly {
			reference, referenceTrace = act, trace
			public := m.Forward(ids)
			if len(public.Hidden) != wantPanels {
				t.Fatal("Model.Forward lost its full diagnostic hidden panels")
			}
			continue
		}
		if !reflect.DeepEqual(lastLogits(act), lastLogits(reference)) ||
			!reflect.DeepEqual(act.Hidden[0], reference.Hidden[len(reference.Hidden)-1]) {
			t.Fatal("serving head input or logits changed")
		}
		if cmp := computetrace.CompareActivationTraces(referenceTrace, trace, computetrace.ActivationTolerance{}); cmp.Verdict != computetrace.TraceEqual {
			t.Fatalf("serving trace changed: %s: %s", cmp.Verdict, cmp.Reason)
		}
		if !s.v41IncrementalEligible() {
			t.Fatal("serving prefill did not retain continuation state")
		}
	}

	// Hand-built full geometry can admit an empty decoder stack. Its head still
	// consumes the embedding panel, even when only the last logits are wanted.
	empty := build()
	empty.Cfg.NumLayers = 0
	empty.Cfg.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
	all, err := empty.forwardV41WithHeadSelection(ids, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	last, err := empty.forwardV41WithHeadSelection(ids, nil, true)
	if err != nil || len(last.Hidden) != 1 || !reflect.DeepEqual(last.Hidden, all.Hidden) || !reflect.DeepEqual(lastLogits(last), lastLogits(all)) {
		t.Fatalf("empty-stack head changed: act=%v err=%v", last != nil, err)
	}

	// A layer-zero hidden collapse still fails before any layer-one work, even
	// though its output is no longer retained for the serving result.
	m := build()
	b := newV41MHCTestBackend(true)
	b.failOp, b.failCall = "pre", 3
	b.cause = &compute.BackendError{Backend: "cpu-ref-mhc", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	s := v41DenseTestSession(t, m, b)
	act, err := m.forwardV41WithHeadSelection(ids, s.v41State(), true)
	var selected *V41MHCOperationError
	if act != nil || !errors.Is(err, b.cause) || !errors.As(err, &selected) || selected.Layer != 0 ||
		b.calls["pre"] != 3 || b.calls["rms-split"] != 2 || len(s.v41State().history) != 0 || !s.BackendSessionClosed() {
		t.Fatalf("intermediate validation failure changed: act=%v err=%v calls=%v history=%v closed=%t", act != nil, err, b.calls, s.v41State().history, s.BackendSessionClosed())
	}
}
