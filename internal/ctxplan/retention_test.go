package ctxplan

import (
	"context"
	"strings"
	"testing"
)

// TestContextPlanPreviewRetentionOutcomes is the #3024 core witness: every requested pin/
// release target gets a typed outcome (honored / missing / released / refused) read from the
// plan's own accounting — no bytes are materialized (PlanCells is pure). In particular a
// sealed retained target is reported refused, never silently honored.
func TestContextPlanPreviewRetentionOutcomes(t *testing.T) {
	st := NewMemStore()
	st.Add("system", DurabilityDurable, []byte("base system prompt"), false)      // span:0
	st.Add("user", DurabilityDurable, []byte("a fact the agent releases"), false) // span:1
	st.Add("tool", DurabilityDurable, []byte("quarantined secret"), true)         // span:2 (sealed)
	spans, _ := st.Spans(context.Background())

	f := Forecast{
		Pins:     []string{"span:0", "span:2", "span:999"},
		Releases: []string{"span:1"},
	}
	plan := PlanCells(spans, f, Budget{Tokens: 1000}, nil)
	got := RetentionOutcomes(plan, f)

	want := map[string]string{
		"span:0":   RetentionHonored,
		"span:1":   RetentionReleased,
		"span:2":   RetentionRefused,
		"span:999": RetentionMissing,
	}
	if len(got) != len(want) {
		t.Fatalf("got %d outcomes, want %d: %+v", len(got), len(want), got)
	}
	for _, o := range got {
		if want[o.ID] != o.Outcome {
			t.Errorf("target %s: outcome=%s, want %s (%+v)", o.ID, o.Outcome, want[o.ID], o)
		}
		if o.ID == "span:2" {
			// Criterion 3: sealed → refused with a real reason, never marked resident.
			if o.Outcome != RetentionRefused || o.Reason != ElideSealed || o.Resident {
				t.Errorf("sealed pinned target must be refused/sealed/non-resident, got %+v", o)
			}
		}
		if o.ID == "span:0" && !o.Resident {
			t.Errorf("honored pin must be marked resident: %+v", o)
		}
	}
}

// TestContextPlanPreviewRetentionOnPreview checks the outcomes ride on the Preview built with a
// forecast (and are rendered in text + markdown), while PreviewOf(plan) alone leaves them nil.
func TestContextPlanPreviewRetentionOnPreview(t *testing.T) {
	st := NewMemStore()
	st.Add("system", DurabilityDurable, []byte("base"), false) // span:0
	st.Add("tool", DurabilityDurable, []byte("sealed"), true)  // span:1 (sealed)
	spans, _ := st.Spans(context.Background())
	f := Forecast{Pins: []string{"span:0", "span:1"}}
	layout := Layout{
		Base:              AreaPolicy{MaxSpans: 4, Precision: PrecisionExact},
		Current:           AreaPolicy{MaxSpans: 2, Precision: PrecisionExact},
		Recent:            AreaPolicy{MaxSpans: 4, Precision: PrecisionPlanned},
		Deep:              AreaPolicy{MaxSpans: 4, Precision: PrecisionPointer},
		IncludeDurability: []string{DurabilityDurable, DurabilityBounded},
		MaxCandidates:     -1,
	}
	pv := PreviewLayout(spans, f, Budget{Tokens: 500}, nil, layout)
	if len(pv.Retention) != 2 {
		t.Fatalf("preview should carry 2 retention outcomes, got %d: %+v", len(pv.Retention), pv.Retention)
	}

	plan := BuildIndex(spans).PlanLayout(f, Budget{Tokens: 500}, nil, layout)
	if PreviewOf(plan).Retention != nil {
		t.Error("PreviewOf(plan) must leave Retention nil — it has no forecast to know requested targets")
	}
	if !strings.Contains(pv.Explain(), "RETENTION") {
		t.Errorf("Explain output missing RETENTION section:\n%s", pv.Explain())
	}
	if !strings.Contains(pv.Markdown(), "Retention / pin outcomes") {
		t.Errorf("Markdown output missing retention section:\n%s", pv.Markdown())
	}
}
