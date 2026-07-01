package ctxplan

import (
	"context"
	"strings"
	"testing"
)

// buildPreviewFixture assembles the same four-area shape layout_test.go uses (a base pin, a
// deep relevant span reached only as a pointer, a run of old irrelevant filler, a recent
// span, and a current span), so the preview test exercises all five regions: pinned
// (base+current), recent, deep (nothing here — the deep candidate is pointer-only), elided
// (the irrelevant filler that loses the knapsack), and query-needed (the pointer span).
func buildPreviewFixture(t *testing.T) (spans []Span, layout Layout, f Forecast) {
	t.Helper()
	ctx := context.Background()
	st := NewMemStore()
	st.Add("system", DurabilityDurable, []byte("base system prompt: obey the policy"), false)              // span:0 base pin
	st.Add("user", DurabilitySession, []byte("goal: rotate the auth token"), false)                        // span:1
	st.Add("WebSearch", DurabilityDurable, []byte("auth token rotation runbook: mint roll revoke"), false) // span:2 deep pointer
	for i := 0; i < 8; i++ {
		st.Add("Bash", DurabilityTurn, []byte("old irrelevant build log line "+itoaTest(i)), false)
	}
	st.Add("Read", DurabilityTurn, []byte("recent note before current"), false)
	st.Add("Bash", DurabilityTurn, []byte("newest tool result for the current turn"), false)

	var err error
	spans, err = st.Spans(ctx)
	if err != nil {
		t.Fatal(err)
	}
	layout = Layout{
		Base:              AreaPolicy{MaxSpans: 1, Precision: PrecisionExact},
		Current:           AreaPolicy{MaxSpans: 1, Precision: PrecisionExact},
		Recent:            AreaPolicy{MaxSpans: 2, Precision: PrecisionPlanned},
		Deep:              AreaPolicy{MaxSpans: 1, Precision: PrecisionPointer},
		IncludeDurability: []string{DurabilityDurable},
		MaxCandidates:     -1,
	}
	f = Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0"}}
	return spans, layout, f
}

// TestContextPlanPreviewFiveRegions is the issue's own witness surface: a preview over a
// PlanLayout output must classify the base/current pins into pinned, the recency-tail span
// into recent, and the pointer-precision deep candidate into query_needed — never silently
// drop or mislabel a region.
func TestContextPlanPreviewFiveRegions(t *testing.T) {
	spans, layout, f := buildPreviewFixture(t)
	pv := PreviewLayout(spans, f, Budget{Tokens: 999}, nil, layout)

	pinnedIDs := previewIDs(pv.Pinned)
	if !pinnedIDs["span:0"] || !pinnedIDs["span:12"] {
		t.Fatalf("base pin span:0 and current span:12 must both be PINNED, got %+v", pv.Pinned)
	}
	if len(pv.Recent) == 0 {
		t.Fatalf("expected at least one RECENT row, got none: %+v", pv)
	}
	qnIDs := previewIDs(pv.QueryNeeded)
	if !qnIDs["span:2"] {
		t.Fatalf("deep pointer span:2 must be QUERY-NEEDED (never faulted in, needs a follow-up query), got %+v", pv.QueryNeeded)
	}
	if pv.PlanID == "" {
		t.Fatal("preview must carry the plan's deterministic plan_id")
	}
	if !pv.Faithful {
		t.Fatalf("preview of a faithful plan must report faithful=true: %+v", pv)
	}
}

// TestContextPlanPreviewCoversEveryCandidate is the faithfulness-of-the-preview witness: no
// candidate the planner considered may go missing from the rendered dry run. Every row of
// Plan.Selected/Elided must land in EXACTLY one of the five regions.
func TestContextPlanPreviewCoversEveryCandidate(t *testing.T) {
	spans, layout, f := buildPreviewFixture(t)
	p := BuildIndex(spans).PlanLayout(f, Budget{Tokens: 999}, nil, layout)
	pv := PreviewOf(p)

	if got, want := pv.RowCount(), p.Candidates; got != want {
		t.Fatalf("preview row count %d must equal plan candidates %d (a candidate went missing from the dry run)", got, want)
	}
	if !Audit(p).Faithful {
		t.Fatalf("fixture plan should be faithful: %+v", Audit(p))
	}
}

// TestContextPlanPreviewNeverMaterializes proves the preview is a pure read/render: it must
// not require (or accept) a Store at all, so it structurally cannot page bytes in through
// the trust gate. A sealed span reachable by the layout must show up ELIDED with a
// sealed reason, never rendered with content, and PreviewOf/PreviewLayout must not panic or
// need store access to get there.
func TestContextPlanPreviewNeverMaterializes(t *testing.T) {
	ctx := context.Background()
	st := NewMemStore()
	st.Add("system", DurabilityDurable, []byte("base pin"), false)                // span:0
	sealed := st.Add("Bash", DurabilityTurn, []byte("secret leaked token"), true) // span:1 sealed
	st.Add("Bash", DurabilityTurn, []byte("newest current turn result"), false)   // span:2 current
	spans, err := st.Spans(ctx)
	if err != nil {
		t.Fatal(err)
	}
	layout := Layout{
		Base:          AreaPolicy{MaxSpans: 1, Precision: PrecisionExact},
		Current:       AreaPolicy{MaxSpans: 1, Precision: PrecisionExact},
		Recent:        AreaPolicy{MaxSpans: 5, Precision: PrecisionPlanned},
		Deep:          AreaPolicy{MaxSpans: 5, Precision: PrecisionPlanned},
		MaxCandidates: -1,
	}
	f := Forecast{Pins: []string{"span:0"}}
	pv := PreviewLayout(spans, f, Budget{Tokens: 999}, nil, layout)

	for _, row := range pv.Pinned {
		if row.ID == sealed.ID {
			t.Fatalf("sealed span must never appear in PINNED: %+v", row)
		}
	}
	found := false
	for _, row := range pv.Elided {
		if row.ID == sealed.ID {
			found = true
			if row.Reason != ElideSealed {
				t.Fatalf("sealed span must be elided with reason=%s, got %q", ElideSealed, row.Reason)
			}
			if row.Resident {
				t.Fatal("a sealed row must never be marked resident")
			}
		}
	}
	if !found {
		t.Fatalf("sealed span:1 must appear somewhere in the preview (elided), got pinned=%v recent=%v deep=%v elided=%v qn=%v",
			pv.Pinned, pv.Recent, pv.Deep, pv.Elided, pv.QueryNeeded)
	}
}

// TestContextPlanPreviewExplainAndMarkdownRenderAllRegions checks the two human-readable
// renders (Explain for a terminal dry run, Markdown for a shareable report) both name every
// region the issue's Done condition lists, so an operator reading either surface sees
// "what would be paged in or left cold" without re-deriving it from raw JSON.
func TestContextPlanPreviewExplainAndMarkdownRenderAllRegions(t *testing.T) {
	spans, layout, f := buildPreviewFixture(t)
	pv := PreviewLayout(spans, f, Budget{Tokens: 999}, nil, layout)

	explain := pv.Explain()
	for _, want := range []string{"PINNED", "RECENT", "DEEP", "ELIDED", "QUERY-NEEDED"} {
		if !strings.Contains(explain, want) {
			t.Errorf("Explain() missing region marker %q:\n%s", want, explain)
		}
	}
	md := pv.Markdown()
	for _, want := range []string{"Pinned", "Recent", "Deep", "Elided", "Query-needed"} {
		if !strings.Contains(md, want) {
			t.Errorf("Markdown() missing region heading %q:\n%s", want, md)
		}
	}
}

func previewIDs(rows []PreviewRow) map[string]bool {
	out := make(map[string]bool, len(rows))
	for _, r := range rows {
		out[r.ID] = true
	}
	return out
}
