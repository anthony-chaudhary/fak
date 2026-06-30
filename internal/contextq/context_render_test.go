package contextq

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func TestRenderKnownUnknownAssumedContext(t *testing.T) {
	res := Result{
		Query: "refund fee",
		Slices: []SliceRef{
			{
				Step: 4, Role: "tool", Descriptor: "refund_fee.md", Bytes: 120,
				Source:         testEntryID("source-known"),
				ViewID:         "view-refund",
				MaterializedBy: MaterializationHit,
			},
		},
		Views: []MemoryViewRecord{
			{ViewID: "view-refund", CacheEntry: cachemeta.Entry{ID: testEntryID("view-known")}},
		},
		Refused: []Refusal{
			{Step: 7, Role: "tool", Descriptor: "poison.md", Reason: "sealed_by_trust_gate", Entry: testEntryID("refused")},
		},
		Omissions: []Omission{
			{Step: 9, Role: "assistant", Descriptor: "old-tail", Reason: "budget_exhausted", Entry: testEntryID("omitted")},
		},
	}
	assumed := []AssumedContext{
		{Key: "customer-tier", Statement: "customer is premium", Source: "user_stated", Confidence: 0.91, Action: "use"},
	}

	rendered := RenderKnownUnknownAssumedContext(res, assumed)
	if rendered.Query != "refund fee" {
		t.Fatalf("query = %q", rendered.Query)
	}
	if len(rendered.Known) != 1 {
		t.Fatalf("known rows = %+v, want one materialized evidence row", rendered.Known)
	}
	known := rendered.Known[0]
	if known.Step != 4 || known.Descriptor != "refund_fee.md" || known.MaterializedBy != MaterializationHit {
		t.Fatalf("known row = %+v", known)
	}
	if known.SourceDigest != "source-known" || known.ViewDigest != "view-known" {
		t.Fatalf("known row did not carry source/view digests: %+v", known)
	}
	if len(rendered.Unknown) != 2 {
		t.Fatalf("unknown rows = %+v, want refusal and omission", rendered.Unknown)
	}
	if len(rendered.Assumed) != 1 || rendered.Assumed[0].Key != "customer-tier" {
		t.Fatalf("assumed rows = %+v", rendered.Assumed)
	}

	md := rendered.Markdown()
	for _, want := range []string{
		"# context evidence split",
		"known: 1",
		"unknown: 2",
		"assumed: 1",
		"refund_fee.md",
		"sealed_by_trust_gate",
		"budget_exhausted",
		"customer is premium",
		"0.91",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func TestRenderContextDoesNotPromoteUnknownToKnown(t *testing.T) {
	res := Result{
		Query: "trust violation",
		Slices: []SliceRef{
			{Step: 1, Role: "user", Descriptor: "safe.md", Source: testEntryID("safe"), MaterializedBy: MaterializationFault},
		},
		Refused: []Refusal{
			{Step: 2, Role: "tool", Descriptor: "sealed.md", Reason: "sealed_by_trust_gate", Entry: testEntryID("sealed")},
		},
		Omissions: []Omission{
			{Step: 3, Role: "assistant", Descriptor: "not-selected.md", Reason: "not_selected_by_ranker", Entry: testEntryID("not-selected")},
		},
	}

	rendered := RenderKnownUnknownAssumedContext(res, []AssumedContext{
		{Key: "maybe-target", Statement: "target may be production", Confidence: 1.7},
		{Statement: "   "},
	})
	if len(rendered.Known) != 1 || rendered.Known[0].Descriptor != "safe.md" {
		t.Fatalf("known rows = %+v, want only the materialized safe slice", rendered.Known)
	}
	for _, row := range rendered.Known {
		if row.Descriptor == "sealed.md" || row.Descriptor == "not-selected.md" {
			t.Fatalf("unknown descriptor promoted into known rows: %+v", rendered.Known)
		}
	}
	if len(rendered.Unknown) != 2 {
		t.Fatalf("unknown rows = %+v, want refusal and omission", rendered.Unknown)
	}
	if len(rendered.Assumed) != 1 {
		t.Fatalf("assumption cleanup = %+v, want one non-empty assumption", rendered.Assumed)
	}
	if rendered.Assumed[0].Confidence != 1 {
		t.Fatalf("assumption confidence = %f, want clamped to 1", rendered.Assumed[0].Confidence)
	}
}

func testEntryID(digest string) cachemeta.EntryID {
	return cachemeta.EntryID{
		Digest:    digest,
		MediaType: cachemeta.MediaRecallPage,
		Length:    16,
		Unit:      cachemeta.UnitBytes,
	}
}
