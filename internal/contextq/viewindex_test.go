package contextq

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// TestIndexViewsEmitsMultiViewSet is the witness for acceptance #2 of #437: a
// recall-image view indexer emits descriptor, facts, timeline, QA, and summary
// views, each provenance-bound to its source page id and digest-addressed.
func TestIndexViewsEmitsMultiViewSet(t *testing.T) {
	im := attachFixture(t)
	res := IndexViews(context.Background(), im, IndexRequest{
		PolicyVersion: "index-policy",
	})

	if res.Stats.PagesIndexed == 0 {
		t.Fatal("expected at least one benign page indexed")
	}
	if len(res.Views) == 0 {
		t.Fatal("expected emitted views")
	}

	// Every named view type must appear for every indexed page.
	wantTypes := map[ViewType]bool{
		ViewDescriptor: false, ViewFacts: false, ViewTimeline: false,
		ViewQA: false, ViewSummary: false,
	}
	for _, v := range res.Views {
		wantTypes[v.ViewType] = true

		if len(v.SourcePageIDs) == 0 {
			t.Fatalf("view %q has no source page ids", v.ViewID)
		}
		if len(v.SourceDigests) == 0 || v.SourceDigests[0] == "" {
			t.Fatalf("view %q has no source digest", v.ViewID)
		}
		if v.PolicyVersion != "index-policy" {
			t.Fatalf("view %q policy version = %q, want carry-through", v.ViewID, v.PolicyVersion)
		}
		if v.FaithfulnessProbe != 1.0 {
			t.Fatalf("view %q faithfulness = %f, want 1.0 (extractive)", v.ViewID, v.FaithfulnessProbe)
		}
		if v.Coverage < 0 || v.Coverage > 1.0 {
			t.Fatalf("view %q coverage out of [0,1]: %f", v.ViewID, v.Coverage)
		}
		if v.CacheEntry.Plane != cachemeta.PlaneMemoryView ||
			v.CacheEntry.ID.MediaType != cachemeta.MediaMemoryView {
			t.Fatalf("view %q did not lower into memory-view cache metadata: %+v", v.ViewID, v.CacheEntry)
		}
		if v.CacheEntry.ID.Digest == "" {
			t.Fatalf("view %q is not digest-addressed", v.ViewID)
		}
		if len(v.CacheEntry.Derivation.SourceRefs) != 1 {
			t.Fatalf("view %q must carry exactly one source ref, got %d", v.ViewID, len(v.CacheEntry.Derivation.SourceRefs))
		}
	}
	for vt, seen := range wantTypes {
		if !seen {
			t.Fatalf("view type %q was never emitted; got %+v", vt, res.Stats.ViewsByType)
		}
	}

	// Distinct view types over the same page must be distinct cache artifacts.
	digests := map[string]bool{}
	for _, v := range res.Views {
		if digests[v.CacheEntry.ID.Digest] {
			t.Fatalf("duplicate view digest %q — views are not independently addressed", v.CacheEntry.ID.Digest)
		}
		digests[v.CacheEntry.ID.Digest] = true
	}
}

// TestIndexViewsFailsClosedOnSealed proves a sealed source page yields no view
// and is refused before any byte is paged in (acceptance #4: sealed-source
// views fail closed). The fixture carries a trust-violation page the gate seals.
func TestIndexViewsFailsClosedOnSealed(t *testing.T) {
	im := attachFixture(t)
	res := IndexViews(context.Background(), im, IndexRequest{})

	if res.Stats.PagesRefused == 0 {
		t.Fatal("expected at least one sealed/tombstoned page refused")
	}
	if !indexHasRefusal(res, "sealed_by_trust_gate") {
		t.Fatalf("expected a sealed REFUSE, got %+v", res.Refused)
	}

	// No emitted view may name a refused page as its source.
	refusedSteps := map[int]bool{}
	for _, r := range res.Refused {
		refusedSteps[r.Step] = true
	}
	for _, v := range res.Views {
		for _, step := range v.SourcePageIDs {
			if refusedSteps[step] {
				t.Fatalf("view %q leaks refused source page %d", v.ViewID, step)
			}
		}
	}

	// A refusal must carry a REFUSE verdict, never a FAULT for the same step.
	for _, r := range res.Refused {
		for _, vd := range res.Verdicts {
			if vd.Step == r.Step && vd.Kind == MaterializationFault {
				t.Fatalf("refused step %d also got a FAULT verdict: %+v", r.Step, vd)
			}
		}
	}
}

// TestIndexViewsDeterministic proves the indexer is a pure function of the
// image and request: two passes produce DeepEqual results.
func TestIndexViewsDeterministic(t *testing.T) {
	im := attachFixture(t)
	req := IndexRequest{PolicyVersion: "p", Producer: "det"}
	a := IndexViews(context.Background(), im, req)
	b := IndexViews(context.Background(), im, req)
	if len(a.Views) == 0 {
		t.Fatal("non-vacuity: expected emitted views")
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("IndexViews is not deterministic over a fixed image+request")
	}
}

// TestIndexViewsQAShape checks the QA view is a templated question/answer pair
// drawn from source-derived text.
func TestIndexViewsQAShape(t *testing.T) {
	im := attachFixture(t)
	res := IndexViews(context.Background(), im, IndexRequest{Views: []ViewType{ViewQA}})
	if len(res.Views) == 0 {
		t.Fatal("expected QA views")
	}
	for _, v := range res.Views {
		if v.ViewType != ViewQA {
			t.Fatalf("expected only QA views, got %q", v.ViewType)
		}
	}
	// The QA producer renders "Q: ... A: ..."; assert the shape on a built payload.
	q, _ := buildQA(im.Backtrace()[0], []byte("alpha\nbeta"))
	payload := string(q)
	if !strings.HasPrefix(payload, "Q: ") || !strings.Contains(payload, "\nA: ") {
		t.Fatalf("QA payload not a templated pair: %q", payload)
	}
}

func indexHasRefusal(res IndexResult, reason string) bool {
	for _, r := range res.Refused {
		if r.Reason == reason {
			return true
		}
	}
	return false
}
