package modelroute

import (
	"fmt"
	"testing"
)

// stubFiller answers a fixed batch of demands from a table keyed by hint — the "big
// model" with no model in the loop, so the whole path is deterministic.
type stubFiller struct {
	byHint map[string]string
	calls  int
}

func (s *stubFiller) Fill(demands []SpanDemand) ([]string, error) {
	s.calls++
	out := make([]string, len(demands))
	for i, d := range demands {
		out[i] = s.byHint[d.Hint] // missing hint -> "" -> rejected as unfilled
	}
	return out, nil
}

func TestParseDraft(t *testing.T) {
	d, err := ParseDraft("The speed of light is [[SPAN:number|c in m/s]] exactly, per [[SPAN:citation|SI]].")
	if err != nil {
		t.Fatalf("ParseDraft: %v", err)
	}
	dem := d.Demands()
	if len(dem) != 2 {
		t.Fatalf("demands = %d, want 2", len(dem))
	}
	if dem[0].Kind != SpanNumber || dem[0].Index != 0 || dem[0].Hint != "c in m/s" {
		t.Errorf("demand0 = %+v", dem[0])
	}
	if dem[1].Kind != SpanCitation || dem[1].Index != 1 {
		t.Errorf("demand1 = %+v", dem[1])
	}
}

func TestParseDraftUnknownKindFoldsToOther(t *testing.T) {
	d, err := ParseDraft("x [[SPAN:wat|h]] y")
	if err != nil {
		t.Fatalf("ParseDraft: %v", err)
	}
	if got := d.Demands()[0].Kind; got != SpanOther {
		t.Errorf("unknown kind = %q, want %q", got, SpanOther)
	}
}

func TestParseDraftUnterminatedFailsClosed(t *testing.T) {
	if _, err := ParseDraft("oops [[SPAN:fact|no close"); err == nil {
		t.Fatal("want error on unterminated marker, got nil")
	}
}

// TestConsultSplicesAndSaves is the spine witness: the local model wrote the whole
// fluent answer and marked three hard spans; the big model fills only those. The
// spliced answer is correct AND the big model generated only a small fraction of the
// bytes — remote spend << big-only. This is exactly the epic's ask for this child.
func TestConsultSplicesAndSaves(t *testing.T) {
	draft := "To sort a slice in Go, call [[SPAN:api|stdlib sort by less]] with a less " +
		"function; it runs in O(n log n) time and was introduced in Go " +
		"[[SPAN:number|generics sort.Slice version]]. See [[SPAN:citation|pkg.go.dev sort]] " +
		"for the full contract and the stability caveat."
	d, err := ParseDraft(draft)
	if err != nil {
		t.Fatalf("ParseDraft: %v", err)
	}
	f := &stubFiller{byHint: map[string]string{
		"stdlib sort by less":         "sort.Slice",
		"generics sort.Slice version": "1.8",
		"pkg.go.dev sort":             "pkg.go.dev/sort",
	}}
	answer, sv, unfilled, err := Consult(d, f)
	if err != nil {
		t.Fatalf("Consult: %v", err)
	}
	if f.calls != 1 {
		t.Errorf("filler called %d times, want 1 (batched)", f.calls)
	}
	if len(unfilled) != 0 {
		t.Errorf("unfilled = %v, want none", unfilled)
	}
	want := "To sort a slice in Go, call sort.Slice with a less function; it runs in " +
		"O(n log n) time and was introduced in Go 1.8. See pkg.go.dev/sort for the " +
		"full contract and the stability caveat."
	if answer != want {
		t.Errorf("answer =\n%q\nwant\n%q", answer, want)
	}
	if sv.TotalSpans != 3 || sv.FilledSpans != 3 {
		t.Errorf("spans filled = %d/%d, want 3/3", sv.FilledSpans, sv.TotalSpans)
	}
	// The witness: the big model generated only the hard spans, a small slice of the
	// whole answer. Everything else was fluent local scaffolding.
	if sv.RemoteFraction >= 0.20 {
		t.Errorf("RemoteFraction = %.3f, want < 0.20 (remote spend should be a small slice)", sv.RemoteFraction)
	}
	if sv.RemoteChars+sv.LocalChars != sv.BigOnlyChars {
		t.Errorf("chars: remote(%d)+local(%d) != bigonly(%d)", sv.RemoteChars, sv.LocalChars, sv.BigOnlyChars)
	}
}

// TestConsultRejectsBadFill proves fail-closed acceptance: a number span whose remote
// fill carries no digit is rejected, surfaces as UNFILLED, and does NOT count as remote
// spend — a bad fill is caught, not trusted.
func TestConsultRejectsBadFill(t *testing.T) {
	d, err := ParseDraft("pi is about [[SPAN:number|pi to 2dp]].")
	if err != nil {
		t.Fatalf("ParseDraft: %v", err)
	}
	f := FillerFunc(func(demands []SpanDemand) ([]string, error) {
		return []string{"three point one four"}, nil // no digit -> rejected
	})
	answer, sv, unfilled, err := Consult(d, f)
	if err != nil {
		t.Fatalf("Consult: %v", err)
	}
	if len(unfilled) != 1 || sv.FilledSpans != 0 || sv.RemoteChars != 0 {
		t.Errorf("bad fill not rejected: unfilled=%v filled=%d remote=%d", unfilled, sv.FilledSpans, sv.RemoteChars)
	}
	if want := "pi is about [[UNFILLED:number|pi to 2dp]]."; answer != want {
		t.Errorf("answer = %q, want %q", answer, want)
	}
}

func TestConsultCountMismatchFailsClosed(t *testing.T) {
	d, _ := ParseDraft("a [[SPAN:fact|x]] b [[SPAN:fact|y]]")
	f := FillerFunc(func(demands []SpanDemand) ([]string, error) {
		return []string{"only-one"}, nil
	})
	if _, _, _, err := Consult(d, f); err == nil {
		t.Fatal("want error on filler count mismatch, got nil")
	}
}

func TestConsultNoDemandsSkipsFiller(t *testing.T) {
	d, _ := ParseDraft("a wholly local answer, no hard spans at all.")
	f := &stubFiller{}
	answer, sv, _, err := Consult(d, f)
	if err != nil {
		t.Fatalf("Consult: %v", err)
	}
	if f.calls != 0 {
		t.Errorf("filler called %d times, want 0 (no demands)", f.calls)
	}
	if sv.RemoteChars != 0 || sv.RemoteFraction != 0 {
		t.Errorf("remote spend on a demand-free draft: %+v", sv)
	}
	if answer != "a wholly local answer, no hard spans at all." {
		t.Errorf("answer = %q", answer)
	}
}

// Example_spanFill is the captured live run: a local draft marks two hard spans, the
// remote filler supplies just those, and the savings show the big model generated a
// small slice of the answer.
func Example_spanFill() {
	d, _ := ParseDraft("The Go module cache lives at [[SPAN:api|GOMODCACHE default]], sized " +
		"[[SPAN:fact|size cap]] by default.")
	f := FillerFunc(func(demands []SpanDemand) ([]string, error) {
		return []string{"$GOPATH/pkg/mod", "with no fixed cap"}, nil
	})
	answer, sv, _, _ := Consult(d, f)
	fmt.Println(answer)
	fmt.Printf("spans=%d/%d remote_fraction=%.2f\n", sv.FilledSpans, sv.TotalSpans, sv.RemoteFraction)
	// Output:
	// The Go module cache lives at $GOPATH/pkg/mod, sized with no fixed cap by default.
	// spans=2/2 remote_fraction=0.40
}
