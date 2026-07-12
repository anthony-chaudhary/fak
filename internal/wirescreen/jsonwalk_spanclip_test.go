package wirescreen

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// jsonwalk_spanclip_test.go covers the leaf-boundary half of the borrow (issue
// #3551, lmnr json_walker.rs:243 apply_spans_and_serialize): mapping a detector span
// expressed in RENDERED-PROSE coordinates back onto the exact JSON string leaves it
// overlaps. The three acceptance tests in jsonwalk_test.go exercise whole-leaf
// redaction, structural-key skipping, and stringified expansion; none place a span on
// a leaf BOUNDARY. These two do — a span straddling the "\n\n" separator must clip
// onto both leaves, and a span landing wholly on a key/separator region must drop —
// which the anchored PIIRedactor cannot be coaxed into emitting from real prose.

// spanStubRedactor is a test Redactor that redacts one caller-chosen substring of the
// rendered prose instead of running a regex over it. It lets a white-box test place a
// span at an EXACT prose coordinate — in particular one straddling the "\n\n" leaf
// separator, or one landing wholly on a key/separator region — that the deterministic
// PIIRedactor never produces. target is the literal prose slice to cover; a target
// absent from the prose yields no span (and the walker falls back).
type spanStubRedactor struct {
	target string
	kind   string
}

func (spanStubRedactor) Name() string { return "spanstub" }

func (s spanStubRedactor) Propose(_ context.Context, prose []byte, _ string) []Span {
	i := strings.Index(string(prose), s.target)
	if i < 0 {
		return nil
	}
	return []Span{{Start: i, End: i + len(s.target), Kind: s.kind}}
}

// TestRedactJSONLeavesSpanStraddlesSeparator proves a detector span that straddles the
// "\n\n" separator between two rendered leaves is CLIPPED onto both: each leaf's
// covered sub-range is redacted, and the separator + key-label bytes between them
// (which back no leaf) never leak into either JSON string value. The re-marshalled
// body still round-trips as JSON.
func TestRedactJSONLeavesSpanStraddlesSeparator(t *testing.T) {
	// Sorted keys render as prose "a: one\n\nb: two". The target "ne\n\nb: t" begins
	// inside leaf a's value ("o[ne]"), crosses the separator and the "b: " key label,
	// and ends inside leaf b's value ("[t]wo").
	body := []byte(`{"a":"one","b":"two"}`)
	r := spanStubRedactor{target: "ne\n\nb: t", kind: "x"}
	jr, ok := RedactJSONLeaves(context.Background(), r, body, "")
	if !ok {
		t.Fatal("RedactJSONLeaves returned ok=false, want the straddling span clipped onto both leaves")
	}
	var got map[string]string
	if err := json.Unmarshal(jr.Redacted, &got); err != nil {
		t.Fatalf("redacted body is not valid JSON: %v (%s)", err, jr.Redacted)
	}
	if got["a"] != "o[REDACTED:x]" {
		t.Fatalf("leaf a = %q, want the tail clipped to o[REDACTED:x]", got["a"])
	}
	if got["b"] != "[REDACTED:x]wo" {
		t.Fatalf("leaf b = %q, want the head clipped to [REDACTED:x]wo", got["b"])
	}
	// The separator/key bytes must never have entered a value: no stray newline or the
	// "b: " label may appear inside either redacted string.
	for k, v := range got {
		if strings.Contains(v, "\n") || strings.Contains(v, "b: ") {
			t.Fatalf("leaf %q leaked separator/key bytes: %q", k, v)
		}
	}
	// Both leaves are named in the pointer-addressed audit.
	if len(jr.Hits) != 2 {
		t.Fatalf("hits = %+v, want one clip on each of the two leaves", jr.Hits)
	}
	seen := map[string]bool{}
	for _, h := range jr.Hits {
		seen[h.Pointer] = true
	}
	if !seen["/a"] || !seen["/b"] {
		t.Fatalf("hits = %+v, want pointers /a and /b", jr.Hits)
	}
}

// TestRedactJSONLeavesSpanOnSeparatorDropped proves the other half of the boundary
// rule: a span landing WHOLLY on a key/separator region — prose bytes that back no
// string leaf — is dropped. Nothing is redacted, so RedactJSONLeaves reports ok=false
// and returns the body verbatim for the caller's flat-path fallback.
func TestRedactJSONLeavesSpanOnSeparatorDropped(t *testing.T) {
	body := []byte(`{"a":"one","b":"two"}`)
	// "\n\nb: " is the separator plus the "b: " key label between the two values; it
	// intersects neither leaf's [start,end) range.
	r := spanStubRedactor{target: "\n\nb: ", kind: "x"}
	jr, ok := RedactJSONLeaves(context.Background(), r, body, "")
	if ok {
		t.Fatalf("RedactJSONLeaves redacted a key/separator span; want ok=false, got %s", jr.Redacted)
	}
	if string(jr.Redacted) != string(body) {
		t.Fatalf("Redacted = %q, want the input body verbatim on a dropped span", jr.Redacted)
	}
}
