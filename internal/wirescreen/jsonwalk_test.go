package wirescreen

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestRedactJSONLeavesNestedResult proves the leaf-aware pass redacts PII inside a
// nested tool_result content string while leaving the JSON scaffolding — the
// structural tool_use_id / role / type identifiers — byte-for-byte intact.
func TestRedactJSONLeavesNestedResult(t *testing.T) {
	body := []byte(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01ABC","content":"reach me at alice@example.com please"}]}`)
	jr, ok := RedactJSONLeaves(context.Background(), PIIRedactor(), body, "")
	if !ok {
		t.Fatal("RedactJSONLeaves returned ok=false, want a redaction on the nested email")
	}
	got := string(jr.Redacted)
	if strings.Contains(got, "alice@example.com") {
		t.Fatalf("email survived redaction: %s", got)
	}
	if !strings.Contains(got, "[REDACTED:email]") {
		t.Fatalf("missing email placeholder: %s", got)
	}
	// Structural identifiers must be untouched — they are never rendered or screened.
	if !strings.Contains(got, `"tool_use_id":"toolu_01ABC"`) {
		t.Fatalf("structural tool_use_id was mangled: %s", got)
	}
	if !strings.Contains(got, `"role":"user"`) || !strings.Contains(got, `"type":"tool_result"`) {
		t.Fatalf("structural role/type was mangled: %s", got)
	}
	// The pointer-addressed audit names the exact leaf that changed.
	if len(jr.Hits) != 1 || jr.Hits[0].Pointer != "/content/0/content" || jr.Hits[0].Kind != "email" {
		t.Fatalf("hits = %+v, want one email hit at /content/0/content", jr.Hits)
	}
	// The output still round-trips as JSON.
	var back any
	if err := json.Unmarshal(jr.Redacted, &back); err != nil {
		t.Fatalf("redacted body is not valid JSON: %v", err)
	}
}

// TestRedactJSONLeavesStructuralKeyLooksLikePII proves a secret-shaped value that
// sits under a STRUCTURAL key is never screened: the walker skips the whole subtree,
// so an id/name/source that happens to match a PII regex is preserved verbatim.
func TestRedactJSONLeavesStructuralKeyLooksLikePII(t *testing.T) {
	// `id` carries an SSN-shaped value, `name` an email-shaped one — both structural,
	// so both must survive. The one prose leaf (note) has no PII, so nothing redacts.
	body := []byte(`{"id":"123-45-6789","name":"svc@corp.io","note":"all clear"}`)
	_, ok := RedactJSONLeaves(context.Background(), PIIRedactor(), body, "")
	if ok {
		t.Fatal("RedactJSONLeaves redacted a value under a structural key; want ok=false (nothing to redact)")
	}

	// Now add a genuine PII leaf under a NON-structural key: only that leaf redacts,
	// the structural look-alikes stay verbatim.
	body = []byte(`{"id":"123-45-6789","name":"svc@corp.io","note":"ping user@real.com now"}`)
	jr, ok := RedactJSONLeaves(context.Background(), PIIRedactor(), body, "")
	if !ok {
		t.Fatal("RedactJSONLeaves returned ok=false, want the note email redacted")
	}
	got := string(jr.Redacted)
	if !strings.Contains(got, `"id":"123-45-6789"`) || !strings.Contains(got, `"name":"svc@corp.io"`) {
		t.Fatalf("structural id/name look-alikes were redacted: %s", got)
	}
	if strings.Contains(got, "user@real.com") || !strings.Contains(got, "[REDACTED:email]") {
		t.Fatalf("non-structural email leaf not redacted: %s", got)
	}
	if len(jr.Hits) != 1 || jr.Hits[0].Pointer != "/note" {
		t.Fatalf("hits = %+v, want one hit at /note", jr.Hits)
	}
}

// TestRedactJSONLeavesStringifiedPayload proves the walker reaches one level into a
// stringified-JSON payload: it redacts a secret in the inner object and re-serializes
// so the outer string slot still holds valid, re-parseable JSON.
func TestRedactJSONLeavesStringifiedPayload(t *testing.T) {
	body := []byte(`{"tool":"send","arguments":"{\"to\":\"bob@example.com\",\"kind\":\"note\"}"}`)
	jr, ok := RedactJSONLeaves(context.Background(), PIIRedactor(), body, "")
	if !ok {
		t.Fatal("RedactJSONLeaves returned ok=false, want the inner email redacted")
	}
	if strings.Contains(string(jr.Redacted), "bob@example.com") {
		t.Fatalf("inner email survived: %s", jr.Redacted)
	}
	// The hit is addressed inside the stringified payload.
	if len(jr.Hits) != 1 || jr.Hits[0].Pointer != "/arguments/to" || jr.Hits[0].Kind != "email" {
		t.Fatalf("hits = %+v, want one email hit at /arguments/to", jr.Hits)
	}
	// The outer body round-trips, AND the re-marshalled arguments string is itself
	// still valid JSON carrying the placeholder — the round-trip guarantee.
	var outer struct {
		Tool      string `json:"tool"`
		Arguments string `json:"arguments"`
	}
	if err := json.Unmarshal(jr.Redacted, &outer); err != nil {
		t.Fatalf("outer body not valid JSON: %v", err)
	}
	var inner map[string]string
	if err := json.Unmarshal([]byte(outer.Arguments), &inner); err != nil {
		t.Fatalf("stringified arguments no longer valid JSON: %v (%s)", err, outer.Arguments)
	}
	if inner["to"] != "[REDACTED:email]" || inner["kind"] != "note" {
		t.Fatalf("inner payload = %+v, want to redacted + kind preserved", inner)
	}
}

// TestRedactJSONLeavesNonJSONFallsBack proves a non-JSON (or bare-scalar) body is left
// untouched with ok=false, so the caller cleanly falls back to the flat path.
func TestRedactJSONLeavesNonJSONFallsBack(t *testing.T) {
	for _, body := range []string{
		"plain text with alice@example.com in it",
		"42",
		"",
	} {
		jr, ok := RedactJSONLeaves(context.Background(), PIIRedactor(), []byte(body), "")
		if ok {
			t.Fatalf("body %q: ok=true, want fallback ok=false", body)
		}
		if string(jr.Redacted) != body {
			t.Fatalf("body %q: Redacted=%q, want the input verbatim on fallback", body, jr.Redacted)
		}
	}
}
