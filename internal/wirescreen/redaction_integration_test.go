package wirescreen_test

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"     // CAS backend backing the redaction witness
	"github.com/anthony-chaudhary/fak/internal/wirescreen" // the redaction floor + Apply/Restore
)

// TestApply_RedactsSpansAndPinsOriginal is the rung witness/acceptance test
// (PROMPTS doc): proposed spans are replaced with placeholders on the outbound
// bytes, the secret bytes are GONE from the redacted output, and the UNREDACTED
// original stays pinned in CAS untouched.
func TestApply_RedactsSpansAndPinsOriginal(t *testing.T) {
	r := wirescreen.PIIRedactor() // exported test accessor for piiRedactor{}
	ctx := context.Background()
	original := []byte("reach alice@example.com and charge 4111 1111 1111 1111, key AKIAIOSFODNN7EXAMPLE")

	red, ok := wirescreen.Apply(ctx, r, original, "user")
	if !ok {
		t.Fatalf("Apply returned ok=false on a body with clear PII/secrets")
	}
	if len(red.Spans) < 3 {
		t.Fatalf("expected >=3 spans (email, card, aws key), got %d: %v", len(red.Spans), red.Spans)
	}
	if red.By != "pii" {
		t.Errorf("By = %q, want \"pii\"", red.By)
	}

	// The secret bytes themselves must be ABSENT from the redacted output.
	for _, secret := range []string{"alice@example.com", "4111 1111 1111 1111", "AKIAIOSFODNN7EXAMPLE"} {
		if bytes.Contains(red.Redacted, []byte(secret)) {
			t.Errorf("redacted output still contains secret %q: %s", secret, red.Redacted)
		}
	}
	// Each redacted span is replaced by a [REDACTED:<kind>] placeholder.
	for _, s := range red.Spans {
		if !strings.Contains(string(red.Redacted), "[REDACTED:"+s.Kind+"]") {
			t.Errorf("redacted output missing placeholder for kind %q", s.Kind)
		}
	}
	// The non-secret framing text is preserved (one-sided: only spans change).
	for _, framing := range []string{"reach", "and charge", ", key"} {
		if !bytes.Contains(red.Redacted, []byte(framing)) {
			t.Errorf("redacted output lost framing text %q: %s", framing, red.Redacted)
		}
	}

	// The CAS original is untouched: Restore returns the body byte-exact.
	if red.Original.Kind != abi.RefBlob || red.Original.Digest == "" {
		t.Errorf("expected a CAS blob handle, got %+v", red.Original)
	}
	got, err := wirescreen.Restore(ctx, red.Original)
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if !bytes.Equal(got, original) {
		t.Errorf("Restore is not byte-exact:\n got=%q\nwant=%q", got, original)
	}
}

// TestApply_RedactionsCounter: each redacted span bumps the lifetime Redactions()
// counter — the redaction peer of Flags()/Screened().
func TestApply_RedactionsCounter(t *testing.T) {
	r := wirescreen.PIIRedactor()
	ctx := context.Background()
	before := wirescreen.Redactions()
	body := []byte("ssn 123-45-6789 and card 4242 4242 4242 4242")
	red, ok := wirescreen.Apply(ctx, r, body, "tool")
	if !ok {
		t.Fatalf("Apply returned ok=false")
	}
	if want := before + int64(len(red.Spans)); wirescreen.Redactions() != want {
		t.Errorf("Redactions() = %d, want %d (before=%d, spans=%d)", wirescreen.Redactions(), want, before, len(red.Spans))
	}
}

// TestApply_NoSpansIsNoOp: a body with no PII/secrets yields ok=false, no CAS pin,
// and the body byte-identical. The inert path costs nothing.
func TestApply_NoSpansIsNoOp(t *testing.T) {
	r := wirescreen.PIIRedactor()
	body := []byte("the build is currently running, see issue #572")
	red, ok := wirescreen.Apply(context.Background(), r, body, "assistant")
	if ok {
		t.Fatalf("benign body must yield ok=false, got %v", red)
	}
	if !bytes.Equal(red.Redacted, body) {
		t.Errorf("benign body must be returned unchanged: got %q", red.Redacted)
	}
	if red.Original.Digest != "" {
		t.Errorf("benign body must not pin a CAS original, got digest %q", red.Original.Digest)
	}
}

// TestEndToEndWithPIIRedactor exercises the full env -> ActiveRedactor -> Apply ->
// Restore path. It is a no-op unless FAK_WIRE_REDACT=pii is set BEFORE the test
// binary starts (init reads the env once), so it proves the opt-in path without
// changing the default run.
func TestEndToEndWithPIIRedactor(t *testing.T) {
	if os.Getenv("FAK_WIRE_REDACT") != "pii" {
		t.Skip("set FAK_WIRE_REDACT=pii to exercise the end-to-end redaction path")
	}
	r := wirescreen.ActiveRedactor()
	if r == nil {
		t.Fatal("FAK_WIRE_REDACT=pii but ActiveRedactor() is nil (selection failed)")
	}
	ctx := context.Background()
	body := []byte("Authorization: Bearer z9z9z9z9z9z9z9z9z9z9z9, email ops@corp.internal")
	red, ok := wirescreen.Apply(ctx, r, body, "user")
	if !ok {
		t.Fatalf("end-to-end: expected redaction, got ok=false")
	}
	if bytes.Contains(red.Redacted, []byte("z9z9z9z9z9z9z9z9z9z9z9")) {
		t.Errorf("end-to-end: bearer token survived redaction: %s", red.Redacted)
	}
	if got, err := wirescreen.Restore(ctx, red.Original); err != nil || !bytes.Equal(got, body) {
		t.Errorf("end-to-end: Restore not byte-exact (err=%v)", err)
	}
}
