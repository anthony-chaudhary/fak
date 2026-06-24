package agent

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wirescreen"
)

// transcript_redact_test.go is the rung-5 WIRE witness (#572): the redaction that
// RedactOutboundMessages runs on the non-passthrough re-marshal path (stream.go) is
// reversible, one-sided, and default-inert. It peers with the wirescreen-package
// Apply/Restore witness (redaction_integration_test.go) by exercising the SAME Apply
// through the agent's outbound-message wire point. The blob CAS backend that backs the
// Restore witness is linked via deps.go (the agent package's blank import of blob).

// TestRedactOutbound_ActiveRedactsAndRestores: with a redactor selected, each flagged
// span in the body that leaves the box is replaced by a [REDACTED:<kind>] placeholder,
// the secret bytes are GONE, the surrounding framing is preserved, an untouched message
// is byte-identical, and the UNREDACTED original pinned in CAS restores byte-exact.
func TestRedactOutbound_ActiveRedactsAndRestores(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "reach alice@example.com and charge 4111 1111 1111 1111 ok"},
		{Role: "assistant", Content: "noted, nothing to scrub here"},
	}
	out, redactions := redactOutbound(wirescreen.PIIRedactor(), msgs)
	if len(redactions) != 1 {
		t.Fatalf("expected exactly one redacted message, got %d: %v", len(redactions), redactions)
	}
	if redactions[0].Index != 0 || redactions[0].By != "pii" {
		t.Errorf("redaction record = {Index:%d By:%q}, want {0, pii}", redactions[0].Index, redactions[0].By)
	}
	body := out[0].Content
	for _, secret := range []string{"alice@example.com", "4111 1111 1111 1111"} {
		if bytes.Contains([]byte(body), []byte(secret)) {
			t.Errorf("redacted body still contains secret %q: %s", secret, body)
		}
	}
	for _, framing := range []string{"reach", "and charge", "ok"} {
		if !bytes.Contains([]byte(body), []byte(framing)) {
			t.Errorf("redacted body lost framing %q: %s", framing, body)
		}
	}
	// One-sided: the message with no secrets is byte-identical (only spans change).
	if out[1].Content != msgs[1].Content {
		t.Errorf("untouched message was modified: got %q want %q", out[1].Content, msgs[1].Content)
	}
	// Reversibility witness: Restore returns the pre-redaction body byte-exact.
	got, err := wirescreen.Restore(context.Background(), redactions[0].Original)
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if !bytes.Equal(got, []byte(msgs[0].Content)) {
		t.Errorf("Restore is not byte-exact:\n got=%q\nwant=%q", got, msgs[0].Content)
	}
}

// TestRedactOutbound_MissLeavesBytesUncorrupted: a PII-shaped span the proposer did NOT
// flag (a 16-digit run that is NOT Luhn-valid, so it is NOT a credit card) must pass
// through UNCORRUPTED — the fail-open contract. Redaction is one-sided: it only rewrites
// flagged spans; an unflagged span is left byte-for-byte as it was.
func TestRedactOutbound_MissLeavesBytesUncorrupted(t *testing.T) {
	// "1234567890123456" is 16 digits but NOT Luhn-valid -> not flagged; the SSN is.
	body := "ref 1234567890123456 and ssn 123-45-6789 end"
	msgs := []Message{{Role: "user", Content: body}}
	out, redactions := redactOutbound(wirescreen.PIIRedactor(), msgs)
	if len(redactions) != 1 {
		t.Fatalf("expected one redacted message (the ssn), got %d", len(redactions))
	}
	got := out[0].Content
	if bytes.Contains([]byte(got), []byte("123-45-6789")) {
		t.Errorf("the SSN should have been redacted: %s", got)
	}
	// The unflagged digit run survived byte-intact (fail-open, no corruption).
	if !bytes.Contains([]byte(got), []byte("1234567890123456")) {
		t.Errorf("an unflagged span was corrupted: %s", got)
	}
	for _, framing := range []string{"ref", "and ssn", "end"} {
		if !bytes.Contains([]byte(got), []byte(framing)) {
			t.Errorf("framing %q was corrupted: %s", framing, got)
		}
	}
}

// TestRedactOutbound_DefaultInertIsByteIdentical: with no redactor active (the default),
// the outbound path is byte-identical to today at zero cost — the messages are returned
// unchanged and NO redaction is recorded. This is the agent-path peer of the spine's
// TestDefaultInertRegistersNoABIScreen / TestDefaultInert_NoActiveRedactor.
func TestRedactOutbound_DefaultInertIsByteIdentical(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "card 4111 1111 1111 1111 here"},
		{Role: "tool", Name: "fetch", Content: "key AKIAIOSFODNN7EXAMPLE"},
	}
	out, redactions := redactOutbound(nil, msgs) // nil redactor = the inert default
	if len(redactions) != 0 {
		t.Fatalf("inert path must record no redactions, got %d", len(redactions))
	}
	if len(out) != len(msgs) {
		t.Fatalf("inert path changed the message count: got %d want %d", len(out), len(msgs))
	}
	for i := range msgs {
		if out[i].Content != msgs[i].Content {
			t.Errorf("inert path changed message %d content: got %q want %q", i, out[i].Content, msgs[i].Content)
		}
	}
}

// TestRedactOutbound_DoesNotMutateInput: the caller's slice is never mutated
// (copy-on-write), matching QuarantineOutboundMessagesDoesNotMutateInput.
func TestRedactOutbound_DoesNotMutateInput(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "ssn 123-45-6789 on file"}}
	orig := msgs[0].Content
	_, _ = redactOutbound(wirescreen.PIIRedactor(), msgs)
	if msgs[0].Content != orig {
		t.Errorf("input slice was mutated: got %q want %q", msgs[0].Content, orig)
	}
}

// TestRedactOutbound_EmptyContentSkipped: a message with no content is not handed to the
// redactor (avoids a pointless zero-length Apply), and yields no redaction record.
func TestRedactOutbound_EmptyContentSkipped(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: ""},
		{Role: "assistant", Content: "ssn 123-45-6789"},
	}
	out, redactions := redactOutbound(wirescreen.PIIRedactor(), msgs)
	if len(redactions) != 1 || redactions[0].Index != 1 {
		t.Fatalf("expected one redaction on index 1, got %v", redactions)
	}
	if out[0].Content != "" {
		t.Errorf("empty content was changed: %q", out[0].Content)
	}
}
