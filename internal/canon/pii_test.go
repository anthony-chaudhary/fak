package canon

import (
	"encoding/base64"
	"strings"
	"testing"
)

// TestScanDetectsPII pins the deterministic general-PII needle set (#5378): each family
// (email/national-id/phone/PAN/IBAN) is flagged on the raw view, and the credential/
// injection axes stay OFF for a body that carries only PII (PII is its own axis).
func TestScanDetectsPII(t *testing.T) {
	for name, body := range map[string]string{
		"email":     `contact: alice.customer@acme.example is the account owner`,
		"ssn":       `SSN on file: 123-45-6789 (do not share)`,
		"phone":     `call the customer back at +1 415-555-0132 tomorrow`,
		"pan-space": `card 4111 1111 1111 1111 was declined`,
		"pan-dash":  `card 4111-1111-1111-1111 was declined`,
		"iban":      `remit to GB82WEST12345698765432 by Friday`,
	} {
		f := Scan([]byte(body))
		if !f.PII {
			t.Errorf("%s: want PII=true, got %+v (body=%q)", name, f, body)
		}
		if f.Secret {
			t.Errorf("%s: PII body wrongly flagged as Secret: %q", name, body)
		}
		if f.Any() {
			t.Errorf("%s: PII must NOT fold into Any() (secret+injection trigger): %q", name, body)
		}
	}
}

// TestScanPIINegatives pins that bare identifiers that are NOT PII-shaped stay clean: the
// numeric families require their canonical separators, so an unseparated digit run does not
// trip them.
func TestScanPIINegatives(t *testing.T) {
	for name, body := range map[string]string{
		"git-sha":       `commit 9f8e7d6c5b4a3f2e1d0c9b8a7654321012345678 landed`,
		"order-number":  `order #123456789 shipped`,
		"bare-16-digit": `token 4111111111111111 (no separators)`,
		"version":       `release 1.2.3 build 456`,
		"plain-prose":   `the quick brown fox jumped over the lazy dog`,
	} {
		if f := Scan([]byte(body)); f.PII {
			t.Errorf("%s: false-positive PII on non-PII body: %q", name, body)
		}
	}
}

// TestRedactPIIMasksAndReScreensClean proves RedactPII masks every raw PII span in place
// and the redacted body re-screens clean (no PII, and the surrounding text survives) — the
// warn-first mask-in-place property normgate depends on.
func TestRedactPIIMasksAndReScreensClean(t *testing.T) {
	body := []byte(`owner alice@acme.example, SSN 123-45-6789, card 4111 1111 1111 1111`)
	red, masked := RedactPII(body)
	if masked != 3 {
		t.Fatalf("want 3 masked spans, got %d (redacted=%q)", masked, red)
	}
	if Scan(red).PII {
		t.Errorf("redacted body still flags PII: %q", red)
	}
	for _, leaked := range []string{"alice@acme.example", "123-45-6789", "4111 1111 1111 1111"} {
		if strings.Contains(string(red), leaked) {
			t.Errorf("redacted body still leaks %q: %q", leaked, red)
		}
	}
	if !strings.Contains(string(red), "owner") || !strings.Contains(string(red), "card") {
		t.Errorf("redaction dropped surrounding legitimate text: %q", red)
	}
}

// TestRawPIIComplete: a raw email is raw-locatable (redactable); a base64-obfuscated email
// is caught by Scan on the decoded view but has NO raw span, so RawPIIComplete is false —
// the signal normgate uses to SEAL rather than mask an obfuscated needle.
func TestRawPIIComplete(t *testing.T) {
	raw := []byte(`email me at bob@acme.example`)
	if !Scan(raw).PII {
		t.Fatal("raw email should be detected")
	}
	if !RawPIIComplete(raw) {
		t.Error("a raw email must be raw-locatable (RawPIIComplete=true)")
	}

	obf := []byte(`decode: ` + base64.StdEncoding.EncodeToString([]byte("carol@acme.example")))
	if !Scan(obf).PII {
		t.Fatal("base64-obfuscated email should be detected on the decoded view")
	}
	if RawPIIComplete(obf) {
		t.Error("an obfuscation-only PII needle must NOT be raw-complete (must seal)")
	}
	if _, masked := RedactPII(obf); masked != 0 {
		t.Errorf("RedactPII must not claim to mask an obfuscation-only needle, masked=%d", masked)
	}
}

func TestContextualPIIClassExemptionPreservesOnlyDeclaredClass(t *testing.T) {
	body := []byte("contact hiring@example.com, SSN 123-45-6789")
	exempt := map[string]bool{PIIClassEmail: true}
	if classes := PIIClasses(body); !classes[PIIClassEmail] || !classes[PIIClassNationalID] {
		t.Fatalf("want email and national-id classes, got %v", classes)
	}
	red, masked := RedactPIIExcept(body, exempt)
	if masked != 1 {
		t.Fatalf("want one protected span masked, got %d: %q", masked, red)
	}
	got := string(red)
	if !strings.Contains(got, "hiring@example.com") {
		t.Fatalf("declared-public email was changed: %q", got)
	}
	if strings.Contains(got, "123-45-6789") || !strings.Contains(got, "[redacted:pii:11B]") {
		t.Fatalf("protected national id was not masked: %q", got)
	}
}

func TestPIIClassNamesAreClosed(t *testing.T) {
	for _, class := range []string{PIIClassEmail, PIIClassPhone, PIIClassNationalID, PIIClassPaymentCard, PIIClassIBAN} {
		if !KnownPIIClass(class) {
			t.Errorf("known class %q rejected", class)
		}
	}
	if KnownPIIClass("all") || KnownPIIClass("emails") {
		t.Error("unknown or blanket class must fail closed")
	}
}
