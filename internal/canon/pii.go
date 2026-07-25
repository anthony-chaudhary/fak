package canon

import (
	"fmt"
	"regexp"
	"strings"
)

// PIIPatterns is the deterministic general-PII needle set the served-path redaction seam
// (internal/normgate) composes IN FRONT of the credential/injection detectors to close the
// #3280 gateway-parity gap (#5378): the secret path masks credential shapes, but a tool
// result or model response carrying a customer's email, phone, national ID, payment card,
// or IBAN passed through unmasked. These are DETERMINISTIC shapes (not ML/NER — a pluggable
// detector can follow, per the #5378 non-goals): each pattern is tight enough that a bare
// identifier does not trip it — the numeric families require their canonical separators, so
// an unseparated digit run (a git SHA, a UUID field, a 9-digit order number) is not treated
// as PII.
//
// Exported (like SecretPatterns) so a caller can audit or extend the set. Detection (Scan)
// and redaction (RedactPII) BOTH read combinedPII, so they can never drift — the same
// invariant the secret axis holds.
var PIIPatterns = []*regexp.Regexp{
	// Email address.
	regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`),
	// US national ID (SSN), dashed — the dashes keep it off any bare 9-digit run.
	regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
	// Phone number — requires the canonical separators (spaces/dots/dashes, optional
	// country code and area-code parens), so a bare 10-digit identifier is not a phone.
	regexp.MustCompile(`(?:\+\d{1,3}[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b`),
	// Payment card (PAN) — the grouped 13–19-digit copy-paste form (four separated
	// groups). The separators keep it off an unseparated long digit run.
	regexp.MustCompile(`\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{1,7}\b`),
	// IBAN — two-letter country code + two check digits + 11–30 alphanumeric.
	regexp.MustCompile(`\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b`),
}

// combinedPII is the single alternation over PIIPatterns — one linear pass instead of
// len(PIIPatterns) separate backtracking runs, built FROM PIIPatterns so the two can never
// drift (mirrors combinedSecret). Each source is wrapped in a NON-CAPTURING group so an
// inline flag stays scoped to its own alternative.
var combinedPII = func() *regexp.Regexp {
	alts := make([]string, len(PIIPatterns))
	for i, re := range PIIPatterns {
		alts[i] = "(?:" + re.String() + ")"
	}
	return regexp.MustCompile(strings.Join(alts, "|"))
}()

// RedactPII masks general-PII spans (PIIPatterns) in body IN PLACE, keeping every other
// byte, and reports how many spans it masked — the warn-first counterpart to the seal, so a
// legitimate tool result carrying one customer email is kept in context with only the email
// masked. It masks the RAW spans of combinedPII (the same matcher Scan reads), so detection
// and redaction can never drift; a PII needle caught only on a de-obfuscated view has NO
// raw span here and is left for the seal (see RawPIIComplete). The mask is a fixed
// length-tagged sentinel that carries none of the original bytes and is not itself
// PII-shaped, so a redacted body re-screens clean (the property normgate's page-in gate
// needs) — verified in pii_test.go.
func RedactPII(body []byte) (redacted []byte, masked int) {
	raw := string(body)
	locs := combinedPII.FindAllStringIndex(raw, -1)
	if len(locs) == 0 {
		return body, 0
	}
	var out []byte
	prev := 0
	for _, loc := range locs {
		out = append(out, raw[prev:loc[0]]...)
		out = append(out, redactionMarkPII(loc[1]-loc[0])...)
		prev = loc[1]
		masked++
	}
	out = append(out, raw[prev:]...)
	return out, masked
}

// RawPIIComplete reports whether the PII canon.Scan detects in body is locatable as a RAW
// span (so RedactPII can mask it). It is false when a PII needle is caught only on a
// de-obfuscated view — the case the caller must seal rather than redact, because a raw-span
// redactor cannot reach an obfuscated needle. Mirrors RawSecretComplete.
func RawPIIComplete(body []byte) bool {
	if !Scan(body).PII {
		return true // no PII at all — trivially complete
	}
	return combinedPII.MatchString(string(body))
}

// redactionMarkPII is the fixed replacement for one masked PII span. It records the masked
// length so a reader sees bytes were removed, without leaking any of them, and is
// deliberately NOT PII-shaped so the redacted body re-screens clean.
func redactionMarkPII(n int) []byte {
	return []byte(fmt.Sprintf("[redacted:pii:%dB]", n))
}
