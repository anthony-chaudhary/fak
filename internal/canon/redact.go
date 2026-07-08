package canon

import (
	"fmt"
)

// RedactSecrets masks credential-shaped spans in body IN PLACE, keeping every other
// byte, and reports how many spans it masked. It is the warn-first counterpart to the
// page-out/seal path: instead of holding a whole legitimate tool result out of context
// because one span looks like a credential, the caller can keep the surrounding output
// (an HTTP status, a JSON body, a diagnostic line) and mask only the secret itself.
//
// Scope and safety, by construction:
//
//   - It masks the RAW spans of combinedSecret (the same matcher canon.Scan uses),
//     minus obvious placeholders (isPlaceholderSecret) — so a real `xoxb-…` token is
//     masked but `AKIAIOSFODNN7EXAMPLE` is left untouched, identical to the detection
//     side. Detection and redaction can never drift because both read combinedSecret.
//   - It masks the RAW body only. A secret that Scan catches solely on a de-obfuscated
//     view (base64/homoglyph/bidi/char-spread) has NO locatable raw span here, so
//     RedactSecrets leaves it untouched and reports whether the raw pass was COMPLETE:
//     the second return names whether a caller may treat the redacted body as
//     secret-clean. An obfuscated secret is exactly the adversarial case where the
//     permissive warn-first path should NOT apply, so the caller falls back to the seal.
//
// The mask is a fixed sentinel that carries no bytes of the original secret and is
// itself never secret-shaped, so a redacted body passes a fresh canon.Scan re-screen
// (the property the page-in gate needs) — verified in redact_test.go.
func RedactSecrets(body []byte) (redacted []byte, masked int) {
	raw := string(body)
	locs := combinedSecret.FindAllStringIndex(raw, -1)
	if len(locs) == 0 {
		return body, 0
	}
	var out []byte
	prev := 0
	for _, loc := range locs {
		span := raw[loc[0]:loc[1]]
		if isPlaceholderSecret(span) {
			continue // a placeholder/example is not a real credential — leave it verbatim
		}
		out = append(out, raw[prev:loc[0]]...)
		out = append(out, redactionMark(len(span))...)
		prev = loc[1]
		masked++
	}
	if masked == 0 {
		return body, 0 // every match was a placeholder — nothing real to mask
	}
	out = append(out, raw[prev:]...)
	return out, masked
}

// RawSecretComplete reports whether every secret canon.Scan detects in body is
// locatable as a RAW span (so RedactSecrets can mask all of them). It is false when a
// secret is caught only on a de-obfuscated view — the case the caller must seal rather
// than redact, because a raw-span redactor cannot reach an obfuscated credential.
func RawSecretComplete(body []byte) bool {
	if !Scan(body).Secret {
		return true // no secret at all — trivially complete
	}
	// A secret exists. It is raw-locatable iff the raw body itself carries a
	// non-placeholder combinedSecret span.
	for _, loc := range combinedSecret.FindAllStringIndex(string(body), -1) {
		if !isPlaceholderSecret(string(body)[loc[0]:loc[1]]) {
			return true
		}
	}
	return false
}

// redactionMark is the fixed replacement for one masked secret span. It records the
// masked length so a reader sees that bytes were removed, without leaking any of them,
// and is deliberately NOT secret-shaped so the redacted body re-screens clean.
func redactionMark(n int) []byte {
	return []byte(fmt.Sprintf("[redacted:secret:%dB]", n))
}
