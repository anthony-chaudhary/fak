package gateway

import "strconv"

// reasonPIIRedacted is the general-PII twin of reasonSecretRedacted (#5378): an
// email/phone/national-id/PAN/IBAN span in a tool result was MASKED IN PLACE (warn-first
// default, internal/normgate). Like the secret redaction it gets a one-line WARN, not the
// "held out of context" banner — and it never baits a re-read (there is nothing paged out
// to retrieve). A local literal mirroring internal/abi.ReasonPIIRedacted, so the served
// note needs no import for one string (the same convention as reasonSecretRedacted). Lives
// in its own file so it composes cleanly on the shared trunk without touching the
// peer-co-edited messages.go.
const reasonPIIRedacted = "PII_REDACTED"

// redactedSpanWarn is the masked-in-place warn rule, written once for every redaction
// class (#5378). The rule: nothing redacted means no line at all (so the warn composes
// with the held-out banner instead of stacking an empty one); the span noun agrees with
// n; and a single line always carries the count, the reason code, the promise that the
// rest of the result is intact, and the posture to set to hold the whole result instead.
// kind labels the spans, subject names what stays masked, and posture is spelled out by
// the caller because each class holds out under a differently-named posture.
func redactedSpanWarn(n int, kind, reason, subject, posture string) string {
	if n <= 0 {
		return ""
	}
	span := "span"
	if n > 1 {
		span = "spans"
	}
	return "[fak] masked " + strconv.Itoa(n) + " " + kind + " " + span +
		" in a tool result (" + reason + ") — the rest of the output is intact and in context. " +
		"Warn-first default: your own output is not withheld, only " + subject + " is masked. " +
		"To hold the whole result instead, set the " + posture + "."
}

// piiRedactedWarn is the general-PII twin of secretRedactedWarn (#5378): the one-line warn
// for masked-in-place PII spans (email/phone/national-id/PAN/IBAN). Empty when nothing was
// redacted, so it composes with the held-out banner and the secret warn.
func piiRedactedWarn(n int) string {
	return redactedSpanWarn(n, "PII", reasonPIIRedacted,
		"the PII itself (email/phone/ID/card)", "fail_closed posture")
}
