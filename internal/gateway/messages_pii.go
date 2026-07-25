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

// piiRedactedWarn is the general-PII twin of secretRedactedWarn (#5378): the one-line warn
// for masked-in-place PII spans (email/phone/national-id/PAN/IBAN). Empty when nothing was
// redacted, so it composes with the held-out banner and the secret warn.
func piiRedactedWarn(n int) string {
	if n <= 0 {
		return ""
	}
	span := "span"
	if n > 1 {
		span = "spans"
	}
	return "[fak] masked " + strconv.Itoa(n) + " PII " + span +
		" in a tool result (PII_REDACTED) — the rest of the output is intact and in context. " +
		"Warn-first default: your own output is not withheld, only the PII itself (email/phone/ID/card) is masked. " +
		"To hold the whole result instead, set the fail_closed posture."
}
