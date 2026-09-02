package sessionctl

// refusalString renders a closed-reason refusal the one way every *Refusal
// error in this package does: the reason token alone, or "REASON: detail"
// when a detail is attached. The string form is for logs and error plumbing;
// callers switch on Reason, never parse Detail (each refusal type's doc).
func refusalString(reason, detail string) string {
	if detail == "" {
		return reason
	}
	return reason + ": " + detail
}
