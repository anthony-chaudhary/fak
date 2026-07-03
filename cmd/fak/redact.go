package main

// redactSecret shows only that a secret is present plus its last 4 bytes,
// never the secret itself: "" -> "(unset)", len<=4 -> "****", else "****"+tail.
// Shared by the chatrelay and slack surfaces (#1419).
func redactSecret(s string) string {
	if s == "" {
		return "(unset)"
	}
	if len(s) <= 4 {
		return "****"
	}
	return "****" + s[len(s)-4:]
}
