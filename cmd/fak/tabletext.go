package main

// truncateTableField truncates a byte-width CLI table field and marks the cut
// with a Unicode ellipsis. It deliberately preserves the historical byte-width
// contract used by fleet, loop-recover, and schedscan; rune-width surfaces use
// truncRunes instead, and cache-value's `~` marker stays a separate behavior.
func truncateTableField(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
