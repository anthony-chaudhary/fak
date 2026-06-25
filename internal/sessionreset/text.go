package sessionreset

import "strings"

// text.go — the pure transcript helpers the contributors share. All deterministic,
// stdlib-only, no clock. They make no provider assumptions beyond Msg{Role, Content}.

// firstUserLine returns the first non-empty user message — the standing objective a
// task usually opens with. "" if there is none.
func firstUserLine(msgs []Msg) string {
	for _, m := range msgs {
		if m.Role == "user" {
			if c := strings.TrimSpace(m.Content); c != "" {
				return c
			}
		}
	}
	return ""
}

// lastUserLine returns the last non-empty user message — the latest ask, the step
// the session was on when it reset. "" if there is none.
func lastUserLine(msgs []Msg) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			if c := strings.TrimSpace(msgs[i].Content); c != "" {
				return c
			}
		}
	}
	return ""
}

// systemPreamble concatenates the leading system messages — the stable prefix that
// does not change across the session and is worth replaying warm. Only the system
// lines at the HEAD of the transcript count as the prefix (a mid-stream system note
// is not part of the stable preamble).
func systemPreamble(msgs []Msg) string {
	var parts []string
	for _, m := range msgs {
		if m.Role != "system" {
			break
		}
		if c := strings.TrimSpace(m.Content); c != "" {
			parts = append(parts, c)
		}
	}
	return strings.Join(parts, "\n")
}

// lastTurns returns the last n non-empty messages, oldest-first within the tail —
// the verbatim recent exchange. A turn here is one message (role+content); n counts
// messages, not request/response pairs, keeping it provider-neutral.
func lastTurns(msgs []Msg, n int) []Msg {
	if n <= 0 {
		return nil
	}
	var tail []Msg
	for i := len(msgs) - 1; i >= 0 && len(tail) < n; i-- {
		if strings.TrimSpace(msgs[i].Content) == "" {
			continue
		}
		tail = append(tail, msgs[i])
	}
	// reverse to oldest-first
	for i, j := 0, len(tail)-1; i < j; i, j = i+1, j-1 {
		tail[i], tail[j] = tail[j], tail[i]
	}
	return tail
}

// clip truncates s to at most max runes, appending an ellipsis marker when it cuts,
// so a single huge line cannot blow up the seed. Rune-safe (never splits a multibyte
// codepoint).
func clip(s string, max int) string {
	if max <= 0 {
		return s
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + " […]"
}

// approxTokens is a cheap, deterministic token estimate (~4 chars/token) for the
// warm-prefix cost gate. It is intentionally a coarse heuristic, not a tokenizer call
// — the warm-prefix contributor only needs an order-of-magnitude size to price the
// replay, and a real tokenizer is a tier-1 dependency this mechanism does not need.
func approxTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}
