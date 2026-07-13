package issuehygiene

import (
	"strconv"
	"strings"
	"time"
)

// itoa is strconv.Itoa under a short name (the defect strings call it a lot).

// stripBOM drops a leading UTF-8 byte-order mark (PowerShell redirection stamps
// one on `gh ... > file`). Built from the rune so no BOM byte sits in source.
func stripBOM(s string) string { return strings.TrimPrefix(s, string(rune(0xFEFF))) }

// plural renders "<n> <noun>" and appends "s" when n != 1 (unless the noun
// already ends in s). Deterministic; no locale.
func plural(n int, noun string) string {
	s := strconv.Itoa(n) + " " + noun
	if n != 1 && !strings.HasSuffix(noun, "s") {
		s += "s"
	}
	return s
}

// sim formats a similarity ratio as a fixed 2-decimal string (0.86).
func sim(f float64) string { return strconv.FormatFloat(f, 'f', 2, 64) }

// staleDaysSince returns whole days between an RFC3339 updatedAt and nowUnix.
// A missing/unparseable timestamp or a zero clock returns 0 (never stale), so
// the soft staleness axis degrades safely rather than flagging on bad input.
func staleDaysSince(updatedAt string, nowUnix int64) int {
	if nowUnix == 0 || strings.TrimSpace(updatedAt) == "" {
		return 0
	}
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(updatedAt))
	if err != nil {
		return 0
	}
	delta := nowUnix - t.Unix()
	if delta < 0 {
		return 0
	}
	return int(delta / 86400)
}

// triageBacklogScore maps the triage-inbox depth to a soft grade contribution:
// full marks when the inbox is empty, shaving at most 40 points as it fills so a
// deep un-triaged backlog nudges the grade without ever gating (the axis carries
// no Defects, only Soft). Deliberately gentle: the idea-scout inbox is SUPPOSED
// to hold items awaiting a human, so depth is advisory, not a failure.
func triageBacklogScore(n int) float64 {
	if n <= 0 {
		return 100
	}
	penalty := n
	if penalty > 40 {
		penalty = 40
	}
	return float64(100 - penalty)
}

// triageBacklogSoft returns the single advisory line for a non-empty triage
// inbox, or nil when it is empty.
func triageBacklogSoft(n int) []string {
	if n <= 0 {
		return nil
	}
	return []string{strconv.Itoa(n) + " open issue(s) carry a triage/provenance label (idea-scout / needs-triage / triage-only) awaiting human triage"}
}
