package devindex

import "strings"

// repeatBenignVerbs is the explicit, reviewable catalog of tools whose repeated
// use is normally progress rather than a stuck loop. Keep this list narrow: a
// tool belongs here only when repeated calls are intrinsic to reading or
// searching through distinct material.
var repeatBenignVerbs = map[string]bool{
	"glob":       true,
	"grep":       true,
	"read":       true,
	"search":     true,
	"search_kb":  true,
	"websearch":  true,
	"web_search": true,
}

// RepeatBenign reports whether repetition detectors should exclude toolName.
// Tool names are matched case-insensitively because transcript providers differ
// in whether they preserve catalog casing (for example, Read versus read).
func RepeatBenign(toolName string) bool {
	return repeatBenignVerbs[strings.ToLower(strings.TrimSpace(toolName))]
}
