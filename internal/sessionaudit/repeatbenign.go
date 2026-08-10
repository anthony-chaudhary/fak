package sessionaudit

import "strings"

// repeatBenignTools is the explicit, reviewable catalog of tools whose repeated
// use is normally progress rather than a stuck loop. Keep this list narrow: a
// tool belongs here only when repeated calls are intrinsic to reading or
// searching through distinct material.
var repeatBenignTools = map[string]bool{
	"glob":       true,
	"grep":       true,
	"read":       true,
	"search":     true,
	"search_kb":  true,
	"websearch":  true,
	"web_search": true,
}

// repeatBenign reports whether repetition detectors should exclude toolName.
// Tool names are matched case-insensitively because transcript providers differ
// in whether they preserve catalog casing (for example, Read versus read).
func repeatBenign(toolName string) bool {
	return repeatBenignTools[strings.ToLower(strings.TrimSpace(toolName))]
}
