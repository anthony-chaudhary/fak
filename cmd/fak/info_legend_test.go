package main

import (
	"strings"
	"testing"
)

// legendVocabulary is the contract between the live status line (renderGuardInfoLine) and the
// guide printed above it (guardInfoLegend): for every quantity the line shows, `shown` is the
// plain word an operator reads on the line and `field` is the /debug/vars field that number is
// decoded from. The legend must carry BOTH, so a watcher can move between the pane, `fak info
// --json` and a raw curl of /debug/vars without guessing which name is which.
var legendVocabulary = []struct {
	shown string // the word renderGuardInfoLine puts on the line
	field string // the /debug/vars name (or concept) the legend must tie it to
}{
	{shown: "cache: ", field: "cache"},
	{shown: "safety: ", field: "floor"},
	{shown: "replies ", field: "inference.turns"},
	{shown: "busy with ", field: "gateway.inflight_requests"},
	{shown: "running ", field: "gateway.uptime_seconds"},
}

// TestGuardInfoLegendCoversEveryLineTerm proves the guide is complete against the line ITSELF,
// not against a frozen wish-list: each term is first asserted to be really on a rendered status
// line, then asserted to be explained in the legend under both the word the operator sees and
// the /debug/vars field name it comes from. Dropping a clause from the line, or an entry from
// the legend, fails here.
func TestGuardInfoLegendCoversEveryLineTerm(t *testing.T) {
	line := renderGuardInfoLine(provenVisualVars())
	legend := guardInfoLegend()
	for _, term := range legendVocabulary {
		if !strings.Contains(line, term.shown) {
			t.Errorf("status line does not show %q — the legend entry would be stale:\n%s", term.shown, line)
		}
		word := strings.TrimSuffix(strings.TrimSpace(term.shown), ":")
		if !strings.Contains(legend, word) {
			t.Errorf("legend has no entry for the shown word %q:\n%s", word, legend)
		}
		if !strings.Contains(legend, term.field) {
			t.Errorf("legend never ties %q to its /debug/vars field %q:\n%s", word, term.field, legend)
		}
	}
}

// TestGuardInfoLegendStatesUnitsAndRange proves the legend answers the two questions a bare
// number cannot: is this a running total or an instantaneous reading, and what does the safety
// count actually cover. "busy with" is the trap — it is a gauge that falls back to 0 between
// turns, so an operator who reads it as a session total mis-reads an idle gateway as a broken
// one.
func TestGuardInfoLegendStatesUnitsAndRange(t *testing.T) {
	legend := guardInfoLegend()
	for _, want := range []string{
		"gauge, not a total", // busy with / inflight is instantaneous
		"drops back to 0",    // ...and returns to zero between calls
		"never decreases",    // replies / turns is a monotone running total
		"counts up from 0",   // running / uptime restarts with the gateway
		"blocked",            // the three floor outcomes, named
		"fixed",
		"set aside",
	} {
		if !strings.Contains(legend, want) {
			t.Errorf("legend does not state %q — an operator cannot tell units/range from the number alone:\n%s", want, legend)
		}
	}
}
