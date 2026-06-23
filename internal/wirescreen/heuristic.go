package wirescreen

import (
	"context"
	"strings"
	"unicode"
)

// heuristicScreener is a deterministic, dependency-free reference Screener. It is NOT
// a model — it exists to PROVE the wiring end to end (a screen that fires routes a
// floor-admitted result into the recoverable quarantine) and to give an operator a
// zero-dependency opt-in via FAK_WIRE_SCREEN=heuristic.
//
// To be a MEANINGFUL additive screen, it flags injection-shaped phrasings that the
// context-MMU's literal injectionMarkers list deliberately does NOT cover (it never
// duplicates the floor — that would add nothing). It normalizes case and collapses
// internal whitespace first, a light de-obfuscation in the spirit of normgate, then
// matches a small set of imperative-override / data-exfil tells. The real semantic
// judgement is the model-backed screener (a gated follow-on); this is the floor for
// the floor's blind spot, not the ceiling.
type heuristicScreener struct{}

func (heuristicScreener) Name() string { return "heuristic" }

// semanticMarkers are injection tells phrased to evade the literal-marker floor:
// social-engineering imperatives, role reassignment, and exfil instructions that
// contain none of ctxmmu's exact strings. Kept lowercase + single-spaced to match the
// normalized body.
var semanticMarkers = []string{
	"do not tell the user",
	"don't tell the user",
	"without telling the user",
	"without informing the user",
	"keep this from the user",
	"from now on you",
	"your real instructions are",
	"your true instructions",
	"your new role is",
	"pretend you are",
	"pretend that you",
	"act as if you",
	"bypass the",
	"disregard your guidelines",
	"forget your instructions",
	"forget the previous",
	"send the contents to",
	"send your context to",
	"forward this conversation to",
	"override your safety",
}

func (heuristicScreener) Flag(_ context.Context, body []byte, _ string) (bool, string) {
	norm := normalize(body)
	for _, m := range semanticMarkers {
		if strings.Contains(norm, m) {
			return true, m
		}
	}
	return false, ""
}

// normalize lowercases and collapses any run of whitespace to a single space, so a
// payload that spaces or newlines out its imperative still matches a single-spaced
// marker. Deterministic and allocation-bounded; no regex backtracking.
func normalize(b []byte) string {
	var sb strings.Builder
	sb.Grow(len(b))
	lastSpace := false
	for _, r := range string(b) {
		if unicode.IsSpace(r) {
			if !lastSpace {
				sb.WriteByte(' ')
				lastSpace = true
			}
			continue
		}
		sb.WriteRune(unicode.ToLower(r))
		lastSpace = false
	}
	return sb.String()
}
