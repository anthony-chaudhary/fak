package hooks

import (
	"fmt"
	"strings"
)

const microharnessWitnessToken = "microharness-witness:"

// gateMicroharnessWitness nudges bounded-harness changes to retain a checkable receipt.
// It is advisory at registration: adoption can be measured before any block-mode promotion.
func gateMicroharnessWitness(d *StagedDiff) ([]Finding, error) {
	var candidates []string
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			continue
		}
		if strings.HasPrefix(p, "cmd/microharnessdemo/") ||
			strings.HasPrefix(p, "internal/microagent/") ||
			strings.HasPrefix(p, "internal/harnesscompose/") {
			candidates = append(candidates, p)
		}
	}
	d.NoteCandidates("MICROHARNESS_WITNESS", len(candidates), "bounded harness source file(s)")
	if len(candidates) == 0 {
		return nil, nil
	}
	for _, line := range d.AddedLines() {
		if strings.Contains(strings.ToLower(line.Text), microharnessWitnessToken) {
			return nil, nil
		}
	}
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		if strings.HasSuffix(p, "_test.go") && (strings.HasPrefix(p, "cmd/microharnessdemo/") ||
			strings.HasPrefix(p, "internal/microagent/") || strings.HasPrefix(p, "internal/harnesscompose/")) {
			return nil, nil
		}
	}
	return []Finding{{
		Gate:   "MICROHARNESS_WITNESS",
		File:   candidates[0],
		Detail: fmt.Sprintf("bounded harness source changed without a staged test or `%s` receipt; run the focused proof and retain its witness", microharnessWitnessToken),
	}}, nil
}
