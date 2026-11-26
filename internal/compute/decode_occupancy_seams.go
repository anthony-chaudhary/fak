package compute

import (
	_ "embed"
	"strings"
)

// decodeSeamsText records, per shortlist candidate, the seam it targets as tab-separated lines. The seams
// name source files and kernels, so they live in this data file rather than in Go literals.
//
//go:embed decode_occupancy_seams.txt
var decodeSeamsText string

// decodeSeamOf returns the recorded seam of a candidate. A candidate with no record says so in its text,
// so an unrecorded seam can never read as an empty or clean one.
func decodeSeamOf(candidate string) string {
	for _, line := range strings.Split(decodeSeamsText, "\n") {
		name, seam, ok := strings.Cut(line, "\t")
		if ok && name == candidate {
			return strings.TrimSpace(seam)
		}
	}
	return "seam not recorded for " + candidate
}
