package dispatchtick

import "regexp"

// prereqMarkerRE matches a "depends-on:/blocked-by: #N" prerequisite marker in an issue body,
// followed by one or more "#N" references (comma/and/&-separated). issueRefRE pulls the bare
// numbers back out of a matched marker's reference run. Ported verbatim from the Python
// dispatcher (tools/issue_resolve_dispatch.py _PREREQ_MARKER / _ISSUE_REF) so the two pickers
// parse the same edges; the (?im) flags make the verb match case-insensitive and per-line.
var (
	prereqMarkerRE = regexp.MustCompile(`(?im)\b(?:depends[ -]?on|blocked[ -]?by)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	issueRefRE     = regexp.MustCompile(`#(\d+)`)
)

// CandidateBlockedBy is the dispatchorder Candidate.BlockedBy list an issue earns from its body:
// the issue numbers it declares as prerequisites via a "depends-on:/blocked-by: #N" marker (one
// or many, comma/and/&-separated), as bare-numeric string IDs ("120", not "#120") in the leaf's
// ID space — the same space dispatch builds from strconv.Itoa(number), so a returned ID lines up
// with a Candidate.ID for the open-prereq presence check.
//
// Prose that merely contains the words ("it depends on the weather") never matches: the marker
// must be immediately followed by a "#N" reference. An issue with no marker maps to an empty
// list, so a marker-free candidate carries no prerequisite edges and dispatches exactly as it did
// before the field existed (the additive, no-regression posture). Order-preserving and
// de-duplicated, so the result is deterministic per body.
func CandidateBlockedBy(body string) []string {
	var out []string
	seen := map[string]bool{}
	for _, marker := range prereqMarkerRE.FindAllStringSubmatch(body, -1) {
		for _, ref := range issueRefRE.FindAllStringSubmatch(marker[1], -1) {
			num := ref[1]
			if seen[num] {
				continue
			}
			seen[num] = true
			out = append(out, num)
		}
	}
	return out
}
