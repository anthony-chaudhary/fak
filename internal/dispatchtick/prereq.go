package dispatchtick

import "regexp"

var (
	startBlockedByMarkerRE    = regexp.MustCompile(`(?im)\b(?:start[ -]?blocked[ -]?by)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	coordinatesWithMarkerRE   = regexp.MustCompile(`(?im)\b(?:coordinates[ -]?with)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	promotionRequiresMarkerRE = regexp.MustCompile(`(?im)\b(?:promotion[ -]?requires|promotes[ -]?after)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	legacyPrereqMarkerRE      = regexp.MustCompile(`(?im)\b(?:depends[ -]?on|blocked[ -]?by)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	prereqMarkerRE            = legacyPrereqMarkerRE
	issueRefRE                = regexp.MustCompile(`#(\d+)`)

	dependenciesSectionHeaderRE = regexp.MustCompile(`(?im)^#{1,6}\s*(?:Dependencies|Dependency markers)\b`)
	shiftLeftSectionHeaderRE    = regexp.MustCompile(`(?im)^#{1,6}\s*(?:Scope\s*/\s*tree|Witness\s*/\s*proof|Placement)\b`)
	markdownHeadingRE           = regexp.MustCompile(`(?m)^#{1,6}\s+`)
	explicitBlockedByRE         = regexp.MustCompile(`(?im)(?:^|[^\w-])(?:blocked[ -]?by)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
	legacyOrStartPrereqRE       = regexp.MustCompile(`(?im)\b(?:start[ -]?blocked[ -]?by|depends[ -]?on|blocked[ -]?by)\b[:\s]*((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)`)
)

// DependencyEdges represents the standardized three-way dependency relations parsed
// from an issue body.
type DependencyEdges struct {
	StartBlockedBy    []string `json:"start_blocked_by,omitempty"`
	CoordinatesWith   []string `json:"coordinates_with,omitempty"`
	PromotionRequires []string `json:"promotion_requires,omitempty"`
}

func extractRefs(re *regexp.Regexp, text string) []string {
	var out []string
	seen := map[string]bool{}
	for _, marker := range re.FindAllStringSubmatch(text, -1) {
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

func extractDependenciesSection(body string) string {
	loc := dependenciesSectionHeaderRE.FindStringIndex(body)
	if loc == nil {
		return ""
	}
	rest := body[loc[1]:]
	nextLoc := markdownHeadingRE.FindStringIndex(rest)
	if nextLoc != nil {
		return rest[:nextLoc[0]]
	}
	return rest
}

func isNewContract(body string) bool {
	return startBlockedByMarkerRE.MatchString(body) ||
		coordinatesWithMarkerRE.MatchString(body) ||
		promotionRequiresMarkerRE.MatchString(body) ||
		dependenciesSectionHeaderRE.MatchString(body) ||
		shiftLeftSectionHeaderRE.MatchString(body)
}

// CandidateCoordinatesWith extracts advisory interface alignment references ('Coordinates with: #N')
// from an issue body. These never block pickup.
func CandidateCoordinatesWith(body string) []string {
	return extractRefs(coordinatesWithMarkerRE, body)
}

// CandidatePromotionRequires extracts promotion gate references ('Promotion requires: #N', 'promotes-after: #N')
// from an issue body. These gate promotion/closure claims, not implementation pickup.
func CandidatePromotionRequires(body string) []string {
	return extractRefs(promotionRequiresMarkerRE, body)
}

// CandidateDependencyEdges extracts all three dependency edge types from an issue body.
func CandidateDependencyEdges(body string) DependencyEdges {
	return DependencyEdges{
		StartBlockedBy:    CandidateBlockedBy(body),
		CoordinatesWith:   CandidateCoordinatesWith(body),
		PromotionRequires: CandidatePromotionRequires(body),
	}
}

// CandidateBlockedBy is the dispatchorder Candidate.BlockedBy list an issue earns from its body:
// the issue numbers that block starting/picking up this issue.
//
// In newly authored contracts (containing startBlockedByMarkerRE, coordinatesWithMarkerRE,
// promotionRequiresMarkerRE, a Dependencies section, or shift-left sections), only 'Start blocked by:'
// (and explicit 'blocked-by:' in a Dependencies section) blocks pickup. Ambiguous free-text
// 'Depends on #N' in prose is NOT inferred as a hard hold.
//
// In legacy issue bodies, legacyPrereqMarkerRE ("depends-on:/blocked-by: #N") and startBlockedByMarkerRE
// are extracted as before.
//
// Advisory ('Coordinates with:') and promotion ('Promotion requires:') edges never populate BlockedBy,
// ensuring open promotion evidence does not produce BLOCKED_BY_OPEN_PREREQ holds.
func CandidateBlockedBy(body string) []string {
	nonBlocking := map[string]bool{}
	for _, num := range CandidateCoordinatesWith(body) {
		nonBlocking[num] = true
	}
	for _, num := range CandidatePromotionRequires(body) {
		nonBlocking[num] = true
	}

	var out []string
	seen := map[string]bool{}
	add := func(num string) {
		if seen[num] || nonBlocking[num] {
			return
		}
		seen[num] = true
		out = append(out, num)
	}

	if isNewContract(body) {
		for _, num := range extractRefs(startBlockedByMarkerRE, body) {
			add(num)
		}
		if depSec := extractDependenciesSection(body); depSec != "" {
			for _, num := range extractRefs(explicitBlockedByRE, depSec) {
				add(num)
			}
		}
	} else {
		for _, marker := range legacyOrStartPrereqRE.FindAllStringSubmatch(body, -1) {
			for _, ref := range issueRefRE.FindAllStringSubmatch(marker[1], -1) {
				add(ref[1])
			}
		}
	}
	return out
}
