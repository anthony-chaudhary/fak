package dispatchtick

import (
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

var (
	startBlockedByMarkerRE    = regexp.MustCompile(`(?im)\b(?:start[ -]?blocked[ -]?by)\b[:\s]*([^\r\n]*)`)
	coordinatesWithMarkerRE   = regexp.MustCompile(`(?im)\b(?:coordinates[ -]?with)\b[:\s]*([^\r\n]*)`)
	promotionRequiresMarkerRE = regexp.MustCompile(`(?im)\b(?:promotion[ -]?requires|promotes[ -]?after)\b[:\s]*([^\r\n]*)`)
	legacyPrereqMarkerRE      = regexp.MustCompile(`(?im)\b(?:depends[ -]?on|blocked[ -]?by)\b[:\s]*([^\r\n]*)`)
	prereqMarkerRE            = legacyPrereqMarkerRE

	dependenciesSectionHeaderRE = regexp.MustCompile(`(?im)^#{1,6}\s*(?:Dependencies|Dependency markers)\b`)
	shiftLeftSectionHeaderRE    = regexp.MustCompile(`(?im)^#{1,6}\s*(?:Scope\s*/\s*tree|Witness\s*/\s*proof|Placement)\b`)
	markdownHeadingRE           = regexp.MustCompile(`(?m)^#{1,6}\s+`)
	explicitBlockedByRE         = regexp.MustCompile(`(?im)(?:^|[^\w-])(?:blocked[ -]?by)\b[:\s]*([^\r\n]*)`)
	legacyOrStartPrereqRE       = regexp.MustCompile(`(?im)\b(?:start[ -]?blocked[ -]?by|depends[ -]?on|blocked[ -]?by)\b[:\s]*([^\r\n]*)`)
)

// Check the entire atom before the canonical parser sees it: its prose extractor
// must not turn a malformed qualified reference into a bare same-number issue.
const dependencyOwnerAtom = `[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?`
const dependencyRepositoryAtom = dependencyOwnerAtom + `/[A-Za-z0-9_.-]+`

var dependencyAtomRE = regexp.MustCompile(`^(?:#[1-9][0-9]*|` + dependencyRepositoryAtom + `#[1-9][0-9]*|https?://github\.com/` + dependencyRepositoryAtom + `/issues/[1-9][0-9]*)$`)

func prerequisiteRefs(value string) (refs []string, malformed string) {
	rest := strings.TrimSpace(value)
	if rest == "" {
		return nil, "missing reference"
	}
	if strings.EqualFold(rest, "none") || strings.EqualFold(rest, "none.") {
		return nil, ""
	}
	for rest != "" {
		end := strings.IndexAny(rest, " \t,&")
		if end < 0 {
			end = len(rest)
		}
		if end == 0 {
			return refs, "unexpected reference separator"
		}
		atom := strings.TrimSuffix(strings.TrimSuffix(rest[:end], "."), ";")
		if !dependencyAtomRE.MatchString(strings.ToLower(atom)) {
			return refs, rest[:end]
		}
		parsed := issuepolicy.CandidatePickupBlockedBy(issuepolicy.ParseIssueDependencies("Start blocked by: " + strings.ToLower(atom)))
		if len(parsed) != 1 {
			return refs, atom
		}
		refs = append(refs, parsed[0])
		rest = strings.TrimSpace(rest[end:])
		if rest == "" {
			break
		}
		if strings.EqualFold(rest, "and") {
			return refs, "missing reference after separator"
		}
		if rest[0] == ',' || rest[0] == '&' {
			rest = strings.TrimSpace(rest[1:])
		} else if len(rest) > 3 && strings.EqualFold(rest[:3], "and") && (rest[3] == ' ' || rest[3] == '\t') {
			rest = strings.TrimSpace(rest[3:])
		} else if !strings.HasPrefix(rest, "#") && !strings.Contains(strings.Fields(rest)[0], "/") {
			break // Ordinary explanation after a completed reference is not another edge.
		}
		if rest == "" {
			return refs, "missing reference after separator"
		}
	}
	return refs, ""
}

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
		refs, _ := prerequisiteRefs(marker[1])
		for _, num := range refs {
			if seen[num] {
				continue
			}
			seen[num] = true
			out = append(out, num)
		}
	}
	return out
}

func blockingMarkerValues(body string) []string {
	var values []string
	add := func(re *regexp.Regexp, text string) {
		for _, marker := range re.FindAllStringSubmatch(text, -1) {
			values = append(values, marker[1])
		}
	}
	if isNewContract(body) {
		add(startBlockedByMarkerRE, body)
		add(explicitBlockedByRE, extractDependenciesSection(body))
	} else {
		add(legacyOrStartPrereqRE, body)
	}
	return values
}

// The live consumer has only local numeric identities. Until repository-aware
// resolution exists, do not mistake an absent qualified ID for a closed issue.
func unresolvedStartPrerequisites(body string) []string {
	var out []string
	seen := map[string]bool{}
	add := func(ref string) {
		if !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	for _, ref := range CandidateBlockedBy(body) {
		if strings.Contains(ref, "#") {
			add(ref)
		}
	}
	for _, value := range blockingMarkerValues(body) {
		refs, malformed := prerequisiteRefs(value)
		if malformed != "" && (isNewContract(body) || len(refs) > 0 || strings.ContainsAny(value, "#/")) {
			add(truncateRunes(malformed, 100))
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
// the canonical references that block starting/picking up this issue. Bare #N
// retains its numeric form; a qualified reference retains owner/repo#N.
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
		for _, num := range extractRefs(legacyOrStartPrereqRE, body) {
			add(num)
		}
	}
	return out
}
