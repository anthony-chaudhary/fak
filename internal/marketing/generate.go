package marketing

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// generate.go — the event -> Artifact generators. Each takes witnessed ships (already
// CLAIMS.md-filtered by the caller via FilterMarketable) and folds them into one Artifact,
// building every claim through NewClaim so an unwitnessed item cannot enter. A generator
// given zero marketable ships returns an HONEST empty artifact ("no witnessed ships"), never
// a fabricated one — the render handles the empty case.
//
// The claim TEXT is derived from the ship subject: strip the `type(scope):` prefix and the
// trailing `(fak <leaf>)` stamp so the bullet reads as product prose, then prefix a framing
// word chosen from the conventional-commits verb (feat -> "New:", fix -> "Fixed:", perf ->
// "Faster:"). The sha — the witness — is attached by NewClaim, not the text.

// subjectPrefixRE captures the conventional-commits type(scope): prefix so claimText can
// strip it. The verb (group 1) drives the framing word; the body (after the colon) is the
// human assertion.
var subjectPrefixRE = regexp.MustCompile(`^([a-z]+)(?:\(([^)]+)\))?:\s*(.*)$`)

// trailerStripRE removes a trailing `(fak <leaf>)` / `(refs fak <leaf>)` stamp from a subject
// so it doesn't appear twice (the leaf is already rendered separately on the bullet).
var trailerStripRE = regexp.MustCompile(`\s*\((?:refs\s+)?fak[ :]+[A-Za-z0-9][\w.\-]*\)\s*$`)

// directStampRE removes a leading `fak/<leaf>:` DIRECT ship-stamp so the bullet reads as the
// assertion alone (the leaf is rendered separately). A direct stamp isn't a conventional
// `type(scope):`, so subjectPrefixRE won't strip it — this does.
var directStampRE = regexp.MustCompile(`^fak/[A-Za-z0-9][\w.\-]*:\s*`)

// frameWord maps a conventional-commits verb to the marketing framing word. An unknown verb
// falls through to "Shipped:" — honest and neutral.
func frameWord(verb string) string {
	switch verb {
	case "feat":
		return "New:"
	case "fix":
		return "Fixed:"
	case "perf":
		return "Faster:"
	case "docs":
		return "Documented:"
	case "refactor":
		return "Improved:"
	default:
		return "Shipped:"
	}
}

// claimText renders a ship's subject as marketing prose: strip the stamp, strip the
// type(scope): prefix, and prefix the verb-derived framing word. E.g.
// "feat(gateway): add the reclaim path (fak gateway)" -> "New: add the reclaim path".
func claimText(s Ship) string {
	body := trailerStripRE.ReplaceAllString(s.Subject, "")
	verb := ""
	if m := subjectPrefixRE.FindStringSubmatch(body); m != nil {
		verb = m[1]
		body = m[3]
	} else if directStampRE.MatchString(body) {
		// A direct `fak/<leaf>:` stamp carries no conventional verb; strip the stamp and
		// frame neutrally (frameWord("") -> "Shipped:").
		body = directStampRE.ReplaceAllString(body, "")
	}
	body = strings.TrimSpace(body)
	if body == "" {
		body = s.Subject // never emit an empty claim; fall back to the raw subject
	}
	return frameWord(verb) + " " + body
}

// buildClaims turns ships into witnessed claims via NewClaim. Any ship that fails the
// witness check is skipped (never fatal); skipped is normally 0 because CollectShips only
// emits {trailer,direct} ships with shas.
func buildClaims(ships []Ship) (claims []Claim, skipped int) {
	return MustClaims(claimText, ships)
}

// WeeklyDigest folds a window's worth of ships into a "what shipped" rollup — the bgloop /
// cron default. when stamps the title; activity carries the honest other-commits count;
// excluded surfaces what the CLAIMS.md gate withheld.
func WeeklyDigest(ships []Ship, activity Activity, excluded []ExcludedShip, when time.Time) Artifact {
	claims, _ := buildClaims(ships)
	title := "fak — what shipped"
	if !when.IsZero() {
		title = fmt.Sprintf("fak — what shipped (week of %s)", when.Format("Jan 2"))
	}
	return Artifact{
		Kind:      KindWeeklyDigest,
		Title:     title,
		Lead:      digestLead(claims),
		Claims:    claims,
		Activity:  activity,
		Excluded:  excluded,
		DedupeKey: dedupeKey("weekly-digest", ships),
	}
}

// LaunchBlurb announces a single feature on its own — the git-hook / per-feature surface.
func LaunchBlurb(s Ship) Artifact {
	claims, _ := buildClaims([]Ship{s})
	return Artifact{
		Kind:      KindLaunchBlurb,
		Title:     "New in fak: " + leafTitle(s.Leaf),
		Claims:    claims,
		Activity:  Activity{Commits: 1, Ships: 1},
		DedupeKey: dedupeKey("launch-blurb", []Ship{s}),
	}
}

// EpicBlurb groups the ships delivered under a closed epic. epicTitle is the human label
// (e.g. the GitHub issue title); the ships are its witnessed deliverables.
func EpicBlurb(epicTitle string, ships []Ship, excluded []ExcludedShip) Artifact {
	claims, _ := buildClaims(ships)
	title := "fak epic shipped"
	if epicTitle != "" {
		title = "Epic shipped: " + epicTitle
	}
	return Artifact{
		Kind:      KindEpicBlurb,
		Title:     title,
		Lead:      fmt.Sprintf("%d witnessed ship%s closed this epic.", len(claims), plural(len(claims))),
		Claims:    claims,
		Activity:  Activity{Commits: len(ships), Ships: len(ships)},
		Excluded:  excluded,
		DedupeKey: dedupeKey("epic-blurb:"+epicTitle, ships),
	}
}

// ReleaseHighlight folds a release's ships into a highlight card. version is the tag
// (e.g. "v0.18.0"); notesLead is an optional one-line summary pulled from the release notes.
func ReleaseHighlight(version, notesLead string, ships []Ship, excluded []ExcludedShip) Artifact {
	claims, _ := buildClaims(ships)
	lead := notesLead
	if lead == "" {
		lead = fmt.Sprintf("%d witnessed ship%s in this release.", len(claims), plural(len(claims)))
	}
	return Artifact{
		Kind:      KindReleaseHighlight,
		Title:     "fak " + version,
		Lead:      lead,
		Claims:    claims,
		Activity:  Activity{Commits: len(ships), Ships: len(ships)},
		Excluded:  excluded,
		DedupeKey: dedupeKey("release-highlight:"+version, ships),
	}
}

// digestLead writes a one-line lead summarizing the leaves that shipped, e.g.
// "Updates across gateway, model, and 2 more." Empty when there are no claims.
func digestLead(claims []Claim) string {
	if len(claims) == 0 {
		return ""
	}
	seen := map[string]bool{}
	var leaves []string
	for _, c := range claims {
		if c.Ship.Leaf != "" && !seen[c.Ship.Leaf] {
			seen[c.Ship.Leaf] = true
			leaves = append(leaves, c.Ship.Leaf)
		}
	}
	sort.Strings(leaves)
	switch len(leaves) {
	case 0:
		return ""
	case 1:
		return "Updates in " + leaves[0] + "."
	case 2:
		return "Updates across " + leaves[0] + " and " + leaves[1] + "."
	default:
		return fmt.Sprintf("Updates across %s, %s, and %d more.", leaves[0], leaves[1], len(leaves)-2)
	}
}

// leafTitle renders a leaf as a readable feature name; "" leaves a generic label.
func leafTitle(leaf string) string {
	if leaf == "" {
		return "an update"
	}
	return leaf
}

// dedupeKey builds the stable identity an idempotent poster keys on: the kind plus the
// sorted ship shas. Re-rendering the same ships in the same window yields the same key, so a
// dedupe-aware post is a no-op — the second idempotency layer behind the high-water mark.
func dedupeKey(kind string, ships []Ship) string {
	shas := make([]string, 0, len(ships))
	for _, s := range ships {
		shas = append(shas, s.SHA)
	}
	sort.Strings(shas)
	return kind + ":" + strings.Join(shas, ",")
}
