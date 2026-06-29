package marketing

import (
	"fmt"
	"sort"
	"strings"
)

// render.go — the marketing Artifact and its Slack render. An Artifact is one rendered
// marketing post, decoupled from which generator built it (generate.go), exactly as
// scoreboard.Update is decoupled from its source. It satisfies the cmd/fak slackCard
// interface (Text() + Blocks()), so it reuses the shared slackPostTail transport verbatim —
// no new Slack code.
//
// The honesty floor lives in the render: every claim bullet prints its witnessing sha
// inline, and the footer states "N witnessed ships · M other commits". Because a Claim
// cannot be constructed without a sha (claim.go NewClaim), there is no code path here that
// can emit an unwitnessed bullet — the render trusts the type, it does not re-validate.

// Kind is the marketing artifact type. The render is shared; the Kind only changes the
// headline framing and emoji, so a new Kind is a small switch arm, not a new renderer.
type Kind string

const (
	// KindLaunchBlurb: one feature/epic announced on its own — "New in fak: <X>".
	KindLaunchBlurb Kind = "launch-blurb"
	// KindWeeklyDigest: a "what shipped" rollup over a window — the bgloop/cron default.
	KindWeeklyDigest Kind = "weekly-digest"
	// KindReleaseHighlight: a docs/releases/v*.md folded into a highlight card.
	KindReleaseHighlight Kind = "release-highlight"
	// KindEpicBlurb: the ships under a closed epic, grouped.
	KindEpicBlurb Kind = "epic-blurb"
)

// Artifact is one rendered marketing post. Title/Lead are the human framing; Claims are the
// witnessed spine (each carrying its sha); Activity is the honest "other commits" tally;
// Excluded is what the CLAIMS.md gate withheld (surfaced, never silent). DedupeKey is the
// stable identity an idempotent poster keys on so a re-render of the same window is a no-op.
type Artifact struct {
	Kind      Kind
	Title     string         // headline, e.g. "fak — what shipped this week"
	Lead      string         // optional one-line framing under the title
	Claims    []Claim        // the witnessed assertions; each renders its sha inline
	Activity  Activity       // commits/ships counts for the honest footer
	Excluded  []ExcludedShip // ships the CLAIMS.md gate withheld, with reasons (surfaced)
	Source    string         // who posted: ci | hook | serve | cron | hostname
	DedupeKey string         // stable identity for idempotent posting (e.g. "weekly-digest:<sha-range>")
}

// emoji picks the headline glyph by kind — a launch is a rocket, a digest a newspaper, a
// release a tag, an epic a checkered flag. Cosmetic; the witness rung is the substance.
func (a Artifact) emoji() string {
	switch a.Kind {
	case KindLaunchBlurb:
		return ":rocket:"
	case KindReleaseHighlight:
		return ":label:"
	case KindEpicBlurb:
		return ":checkered_flag:"
	default:
		return ":newspaper:"
	}
}

// title returns the headline, defaulting by kind when Title is unset.
func (a Artifact) title() string {
	if a.Title != "" {
		return a.Title
	}
	switch a.Kind {
	case KindLaunchBlurb:
		return "New in fak"
	case KindReleaseHighlight:
		return "fak release"
	case KindEpicBlurb:
		return "fak epic shipped"
	default:
		return "fak — what shipped"
	}
}

// claimBullet renders one claim as a witnessed line: the text, then the sha + leaf + label.
// This is the single place a claim becomes prose; it ALWAYS includes the sha, so a reader
// (or an answer engine) can trace the assertion to the commit. There is no unwitnessed form.
func claimBullet(c Claim) string {
	leaf := c.Ship.Leaf
	if leaf != "" {
		leaf = " (fak " + leaf + ")"
	}
	return fmt.Sprintf("• %s — `%s`%s _%s_", c.Text, c.Ship.SHA, leaf, c.Label)
}

// footer renders the honest provenance tally: how many witnessed ships back this artifact
// and how many other commits the window held (never folded into the ship count), plus a
// note when ships were withheld by the CLAIMS.md gate.
func (a Artifact) footer() string {
	parts := []string{fmt.Sprintf("%d witnessed ship%s", len(a.Claims), plural(len(a.Claims)))}
	other := a.Activity.Commits - a.Activity.Ships
	if other < 0 {
		other = 0
	}
	parts = append(parts, fmt.Sprintf("%d other commit%s", other, plural(other)))
	if n := len(a.Excluded); n > 0 {
		parts = append(parts, fmt.Sprintf("%d held (CLAIMS.md stub/simulated)", n))
	}
	return strings.Join(parts, " · ")
}

// Text renders the plain-text fallback — the notification line and the --dry-run / test
// surface. An artifact with zero claims renders an honest "no witnessed ships" line, never
// an empty boast.
func (a Artifact) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s *%s*", a.emoji(), a.title())
	if a.Lead != "" {
		fmt.Fprintf(&b, "\n%s", a.Lead)
	}
	if len(a.Claims) == 0 {
		b.WriteString("\nNo witnessed ships in this window.")
	}
	for _, c := range sortedClaims(a.Claims) {
		fmt.Fprintf(&b, "\n%s", claimBullet(c))
	}
	fmt.Fprintf(&b, "\n_%s_", a.footer())
	if a.Source != "" {
		fmt.Fprintf(&b, "\n_posted by %s_", a.Source)
	}
	return b.String()
}

// Blocks renders the Block Kit payload, carrying the same facts as Text (sha on every
// bullet, the honest footer in a context block) so a non-Block client loses nothing.
func (a Artifact) Blocks() []any {
	header := fmt.Sprintf("%s %s", a.emoji(), a.title())
	blocks := []any{
		map[string]any{"type": "section", "text": map[string]any{"type": "mrkdwn", "text": "*" + header + "*"}},
	}
	if a.Lead != "" {
		blocks = append(blocks, map[string]any{"type": "section", "text": map[string]any{"type": "mrkdwn", "text": a.Lead}})
	}
	if len(a.Claims) == 0 {
		blocks = append(blocks, map[string]any{"type": "section", "text": map[string]any{"type": "mrkdwn", "text": "No witnessed ships in this window."}})
	} else {
		var lines []string
		for _, c := range sortedClaims(a.Claims) {
			lines = append(lines, claimBullet(c))
		}
		blocks = append(blocks, map[string]any{"type": "section", "text": map[string]any{"type": "mrkdwn", "text": strings.Join(lines, "\n")}})
	}
	ctxParts := []string{a.footer()}
	if a.Source != "" {
		ctxParts = append(ctxParts, "posted by "+a.Source)
	}
	blocks = append(blocks, map[string]any{
		"type":     "context",
		"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
	})
	return blocks
}

// sortedClaims returns the claims newest-first by ship date (stable), so the freshest ship
// leads the card. A copy — the input order is not mutated.
func sortedClaims(claims []Claim) []Claim {
	out := append([]Claim(nil), claims...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Ship.Date.After(out[j].Ship.Date) })
	return out
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
