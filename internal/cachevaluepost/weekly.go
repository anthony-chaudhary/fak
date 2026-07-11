package cachevaluepost

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/slackmeta"
)

// weeklyCardTitle is the fixed headline for the weekly fleet cache-health digest
// (#3646). It names the OPERATIONAL question the card answers — "is the cache
// machinery working across the fleet this week" — distinct from the daily Track-1/$
// card's "what did it save".
const weeklyCardTitle = "fak cache-health — weekly fleet digest (WITNESSED)"

// WeeklyCard is one weekly cache-health digest post: the folded digest plus who
// posted it. Like Card, the render reads only the digest, so the card is
// deterministic given the digest; it satisfies the shared slackCard interface
// (Text/Blocks) and drains through the same durable outbox tail as every feeder.
type WeeklyCard struct {
	Digest cachevaluereport.WeeklyDigest // the folded weekly fleet cache-health digest
	Source string                        // who posted: "ci" | "agent" | <hostname> (optional)
}

// FoldWeekly builds the channel WeeklyCard from a folded weekly digest. It is pure;
// Source is stamped by the caller after the fold, mirroring Fold/FoldTwoTrack.
func FoldWeekly(d cachevaluereport.WeeklyDigest) WeeklyCard {
	return WeeklyCard{Digest: d}
}

func (c WeeklyCard) glyph() string {
	if c.Digest.Verdict == "MEASURED" {
		return ":bar_chart:"
	}
	return ":hourglass_flowing_sand:"
}

// pct renders an optional 0..100 percentage, "n/a" when the window carried no
// evidence for it — never a fabricated zero.
func pct(p *float64) string {
	if p == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", *p)
}

// adoptionLine reads "posture-active 2/3 sessions (67%), prior 1/2 (50%) ↑ improved".
func (c WeeklyCard) adoptionLine() string {
	tw, pw := c.Digest.ThisWeek, c.Digest.PriorWeek
	return fmt.Sprintf("posture adoption: %d/%d sessions (%s), prior %d/%d (%s)  %s",
		tw.PostureActiveSessions, tw.ExitSessions, pct(tw.PostureAdoptionPct),
		pw.PostureActiveSessions, pw.ExitSessions, pct(pw.PostureAdoptionPct),
		trendArrow(c.Digest.AdoptionTrend))
}

// reuseLine reads "reuse: 75.0% over 20 m-turns, prior 50.0% ↑ improved (thin)".
func (c WeeklyCard) reuseLine() string {
	tw, pw := c.Digest.ThisWeek, c.Digest.PriorWeek
	this := "n/a"
	if tw.ReuseRatio != nil {
		this = fmt.Sprintf("%.1f%%", 100**tw.ReuseRatio)
	}
	prior := "n/a"
	if pw.ReuseRatio != nil {
		prior = fmt.Sprintf("%.1f%%", 100**pw.ReuseRatio)
	}
	thin := ""
	if tw.ReuseThin {
		thin = " (thin)"
	}
	return fmt.Sprintf("reuse: %s over %d m-turns, prior %s  %s%s",
		this, tw.MultiTurnTurns, prior, trendArrow(c.Digest.ReuseTrend), thin)
}

// shedLine reads "shed: 40000 tok over 2 fire(s) (~20000/fire, 1 bail), prior 10000 tok / 1 fire".
func (c WeeklyCard) shedLine() string {
	tw, pw := c.Digest.ThisWeek, c.Digest.PriorWeek
	per := ""
	if tw.ShedTokensPerFire != nil {
		per = fmt.Sprintf(" (~%.0f/fire, %d bail)", *tw.ShedTokensPerFire, tw.CompactionBailed)
	} else if tw.CompactionBailed > 0 {
		per = fmt.Sprintf(" (%d bail, never fired)", tw.CompactionBailed)
	}
	return fmt.Sprintf("shed: %d tok over %d fire(s)%s, prior %d tok / %d fire(s)",
		tw.ShedTokens, tw.CompactionFired, per, pw.ShedTokens, pw.CompactionFired)
}

// refusedLine reads "refused upgrades: 1 of 4 heads (25%), prior 2 of 2 (100%)".
func (c WeeklyCard) refusedLine() string {
	tw, pw := c.Digest.ThisWeek, c.Digest.PriorWeek
	return fmt.Sprintf("refused upgrades: %d of %d head(s) (%s), prior %d of %d (%s)",
		tw.TTLRefusals, tw.TTLUpgrades+tw.TTLRefusals, pct(tw.RefusedUpgradePct),
		pw.TTLRefusals, pw.TTLUpgrades+pw.TTLRefusals, pct(pw.RefusedUpgradePct))
}

func (c WeeklyCard) windowLine() string {
	return fmt.Sprintf("window: %s → %s (vs %s → %s)",
		c.Digest.ThisWeek.Start, c.Digest.ThisWeek.End,
		c.Digest.PriorWeek.Start, c.Digest.PriorWeek.End)
}

func (c WeeklyCard) fence() string {
	return "fence: " + cachevaluereport.PublishableValueFamily
}

func (c WeeklyCard) bodyLines() []string {
	return []string{
		c.windowLine(),
		c.adoptionLine(),
		c.reuseLine(),
		c.shedLine(),
		c.refusedLine(),
	}
}

// Text renders the plain-text fallback — what Slack notifications, tests, and
// --dry-run show. It leads with the verdict + headline, then the finding, the four
// signal lines, the next action, and the honesty fence, mirroring Card.Text.
func (c WeeklyCard) Text() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s *%s* — %s", c.glyph(), weeklyCardTitle, c.Digest.Verdict)
	if f := strings.TrimSpace(c.Digest.Finding); f != "" {
		fmt.Fprintf(&sb, "\n%s", f)
	}
	for _, ln := range c.bodyLines() {
		fmt.Fprintf(&sb, "\n• %s", ln)
	}
	if na := strings.TrimSpace(c.Digest.NextAction); na != "" {
		fmt.Fprintf(&sb, "\nnext: %s", na)
	}
	fmt.Fprintf(&sb, "\n%s", c.fence())
	if src := strings.TrimSpace(c.Source); src != "" {
		fmt.Fprintf(&sb, "\n_posted by %s_", src)
	}
	return slackmeta.AppendText(sb.String(), c.signalNoise())
}

// Blocks renders the Block Kit payload, carrying the same facts as Text so a
// non-Block client loses nothing.
func (c WeeklyCard) Blocks() []any {
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s* — %s", c.glyph(), weeklyCardTitle, c.Digest.Verdict)},
		},
	}
	if f := strings.TrimSpace(c.Digest.Finding); f != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": f},
		})
	}
	body := make([]string, 0, len(c.bodyLines())+1)
	for _, ln := range c.bodyLines() {
		body = append(body, "• "+ln)
	}
	if na := strings.TrimSpace(c.Digest.NextAction); na != "" {
		body = append(body, "next: "+na)
	}
	blocks = append(blocks, map[string]any{
		"type": "section",
		"text": map[string]any{"type": "mrkdwn", "text": strings.Join(body, "\n")},
	})

	ctxParts := []string{c.fence()}
	if s := strings.TrimSpace(c.Digest.Schema); s != "" {
		ctxParts = append(ctxParts, "schema: "+s)
	}
	if src := strings.TrimSpace(c.Source); src != "" {
		ctxParts = append(ctxParts, "posted by "+src)
	}
	blocks = append(blocks, map[string]any{
		"type":     "context",
		"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
	})
	return slackmeta.AppendContext(blocks, c.signalNoise())
}

func (c WeeklyCard) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(c.Digest.Finding, c.Digest.NextAction, c.fence()) + len(c.bodyLines())
	noise := 1 + slackmeta.NonEmpty(c.Digest.Schema, c.Source)
	return slackmeta.New(signal, noise, "four weekly fleet cache-health signals with week-over-week direction vs schema/source")
}
