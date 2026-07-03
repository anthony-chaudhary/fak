package sidecar

// render.go holds the two surfaces of the sidecar pane — terminal text and Slack
// Block Kit. They are deliberately in one file, side by side, so the parity
// contract is visible at a glance: BOTH functions walk p.Sections(), the single
// producer of the ordered render model, and each renders that model's plane
// order and per-plane Line order verbatim. Neither reaches into the raw inputs.
// A field can only appear on one surface if it appears in Sections(), where the
// other surface sees it too — so a divergence is a change to Sections(), which
// both observe, not a drift a single surface can acquire alone.

import (
	"fmt"
	"strings"
)

// planeGlyph and planeProvGlyph keep the two surfaces cosmetically consistent
// without letting cosmetics carry a field the other surface lacks.
func planeTitle(plane string) string {
	switch plane {
	case PlaneSessions:
		return "Sessions"
	case PlaneAccounts:
		return "Accounts"
	case PlaneLanes:
		return "Lanes"
	case PlanePosture:
		return "Context / cache posture"
	default:
		return plane
	}
}

// RenderText renders the pane as terminal text. It walks Sections() in order:
// one header line per plane (name · provenance · summary), then one line per
// fact. This is the plain-text fallback Slack shows in notifications too, and the
// form tests assert on.
func RenderText(p Pane) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s\n", statusGlyphText(p.OK), p.Headline)
	if p.Workspace != "" || p.Host != "" {
		fmt.Fprintf(&b, "  %s\n", identityLine(p))
	}
	b.WriteString("\n")
	for _, sec := range p.Sections() {
		fmt.Fprintf(&b, "%s  [%s]  %s\n", planeTitle(sec.Plane), sec.Prov, sec.Summary)
		for _, ln := range sec.Lines {
			fmt.Fprintf(&b, "  - %s\n", ln.Value)
		}
		b.WriteString("\n")
	}
	return b.String()
}

// RenderSlack renders the pane as a Slack Block Kit payload ([]any of block
// objects, matching the outbox Blocks contract). It walks the SAME Sections() in
// the SAME order: a header block, one section block per plane (its header line
// plus its fact lines as mrkdwn), and a trailing context block with schema +
// identity. Every field it emits comes from Sections(); RenderText emits the same
// set from the same walk.
func RenderSlack(p Pane) []any {
	blocks := []any{
		map[string]any{
			"type": "header",
			"text": map[string]any{"type": "plain_text", "text": "fak sidecar"},
		},
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("%s %s", statusGlyphSlack(p.OK), p.Headline)},
		},
	}
	for _, sec := range p.Sections() {
		var body strings.Builder
		fmt.Fprintf(&body, "*%s*  _[%s]_  %s", planeTitle(sec.Plane), sec.Prov, sec.Summary)
		for _, ln := range sec.Lines {
			fmt.Fprintf(&body, "\n• %s", ln.Value)
		}
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": body.String()},
		})
	}
	ctx := []string{"schema: " + p.Schema}
	if line := identityLine(p); line != "" {
		ctx = append(ctx, line)
	}
	blocks = append(blocks, map[string]any{
		"type":     "context",
		"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctx, "  ·  ")}},
	})
	return blocks
}

func identityLine(p Pane) string {
	parts := make([]string, 0, 3)
	if p.Host != "" {
		parts = append(parts, "host "+p.Host)
	}
	if p.Workspace != "" {
		parts = append(parts, "workspace "+p.Workspace)
	}
	if p.GeneratedAt != "" {
		parts = append(parts, p.GeneratedAt)
	}
	return strings.Join(parts, "  ·  ")
}

func statusGlyphText(ok bool) string {
	if ok {
		return "[OK]"
	}
	return "[gap]"
}

func statusGlyphSlack(ok bool) string {
	if ok {
		return ":white_check_mark:"
	}
	return ":warning:"
}
