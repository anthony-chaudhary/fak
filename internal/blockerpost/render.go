package blockerpost

import (
	"fmt"
	"strings"
)

// glyph maps a severity to its leading status emoji so the channel reads a blocker's
// urgency at a glance: operator is a red rotating light (it pages), status is a muted
// hourglass (ongoing), clear is a green check (all-clear heartbeat).
func (b Blocker) glyph() string {
	switch b.Severity {
	case SeverityOperator:
		return ":rotating_light:"
	case SeverityClear:
		return ":white_check_mark:"
	default: // SeverityStatus and the zero value
		return ":hourglass_flowing_sand:"
	}
}

// mention returns the broadcast/owner mention for an OPERATOR blocker — the token that
// makes Slack actually page. It is Owner when set (e.g. "<@U123>" to page one person),
// else "<!here>" (page the channel's active members). Status/clear blockers return ""
// so they record silently. This is the single point that decides "surfaced vs
// background", so the contract lives in one tested place.
func (b Blocker) mention() string {
	if b.Severity != SeverityOperator {
		return ""
	}
	if m := strings.TrimSpace(b.Owner); m != "" {
		return m
	}
	return "<!here>"
}

// headline is the bold title line, prefixed with an OPERATOR-NEEDED banner for an
// operator blocker so the surfaced state is unmistakable even before the mention.
func (b Blocker) headline() string {
	title := strings.TrimSpace(b.Title)
	if title == "" {
		title = "blocker"
	}
	switch b.Severity {
	case SeverityOperator:
		return "BLOCKER — needs operator · " + title
	case SeverityClear:
		return title + " — all clear"
	default:
		return title + " — ongoing"
	}
}

// Text renders the plain-text fallback — the line Slack shows in notifications (and the
// field that triggers a broadcast page on an operator blocker) and any client without
// Block Kit. It is also what tests and --dry-run assert on. The mention rides on the
// FIRST line so the notification preview carries it.
func (b Blocker) Text() string {
	var sb strings.Builder
	head := fmt.Sprintf("%s *%s*", b.glyph(), b.headline())
	if m := b.mention(); m != "" {
		head += " " + m
	}
	sb.WriteString(head)
	if d := strings.TrimSpace(b.Detail); d != "" {
		fmt.Fprintf(&sb, "\n%s", d)
	}
	if a := b.actionLine(); a != "" {
		fmt.Fprintf(&sb, "\n• %s", a)
	}
	for _, ln := range b.Lines {
		fmt.Fprintf(&sb, "\n• %s", ln)
	}
	if ref := strings.TrimSpace(b.Ref); ref != "" {
		fmt.Fprintf(&sb, "\nref: %s", ref)
	}
	if src := strings.TrimSpace(b.Source); src != "" {
		fmt.Fprintf(&sb, "\n_posted by %s_", src)
	}
	return sb.String()
}

// actionLine renders the "do this next" affordance as one text line: the label, then
// the URL when present. It is the fallback for the Block Kit link-button; an operator
// blocker with an Action gives the human a concrete next step.
func (b Blocker) actionLine() string {
	label := strings.TrimSpace(b.Action)
	url := strings.TrimSpace(b.ActionURL)
	switch {
	case label == "" && url == "":
		return ""
	case url == "":
		return "do this next: " + label
	case label == "":
		return "do this next: " + url
	default:
		return "do this next: " + label + " — " + url
	}
}

// Blocks renders the Block Kit payload for a richer card. It carries the same facts as
// Text so a non-Block client loses nothing. The mention is emitted as its own lead
// section (mrkdwn) — a broadcast in a section block pages exactly as it does in the
// notification text, so the operator state is surfaced in both render paths.
func (b Blocker) Blocks() []any {
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s*", b.glyph(), b.headline())},
		},
	}
	if m := b.mention(); m != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": m + " — operator action needed"},
		})
	}
	if d := strings.TrimSpace(b.Detail); d != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": d},
		})
	}
	if len(b.Lines) > 0 {
		body := "• " + strings.Join(b.Lines, "\n• ")
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": body},
		})
	}
	// A link-button needs NO interactivity endpoint, so the card stays stateless and
	// within the "just chat.postMessage" constraint. Emit it only when there is a URL;
	// a URL-less action already rode the Text() fallback.
	if label, url := strings.TrimSpace(b.Action), strings.TrimSpace(b.ActionURL); label != "" && url != "" {
		blocks = append(blocks, map[string]any{
			"type": "actions",
			"elements": []any{map[string]any{
				"type": "button",
				"text": map[string]any{"type": "plain_text", "text": label, "emoji": true},
				"url":  url,
			}},
		})
	}
	var ctxParts []string
	if ref := strings.TrimSpace(b.Ref); ref != "" {
		ctxParts = append(ctxParts, "ref: "+ref)
	}
	if src := strings.TrimSpace(b.Source); src != "" {
		ctxParts = append(ctxParts, "posted by "+src)
	}
	if len(ctxParts) > 0 {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
		})
	}
	return blocks
}
