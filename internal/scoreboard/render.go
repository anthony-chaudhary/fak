package scoreboard

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/slackmeta"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Update is one scoreboard post, decoupled from where it came from. A scorecard
// Payload folds into this (FromPayload), and an ad-hoc `--kpi/--value` post builds
// one directly, so the renderer has a single input shape.
type Update struct {
	Title   string   // headline, e.g. "conflation scorecard" or "code-debt"
	Grade   string   // A-F (optional)
	Score   string   // composite, pre-formatted (optional)
	DebtKey string   // name of the debt integer, e.g. "conflation_debt"
	Debt    string   // the debt value, pre-formatted (optional)
	Verdict string   // OK | ACTION (optional)
	Detail  string   // one-line finding/reason
	Notes   string   // free-form multi-paragraph body (e.g. a product-direction note); rendered as-is
	Lines   []string // optional extra lines (e.g. per-KPI scores)
	Source  string   // who posted: "ci" | "agent" | hostname (optional)
	Actions []Action // "do this next" affordances (e.g. an alert -> owning skill)
}

// Action is a "do this next" affordance attached to a card — e.g. an alert that
// points the heaviest drift signal at the skill that retires it. A URL renders a
// Slack link-button (a button with a `url` needs NO interactivity endpoint, so the
// card stays stateless and within the "just chat.postMessage" constraint); when URL
// is empty the action degrades to a plain text line in the fallback.
type Action struct {
	Label string // button text, e.g. "Run /steerability-score"
	URL   string // optional link target (a docs/skill/repo URL)
}

// FromPayload folds a scorecard control-pane Payload into an Update. debtKey selects
// which corpus integer is the headline (e.g. conflationscore.DebtKey); if it is empty
// the renderer omits the debt line. A short per-KPI breakdown is attached.
func FromPayload(title string, p scorecard.Payload, debtKey string) Update {
	u := Update{
		Title:   title,
		DebtKey: debtKey,
		Verdict: p.Verdict,
		Detail:  firstNonEmpty(p.Finding, p.Reason),
	}
	if p.Corpus != nil {
		u.Grade = asString(p.Corpus["grade"])
		u.Score = asString(p.Corpus["score"])
		if debtKey != "" {
			u.Debt = asString(p.Corpus[debtKey])
		}
	}
	for _, k := range p.KPIs {
		u.Lines = append(u.Lines, fmt.Sprintf("%s: %s", k.Key, trimFloat(k.Score)))
	}
	sort.Strings(u.Lines)
	return u
}

// gradeEmoji maps an A-F grade to a status glyph so the channel scans at a glance.
// An ACTION verdict ALWAYS shows the action glyph, even when the grade is an A: a
// scorecard whose own verdict is ACTION (e.g. the industry map grades A on honesty
// but reports weak_standing) must not flash green in the channel — the verdict, not
// the headline letter, decides whether the surface needs attention. Without this an
// A-graded ACTION post read as "all good", the exact map-vs-standing conflation the
// industry scorecard's standing_score was added to surface.
func gradeEmoji(grade, verdict string) string {
	if verdict == "ACTION" {
		return ":red_circle:"
	}
	switch {
	case strings.HasPrefix(grade, "A"):
		return ":large_green_circle:"
	case strings.HasPrefix(grade, "B"):
		return ":large_yellow_circle:"
	case verdict == "OK":
		return ":white_check_mark:"
	case grade == "" && verdict == "":
		return ":bar_chart:"
	default:
		return ":red_circle:"
	}
}

// Text renders the plain-text fallback — the line Slack shows in notifications and
// any client without Block Kit. It is also what tests and --dry-run assert on.
func (u Update) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s *%s*", gradeEmoji(u.Grade, u.Verdict), u.Title)
	var meta []string
	if u.Grade != "" {
		meta = append(meta, "grade "+u.Grade)
	}
	if u.Score != "" {
		meta = append(meta, "score "+u.Score)
	}
	if u.Debt != "" {
		key := u.DebtKey
		if key == "" {
			key = "debt"
		}
		meta = append(meta, fmt.Sprintf("%s %s", key, u.Debt))
	}
	if u.Verdict != "" {
		meta = append(meta, u.Verdict)
	}
	if len(meta) > 0 {
		fmt.Fprintf(&b, " — %s", strings.Join(meta, " · "))
	}
	if u.Detail != "" {
		fmt.Fprintf(&b, "\n%s", u.Detail)
	}
	if u.Notes != "" {
		fmt.Fprintf(&b, "\n%s", u.Notes)
	}
	if len(u.Lines) > 0 {
		fmt.Fprintf(&b, "\n`%s`", strings.Join(u.Lines, "`  `"))
	}
	for _, a := range u.Actions {
		if a.URL != "" {
			fmt.Fprintf(&b, "\n• %s — %s", a.Label, a.URL)
		} else {
			fmt.Fprintf(&b, "\n• %s", a.Label)
		}
	}
	if u.Source != "" {
		fmt.Fprintf(&b, "\n_posted by %s_", u.Source)
	}
	return slackmeta.AppendText(b.String(), u.signalNoise())
}

// Blocks renders the Block Kit payload for a richer card. It carries the same
// information as Text so a non-Block client loses nothing.
func (u Update) Blocks() []any {
	header := fmt.Sprintf("%s %s", gradeEmoji(u.Grade, u.Verdict), u.Title)
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": "*" + header + "*"},
		},
	}
	var fields []any
	add := func(label, val string) {
		if val != "" {
			fields = append(fields, map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s*\n%s", label, val)})
		}
	}
	add("Grade", u.Grade)
	add("Score", u.Score)
	if u.Debt != "" {
		key := u.DebtKey
		if key == "" {
			key = "Debt"
		}
		add(key, u.Debt)
	}
	add("Verdict", u.Verdict)
	if len(fields) > 0 {
		blocks = append(blocks, map[string]any{"type": "section", "fields": fields})
	}
	if u.Detail != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": u.Detail},
		})
	}
	if u.Notes != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": u.Notes},
		})
	}
	if elems := u.actionElements(); len(elems) > 0 {
		blocks = append(blocks, map[string]any{"type": "actions", "elements": elems})
	}
	ctxParts := append([]string{}, u.Lines...)
	if u.Source != "" {
		ctxParts = append(ctxParts, "posted by "+u.Source)
	}
	if len(ctxParts) > 0 {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": strings.Join(ctxParts, "  ·  ")}},
		})
	}
	return slackmeta.AppendContext(blocks, u.signalNoise())
}

func (u Update) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(u.Grade, u.Score, u.Debt, u.Verdict, u.Detail, u.Notes) + len(u.Lines) + len(u.Actions)
	noise := 1 + slackmeta.NonEmpty(u.Source)
	return slackmeta.New(signal, noise, "headline fields, evidence rows, and actions vs source/context")
}

// actionElements renders u.Actions as Block Kit button elements. Slack caps an
// actions block at 5 elements, so extras are dropped (the Text() fallback still
// carries all of them). A button is only emitted when it has a URL — a link-button
// needs no interactivity endpoint; a URL-less action lives only in the text fallback.
func (u Update) actionElements() []any {
	var elems []any
	for _, a := range u.Actions {
		if a.URL == "" || a.Label == "" {
			continue
		}
		elems = append(elems, map[string]any{
			"type": "button",
			"text": map[string]any{"type": "plain_text", "text": a.Label, "emoji": true},
			"url":  a.URL,
		})
		if len(elems) == 5 {
			break
		}
	}
	return elems
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func asString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case float64:
		return trimFloat(t)
	case int:
		return fmt.Sprintf("%d", t)
	default:
		return fmt.Sprintf("%v", t)
	}
}

// trimFloat renders a float without a trailing .0 (12.0 -> "12", 12.5 -> "12.5"),
// matching how the scorecard corpus prints integers and one-decimal scores.
func trimFloat(f float64) string {
	s := fmt.Sprintf("%.1f", f)
	s = strings.TrimSuffix(s, ".0")
	return s
}
