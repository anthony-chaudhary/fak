package grafanapost

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/slackmeta"
)

// Post is one #grafana-channel message, decoupled from which fold produced it. The
// three folds (SnapshotPost, DashboardPost, LinksRollup) each build one, so the
// renderer (Text/Blocks) has a single input shape — the same pattern as
// benchpost.Post and scoreboard.Update.
type Post struct {
	Emoji  string   // leading status glyph
	Title  string   // headline
	Lead   string   // one-line summary / context under the title
	Lines  []string // the body: one line per link / annotation
	Source string   // who posted: "ci" | "agent" | hostname (optional)
}

// defaultEmoji is the surface glyph used when a fold leaves Emoji unset.
const defaultEmoji = ":chart_with_upwards_trend:"

// Text renders the plain-text fallback — the line Slack shows in notifications and any
// client without Block Kit, and what tests and --dry-run assert on.
func (p Post) Text() string {
	var b strings.Builder
	emoji := p.Emoji
	if emoji == "" {
		emoji = defaultEmoji
	}
	fmt.Fprintf(&b, "%s *%s*", emoji, p.Title)
	if p.Lead != "" {
		fmt.Fprintf(&b, "\n%s", p.Lead)
	}
	for _, ln := range p.Lines {
		fmt.Fprintf(&b, "\n• %s", ln)
	}
	if p.Source != "" {
		fmt.Fprintf(&b, "\n_posted by %s_", p.Source)
	}
	return slackmeta.AppendText(b.String(), p.signalNoise())
}

// Blocks renders the Block Kit payload. It carries the same facts as Text so a
// non-Block client loses nothing.
func (p Post) Blocks() []any {
	emoji := p.Emoji
	if emoji == "" {
		emoji = defaultEmoji
	}
	blocks := []any{
		map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": fmt.Sprintf("*%s %s*", emoji, p.Title)},
		},
	}
	if p.Lead != "" {
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": p.Lead},
		})
	}
	if len(p.Lines) > 0 {
		body := "• " + strings.Join(p.Lines, "\n• ")
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{"type": "mrkdwn", "text": body},
		})
	}
	if p.Source != "" {
		blocks = append(blocks, map[string]any{
			"type":     "context",
			"elements": []any{map[string]any{"type": "mrkdwn", "text": "posted by " + p.Source}},
		})
	}
	return slackmeta.AppendContext(blocks, p.signalNoise())
}

func (p Post) signalNoise() slackmeta.Score {
	signal := 1 + slackmeta.NonEmpty(p.Lead) + len(p.Lines)
	noise := 1 + slackmeta.NonEmpty(p.Source)
	return slackmeta.New(signal, noise, "grafana snapshot/link facts vs source/context")
}

// SnapshotPost folds one exported Grafana snapshot into a Post: the headline, a lead
// summarizing the source dashboard / window / expiry, and the snapshot link. A
// snapshot with no URL still renders (so a --dry-run preview works) but the lead flags
// the missing link rather than implying one exists.
func SnapshotPost(s Snapshot) Post {
	title := strings.TrimSpace(s.Title)
	if title == "" {
		title = "Grafana snapshot"
	}
	p := Post{
		Emoji: ":camera_with_flash:",
		Title: "grafana snapshot — " + title,
	}

	var ctx []string
	if d := strings.TrimSpace(s.Dashboard); d != "" {
		ctx = append(ctx, d)
	}
	if r := strings.TrimSpace(s.TimeRange); r != "" {
		ctx = append(ctx, r)
	}
	if e := strings.TrimSpace(s.Expires); e != "" {
		ctx = append(ctx, "expires "+e)
	}
	if len(ctx) > 0 {
		p.Lead = strings.Join(ctx, " · ")
	}

	if n := strings.TrimSpace(s.Note); n != "" {
		p.Lines = append(p.Lines, n)
	}
	if u := strings.TrimSpace(s.URL); u != "" {
		p.Lines = append(p.Lines, "snapshot: "+u)
	} else {
		p.Lines = append(p.Lines, "_(no snapshot URL — export the share/snapshot link from Grafana and pass --url)_")
	}
	return p
}

// DashboardPost folds one registry link into a single-link card — the "here is the
// long-lived dashboard" post. base is the Grafana base URL used to resolve a
// uid-only link (the registry's own base unless overridden).
func DashboardPost(l Link, base string) Post {
	title := strings.TrimSpace(l.Title)
	if title == "" {
		title = strings.TrimSpace(l.UID)
	}
	if title == "" {
		title = "Grafana dashboard"
	}
	p := Post{
		Emoji: ":bar_chart:",
		Title: "grafana dashboard — " + title,
		Lead:  linkLead(l),
	}
	if d := strings.TrimSpace(l.Description); d != "" {
		p.Lines = append(p.Lines, d)
	}
	if u := l.ResolveURL(base); u != "" {
		p.Lines = append(p.Lines, u)
	} else {
		p.Lines = append(p.Lines, "_(no url — set a url or uid for this link in the registry)_")
	}
	return p
}

// rollupOrder is the category order the rollup groups links in: public demos first
// (the headline value), then debug links, then saved rollups.
var rollupOrder = []string{CategoryPublicDemo, CategoryDebug, CategoryRollup}

// LinksRollup folds the registry into a grouped rollup. category "" or "all" includes
// every link; otherwise only links of that category. Links are grouped by category
// (public-demo, debug, rollup, then anything else) with a header line per non-empty
// group. An empty result yields an honest "nothing registered" card rather than a
// blank post.
func LinksRollup(r *Registry, category string) Post {
	want := strings.ToLower(strings.TrimSpace(category))
	all := want == "" || want == "all"

	base := r.Base()
	groups := map[string][]Link{}
	var order []string
	seen := map[string]bool{}
	addGroup := func(cat string) {
		if !seen[cat] {
			seen[cat] = true
			order = append(order, cat)
		}
	}
	for _, c := range rollupOrder {
		addGroup(c)
	}

	total := 0
	for _, l := range r.Links {
		cat := strings.ToLower(strings.TrimSpace(l.Category))
		if cat == "" {
			cat = "uncategorized"
		}
		if !all && cat != want {
			continue
		}
		addGroup(cat)
		groups[cat] = append(groups[cat], l)
		total++
	}

	scope := "all categories"
	if !all {
		scope = want
	}
	if total == 0 {
		return Post{
			Emoji: defaultEmoji,
			Title: "grafana links — none registered",
			Lead:  fmt.Sprintf("no links for %s yet — add them to docs/grafana/links.json (schema fak-grafana-links/1)", scope),
		}
	}

	p := Post{
		Emoji: defaultEmoji,
		Title: "grafana links — dashboards & debug views",
		Lead:  fmt.Sprintf("%d link(s) · %s · base %s", total, scope, base),
	}
	for _, cat := range order {
		ls := groups[cat]
		if len(ls) == 0 {
			continue
		}
		sort.SliceStable(ls, func(i, j int) bool { return ls[i].Title < ls[j].Title })
		p.Lines = append(p.Lines, fmt.Sprintf("*%s* (%d)", cat, len(ls)))
		for _, l := range ls {
			p.Lines = append(p.Lines, "  "+rollupLine(l, base))
		}
	}
	return p
}

// linkLead renders the lifetime tag for a single-link card, e.g. "long-lived" or
// "stack-local". Empty when no lifetime is recorded.
func linkLead(l Link) string {
	lt := strings.TrimSpace(l.Lifetime)
	if lt == "" {
		return ""
	}
	return lt
}

// rollupLine renders one link inside a rollup group: title, lifetime, and the resolved
// URL (or a flag when none resolves).
func rollupLine(l Link, base string) string {
	title := strings.TrimSpace(l.Title)
	if title == "" {
		title = strings.TrimSpace(l.UID)
	}
	parts := []string{title}
	if lt := strings.TrimSpace(l.Lifetime); lt != "" {
		parts = append(parts, lt)
	}
	if u := l.ResolveURL(base); u != "" {
		parts = append(parts, u)
	} else {
		parts = append(parts, "(no url)")
	}
	line := strings.Join(parts, " · ")
	if d := strings.TrimSpace(l.Description); d != "" {
		line += " — " + d
	}
	return line
}
