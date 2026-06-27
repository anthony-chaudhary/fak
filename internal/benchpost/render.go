package benchpost

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Post is one bench-channel message, decoupled from which fold produced it. The three
// folds (RollupFromCatalog, RegressionFromCatalogVsBaseline, RequestFromPlan) each
// build one, so the renderer (Text/Blocks) has a single input shape — the same pattern
// as scoreboard.Update.
type Post struct {
	Emoji  string   // leading status glyph
	Title  string   // headline, e.g. "bench rollup — latest runs"
	Lead   string   // one-line summary / honesty banner under the title
	Lines  []string // the body: one line per run / plan row / regression
	Source string   // who posted: "ci" | "agent" | hostname (optional)
}

// Text renders the plain-text fallback — the line Slack shows in notifications and any
// client without Block Kit, and what tests and --dry-run assert on.
func (p Post) Text() string {
	var b strings.Builder
	emoji := p.Emoji
	if emoji == "" {
		emoji = ":bar_chart:"
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
	return b.String()
}

// Blocks renders the Block Kit payload. It carries the same facts as Text so a
// non-Block client loses nothing.
func (p Post) Blocks() []any {
	emoji := p.Emoji
	if emoji == "" {
		emoji = ":bar_chart:"
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
	return blocks
}

// provenanceLabel maps a run's recorded provenance to the conflation-honest WITNESSED
// vs OBSERVED label: a number fak AUTHORED (measured by fak's own kernel/harness) is
// WITNESSED; a number relayed from an external engine (llama.cpp, ollama, a vendor
// endpoint) is OBSERVED; anything else is UNLABELED. This keeps the channel honest
// about whose number it is (the conflation-scorecard discipline).
func provenanceLabel(r Run) string {
	p := strings.ToLower(strings.TrimSpace(r.Provenance))
	switch p {
	case "measured", "witnessed", "fak-native", "authored":
		return "WITNESSED"
	case "observed", "relayed", "external", "modeled":
		return "OBSERVED"
	}
	// Fall back to tags: an explicit external-engine tag means OBSERVED; a fak-native
	// tag means WITNESSED.
	for _, t := range r.Tags {
		t = strings.ToLower(t)
		switch {
		case strings.Contains(t, "fak-native"), strings.Contains(t, "fak-kernel"):
			return "WITNESSED"
		case strings.Contains(t, "llama"), strings.Contains(t, "ollama"), strings.Contains(t, "vllm"), strings.Contains(t, "external"):
			return "OBSERVED"
		}
	}
	return "UNLABELED"
}

// fmtTok renders a tok/s value, or "—" for a measurement gap (no real number).
func fmtTok(r Run) string {
	if v, ok := r.Val(); ok {
		s := fmt.Sprintf("%.2f", v)
		s = strings.TrimSuffix(s, ".00")
		return s + " tok/s"
	}
	return "—"
}

// RollupFromCatalog folds the latest n runs (most recent timestamp first) into a Post.
// Each line is `machine · model/precision · <tok/s> · <provenance label>`. The lead
// summarizes the corpus (total runs, machines covered).
func RollupFromCatalog(cat *Catalog, n int) Post {
	runs := append([]Run(nil), cat.Runs...)
	sort.SliceStable(runs, func(i, j int) bool { return runs[i].Timestamp > runs[j].Timestamp })

	machines := map[string]struct{}{}
	for _, r := range cat.Runs {
		if r.MachineID != "" {
			machines[r.MachineID] = struct{}{}
		}
	}
	if n <= 0 || n > len(runs) {
		n = len(runs)
	}

	p := Post{
		Emoji: ":bar_chart:",
		Title: "bench rollup — latest runs",
		Lead:  fmt.Sprintf("%d runs on record across %d machines · showing the latest %d", len(cat.Runs), len(machines), n),
	}
	for _, r := range runs[:n] {
		mp := r.Model
		if r.Precision != "" && !strings.EqualFold(r.Precision, "n/a") && !strings.EqualFold(r.Precision, "none") {
			mp = r.Model + "/" + r.Precision
		}
		when := r.Timestamp
		if i := strings.IndexByte(when, 'T'); i > 0 {
			when = when[:i]
		}
		p.Lines = append(p.Lines,
			fmt.Sprintf("`%s` · %s · %s · %s · %s", r.MachineID, mp, fmtTok(r), provenanceLabel(r), when))
	}
	return p
}

var nonSlug = regexp.MustCompile(`[^a-z0-9._-]+`)
var nonKey = regexp.MustCompile(`[^a-z0-9.]+`)
var dashRun = regexp.MustCompile(`-{2,}`)

// slug mirrors tools/bench_signal.py _slug (machine half of the regression key).
func slug(text string) string {
	s := dashRun.ReplaceAllString(nonSlug.ReplaceAllString(strings.ToLower(text), "-"), "-")
	s = strings.Trim(s, "-")
	if s == "" {
		return "x"
	}
	return s
}

// benchKey mirrors tools/bench_signal.py benchmark_key (model+precision half). The
// same model at a different quant tracks apart; the machine and timestamp are excluded.
func benchKey(r Run) string {
	model := r.Model
	if model == "" {
		model = "model"
	}
	raw := model
	prec := strings.ToLower(strings.TrimSpace(r.Precision))
	if prec != "" && prec != "n/a" && prec != "none" {
		raw = model + "-" + r.Precision
	}
	s := dashRun.ReplaceAllString(nonKey.ReplaceAllString(strings.ToLower(raw), "-"), "-")
	s = strings.Trim(s, "-")
	if s == "" {
		return "bench"
	}
	return s
}

// fullKey is the machine-scoped regression key `<slug(machine)>/<benchKey>`, matching
// tools/bench_signal.py full_key and the keys in tools/bench_baseline.json.
func fullKey(r Run) string { return slug(r.MachineID) + "/" + benchKey(r) }

// RegressionFromCatalogVsBaseline compares each baselined `<machine>/<key>` against the
// CURRENT catalog value (the latest run by timestamp with a real number) and surfaces
// drops past BOTH thresholds: a relative drop >= minDropPct AND an absolute drop >=
// minAbs tok/s — the same dual gate as tools/bench_signal.py, so a 5% wobble or a
// sub-tok/s micro-drift is not flagged. An empty result yields a clean ":white_check_
// mark: no regressions" post (the channel still sees the all-clear).
func RegressionFromCatalogVsBaseline(cat *Catalog, bl *Baseline, minDropPct, minAbs float64) Post {
	// CURRENT value per key: the latest (by timestamp) run that has a real number.
	type cur struct {
		val float64
		ts  string
	}
	current := map[string]cur{}
	for _, r := range cat.Runs {
		v, ok := r.Val()
		if !ok {
			continue
		}
		k := fullKey(r)
		if c, seen := current[k]; !seen || r.Timestamp > c.ts {
			current[k] = cur{val: v, ts: r.Timestamp}
		}
	}

	var keys []string
	if bl != nil {
		for k := range bl.Baselines {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	type drop struct {
		key            string
		base, now, pct float64
	}
	var drops []drop
	for _, k := range keys {
		base := bl.Baselines[k]
		c, ok := current[k]
		if !ok || base <= 0 {
			continue // a measurement gap, not a regression
		}
		absDrop := base - c.val
		if absDrop < minAbs {
			continue
		}
		pct := absDrop / base * 100
		if pct < minDropPct {
			continue
		}
		drops = append(drops, drop{key: k, base: base, now: c.val, pct: pct})
	}
	sort.SliceStable(drops, func(i, j int) bool { return drops[i].pct > drops[j].pct })

	if len(drops) == 0 {
		return Post{
			Emoji: ":white_check_mark:",
			Title: "bench regression check — clean",
			Lead:  fmt.Sprintf("no tok/s regressions: %d baselined benchmarks all within threshold (≥%.0f%% AND ≥%.1f tok/s to flag)", len(keys), minDropPct, minAbs),
		}
	}
	p := Post{
		Emoji: ":red_circle:",
		Title: "bench regression check — drops",
		Lead:  fmt.Sprintf("%d benchmark(s) dropped past threshold (≥%.0f%% AND ≥%.1f tok/s)", len(drops), minDropPct, minAbs),
	}
	for _, d := range drops {
		p.Lines = append(p.Lines,
			fmt.Sprintf("`%s` · %.2f → %.2f tok/s (−%.0f%%)", d.key, d.base, d.now, d.pct))
	}
	return p
}
