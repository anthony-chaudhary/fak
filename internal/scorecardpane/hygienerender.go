package scorecardpane

// hygienerender.go — the renderers for the repo-hygiene fold: human snapshot, the
// committed markdown body, and the --compare delta. Ported from the Python
// render/render_markdown/render_compare so `--markdown` regenerates a byte-equal
// snapshot and `--compare` proves the debt moved.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ParseHygienePayload decodes a prior --json hygiene payload (the --compare base).
func ParseHygienePayload(b []byte) (HygienePayload, error) {
	var p HygienePayload
	err := json.Unmarshal(b, &p)
	return p, err
}

// RenderHygiene is the human scorecard, ported from the Python render().
func RenderHygiene(p HygienePayload) string {
	c := p.Corpus
	var b strings.Builder
	fmt.Fprintf(&b, "repo-hygiene-scorecard: %s (%s)\n", p.Verdict, p.Finding)
	fmt.Fprintf(&b, "  %s\n\n", p.Reason)
	fmt.Fprintf(&b, "score %s/100 (grade %s) · HYGIENE-DEBT %d (a11y-debt %d) · %d advisory\n",
		fmtFloat(c.Score), orQ(c.Grade), c.HygieneDebt, c.A11yDebt, c.SoftSignals)
	var grpParts []string
	for _, g := range hygieneGroups {
		grpParts = append(grpParts, fmt.Sprintf("%s:%d", g, c.DebtByGroup[g]))
	}
	fmt.Fprintf(&b, "debt by group: %s\n\n", strings.Join(grpParts, "  "))
	b.WriteString("per-KPI (worst first):\n")
	fmt.Fprintf(&b, "  %5s %4s  %-13s %-15s detail\n", "score", "debt", "group", "kpi")
	for _, row := range c.Breakdown {
		fmt.Fprintf(&b, "  %5d %4d  %-13s %-15s %s\n", row.Score, row.Debt, row.Group, row.KPI, row.Detail)
	}
	b.WriteString("\nhygiene-debt work-list:\n")
	anyDefect := false
	for _, k := range sortKPIsByDebt(p.KPIs) {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		fmt.Fprintf(&b, "  %s (%d):\n", k.KPI, len(k.Defects))
		for i, it := range k.Defects {
			if i >= 12 {
				break
			}
			fmt.Fprintf(&b, "      - %s\n", it)
		}
		if len(k.Defects) > 12 {
			fmt.Fprintf(&b, "      ... and %d more\n", len(k.Defects)-12)
		}
	}
	if !anyDefect {
		b.WriteString("  (none — zero hygiene-debt)\n")
	}
	if len(c.WorktreeClutter) > 0 {
		fmt.Fprintf(&b, "\nworktree clutter (advisory, not debt — %d untracked scratch file(s)):\n", len(c.WorktreeClutter))
		for i, f := range c.WorktreeClutter {
			if i >= 12 {
				break
			}
			fmt.Fprintf(&b, "      · %s\n", f)
		}
	}
	fmt.Fprintf(&b, "\nnext: %s\n", p.NextAction)
	return b.String()
}

// RenderHygieneMarkdown re-renders the committed snapshot body, ported from the
// Python render_markdown().
func RenderHygieneMarkdown(p HygienePayload, stamp string) string {
	c := p.Corpus
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString(`title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"` + "\n")
	b.WriteString(`description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across ` +
		`verbosity, organization, indexing, and accessibility, folded into a composite ` +
		`score and the headline hygiene-debt metric, re-derived from the git-tracked tree."` + "\n")
	b.WriteString("---\n\n")
	b.WriteString("# Repo-hygiene scorecard\n\n")
	if stamp != "" {
		fmt.Fprintf(&b, "<!-- repo-hygiene-scorecard: %s · process: tools/repo_hygiene_scorecard.py -->\n\n", stamp)
	}
	b.WriteString("This is the measuring stick for the repo-3x program — the structural counterpart " +
		"of the docs and code scorecards. Every number below is re-derived from the " +
		"git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline " +
		"metric is **hygiene-debt**: the count of concrete, mechanical structural defects " +
		"you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an " +
		"oversized doc, root clutter, a misplaced dated note, an orphaned doc no index " +
		"links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo " +
		"lean and findable as it grows.\n\n")
	b.WriteString("> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`\n\n")
	b.WriteString("## Headline\n\n")
	b.WriteString("| Metric | Value |\n")
	b.WriteString("|---|---|\n")
	fmt.Fprintf(&b, "| **Hygiene-debt (total HARD defects)** | **%d** |\n", c.HygieneDebt)
	fmt.Fprintf(&b, "| **a11y-debt (accessibility HARD defects)** | **%d** |\n", c.A11yDebt)
	fmt.Fprintf(&b, "| Composite score | %s/100 (grade %s) |\n", fmtFloat(c.Score), orQ(c.Grade))
	fmt.Fprintf(&b, "| Advisory (soft) signals | %d |\n", c.SoftSignals)
	fmt.Fprintf(&b, "| Debt by group | verbosity:%d · organization:%d · indexing:%d · accessibility:%d |\n",
		c.DebtByGroup["verbosity"], c.DebtByGroup["organization"], c.DebtByGroup["indexing"], c.DebtByGroup["accessibility"])
	b.WriteString("\n## Per-KPI\n\n")
	b.WriteString("Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. " +
		"The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. " +
		"`jargon` and `plain_language` are advisory (they score but emit no hard debt — " +
		"gaming a gloss is not clarity).\n\n")
	b.WriteString("| Group | KPI | Score | Debt | Detail |\n")
	b.WriteString("|---|---|---:|:--:|---|\n")
	for _, row := range c.Breakdown {
		fmt.Fprintf(&b, "| %s | `%s` | %d | %d | %s |\n", row.Group, row.KPI, row.Score, row.Debt, row.Detail)
	}
	b.WriteString("\n## Hygiene-debt work-list\n\n")
	anyDefect := false
	for _, k := range sortKPIsByDebt(p.KPIs) {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		fmt.Fprintf(&b, "### `%s` (%s) — %d defect(s), score %d\n", k.KPI, k.Group, len(k.Defects), k.Score)
		for _, it := range k.Defects {
			fmt.Fprintf(&b, "- %s\n", it)
		}
		b.WriteString("\n")
	}
	if !anyDefect {
		b.WriteString("No hygiene-debt: the tree is lean, well-placed, fully indexed, and reads plainly. 🎉\n\n")
	}
	return b.String()
}

// RenderHygieneCompare prints the hygiene-debt delta vs a prior baseline, ported
// from the Python render_compare() (the Nx gate).
func RenderHygieneCompare(baseline, current HygienePayload) string {
	b := baseline.Corpus
	cur := current.Corpus
	bd, cd := b.HygieneDebt, cur.HygieneDebt
	bo, co := b.Score, cur.Score
	ba, ca := b.A11yDebt, cur.A11yDebt
	ratio := "∞ (zero)"
	if cd != 0 {
		ratio = fmt.Sprintf("%.1f×", float64(bd)/float64(cd))
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "hygiene-debt: %d -> %d   (%s fewer defects)\n", bd, cd, ratio)
	fmt.Fprintf(&sb, "a11y-debt:    %d -> %d\n", ba, ca)
	fmt.Fprintf(&sb, "score:        %s/100 -> %s/100   (+%s)\n", fmtFloat(bo), fmtFloat(co), fmtFloat(round1(co-bo)))
	for _, gp := range hygieneGroups {
		fmt.Fprintf(&sb, "  %-13s %d -> %d\n", gp, b.DebtByGroup[gp], cur.DebtByGroup[gp])
	}
	target := bd / 3
	if target < 1 {
		target = 1
	}
	if cd <= target {
		fmt.Fprintf(&sb, "VERDICT: >=3x hygiene-debt reduction achieved (%d -> %d).", bd, cd)
	} else {
		fmt.Fprintf(&sb, "VERDICT: not yet 3x — need hygiene-debt <= %d (now %d).", target, cd)
	}
	return sb.String()
}

func sortKPIsByDebt(kpis []HygieneKPI) []HygieneKPI {
	out := append([]HygieneKPI(nil), kpis...)
	sort.SliceStable(out, func(i, j int) bool { return len(out[i].Defects) > len(out[j].Defects) })
	return out
}

func orQ(s string) string {
	if s == "" {
		return "?"
	}
	return s
}
