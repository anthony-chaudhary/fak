package seoaeoscore

import (
	"fmt"
	"sort"
	"strings"
)

// gradeOrder is the fixed column order for the grade distribution.
var gradeOrder = []string{"S", "A+", "A", "B", "C", "D", "F"}

func joinOrNone(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return strings.Join(s, ", ")
}

// pagesByWorst returns pages sorted by (score asc, n_defects desc).
func pagesByWorst(pages []Page) []Page {
	out := append([]Page(nil), pages...)
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score < out[b].Score
		}
		return out[a].NDefects > out[b].NDefects
	})
	return out
}

// Render is the default human view (mirrors python render()).
func Render(p Payload) string {
	c := p.Corpus
	site := p.Site
	var lines []string
	siteCreditArrow := ""
	if c.ExcellenceCredit.Site != 0 {
		siteCreditArrow = "→" + ftoa(c.SiteScoreWithCredit)
	}
	lines = append(lines,
		fmt.Sprintf("seo-aeo-scorecard: %s (%s)  [scope=%s]", p.Verdict, p.Finding, p.Scope),
		"  "+p.Reason,
		"",
		fmt.Sprintf("corpus: %d pages · overall %s (grade %s) (pages %s · site %s%s) · credit +%d (page %d · site %d) · SEO-DEBT %d",
			c.NPages, ftoa(c.OverallScore), c.Grade, ftoa(c.PageMeanScore), ftoa(c.SiteScore), siteCreditArrow,
			c.ExcellenceCredit.Total, c.ExcellenceCredit.Page, c.ExcellenceCredit.Site, c.SEODebt),
		fmt.Sprintf("meta coverage: %s%%  ·  site checks: %s  ·  JSON-LD present: %s",
			ftoa(c.MetaCoveragePct), c.SiteChecksOK, joinOrNone(c.PresentJSONLD)),
		fmt.Sprintf("discovery orphans (published, not front-door-reachable): %d", c.DiscoveryOrphans),
		fmt.Sprintf("success KPIs: crawl-404 %d  ·  meta-duplicates %d  ·  dead citations %d  ·  llms-full unresolved %d (advisory)",
			c.Crawl404, c.MetaDuplicates, c.CitationDead, c.LLMSFullUnresolved),
	)
	var gparts []string
	for _, g := range gradeOrder {
		gparts = append(gparts, fmt.Sprintf("%s:%d", g, c.GradeDistribution[g]))
	}
	lines = append(lines, "grades: "+strings.Join(gparts, " "))
	lines = append(lines, "next: "+p.NextAction, "", "per-page (worst first):")
	lines = append(lines, fmt.Sprintf("  %5s %2s %3s  %3s %3s %3s %3s %3s %3s %3s  path",
		"score", "gr", "def", "ttl", "dsc", "hdg", "lnk", "crl", "ans", "alt"))
	for _, d := range pagesByWorst(p.Pages) {
		k := d.KPIs
		lines = append(lines, fmt.Sprintf("  %5s %2s %3d  %3d %3d %3d %3d %3d %3d %3d  %s",
			ftoa(d.Score), d.Grade, d.NDefects,
			k["title"], k["description"], k["headings"], k["links"], k["links_crawlable"], k["answerability"], k["alt_text"], d.Path))
	}
	lines = append(lines, "", "site-level checks:")
	for _, ch := range site.Checks {
		mark := "ok "
		if !ch.OK {
			if ch.Hard {
				mark = "!! "
			} else {
				mark = "~  "
			}
		}
		lines = append(lines, fmt.Sprintf("  [%s] %-20s %s", mark, ch.Name, ch.Detail))
	}
	lines = append(lines, "", "seo-debt detail (top defectful pages):")
	worst := append([]Page(nil), p.Pages...)
	sort.SliceStable(worst, func(a, b int) bool { return worst[a].NDefects > worst[b].NDefects })
	worst = firstNPage(worst, 8)
	for _, d := range worst {
		if len(d.Defects) == 0 {
			continue
		}
		lines = append(lines, fmt.Sprintf("  %s (%d):", d.Path, d.NDefects))
		for _, it := range firstNStr(d.Defects, 6) {
			lines = append(lines, "      - "+it)
		}
	}
	if len(site.Defects) > 0 {
		lines = append(lines, "  site:")
		for _, it := range site.Defects {
			lines = append(lines, "      - "+it)
		}
	}
	return strings.Join(lines, "\n")
}

// Markdown renders the PRIVATE SEO-AEO-SCORECARD.md body (mirrors render_markdown()).
func Markdown(p Payload, stamp string) string {
	c := p.Corpus
	site := p.Site
	scope := p.Scope
	if scope == "" {
		scope = "core"
	}
	gd := c.GradeDistribution
	var out []string
	out = append(out, "# SEO / AEO scorecard (PRIVATE)", "")
	out = append(out, "> **Private — do not publish.** Discoverability scores are go-to-market "+
		"positioning (the same strategic class as the ICP memo). The measuring "+
		"TOOL is public (`fak score seo`); these SCORES live only in "+
		"the private repo. The public `.gitignore` blocks this file from a public commit.", "")
	if stamp != "" {
		out = append(out, fmt.Sprintf("<!-- seo-aeo-scorecard: %s · scope=%s · process: fak score seo -->", stamp, scope), "")
	}
	out = append(out, fmt.Sprintf("> Scope: **%s**. Regenerate: "+
		"`fak score seo --scope %s --markdown --stamp DATE`, writing the output to "+
		"SEO-AEO-SCORECARD.md in the private repo. There is no `--transfer` flag: the "+
		"private-archive copy is a manual step.", scope, scope), "")
	out = append(out, "Headline metric is **seo-debt**: the count of concrete, re-derivable "+
		"discoverability defects (a page with no meta description, a missing "+
		"JSON-LD type, a stale `llms-full.txt`, a dead link). Drive it toward zero.", "")
	out = append(out, "## Corpus", "")
	out = append(out, "| Metric | Value |", "|---|---|")
	out = append(out, fmt.Sprintf("| Published pages scored | %d |", c.NPages))
	out = append(out, fmt.Sprintf("| **SEO-debt (total defects)** | **%d** (%d in-page + %d site) |",
		c.SEODebt, c.SEODebtInPages, c.SEODebtInSite))
	out = append(out, fmt.Sprintf("| Overall score (unbounded) | %s (grade %s) |", ftoa(c.OverallScore), c.Grade))
	ec := c.ExcellenceCredit
	siteArrow := ""
	if ec.Site != 0 {
		siteArrow = " → " + ftoa(c.SiteScoreWithCredit)
	}
	out = append(out, fmt.Sprintf("| — page mean / site | %s / %s%s |", ftoa(c.PageMeanScore), ftoa(c.SiteScore), siteArrow))
	out = append(out, fmt.Sprintf("| Excellence credit (page + site) | +%d (%d page + %d site) |", ec.Total, ec.Page, ec.Site))
	out = append(out, fmt.Sprintf("| Meta coverage (title+desc) | %s%% |", ftoa(c.MetaCoveragePct)))
	out = append(out, fmt.Sprintf("| Site checks passing | %s |", c.SiteChecksOK))
	out = append(out, fmt.Sprintf("| JSON-LD types present | %s |", joinOrNone(c.PresentJSONLD)))
	out = append(out, fmt.Sprintf("| Discovery orphans (not front-door-reachable) | %d |", c.DiscoveryOrphans))
	out = append(out, fmt.Sprintf("| Grade distribution | S:%d A+:%d A:%d B:%d C:%d D:%d F:%d |",
		gd["S"], gd["A+"], gd["A"], gd["B"], gd["C"], gd["D"], gd["F"]))
	out = append(out, "")
	out = append(out, "## Per-page scores", "")
	out = append(out, "| Score | Grade | Debt | title | desc | head | link | crawl | ans | alt | Page |")
	out = append(out, "|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|")
	for _, d := range pagesByWorst(p.Pages) {
		k := d.KPIs
		out = append(out, fmt.Sprintf("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | `%s` |",
			ftoa(d.Score), d.Grade, d.NDefects,
			k["title"], k["description"], k["headings"], k["links"], k["links_crawlable"], k["answerability"], k["alt_text"], d.Path))
	}
	out = append(out, "")
	out = append(out, "## Site-level checks", "")
	out = append(out, "| State | Check | Detail |", "|:--:|---|---|")
	for _, ch := range site.Checks {
		mark := "✅"
		if !ch.OK {
			if ch.Hard {
				mark = "❌"
			} else {
				mark = "⚠️"
			}
		}
		out = append(out, fmt.Sprintf("| %s | `%s` | %s |", mark, ch.Name, ch.Detail))
	}
	out = append(out, "")
	out = append(out, "## SEO-debt work-list", "")
	anyDefect := false
	worst := append([]Page(nil), p.Pages...)
	sort.SliceStable(worst, func(a, b int) bool { return worst[a].NDefects > worst[b].NDefects })
	for _, d := range worst {
		if len(d.Defects) == 0 {
			continue
		}
		anyDefect = true
		out = append(out, fmt.Sprintf("### `%s` — %d defect(s), score %s (%s)", d.Path, d.NDefects, ftoa(d.Score), d.Grade))
		for _, it := range d.Defects {
			out = append(out, "- "+it)
		}
		out = append(out, "")
	}
	if len(site.Defects) > 0 {
		anyDefect = true
		out = append(out, "### site")
		for _, it := range site.Defects {
			out = append(out, "- "+it)
		}
		out = append(out, "")
	}
	if !anyDefect {
		out = append(out, "No seo-debt: every published page + site check is clean. \U0001f389", "")
	}
	return strings.Join(out, "\n")
}

// ---------------------------------------------------------------------------
// Compare (prove the debt moved) — mirrors render_compare(baseline, current).
// ---------------------------------------------------------------------------

func baselineCorpus(baseline map[string]any) map[string]any {
	if c, ok := baseline["corpus"].(map[string]any); ok {
		return c
	}
	return map[string]any{}
}

func mapInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	}
	return 0
}

func mapFloat(m map[string]any, key string) float64 {
	switch v := m[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return 0
}

func mapStr(m map[string]any, key string) string {
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func mapCreditTotal(m map[string]any) int {
	if ec, ok := m["excellence_credit"].(map[string]any); ok {
		return mapInt(ec, "total")
	}
	return 0
}

func mapPresentJSONLD(m map[string]any) []string {
	raw, ok := m["present_jsonld"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, x := range raw {
		if s, ok := x.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// Compare prints the seo-debt delta of the current payload vs a prior baseline JSON.
func Compare(current Payload, baseline map[string]any) string {
	b := baselineCorpus(baseline)
	cur := current.Corpus
	bd := mapInt(b, "seo_debt")
	cd := cur.SEODebt
	bo := mapFloat(b, "overall_score")
	co := cur.OverallScore
	bc := mapCreditTotal(b)
	cc := cur.ExcellenceCredit.Total
	ratioS := "∞ (zero)"
	if cd != 0 {
		ratioS = fmt.Sprintf("%.1f×", float64(bd)/float64(cd))
	}
	lines := []string{
		fmt.Sprintf("seo-debt: %d -> %d   (%s fewer defects)", bd, cd, ratioS),
		fmt.Sprintf("overall:  %s -> %s   (+%s, unbounded above 100)", ftoa(bo), ftoa(co), ftoa(round1(co-bo))),
		fmt.Sprintf("credit:   +%d -> +%d   (excellence, earned only when spotless/clean)", bc, cc),
		fmt.Sprintf("meta cov: %s%% -> %s%%", ftoa(mapFloat(b, "meta_coverage_pct")), ftoa(cur.MetaCoveragePct)),
		fmt.Sprintf("site:     %s -> %s", mapStrOr(b, "site_checks_ok", "?"), cur.SiteChecksOK),
		fmt.Sprintf("JSON-LD:  %s -> %s", joinOrNone(mapPresentJSONLD(b)), joinOrNone(cur.PresentJSONLD)),
	}
	if cd <= maxInt(1, bd)/10 {
		lines = append(lines, fmt.Sprintf("VERDICT: >=10x debt reduction achieved (%d -> %d).", bd, cd))
	} else {
		need := maxInt(1, bd/10)
		lines = append(lines, fmt.Sprintf("VERDICT: not yet 10x — need seo-debt <= %d (now %d).", need, cd))
	}
	return strings.Join(lines, "\n")
}

func mapStrOr(m map[string]any, key, def string) string {
	if s := mapStr(m, key); s != "" {
		return s
	}
	return def
}
