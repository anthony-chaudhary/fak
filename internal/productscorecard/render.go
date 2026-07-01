package productscorecard

import (
	"fmt"
	"sort"
	"strings"
)

var marks = map[string]string{
	"durable-product": "*", "usable-today": "o", "real-not-easy": "~",
	"honest-stub": ".", "concept-only": "-",
}

func Render(p Payload) string {
	c := p.Corpus
	cov := mapValue(c["coverage"])
	pos := mapValue(c["standing"])
	var lines []string
	lines = append(lines,
		fmt.Sprintf("product-scorecard: %s (%s)", p.Verdict, p.Finding),
		fmt.Sprintf("  %s", p.Reason),
		"",
		fmt.Sprintf("score %s/100 (grade %s) - PRODUCT-DEBT %d (honesty %d + coverage %d + managed-context %d) - %d advisory",
			formatNumber(c["score"]), stringAny(c["grade"]), intValue(c["product_debt"]), intValue(c["honesty_defects"]), intValue(c["coverage_debt"]), intValue(c["managed_context_debt"]), intValue(c["soft_signals"])),
		fmt.Sprintf("coverage: %s%% (%d/%d concept sections positioned) - %d concepts scored - %d durable products",
			formatNumber(cov["coverage_pct"]), intValue(cov["covered"]), intValue(cov["catalog_total"]), intValue(c["rows"]), intValue(c["durable_products"])),
		fmt.Sprintf("standing: %d durable-product - %d usable-today - %d real-not-easy - %d honest-stub - %d concept-only",
			intValue(pos["durable-product"]), intValue(pos["usable-today"]), intValue(pos["real-not-easy"]), intValue(pos["honest-stub"]), intValue(pos["concept-only"])),
		"debt by group: "+debtGroupLine(mapValue(c["debt_by_group"])),
		"",
		"product concepts (best verdict first):",
		fmt.Sprintf("  %-16s %-10s %-11s %-7s concept", "verdict", "maturity", "cat", "today?"),
	)
	if mc := mapValue(c["managed_context"]); len(mc) > 0 {
		lines = append(lines, fmt.Sprintf("managed-context SLOs: score %s/100 - debt %d/%d - passed %d",
			formatNumber(mc["score"]), intValue(mc["debt"]), intValue(mc["total"]), intValue(mc["passed"])), "")
	}
	for _, row := range sortedLeaderboard(c["leaderboard"]) {
		verdict := stringAny(row["verdict"])
		today := "-"
		if boolAny(row["offline"]) {
			today = "laptop"
		} else if NonEmpty(row["first_command"]) {
			today = "cmd"
		}
		flag := ""
		if verdict != stringAny(row["expected_verdict"]) {
			flag = "  expected " + stringAny(row["expected_verdict"])
		}
		lines = append(lines, fmt.Sprintf("  %s %-14s %-10s %-11s %-7s %s%s",
			marks[verdict], verdict, stringAny(row["maturity"]), stringAny(row["category"]), today, stringAny(row["concept"]), flag))
	}
	lines = append(lines, "", "per-KPI (worst first):", fmt.Sprintf("  %5s %4s  %-13s %-20s detail", "score", "debt", "group", "kpi"))
	for _, b := range mapSlice(c["breakdown"]) {
		lines = append(lines, fmt.Sprintf("  %5d %4d  %-13s %-20s %s",
			intValue(b["score"]), intValue(b["debt"]), stringAny(b["group"]), stringAny(b["kpi"]), stringAny(b["detail"])))
	}
	lines = append(lines, "")
	if unc := sectionSlice(cov["uncovered"]); len(unc) > 0 {
		lines = append(lines, fmt.Sprintf("coverage gaps (%d concept sections with no row):", len(unc)))
		limit := len(unc)
		if limit > 15 {
			limit = 15
		}
		for _, sec := range unc[:limit] {
			lines = append(lines, "      - "+sec.Section)
		}
		lines = append(lines, "")
	}
	lines = append(lines, "product-debt work-list:")
	anyDefect := false
	kpis := append([]KPI(nil), p.KPIs...)
	sort.SliceStable(kpis, func(i, j int) bool { return len(kpis[i].Defects) > len(kpis[j].Defects) })
	for _, k := range kpis {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		lines = append(lines, fmt.Sprintf("  %s (%d):", k.KPI, len(k.Defects)))
		limit := len(k.Defects)
		if limit > 12 {
			limit = 12
		}
		for _, d := range k.Defects[:limit] {
			lines = append(lines, "      - "+d)
		}
		if len(k.Defects) > 12 {
			lines = append(lines, fmt.Sprintf("      ... and %d more", len(k.Defects)-12))
		}
	}
	if !anyDefect {
		lines = append(lines, "  (none - zero honesty-debt; every product claim is honest)")
	}
	if mc := mapValue(c["managed_context"]); len(mc) > 0 && intValue(mc["debt"]) > 0 {
		lines = append(lines, "", "managed-context SLO work-list:")
		for _, row := range mapSlice(mc["rows"]) {
			if intValue(row["debt"]) == 0 {
				continue
			}
			lines = append(lines, fmt.Sprintf("  %s [%s]: %s", stringAny(row["id"]), stringAny(row["status"]), stringAny(row["detail"])))
			if next := stringAny(row["next_action"]); next != "" {
				lines = append(lines, "      - "+next)
			}
		}
	}
	lines = append(lines, "", "next: "+p.NextAction)
	return strings.Join(lines, "\n")
}

func RenderCritical(p Payload) string {
	lines := []string{"product critical backlog (progress worst-first):", ""}
	for _, it := range mapSlice(p.Corpus["critical"]) {
		if intValue(it["debt"]) == 0 && intValue(it["distance"]) <= verdictRank["real-not-easy"] {
			continue
		}
		lines = append(lines, fmt.Sprintf("  [%d debt - %s] %s (%s)", intValue(it["debt"]), stringAny(it["verdict"]), stringAny(it["id"]), stringAny(it["category"])))
		for i, g := range stringSlice(it["gaps"]) {
			if i >= 4 {
				break
			}
			lines = append(lines, "      - "+g)
		}
	}
	lines = append(lines, "", "(rows with 0 debt and a durable/usable/real verdict are omitted; stubs/concepts with gaps remain visible.)")
	return strings.Join(lines, "\n")
}

func RenderGaps(p Payload) string {
	cov := mapValue(p.Corpus["coverage"])
	unc := sectionSlice(cov["uncovered"])
	lines := []string{"product coverage backlog (position every concept section):", "", fmt.Sprintf("UNPOSITIONED - %d CLAIMS.md concept section(s) with no product row:", len(unc))}
	if len(unc) == 0 {
		lines = append(lines, "  (none - every concept section is positioned)")
	}
	for _, sec := range unc {
		lines = append(lines, "  - "+sec.Section)
	}
	return strings.Join(lines, "\n")
}

func RenderCompare(baseline map[string]any, current Payload) string {
	b, cur := mapValue(baseline["corpus"]), current.Corpus
	bd, cd := intValue(b["product_debt"]), intValue(cur["product_debt"])
	ratio := "inf (zero)"
	if cd != 0 {
		ratio = fmt.Sprintf("%.1fx", float64(bd)/float64(cd))
	}
	lines := []string{
		fmt.Sprintf("product-debt: %d -> %d   (%s fewer defects+gaps)", bd, cd, ratio),
		fmt.Sprintf("  honesty:    %d -> %d", intValue(b["honesty_defects"]), intValue(cur["honesty_defects"])),
		fmt.Sprintf("  coverage:   %d -> %d", intValue(b["coverage_debt"]), intValue(cur["coverage_debt"])),
		fmt.Sprintf("score:        %s/100 -> %s/100   (+%s)", formatNumber(b["score"]), formatNumber(cur["score"]), formatNumber(floatValue(cur["score"])-floatValue(b["score"]))),
		fmt.Sprintf("durable:      %d -> %d durable products", intValue(b["durable_products"]), intValue(cur["durable_products"])),
	}
	bg, cg := mapValue(b["debt_by_group"]), mapValue(cur["debt_by_group"])
	for _, gp := range groups {
		lines = append(lines, fmt.Sprintf("  %-13s %d -> %d", gp, intValue(bg[gp]), intValue(cg[gp])))
	}
	target3 := (bd + 2) / 3
	target2 := bd / 2
	switch {
	case cd <= target3:
		lines = append(lines, fmt.Sprintf("VERDICT: >=3x reduction achieved (product-debt %d->%d, target <=%d).", bd, cd, target3))
	case cd <= target2:
		lines = append(lines, fmt.Sprintf("VERDICT: >=2x (not yet 3x) - product-debt %d->%d; 3x needs <=%d.", bd, cd, target3))
	default:
		lines = append(lines, fmt.Sprintf("VERDICT: not yet 2x - need product-debt <=%d (now %d); 3x target <=%d.", target2, cd, target3))
	}
	return strings.Join(lines, "\n")
}

func RenderChart(p Payload) string {
	c := p.Corpus
	pos := mapValue(c["standing"])
	cov := mapValue(c["coverage"])
	lb := sortedLeaderboard(c["leaderboard"])
	lines := []string{
		fmt.Sprintf("product standing chart - %d concepts - score %s/100 (grade %s) - product-debt %d", intValue(c["rows"]), formatNumber(c["score"]), stringAny(c["grade"]), intValue(c["product_debt"])),
		"",
		"verdict ladder (count of concepts, best -> roadmap):",
	}
	maxn := 0
	for _, v := range verdicts {
		if n := intValue(pos[v]); n > maxn {
			maxn = n
		}
	}
	for _, v := range verdicts {
		n := intValue(pos[v])
		lines = append(lines, fmt.Sprintf("  %s %-15s %s %d", marks[v], v, bar(n, maxn, 28), n))
	}
	lines = append(lines, "", "verdict mix by category (each cell = one concept):")
	byCat := map[string][]string{}
	for _, r := range lb {
		byCat[stringAny(r["category"])] = append(byCat[stringAny(r["category"])], stringAny(r["verdict"]))
	}
	cats := sortedMapKeys(byCat)
	for _, cat := range cats {
		vs := byCat[cat]
		sort.SliceStable(vs, func(i, j int) bool { return rankOf(vs[i]) < rankOf(vs[j]) })
		var spark strings.Builder
		durable, usable := 0, 0
		for _, v := range vs {
			spark.WriteString(marks[v])
			if v == "durable-product" {
				durable++
			}
			if v == "usable-today" {
				usable++
			}
		}
		lines = append(lines, fmt.Sprintf("  %-12s %-16s (%d concept(s); %d durable, %d usable-today)", cat, spark.String(), len(vs), durable, usable))
	}
	laptop, needs, nocmd := 0, 0, 0
	for _, r := range lb {
		if boolAny(r["offline"]) {
			laptop++
		} else if NonEmpty(r["first_command"]) {
			needs++
		} else {
			nocmd++
		}
	}
	total := len(lb)
	lines = append(lines, "", "can a person run it today?",
		fmt.Sprintf("  laptop (offline)   %s %d", bar(laptop, total, 28), laptop),
		fmt.Sprintf("  needs gpu/key/net  %s %d", bar(needs, total, 28), needs),
		fmt.Sprintf("  no direct command  %s %d", bar(nocmd, total, 28), nocmd),
		"",
		fmt.Sprintf("coverage  [%s] %s%%  (%d/%d concept sections positioned)", bar(int(floatValue(cov["coverage_pct"])+0.5), 100, 32), formatNumber(cov["coverage_pct"]), intValue(cov["covered"]), intValue(cov["catalog_total"])),
		"",
		"legend: "+legend(),
	)
	return strings.Join(lines, "\n")
}

func RenderDocIndex(p Payload, stamp string) string {
	c := p.Corpus
	cov := mapValue(c["coverage"])
	lines := []string{
		"---",
		"title: \"fak product scorecard - which concepts are durable, real, useful-today products\"",
		"description: \"Inward product-concept scorecard with one honest verdict per concept.\"",
		"---",
		"",
		"# Product scorecard - durable, real, useful-today",
		"",
	}
	if stamp != "" {
		lines = append(lines, fmt.Sprintf("<!-- product-scorecard: %s - process: fak product-scorecard - data: tools/product_scorecard.data/ -->", stamp), "")
	}
	lines = append(lines,
		"Every number below is re-derived from the product scorecard data and cross-checked against the real tree: CLAIMS tags, first commands, witnesses, and entry docs.",
		"",
		"## Headline",
		"",
		"| Metric | Value |",
		"|---|---|",
		fmt.Sprintf("| **Coverage** | **%s%%** (%d/%d concept sections positioned) |", formatNumber(cov["coverage_pct"]), intValue(cov["covered"]), intValue(cov["catalog_total"])),
		fmt.Sprintf("| **Product-debt** | **%d** (honesty %d + coverage %d) |", intValue(c["product_debt"]), intValue(c["honesty_defects"]), intValue(c["coverage_debt"])),
		fmt.Sprintf("| Composite score | %s/100 (grade %s) |", formatNumber(c["score"]), stringAny(c["grade"])),
		fmt.Sprintf("| Durable products | %d of %d concepts |", intValue(c["durable_products"]), intValue(c["rows"])),
		fmt.Sprintf("| As of | %s (fak %s) |", stringAny(c["as_of"]), stringAny(c["fak_version"])),
		"",
		"## Standing at a glance",
		"",
		"```text",
		RenderChart(p),
		"```",
		"",
		"## The verdict ladder",
		"",
		"| Verdict | Means |",
		"|---|---|",
		"| * durable-product | shipped + an offline first command + witness + entry doc |",
		"| o usable-today | shipped + a first command, but it needs a GPU / key / network |",
		"| ~ real-not-easy | shipped/real, but no copy-pasteable command |",
		"| . honest-stub | a STUB / SIMULATED seam, labeled honestly |",
		"| - concept-only | a roadmap idea, not built |",
		"",
		"## The product concepts (best verdict first)",
		"",
		"| | Verdict | Maturity | Category | Use today? | Concept - what you get |",
		"|---|---|---|---|---|---|",
	)
	for _, row := range sortedLeaderboard(c["leaderboard"]) {
		verdict := stringAny(row["verdict"])
		today := "-"
		if boolAny(row["offline"]) {
			today = "laptop"
		} else if NonEmpty(row["first_command"]) {
			today = "needs gpu/key"
		}
		lines = append(lines, fmt.Sprintf("| %s | %s | %s | %s | %s | **%s** - %s |",
			marks[verdict], verdict, stringAny(row["maturity"]), stringAny(row["category"]), today, stringAny(row["concept"]), stringAny(row["what_you_get"])))
	}
	lines = append(lines, "", "## Per-KPI", "", "| Group | KPI | Score | Debt | Detail |", "|---|---|---:|:--:|---|")
	for _, b := range mapSlice(c["breakdown"]) {
		lines = append(lines, fmt.Sprintf("| %s | `%s` | %d | %d | %s |", stringAny(b["group"]), stringAny(b["kpi"]), intValue(b["score"]), intValue(b["debt"]), stringAny(b["detail"])))
	}
	if unc := sectionSlice(cov["uncovered"]); len(unc) > 0 {
		lines = append(lines, "", "## Coverage gaps", "")
		for _, sec := range unc {
			lines = append(lines, "- "+sec.Section)
		}
	}
	return strings.Join(lines, "\n")
}

func RenderDocFolder(p Payload, stamp string) map[string]string {
	return map[string]string{"README.md": RenderDocIndex(p, stamp)}
}

func bar(n, scale, width int) string {
	if scale <= 0 {
		return strings.Repeat(".", width)
	}
	cells := int(mathRound(float64(width*n) / float64(scale)))
	if n > 0 && cells == 0 {
		cells = 1
	}
	if cells > width {
		cells = width
	}
	return strings.Repeat("#", cells) + strings.Repeat(".", width-cells)
}

func mathRound(f float64) float64 {
	if f < 0 {
		return 0
	}
	return float64(int(f + 0.5))
}

func debtGroupLine(m object) string {
	parts := make([]string, 0, len(groups))
	for _, g := range groups {
		parts = append(parts, fmt.Sprintf("%s:%d", g, intValue(m[g])))
	}
	return strings.Join(parts, "  ")
}

func sortedLeaderboard(v any) []map[string]any {
	rows := mapSlice(v)
	sort.SliceStable(rows, func(i, j int) bool {
		ri, rj := rankOf(stringAny(rows[i]["verdict"])), rankOf(stringAny(rows[j]["verdict"]))
		if ri != rj {
			return ri < rj
		}
		return stringAny(rows[i]["category"]) < stringAny(rows[j]["category"])
	})
	return rows
}

func mapSlice(v any) []map[string]any {
	if rows, ok := v.([]map[string]any); ok {
		return rows
	}
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, it := range raw {
		if m, ok := it.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func sectionSlice(v any) []Section {
	if ss, ok := v.([]Section); ok {
		return ss
	}
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]Section, 0, len(raw))
	for _, it := range raw {
		switch s := it.(type) {
		case Section:
			out = append(out, s)
		case map[string]any:
			out = append(out, Section{Section: stringAny(s["section"]), Norm: stringAny(s["norm"])})
		}
	}
	return out
}

func stringSlice(v any) []string {
	if ss, ok := v.([]string); ok {
		return ss
	}
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, it := range raw {
		if s, ok := it.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func stringAny(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func boolAny(v any) bool {
	b, _ := v.(bool)
	return b
}

func formatNumber(v any) string {
	f := floatValue(v)
	if f == float64(int(f)) {
		return fmt.Sprintf("%d", int(f))
	}
	return fmt.Sprintf("%.1f", f)
}

func sortedMapKeys[T any](m map[string]T) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func legend() string {
	parts := make([]string, 0, len(verdicts))
	for _, v := range verdicts {
		parts = append(parts, fmt.Sprintf("%s %s", marks[v], v))
	}
	return strings.Join(parts, "   ")
}
