package scorecard

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// view resolves the corpus a renderer reads; an AUDIT_ERROR payload carries an
// empty corpus, which the Python renderers read through their .get defaults.
func (p *Payload) view() (*Corpus, bool) {
	if p.Corpus == nil {
		return &Corpus{Grade: "?"}, true
	}
	return p.Corpus, false
}

func floatOr(empty bool, f PyFloat, def string) string {
	if empty {
		return def
	}
	return f.String()
}

func bar(n, scale, width int) string {
	if scale <= 0 {
		return strings.Repeat(".", width)
	}
	cells := pyRoundInt(float64(width*max(0, n)) / float64(scale))
	cells = max(0, min(width, cells))
	if n > 0 && cells == 0 {
		cells = 1
	}
	return strings.Repeat("#", cells) + strings.Repeat(".", width-cells)
}

func sortKeyStr(v Value) string {
	if truthy(v) {
		return pyStr(v)
	}
	return ""
}

func sortedLeaderboard(lb []LeaderRow) []LeaderRow {
	out := append([]LeaderRow(nil), lb...)
	sort.SliceStable(out, func(i, j int) bool {
		ri, rj := rankOf(out[i].Verdict, 9), rankOf(out[j].Verdict, 9)
		if ri != rj {
			return ri < rj
		}
		return sortKeyStr(out[i].Family) < sortKeyStr(out[j].Family)
	})
	return out
}

func sortedPerFamily(pf []FamilyCoverage) []FamilyCoverage {
	out := append([]FamilyCoverage(nil), pf...)
	sort.SliceStable(out, func(i, j int) bool {
		gi, gj := out[i].Covered-out[i].Discovered, out[j].Covered-out[j].Discovered
		if gi != gj {
			return gi < gj
		}
		return pyStr(out[i].Family) < pyStr(out[j].Family)
	})
	return out
}

// RenderText is the default human scorecard.
func (p *Payload) RenderText() string {
	c, empty := p.view()
	cov := c.Coverage
	pos := c.Standing
	sp := c.Separation
	ix := c.Index
	ru := c.Rollup
	score := floatOr(empty, c.Score, "0")
	tenth := "0.0"
	if !empty {
		tenth = pyFloatRepr(pyRound(float64(c.Score)/10.0, 1))
	}
	groups := make([]string, len(Groups))
	for i, g := range Groups {
		groups[i] = fmt.Sprintf("%s:%d", g, c.DebtByGroup.Get(g))
	}
	lines := []string{
		fmt.Sprintf("concept-disambiguation: %s (%s)", p.Verdict, p.Finding),
		"  " + p.Reason,
		"",
		fmt.Sprintf("score %s/100 (grade %s) = %s/10 - DISAMBIGUATION-DEBT %d (clarity %d + coverage %d) - %d advisory",
			score, c.Grade, tenth, c.DisambiguationDebt, c.ClarityDefects, c.CoverageDebt, c.SoftSignals),
		fmt.Sprintf("coverage: %s%% (%d/%d confusable tree tokens positioned) - %d concepts scored - %d crystal",
			floatOr(empty, cov.CoveragePct, "0"), cov.Covered, cov.Discovered, c.Rows, c.CrystalConcepts),
		fmt.Sprintf("standing: %d crystal - %d defined - %d drifting - %d entangled - %d colliding - %d undocumented",
			pos.Get("crystal"), pos.Get("defined"), pos.Get("drifting"), pos.Get("entangled"), pos.Get("colliding"), pos.Get("undocumented")),
		fmt.Sprintf("separation: %d confusable name-pair(s) - %d mutual, %d one-sided, %d undrawn - %d boundaries drawn (%d one-way); see --pairs",
			sp.ConfusablePairs, sp.Mutual, sp.OneSided, sp.Undrawn, sp.Boundaries, sp.OneWayBoundaries),
		fmt.Sprintf("index: %d lookup name(s) -> %d concept(s) - %d shared by several, %d of those unseparated; see --index",
			ix.Keys, c.Rows, ix.AmbiguousKeys, ix.UnresolvedSharedNames),
		"debt by group: " + strings.Join(groups, "  "),
		fmt.Sprintf("roll-up: %d top-level abstraction(s) over %d concepts (weakest-link) - %d overclaim(s); see --rollup",
			ru.Roots, ru.ForestNodes, len(ru.Overclaims)),
		"",
		"concepts (best verdict first):",
		fmt.Sprintf("  %s %s %s canonical", padRight("verdict", 13), padRight("kind", 10), padRight("family", 16)),
	}
	for _, row := range sortedLeaderboard(c.Leaderboard) {
		flag := ""
		if !valueEqual(row.Verdict, row.ExpectedVerdict) {
			flag = "  ! expected " + row.ExpectedVerdict
		}
		lines = append(lines, fmt.Sprintf("  %s %s %s %s %s%s", markOf(row.Verdict), padRight(pyStr(row.Verdict), 11),
			padRight(pyStr(row.Kind), 10), padRight(pyStr(row.Family), 16), pyStr(row.Canonical), flag))
	}
	lines = append(lines, "", "per-KPI (worst first):",
		fmt.Sprintf("  %s %s  %s %s detail", padLeft("score", 5), padLeft("debt", 4), padRight("group", 13), padRight("kpi", 22)))
	for _, b := range c.Breakdown {
		lines = append(lines, fmt.Sprintf("  %s %s  %s %s %s", padLeft(strconv.Itoa(b.Score), 5), padLeft(strconv.Itoa(b.Debt), 4),
			padRight(b.Group, 13), padRight(b.Kpi, 22), b.Detail))
	}
	lines = append(lines, "")
	if len(cov.PerFamily) > 0 {
		lines = append(lines, "coverage by family (covered / discovered):")
		for _, f := range sortedPerFamily(cov.PerFamily) {
			lines = append(lines, fmt.Sprintf("  %s %s/%s  (%d unpositioned)", padRight(pyStr(f.Family), 16),
				padLeft(strconv.Itoa(f.Covered), 3), padRight(strconv.Itoa(f.Discovered), 3), f.Discovered-f.Covered))
		}
		lines = append(lines, "")
	}
	lines = append(lines, "disambiguation-debt work-list:")
	kpis := append([]KPI(nil), p.Kpis...)
	sort.SliceStable(kpis, func(i, j int) bool { return len(kpis[i].Defects) > len(kpis[j].Defects) })
	anyDefect := false
	for _, k := range kpis {
		if len(k.Defects) == 0 {
			continue
		}
		anyDefect = true
		lines = append(lines, fmt.Sprintf("  %s (%d):", k.Kpi, len(k.Defects)))
		for _, it := range k.Defects[:min(12, len(k.Defects))] {
			lines = append(lines, "      - "+it)
		}
		if len(k.Defects) > 12 {
			lines = append(lines, fmt.Sprintf("      ... and %d more", len(k.Defects)-12))
		}
	}
	if !anyDefect {
		lines = append(lines, "  (none - every positioned concept is clean)")
	}
	lines = append(lines, "", "next: "+p.NextAction)
	return strings.Join(lines, "\n")
}

// RenderCritical is the worst-first clarity backlog.
func (p *Payload) RenderCritical() string {
	c, _ := p.view()
	lines := []string{"concept-disambiguation critical backlog (clarify worst-first):", ""}
	if len(c.Critical) == 0 {
		lines = append(lines, "  (no concepts scored)")
		return strings.Join(lines, "\n")
	}
	shown := false
	for _, it := range c.Critical {
		if it.Debt == 0 && it.Distance <= verdictRank["defined"] {
			continue
		}
		shown = true
		lines = append(lines, fmt.Sprintf("  [%d debt - %s] %s (%s)", it.Debt, pyStr(it.Verdict), pyStr(it.ID), pyStr(it.Family)))
		gaps := iterValue(orEmptyList(it.Gaps))
		for _, g := range gaps[:min(4, len(gaps))] {
			lines = append(lines, "      - "+pyStr(g))
		}
	}
	if !shown {
		lines = append(lines, "  (no critical rows - every concept is defined-or-better with 0 debt)")
	}
	lines = append(lines, "", "(rows with 0 debt and a crystal/defined verdict are omitted - they are not critical.)")
	return strings.Join(lines, "\n")
}

// RenderGaps is the coverage backlog.
func (p *Payload) RenderGaps() string {
	c, _ := p.view()
	lines := []string{"concept-disambiguation coverage backlog (position every confusable token):", ""}
	pf := c.Coverage.PerFamily
	for _, f := range sortedPerFamily(pf) {
		lines = append(lines, fmt.Sprintf("FAMILY %s: %d/%d positioned (%d unpositioned)", pyStr(f.Family), f.Covered, f.Discovered, f.Discovered-f.Covered))
		for _, tok := range f.Uncovered {
			lines = append(lines, "  - "+tok)
		}
		lines = append(lines, "")
	}
	if len(pf) == 0 {
		lines = append(lines, "  (no families declared)")
	}
	return strings.Join(lines, "\n")
}

// PairsText is the pairwise separation backlog.
func (p *Payload) PairsText() string {
	c, _ := p.view()
	sp := c.Separation
	order := map[string]int{"undrawn": 0, "one_sided": 1, "mutual": 2}
	mark := map[string]string{"undrawn": "x", "one_sided": ">", "mutual": "="}
	lines := []string{
		fmt.Sprintf("concept-disambiguation pairwise separation: %d confusable name-pair(s) discovered - %d undrawn, %d one-sided, %d mutual",
			sp.ConfusablePairs, sp.Undrawn, sp.OneSided, sp.Mutual),
		"",
		"A pair is CONFUSABLE when its two names collide by spelling alone, in one of three",
		"ways: homonym (identical once the parenthetical gloss is stripped), permuted (the",
		fmt.Sprintf("same words in a different order), or near (within %d edits AND sharing a head or", maxPairEdits),
		fmt.Sprintf("tail run of %d+ characters, so a bare family root is not enough).", minSharedAffix),
		"Pairs are DISCOVERED from the names themselves, never declared, so landing a",
		"near-twin name raises the bar automatically. Being clean per-row does not separate",
		"a pair: a concept in a large family satisfies `disambiguated` by naming ANY sibling,",
		"while the twin it is genuinely mistakable for goes undrawn.",
		"",
		"  x undrawn    neither names the other - the twins sit in the catalog unseparated",
		"  > one-sided  drawn from one side only - a reader arriving at the other is unwarned",
		"  = mutual     both name each other - a reader arriving from EITHER side is told",
		"",
	}
	pairs := append([]GradedPair(nil), sp.Pairs...)
	stateOrder := func(s string) int {
		if o, ok := order[s]; ok {
			return o
		}
		return 9
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		a, b := pairs[i], pairs[j]
		if stateOrder(a.State) != stateOrder(b.State) {
			return stateOrder(a.State) < stateOrder(b.State)
		}
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.A != b.A {
			return a.A < b.A
		}
		return a.B < b.B
	})
	if len(pairs) == 0 {
		lines = append(lines, "  (no confusable name-pairs discovered - every canonical name is far",
			"   from every other; the namespace separates itself by spelling alone)")
		return strings.Join(lines, "\n")
	}
	for _, pr := range pairs {
		arrow := "   "
		switch {
		case pr.State == "mutual":
			arrow = "<->"
		case pr.AToB:
			arrow = "-->"
		case pr.BToA:
			arrow = "<--"
		}
		m, ok := mark[pr.State]
		if !ok {
			m = " "
		}
		lines = append(lines, fmt.Sprintf("  %s [%s] %s %s %s", m, padRight(pr.Kind, 8), pyRepr(pr.ACanonical), arrow, pyRepr(pr.BCanonical)))
		lines = append(lines, fmt.Sprintf("      %s   (%s / %s)", pr.Why, pr.A, pr.B))
	}
	lines = append(lines, "",
		fmt.Sprintf("whole-graph mutuality: %d/%d boundaries are drawn from both sides (%d one-way; advisory outside confusable pairs)",
			sp.MutualBoundaries, sp.Boundaries, sp.OneWayBoundaries),
		fmt.Sprintf("dangling references: %d", sp.DanglingReferences))
	return strings.Join(lines, "\n")
}

// IndexTarget is one concept an index key lands on, joined with its contrast set.
type IndexTarget struct {
	ID                  string
	Canonical           Value
	Family              Value
	Kind                Value
	Verdict             Value
	Definition          Value
	Anchor              Value
	NotToBeConfusedWith []string
	Via                 []string
}

// IndexRow is an index entry with every target's catalog facts.
type IndexRow struct {
	IndexEntry
	TargetsFull []IndexTarget
}

func (p *Payload) indexRows() []IndexRow {
	c, _ := p.view()
	byid := map[string]LeaderRow{}
	for _, r := range c.Leaderboard {
		if s, ok := r.ID.(string); ok {
			byid[s] = r
		}
	}
	contrast := map[string][]string{}
	for _, pr := range c.Separation.Pairs {
		contrast[pr.A] = append(contrast[pr.A], pr.B)
		contrast[pr.B] = append(contrast[pr.B], pr.A)
	}
	var entries []IndexEntry
	if p.Index != nil {
		entries = p.Index.Entries
	}
	out := make([]IndexRow, 0, len(entries))
	for _, e := range entries {
		targets := make([]IndexTarget, 0, len(e.Targets))
		for _, tid := range e.Targets {
			r, ok := byid[tid]
			twins := map[string]bool{}
			for _, x := range contrast[tid] {
				twins[x] = true
			}
			for _, x := range e.Targets {
				if x != tid {
					twins[x] = true
				}
			}
			nc := map[string]bool{}
			if ok {
				for _, x := range iterValue(orEmptyList(r.DistinctFrom)) {
					nc[pyStr(x)] = true
				}
			}
			for x := range twins {
				nc[x] = true
			}
			t := IndexTarget{ID: tid, NotToBeConfusedWith: sortedSet(nc), Via: nonNil(e.Via.Get(tid)), Anchor: ""}
			if ok {
				t.Canonical, t.Family, t.Kind, t.Verdict, t.Definition, t.Anchor = r.Canonical, r.Family, r.Kind, r.Verdict, r.Definition, r.GlossaryAnchor
			}
			targets = append(targets, t)
		}
		out = append(out, IndexRow{IndexEntry: e, TargetsFull: targets})
	}
	return out
}

func indexLookup(rows []IndexRow, name string) ([]IndexRow, bool) {
	key := normToken(bareName(name))
	if key == "" {
		key = normToken(name)
	}
	if key == "" {
		return nil, false
	}
	var exact []IndexRow
	for _, e := range rows {
		if e.Key == key {
			exact = append(exact, e)
		}
	}
	if len(exact) > 0 {
		return exact, true
	}
	var near []IndexRow
	for _, e := range rows {
		if strings.Contains(e.Key, key) || strings.Contains(key, e.Key) {
			near = append(near, e)
		}
	}
	return near, false
}

// pySlice is rows[:limit] with Python's negative-index semantics.
func pySlice[T any](rows []T, limit int) []T {
	if limit < 0 {
		limit += len(rows)
		if limit < 0 {
			limit = 0
		}
	}
	if limit > len(rows) {
		limit = len(rows)
	}
	return rows[:limit]
}

// IndexText is the name -> concept lookup index, optionally one lookup.
func (p *Payload) IndexText(limit int, lookup string) string {
	c, _ := p.view()
	ix := c.Index
	rows := p.indexRows()
	var lines []string
	if lookup != "" {
		hits, exact := indexLookup(rows, lookup)
		if len(hits) == 0 {
			return fmt.Sprintf("no lookup name matches %s among %d name(s) over %d concept(s).\n", pyStrString(lookup), ix.Keys, c.Rows) +
				"The name is unpositioned: `--gaps` lists the tree tokens still waiting for a concept, and `fak concept position` lands one."
		}
		rows = hits
		suffix := ""
		if !exact {
			suffix = " (no exact match; nearest keys)"
		}
		lines = []string{fmt.Sprintf("%s -> %d lookup name(s)%s", pyStrString(lookup), len(rows), suffix), ""}
	} else {
		lines = []string{
			fmt.Sprintf("concept-disambiguation name index: %d lookup name(s) over %d concept(s) - %d ambiguous", ix.Keys, c.Rows, ix.AmbiguousKeys),
			"",
			"Every surface a name can arrive as (canonical / alias / grounding token) resolves",
			"to the concept it denotes plus that concept's contrast set. '!' marks a name that",
			"lands on more than one concept - a LOOKUP ambiguity the canonical names do not show.",
			"",
		}
	}
	shown := rows
	if limit != 0 {
		shown = pySlice(rows, limit)
	}
	for _, e := range shown {
		flag := " "
		if e.Ambiguous {
			flag = "!"
		}
		line := flag + " " + e.Spellings[0]
		if len(e.Spellings) > 1 {
			line += "   (also: " + strings.Join(e.Spellings[1:], ", ") + ")"
		}
		lines = append(lines, line)
		for _, t := range e.TargetsFull {
			lines = append(lines, fmt.Sprintf("    -> %s  [%s / %s / %s]  via %s", pyStr(t.Canonical), pyStr(t.Family), pyStr(t.Kind), pyStr(t.Verdict), strings.Join(t.Via, "+")))
			if truthy(t.Definition) {
				lines = append(lines, "       "+pyStr(t.Definition))
			}
			if len(t.NotToBeConfusedWith) > 0 {
				lines = append(lines, "       not to be confused with: "+strings.Join(t.NotToBeConfusedWith, ", "))
			}
		}
	}
	if limit != 0 && len(rows) > limit {
		lines = append(lines, fmt.Sprintf("  ... and %d more name(s) (omit --limit for all)", len(rows)-limit))
	}
	return strings.Join(lines, "\n")
}

// RenderChart is the at-a-glance ASCII chart.
func (p *Payload) RenderChart() string {
	c, empty := p.view()
	pos := c.Standing
	cov := c.Coverage
	lines := []string{
		fmt.Sprintf("concept-disambiguation chart - %d concepts - score %s/100 (grade %s) - disambiguation-debt %d",
			c.Rows, floatOr(empty, c.Score, "0"), c.Grade, c.DisambiguationDebt),
		"",
		"clarity ladder (count of concepts, best -> fog):",
	}
	maxn := 0
	for _, v := range Verdicts {
		maxn = max(maxn, pos.Get(v))
	}
	for _, v := range Verdicts {
		n := pos.Get(v)
		lines = append(lines, fmt.Sprintf("  %s %s %s %d", verdictMark[v], padRight(v, 13), bar(n, maxn, 28), n))
	}
	lines = append(lines, "")
	var famOrder []string
	byFam := map[string][]Value{}
	for _, r := range c.Leaderboard {
		fam := "?"
		if truthy(r.Family) {
			fam = pyStr(r.Family)
		}
		if _, ok := byFam[fam]; !ok {
			famOrder = append(famOrder, fam)
		}
		byFam[fam] = append(byFam[fam], r.Verdict)
	}
	sort.Strings(famOrder)
	lines = append(lines, "clarity mix by family (each cell = one concept):")
	for _, fam := range famOrder {
		verds := append([]Value(nil), byFam[fam]...)
		sort.SliceStable(verds, func(i, j int) bool { return rankOf(verds[i], 9) < rankOf(verds[j], 9) })
		var spark strings.Builder
		crystal := 0
		for _, v := range verds {
			spark.WriteString(markOf(v))
			if valueEqual(v, "crystal") {
				crystal++
			}
		}
		lines = append(lines, fmt.Sprintf("  %s %s (%d concept(s); %d crystal)", padRight(fam, 16), padRight(spark.String(), 18), len(verds), crystal))
	}
	lines = append(lines, "", "coverage by family (positioned / discovered):")
	for _, f := range sortedPerFamily(cov.PerFamily) {
		lines = append(lines, fmt.Sprintf("  %s %s %d/%d", padRight(pyStr(f.Family), 16), bar(f.Covered, max(1, f.Discovered), 28), f.Covered, f.Discovered))
	}
	lines = append(lines, "")
	pct := float64(cov.CoveragePct)
	pctS := floatOr(empty, cov.CoveragePct, "0.0")
	lines = append(lines, fmt.Sprintf("namespace coverage  [%s] %s%%  (%d/%d confusable tokens positioned)",
		bar(pyRoundInt(pct), 100, 32), pctS, cov.Covered, cov.Discovered))
	lines = append(lines, "")
	sp := c.Separation
	npair := sp.ConfusablePairs
	lines = append(lines, "pairwise separation (of the name-pairs a reader cannot keep apart):")
	for _, sl := range []struct {
		n     int
		label string
	}{{sp.Mutual, "mutual"}, {sp.OneSided, "one-sided"}, {sp.Undrawn, "undrawn"}} {
		lines = append(lines, fmt.Sprintf("  %s %s %d", padRight(sl.label, 12), bar(sl.n, max(1, npair), 28), sl.n))
	}
	lines = append(lines, fmt.Sprintf("  pairs separated   [%s] %d/%d", bar(sp.Separated, max(1, npair), 32), sp.Separated, npair))
	ix := c.Index
	lines = append(lines, "", fmt.Sprintf("name index: %d lookup name(s) -> %d concept(s), %d ambiguous", ix.Keys, c.Rows, ix.AmbiguousKeys), "")
	legend := make([]string, len(Verdicts))
	for i, v := range Verdicts {
		legend[i] = verdictMark[v] + " " + v
	}
	lines = append(lines, "legend: "+strings.Join(legend, "   "))
	return strings.Join(lines, "\n")
}

func mixSpark(mix OrderedCounts) string {
	total := 0
	for _, v := range Verdicts {
		total += mix.Get(v)
	}
	if total <= 24 {
		var b strings.Builder
		for _, v := range Verdicts {
			b.WriteString(strings.Repeat(verdictMark[v], mix.Get(v)))
		}
		return b.String()
	}
	var parts []string
	for _, v := range Verdicts {
		if n := mix.Get(v); n != 0 {
			parts = append(parts, fmt.Sprintf("%s%d", verdictMark[v], n))
		}
	}
	return strings.Join(parts, " ")
}

// RollupText is the operator view at the abstraction heads.
func (p *Payload) RollupText() string {
	c, _ := p.view()
	ru := c.Rollup
	lines := []string{
		fmt.Sprintf("concept-disambiguation roll-up: %d top-level abstraction(s), %d head(s) total, max depth %d (%d concepts in the forest)",
			ru.Roots, ru.Heads, ru.MaxDepth, ru.ForestNodes),
		"",
		"Each abstraction rolls up WEAKEST-LINK: only as crystal-clear as its foggiest",
		"descendant. '!' = the head verdict reads clearer than the subtree supports.",
		"",
		fmt.Sprintf("     %s %s %s %s  %s family / canonical", padRight("rolled", 12), padRight("head", 12), padLeft("size", 4), padLeft("debt", 4), padRight("mix", 20)),
	}
	if len(ru.Abstractions) == 0 {
		lines = append(lines, "  (no hierarchy positioned - add a `parent` edge to a row to roll it up)")
		return strings.Join(lines, "\n")
	}
	for _, a := range ru.Abstractions {
		if a.Depth != 0 {
			continue
		}
		flag := " "
		if a.Overclaim {
			flag = "!"
		}
		lines = append(lines, fmt.Sprintf("  %s%s %s %s %s %s  %s %s / %s", flag, markOf(a.RolledVerdict), padRight(a.RolledVerdict, 11),
			padRight(pyStr(a.DeclaredVerdict), 12), padLeft(strconv.Itoa(a.SubtreeSize), 4), padLeft(strconv.Itoa(a.SubtreeOwed), 4),
			padRight(mixSpark(a.VerdictMix), 20), pyStr(a.Family), pyStr(a.Canonical)))
	}
	lines = append(lines, "")
	if len(ru.Overclaims) > 0 {
		lines = append(lines, fmt.Sprintf("abstraction overclaims (%d) - head reads clearer than its subtree supports:", len(ru.Overclaims)))
		for _, o := range ru.Overclaims[:min(20, len(ru.Overclaims))] {
			lines = append(lines, "  ! "+o)
		}
		if len(ru.Overclaims) > 20 {
			lines = append(lines, fmt.Sprintf("  ... and %d more", len(ru.Overclaims)-20))
		}
	} else {
		lines = append(lines, "abstraction overclaims: none - every head verdict is backed by its whole subtree")
	}
	lines = append(lines, "", "(weakest-link: one 'defined' leaf keeps its abstraction from rolling up to crystal - anchor the leaf, or accept the honest lower roll-up.)")
	return strings.Join(lines, "\n")
}

// RenderCompare proves the debt moved against a baseline --json payload.
func RenderCompare(baseline Value, current *Payload) string {
	b := objField(baseline, "corpus")
	cur := current.compareView()
	bd, cd := numOr(b, "disambiguation_debt"), numOr(cur, "disambiguation_debt")
	bo, co := numOr(b, "score"), numOr(cur, "score")
	ratio := "inf (zero)"
	if !cd.isZero() {
		ratio = pyFixed(bd.float()/cd.float(), 1) + "x"
	}
	bs, cs := objField(b, "separation"), objField(cur, "separation")
	bi, ci := objField(b, "index"), objField(cur, "index")
	lines := []string{
		fmt.Sprintf("disambiguation-debt: %s -> %s   (%s fewer defects+gaps)", bd, cd, ratio),
		fmt.Sprintf("  clarity:    %s -> %s", numOr(b, "clarity_defects"), numOr(cur, "clarity_defects")),
		fmt.Sprintf("  coverage:   %s -> %s", numOr(b, "coverage_debt"), numOr(cur, "coverage_debt")),
		fmt.Sprintf("score:        %s/100 -> %s/100   (+%s)", bo, co, co.sub(bo).round1()),
		fmt.Sprintf("crystal:      %s -> %s crystal concepts", numOr(b, "crystal_concepts"), numOr(cur, "crystal_concepts")),
		fmt.Sprintf("pairs undrawn: %s -> %s   (mutual %s -> %s)", numOr(bs, "undrawn"), numOr(cs, "undrawn"), numOr(bs, "mutual"), numOr(cs, "mutual")),
		fmt.Sprintf("ambiguous names: %s -> %s", numOr(bi, "ambiguous_keys"), numOr(ci, "ambiguous_keys")),
	}
	bg, cg := objField(b, "debt_by_group"), objField(cur, "debt_by_group")
	for _, g := range Groups {
		lines = append(lines, fmt.Sprintf("  %s %s -> %s", padRight(g, 13), numOr(bg, g), numOr(cg, g)))
	}
	target3 := bd.add(pyNum{i: 2}).floorDiv(3).maxZero()
	target2 := bd.floorDiv(2).maxZero()
	switch {
	case cd.le(target3):
		lines = append(lines, fmt.Sprintf("VERDICT: >=3x reduction achieved (debt %s->%s, target <=%s).", bd, cd, target3))
	case cd.le(target2):
		lines = append(lines, fmt.Sprintf("VERDICT: >=2x (not yet 3x) - debt %s->%s; 3x needs <=%s.", bd, cd, target3))
	default:
		lines = append(lines, fmt.Sprintf("VERDICT: not yet 2x - need debt <=%s (now %s); 3x target <=%s.", target2, cd, target3))
	}
	return strings.Join(lines, "\n")
}

// compareView is the slice of the current corpus RenderCompare reads, in the
// same generic shape as the baseline it is compared with.
func (p *Payload) compareView() Value {
	o := newObject()
	if p.Corpus == nil {
		return o
	}
	c := p.Corpus
	o.Set("disambiguation_debt", Number(strconv.Itoa(c.DisambiguationDebt)))
	o.Set("score", Number(c.Score.String()))
	o.Set("clarity_defects", Number(strconv.Itoa(c.ClarityDefects)))
	o.Set("coverage_debt", Number(strconv.Itoa(c.CoverageDebt)))
	o.Set("crystal_concepts", Number(strconv.Itoa(c.CrystalConcepts)))
	sep := newObject()
	sep.Set("undrawn", Number(strconv.Itoa(c.Separation.Undrawn)))
	sep.Set("mutual", Number(strconv.Itoa(c.Separation.Mutual)))
	o.Set("separation", sep)
	ix := newObject()
	ix.Set("ambiguous_keys", Number(strconv.Itoa(c.Index.AmbiguousKeys)))
	o.Set("index", ix)
	g := newObject()
	for _, kv := range c.DebtByGroup {
		g.Set(kv.Key, Number(strconv.Itoa(kv.N)))
	}
	o.Set("debt_by_group", g)
	return o
}

// objField is `(v.get(key) or {})` over a generic value.
func objField(v Value, key string) Value {
	o, ok := v.(*Object)
	if !ok {
		return newObject()
	}
	f := get(o, key)
	if !truthy(f) {
		return newObject()
	}
	return f
}

// pyNum is a Python int or float read from JSON, with the arithmetic
// RenderCompare performs on it.
type pyNum struct {
	isFloat bool
	i       int
	f       float64
	raw     Value
}

func numOr(v Value, key string) pyNum {
	o, ok := v.(*Object)
	if !ok {
		return pyNum{}
	}
	x, present := o.Get(key)
	if !present {
		return pyNum{}
	}
	if n, ok := x.(Number); ok {
		if isIntLiteral(string(n)) {
			i, err := strconv.Atoi(string(n))
			if err == nil {
				return pyNum{i: i}
			}
		}
		f, _ := strconv.ParseFloat(string(n), 64)
		return pyNum{isFloat: true, f: f}
	}
	if b, ok := x.(bool); ok {
		if b {
			return pyNum{i: 1, raw: x}
		}
		return pyNum{raw: x}
	}
	return pyNum{raw: x, isFloat: true, f: math.NaN()}
}

func (n pyNum) String() string {
	if n.raw != nil {
		return pyStr(n.raw)
	}
	if n.isFloat {
		return pyFloatRepr(n.f)
	}
	return strconv.Itoa(n.i)
}

func (n pyNum) float() float64 {
	if n.isFloat {
		return n.f
	}
	return float64(n.i)
}

func (n pyNum) isZero() bool { return n.float() == 0 }

func (n pyNum) sub(o pyNum) pyNum {
	if n.isFloat || o.isFloat {
		return pyNum{isFloat: true, f: n.float() - o.float()}
	}
	return pyNum{i: n.i - o.i}
}

func (n pyNum) add(o pyNum) pyNum {
	if n.isFloat || o.isFloat {
		return pyNum{isFloat: true, f: n.float() + o.float()}
	}
	return pyNum{i: n.i + o.i}
}

func (n pyNum) round1() pyNum {
	if n.isFloat {
		return pyNum{isFloat: true, f: pyRound(n.f, 1)}
	}
	return n
}

func (n pyNum) floorDiv(d int) pyNum {
	if n.isFloat {
		return pyNum{isFloat: true, f: math.Floor(n.f / float64(d))}
	}
	q := n.i / d
	if (n.i%d != 0) && ((n.i < 0) != (d < 0)) {
		q--
	}
	return pyNum{i: q}
}

func (n pyNum) maxZero() pyNum {
	if n.float() < 0 {
		return pyNum{}
	}
	return n
}

func (n pyNum) le(o pyNum) bool { return n.float() <= o.float() }
