package scorecard

import (
	"fmt"
	"sort"
	"strings"
)

// Data is the merged catalog: _meta.json's meta + families and every rows file.
type Data struct {
	Meta     Value
	Families Value
	Rows     []*Object
}

// Tree is the set of real-tree facts the KPIs cross-check rows against.
type Tree struct {
	SymFiles    map[string]int
	Structural  map[string]bool
	InTree      func(string) bool
	WirePresent func(string) bool
	Exists      func(string) bool
	DocVerbs    map[string]bool
}

type edgeSet map[[2]string]bool

func (e edgeSet) has(a, b string) bool { return e[[2]string{a, b}] }

func clamp(score float64) int {
	r := pyRoundInt(score)
	if r < 0 {
		return 0
	}
	if r > 100 {
		return 100
	}
	return r
}

// GradeLetter maps a score to its letter grade.
func GradeLetter(score float64) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	}
	return "F"
}

func ratio3(score int) PyFloat { return PyFloat(pyRound(float64(score)/100, 3)) }

// ---------------------------------------------------------------------------
// The fold.
// ---------------------------------------------------------------------------

func auditError(workspace, reason string) *Payload {
	return &Payload{
		Schema: Schema, OK: false, Verdict: "AUDIT_ERROR", Finding: "tooling_error", Reason: reason,
		NextAction: fmt.Sprintf("fix the read (run from repo ROOT; check %s/), then re-run", DataDirRel),
		Workspace:  workspace, Kpis: []KPI{},
	}
}

func objects(v Value) []*Object {
	var out []*Object
	for _, x := range iterValue(orEmptyList(v)) {
		if o, ok := x.(*Object); ok {
			out = append(out, o)
		}
	}
	return out
}

// BuildPayload folds data against tree into the scorecard payload.
func BuildPayload(workspace string, data *Data, tree *Tree, loadErr string) *Payload {
	if loadErr != "" || data == nil {
		if loadErr == "" {
			loadErr = "no data"
		}
		return auditError(workspace, loadErr)
	}
	meta, _ := data.Meta.(*Object)
	if meta == nil {
		meta = newObject()
	}
	rows := data.Rows
	if rows == nil {
		rows = []*Object{}
	}
	famDefs := objects(data.Families)
	if famDefs == nil {
		famDefs = []*Object{}
	}
	families := map[string]bool{}
	for _, f := range famDefs {
		if id := get(f, "id"); nonempty(id) {
			families[id.(string)] = true
		}
	}
	existsCache := map[string]bool{}
	exists := func(p string) bool {
		if v, ok := existsCache[p]; ok {
			return v
		}
		v := tree.Exists(p)
		existsCache[p] = v
		return v
	}

	collisions, collidingIDs := findCollisions(rows)
	sizes := clusterSizes(rows)
	byid := byIDMap(rows)

	index := buildIndex(rows)
	edges, unresolved := separationEdges(rows, index.byKeyMap())
	pairs := confusablePairs(rows)
	ipairs := indexPairs(index, pairs)
	sep := separationReport(byid, edges, pairs)
	contrast := contrastCount(edges, append(append([]Pair(nil), pairs...), ipairs...))
	entangled := entangledRows(sep, edges, ipairs)
	unseparated := unseparatedPairs(sep, ipairs, edges)
	expected := make([]verdictWhy, len(rows))
	for i, r := range rows {
		colliding, twin := false, ""
		if id, isStr := idString(r); isStr {
			colliding, twin = collidingIDs[id], entangled[id]
		}
		v, why := expectedVerdict(r, colliding, exists, sizes, twin)
		expected[i] = verdictWhy{v, why}
	}

	kpis := []KPI{
		kpiWellFormed(rows, families),
		kpiCanonicalUnique(collisions),
		kpiDefined(rows),
		kpiDisambiguated(rows, sizes),
		kpiReferenceResolves(unresolved),
		kpiPairSeparated(sep),
		kpiPairMutual(sep),
		kpiGrounded(rows, tree.InTree),
		kpiAliasResolved(rows, tree.InTree, tree.WirePresent),
		kpiAnchored(rows, exists),
		kpiIndexResolves(ipairs, byid, edges, index),
		kpiClarityConsistent(rows, expected),
		kpiKindGroundingSoft(rows, tree.DocVerbs),
		kpiHierarchySoft(rows, byid),
		kpiMutualitySoft(sep),
	}
	byName := map[string]KPI{}
	for _, k := range kpis {
		byName[k.Kpi] = k
	}
	terms := make([]float64, 0, len(kpiWeights))
	for _, w := range kpiWeights {
		terms = append(terms, float64(w.weight*float64(byName[w.name].Score)))
	}
	clarityScore := pyRound(pySum(terms), 1)
	clarityDefects, nSoft := 0, 0
	for _, k := range kpis {
		clarityDefects += len(k.Defects)
		nSoft += len(k.Soft)
	}

	cov := coverageReport(famDefs, rows, tree)
	debt := clarityDefects + cov.CoverageDebt
	covPct := float64(cov.CoveragePct)
	if cov.Discovered == 0 {
		covPct = 100.0
	}
	score := pyRound(float64(clarityWeight*clarityScore)+float64(coverageWeight*covPct), 1)
	grade := GradeLetter(score)

	debtByGroup := make(OrderedCounts, len(Groups))
	for i, g := range Groups {
		debtByGroup[i] = KV{g, 0}
	}
	for _, k := range kpis {
		for i := range debtByGroup {
			if debtByGroup[i].Key == k.Group {
				debtByGroup[i].N += len(k.Defects)
			}
		}
	}
	breakdown := make([]Breakdown, 0, len(kpis))
	for _, k := range kpis {
		breakdown = append(breakdown, Breakdown{Kpi: k.Kpi, Group: k.Group, Score: k.Score, Value: ratio3(k.Score), Debt: len(k.Defects), Detail: k.Detail})
	}
	sort.SliceStable(breakdown, func(i, j int) bool {
		if breakdown[i].Debt != breakdown[j].Debt {
			return breakdown[i].Debt > breakdown[j].Debt
		}
		return breakdown[i].Score < breakdown[j].Score
	})

	rowOwed := perRowDebt(rows, kpis)
	pos := standing(rows)
	lb := make([]LeaderRow, 0, len(rows))
	for i, r := range rows {
		lb = append(lb, LeaderRow{
			ID: get(r, "id"), Canonical: get(r, "canonical"), Family: get(r, "family"),
			Kind: get(r, "kind"), Verdict: get(r, "verdict"), ExpectedVerdict: expected[i].verdict,
			Definition: get(r, "definition"), DistinctFrom: orEmptyList(get(r, "distinct_from")),
			Grounding: get(r, "grounding"), GlossaryAnchor: getOr(r, "glossary_anchor", ""),
		})
	}
	crit := criticalBacklog(rows, rowOwed)
	rollup := rollUp(rows, byid, rowOwed)

	kpiScores := make(OrderedCounts, 0, len(kpis))
	debtByKpi := make(OrderedCounts, 0, len(kpis))
	for _, k := range kpis {
		kpiScores = append(kpiScores, KV{k.Kpi, k.Score})
		debtByKpi = append(debtByKpi, KV{k.Kpi, len(k.Defects)})
	}
	unresolvedShared := 0
	for _, p := range ipairs {
		if !edges.has(p.A, p.B) || !edges.has(p.B, p.A) {
			unresolvedShared++
		}
	}
	corpus := &Corpus{
		Value: PyFloat(pyRound(score/100, 3)), ValueUnit: "quality_ratio",
		Score: PyFloat(score), LegacyScore: PyFloat(score), LegacyScoreScale: 100, Grade: grade,
		ClarityScore: PyFloat(clarityScore), ClarityValue: PyFloat(pyRound(clarityScore/100, 3)),
		DisambiguationDebt: debt, ClarityDefects: clarityDefects, CoverageDebt: cov.CoverageDebt,
		Coverage: cov, SoftSignals: nSoft, Rows: len(rows), CrystalConcepts: pos.Get("crystal"),
		Families: len(famDefs), AsOf: getOr(meta, "as_of", ""), FakVersion: getOr(meta, "fak_version", ""),
		Standing: pos, DebtByGroup: debtByGroup, KpiScores: kpiScores, DebtByKpi: debtByKpi,
		Breakdown: breakdown, Leaderboard: lb, Critical: crit, Rollup: rollup,
		Separation: Separation{
			ConfusablePairs: len(sep.pairs), Separated: sep.mutual + sep.oneSided,
			Mutual: sep.mutual, OneSided: sep.oneSided, Undrawn: sep.undrawn,
			EntangledConcepts: len(entangled), Boundaries: sep.edges, MutualBoundaries: sep.mutualEdges,
			OneWayBoundaries: len(sep.oneWay), DanglingReferences: len(unresolved),
			Pairs: sep.pairs, Unseparated: unseparated,
		},
		Index: IndexSummary{
			Keys: index.Keys, AmbiguousKeys: index.AmbiguousKeys, SharedNamePairs: len(ipairs),
			UnresolvedSharedNames: unresolvedShared, Ambiguous: index.Ambiguous, ContrastConcepts: contrast,
		},
	}

	standingLine := fmt.Sprintf("%d crystal / %d defined / %d drifting / %d entangled / %d colliding / %d undocumented",
		pos.Get("crystal"), pos.Get("defined"), pos.Get("drifting"), pos.Get("entangled"), pos.Get("colliding"), pos.Get("undocumented"))
	covLine := fmt.Sprintf("coverage %s%% (%d/%d confusable tree tokens positioned)", cov.CoveragePct, cov.Covered, cov.Discovered)
	p := &Payload{Schema: Schema, Workspace: workspace, Corpus: corpus, Kpis: kpis,
		Data: &DataEcho{Rows: rows, Families: famDefs}, Index: index}
	scoreS := pyFloatRepr(score)
	switch {
	case debt == 0:
		p.OK, p.Verdict, p.Finding = true, "OK", "namespace_crystal_clear"
		p.Reason = fmt.Sprintf("namespace crystal-clear: score %s/100 (grade %s); %s; zero disambiguation-debt across %d KPIs over %d concepts (%s; %d advisory)",
			scoreS, grade, covLine, len(kpis), len(rows), standingLine, nSoft)
		p.NextAction = "hold the line; when a new confusable name lands in the tree, coverage drops - position + disambiguate it; re-run to keep debt at 0"
	case clarityDefects == 0 && cov.CoverageDebt > 0:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "coverage_debt"
		p.Reason = fmt.Sprintf("%d confusable tree token(s) not yet positioned; %s; score %s/100 (grade %s); positioned rows are clean (0 clarity-debt); standing %s",
			cov.CoverageDebt, covLine, scoreS, grade, standingLine)
		p.NextAction = "close coverage (see --gaps): add a disambiguated row for each unpositioned confusable token, worst family first; re-run"
	default:
		p.OK, p.Verdict, p.Finding = false, "ACTION", "disambiguation_debt"
		worst := breakdown[0]
		p.Reason = fmt.Sprintf("%d clarity defect(s) + %d coverage gap(s) = disambiguation-debt %d; score %s/100 (grade %s); heaviest KPI: %s (%d); %s; standing %s",
			clarityDefects, cov.CoverageDebt, debt, scoreS, grade, worst.Kpi, worst.Debt, covLine, standingLine)
		p.NextAction = "retire disambiguation-debt worst-first (--critical + per-KPI defects): rename true collisions, write missing definitions, draw + anchor the distinctions; then close coverage (--gaps); re-run to prove the drop"
	}
	return p
}

func standing(rows []*Object) OrderedCounts {
	counts := make(OrderedCounts, len(Verdicts))
	for i, v := range Verdicts {
		counts[i] = KV{v, 0}
	}
	for _, r := range rows {
		if s, ok := get(r, "verdict").(string); ok {
			if rk, ok := verdictRank[s]; ok {
				counts[rk].N++
			}
		}
	}
	return counts
}

func perRowDebt(rows []*Object, kpis []KPI) map[string]int {
	out := map[string]int{}
	for i, r := range rows {
		v, ok := r.Get("id")
		switch {
		case !ok:
			out[fmt.Sprintf("row[%d]", i)] = 0
		case isString(v):
			out[v.(string)] = 0
		}
	}
	for _, k := range kpis {
		for _, d := range k.Defects {
			rid, _, _ := strings.Cut(d, ":")
			if _, ok := out[rid]; ok {
				out[rid]++
			}
		}
	}
	return out
}

func criticalBacklog(rows []*Object, rowOwed map[string]int) []CriticalRow {
	out := make([]CriticalRow, 0, len(rows))
	sortKey := make([]string, 0, len(rows))
	for _, r := range rows {
		id := get(r, "id")
		debt := 0
		if s, ok := id.(string); ok {
			debt = rowOwed[s]
		}
		out = append(out, CriticalRow{ID: id, Canonical: get(r, "canonical"), Family: get(r, "family"),
			Verdict: get(r, "verdict"), Debt: debt, Distance: rankOf(get(r, "verdict"), 9), Gaps: orEmptyList(get(r, "gaps"))})
		k := ""
		if truthy(id) {
			k = pyStr(id)
		}
		sortKey = append(sortKey, k)
	}
	idx := make([]int, len(out))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		x, y := out[idx[a]], out[idx[b]]
		if x.Debt != y.Debt {
			return x.Debt > y.Debt
		}
		if x.Distance != y.Distance {
			return x.Distance > y.Distance
		}
		return sortKey[idx[a]] < sortKey[idx[b]]
	})
	sorted := make([]CriticalRow, len(out))
	for i, j := range idx {
		sorted[i] = out[j]
	}
	return sorted
}
