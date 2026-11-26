package scorecard

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// KPIs.
// ---------------------------------------------------------------------------

func newKPI(name string, defects []string, okDetail, badDetail string, soft []string) KPI {
	if soft == nil {
		soft = []string{}
	}
	defects = nonNil(defects)
	detail := okDetail
	if len(defects) > 0 {
		detail = badDetail
		if detail == "" {
			detail = fmt.Sprintf("%d defect(s)", len(defects))
		}
	}
	score := clamp(float64(100 - kpiPenalty[name]*len(defects) - min(10, 2*len(soft))))
	return KPI{Kpi: name, Group: kpiGroup[name], Score: score, Value: ratio3(score), Detail: detail, Defects: defects, Soft: soft}
}

func softKPI(name, group string, score int, okDetail, badDetail string, soft []string) KPI {
	detail := okDetail
	if len(soft) > 0 {
		detail = badDetail
	}
	return KPI{Kpi: name, Group: group, Score: score, Value: ratio3(score), Detail: detail, Defects: []string{}, Soft: nonNil(soft)}
}

func inSet(set map[string]bool, v Value) bool {
	s, ok := v.(string)
	return ok && set[s]
}

func kpiWellFormed(rows []*Object, families map[string]bool) KPI {
	var defects []string
	seen := map[string]bool{}
	for i, r := range rows {
		rid := ridIndexed(r, i)
		for _, f := range requiredFields {
			if _, ok := r.Get(f); !ok {
				defects = append(defects, fmt.Sprintf("%s: missing field '%s'", rid, f))
			}
		}
		id := get(r, "id")
		switch {
		case !nonempty(id):
			defects = append(defects, rid+": missing id")
		case seen[id.(string)]:
			defects = append(defects, rid+": duplicate id")
		default:
			seen[id.(string)] = true
		}
		if len(families) > 0 && !inSet(families, get(r, "family")) {
			defects = append(defects, fmt.Sprintf("%s: family %s not declared in _meta.json", rid, pyRepr(get(r, "family"))))
		}
		if !inSet(kinds, get(r, "kind")) {
			defects = append(defects, fmt.Sprintf("%s: kind %s not in %s", rid, pyRepr(get(r, "kind")), pyRepr(kindNames)))
		}
		if !inSet(groundingKinds, get(r, "grounding_kind")) {
			defects = append(defects, fmt.Sprintf("%s: grounding_kind %s not in %s", rid, pyRepr(get(r, "grounding_kind")), pyRepr(groundingKindNames)))
		}
		if _, ok := verdictRank[getString(r, "verdict")]; !ok || !isString(get(r, "verdict")) {
			defects = append(defects, fmt.Sprintf("%s: verdict %s not in %s", rid, pyRepr(get(r, "verdict")), pyRepr(Verdicts)))
		}
		for _, lf := range []string{"distinct_from", "aliases", "gaps"} {
			if _, ok := get(r, lf).([]Value); !ok {
				defects = append(defects, fmt.Sprintf("%s: %s must be a list", rid, lf))
			}
		}
		if !nonempty(get(r, "canonical")) {
			defects = append(defects, rid+": missing canonical (the one true name)")
		}
		if !nonempty(get(r, "grounding")) {
			defects = append(defects, rid+": missing grounding (a token that must appear in the tree)")
		}
	}
	return newKPI("well_formed", defects, fmt.Sprintf("all %d rows well-formed", len(rows)),
		fmt.Sprintf("%d malformed field(s)", len(defects)), nil)
}

func isString(v Value) bool { _, ok := v.(string); return ok }

func kpiCanonicalUnique(coll []collision) KPI {
	sorted := append([]collision(nil), coll...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].id < sorted[j].id })
	var defects []string
	for _, c := range sorted {
		defects = append(defects, fmt.Sprintf("%s: canonical name collides with %s - rename one", c.id, strings.Join(c.others, ", ")))
	}
	return newKPI("canonical_unique", defects, "every concept has a unique canonical name",
		fmt.Sprintf("%d canonical-name collision(s)", len(defects)), nil)
}

func kpiDefined(rows []*Object) KPI {
	var defects []string
	for i, r := range rows {
		if !nonempty(get(r, "definition")) {
			defects = append(defects, ridDefault(r, i)+": no definition - one sentence on what it IS")
		}
	}
	return newKPI("defined", defects, "every concept has a definition",
		fmt.Sprintf("%d undefined concept(s)", len(defects)), nil)
}

func kpiDisambiguated(rows []*Object, sizes map[string]int) KPI {
	ids := idSet(rows)
	canon := map[string]bool{}
	for _, r := range rows {
		canon[normValue(getOr(r, "canonical", ""))] = true
	}
	var defects []string
	for i, r := range rows {
		rid := ridDefault(r, i)
		fam := get(r, "family")
		if familySize(sizes, fam) < 2 {
			continue
		}
		famS := pyStr(fam)
		if !nonempty(get(r, "distinction")) {
			defects = append(defects, fmt.Sprintf("%s: in family '%s' with siblings but no distinction line (what is it NOT?)", rid, famS))
		}
		var refs []string
		for _, x := range iterValue(orEmptyList(get(r, "distinct_from"))) {
			if nonempty(x) {
				refs = append(refs, x.(string))
			}
		}
		resolved := 0
		for _, x := range refs {
			if ids[x] || canon[normToken(x)] {
				resolved++
			}
		}
		if len(refs) == 0 {
			defects = append(defects, fmt.Sprintf("%s: distinct_from is empty but family '%s' has siblings", rid, famS))
		} else if resolved == 0 {
			defects = append(defects, fmt.Sprintf("%s: distinct_from %s resolves to no catalog id/canonical", rid, pyRepr(refs)))
		}
	}
	return newKPI("disambiguated", defects, "every confusable concept names what it is NOT",
		fmt.Sprintf("%d undisambiguated confusable concept(s)", len(defects)), nil)
}

func kpiReferenceResolves(unresolved []string) KPI {
	return newKPI("reference_resolves", append([]string(nil), unresolved...),
		"every distinct_from reference resolves to a real concept",
		fmt.Sprintf("%d dangling distinct_from reference(s)", len(unresolved)), nil)
}

func kpiPairSeparated(sep sepReport) KPI {
	var defects []string
	for _, p := range sep.pairs {
		if p.State == "undrawn" {
			defects = append(defects, fmt.Sprintf("%s: confusable with %s (%s) but neither names the other - add %s to %s's distinct_from (and back)",
				p.A, p.B, p.Why, p.B, p.A))
		}
	}
	return newKPI("pair_separated", defects, fmt.Sprintf("all %d confusable name-pair(s) are separated", len(sep.pairs)),
		fmt.Sprintf("%d confusable pair(s) with no line drawn", len(defects)), nil)
}

func kpiPairMutual(sep sepReport) KPI {
	var defects []string
	for _, p := range sep.pairs {
		if p.State != "one_sided" {
			continue
		}
		src, dst := p.B, p.A
		if p.AToB {
			src, dst = p.A, p.B
		}
		defects = append(defects, fmt.Sprintf("%s: %s separates from it, but it does not separate back (%s) - a reader arriving at %s is not warned",
			dst, src, p.Why, dst))
	}
	return newKPI("pair_mutual", defects, "every confusable pair draws its line from both sides",
		fmt.Sprintf("%d one-sided boundary on a confusable pair", len(defects)), nil)
}

func kpiIndexResolves(ipairs []Pair, byid map[string]*Object, edges edgeSet, idx *Index) KPI {
	var defects []string
	for _, p := range ipairs {
		if edges.has(p.A, p.B) && edges.has(p.B, p.A) {
			continue
		}
		defects = append(defects, fmt.Sprintf("%s: shares a lookup name with %s (%s vs %s) - %s - but they do not separate from each other; add %s to %s's distinct_from (and back)",
			p.A, p.B, pyStr(rowField(byid, p.A, "canonical")), pyStr(rowField(byid, p.B, "canonical")), p.Why, p.B, p.A))
	}
	return newKPI("index_resolves", defects,
		fmt.Sprintf("every one of %d lookup name(s) resolves - %d land on several concepts, all separated", idx.Keys, idx.AmbiguousKeys),
		fmt.Sprintf("%d unresolvable shared lookup name(s)", len(defects)), nil)
}

func kpiMutualitySoft(sep sepReport) KPI {
	score := clamp(float64(100 - min(20, len(sep.oneWay)/50)))
	soft := sep.oneWay
	if len(soft) > 12 {
		soft = soft[:12]
	}
	return softKPI("mutuality_soft", "separation", score, "every boundary is drawn from both sides",
		fmt.Sprintf("%d/%d boundaries drawn one-way only", len(sep.oneWay), sep.edges), append([]string(nil), soft...))
}

func kpiGrounded(rows []*Object, inTree func(string) bool) KPI {
	var defects []string
	for i, r := range rows {
		g := getOr(r, "grounding", "")
		if !nonempty(g) {
			continue
		}
		if !inTree(normToken(g.(string))) {
			defects = append(defects, fmt.Sprintf("%s: grounding '%s' does not appear in the production corpus", ridDefault(r, i), g.(string)))
		}
	}
	return newKPI("grounded", defects, "every concept's grounding token appears in the tree",
		fmt.Sprintf("%d ungrounded concept(s)", len(defects)), nil)
}

func kpiAliasResolved(rows []*Object, inTree, wirePresent func(string) bool) KPI {
	var defects []string
	for i, r := range rows {
		seen := map[string]bool{}
		for _, av := range iterValue(orEmptyList(get(r, "aliases"))) {
			a, ok := av.(string)
			if !ok || pyStrip(a) == "" || seen[a] {
				continue
			}
			seen[a] = true
			if inTree(normToken(a)) {
				continue
			}
			if isWireKey(a) && wirePresent(a) {
				continue
			}
			defects = append(defects, fmt.Sprintf("%s: alias '%s' resolves nowhere in the tree - a fabricated or stale alias retires coverage debt dishonestly", ridDefault(r, i), a))
		}
	}
	return newKPI("alias_resolved", defects, "every alias resolves in the tree (token or wire key)",
		fmt.Sprintf("%d unresolvable alias(es)", len(defects)), nil)
}

func kpiAnchored(rows []*Object, exists func(string) bool) KPI {
	var defects []string
	for i, r := range rows {
		rid := ridDefault(r, i)
		anchor := getOr(r, "glossary_anchor", "")
		if valueEqual(get(r, "verdict"), "crystal") && !nonempty(anchor) {
			defects = append(defects, rid+": verdict 'crystal' but no glossary_anchor - where is the distinction written?")
			continue
		}
		if nonempty(anchor) && !exists(anchor.(string)) {
			defects = append(defects, fmt.Sprintf("%s: glossary_anchor '%s' does not exist in the tree", rid, anchor.(string)))
		}
	}
	return newKPI("anchored", defects, "every crystal concept's distinction is anchored on disk",
		fmt.Sprintf("%d missing/dangling anchor(s)", len(defects)), nil)
}

type verdictWhy struct{ verdict, why string }

func kpiClarityConsistent(rows []*Object, expected []verdictWhy) KPI {
	var defects []string
	for i, r := range rows {
		declared := get(r, "verdict")
		if !valueEqual(declared, expected[i].verdict) {
			defects = append(defects, fmt.Sprintf("%s: claims '%s' but evidence implies '%s' - %s", ridDefault(r, i), pyStr(declared), expected[i].verdict, expected[i].why))
		}
	}
	return newKPI("clarity_consistent", defects, "every verdict matches its evidence",
		fmt.Sprintf("%d verdict overclaim(s)", len(defects)), nil)
}

func kpiKindGroundingSoft(rows []*Object, docVerbs map[string]bool) KPI {
	var soft []string
	for i, r := range rows {
		rid := ridDefault(r, i)
		kind, gk := get(r, "kind"), get(r, "grounding_kind")
		if valueEqual(kind, "cli-verb") {
			verb := normValue(getOr(r, "grounding", ""))
			if !valueEqual(gk, "verb") {
				soft = append(soft, fmt.Sprintf("%s: kind 'cli-verb' but grounding_kind '%s' (expected 'verb')", rid, pyStr(gk)))
			} else if verb != "" && !docVerbs[verb] {
				soft = append(soft, fmt.Sprintf("%s: cli-verb '%s' not documented in cli-reference", rid, pyStr(get(r, "grounding"))))
			}
		}
		if valueEqual(kind, "metric") && !valueEqual(gk, "metric") {
			soft = append(soft, fmt.Sprintf("%s: kind 'metric' but grounding_kind '%s' (expected 'metric')", rid, pyStr(gk)))
		}
	}
	score := clamp(float64(100 - min(40, 6*len(soft))))
	return softKPI("kind_grounding_soft", "honesty", score, "kind agrees with grounding",
		fmt.Sprintf("%d kind/grounding mismatch", len(soft)), soft)
}
