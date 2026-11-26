package scorecard

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Coverage.
// ---------------------------------------------------------------------------

type familyToken struct {
	token    string
	presence int
}

func listStrings(v Value) []string {
	var out []string
	for _, x := range iterValue(orEmptyList(v)) {
		if nonempty(x) {
			out = append(out, normToken(x.(string)))
		}
	}
	return out
}

func minFiles(v Value, present bool) int {
	if !present {
		return 2
	}
	switch x := v.(type) {
	case Number:
		if n, ok := numberInt(string(x)); ok {
			return n
		}
	case string:
		var n int
		if _, err := fmt.Sscanf(pyStrip(x), "%d", &n); err == nil {
			return n
		}
	case bool:
		if x {
			return 1
		}
		return 0
	}
	return 2
}

type rootHits struct {
	tokens []string
	cache  map[string][]string
}

func (h *rootHits) hits(root string) []string {
	if got, ok := h.cache[root]; ok {
		return got
	}
	var out []string
	for _, t := range h.tokens {
		if strings.Contains(t, root) {
			out = append(out, t)
		}
	}
	h.cache[root] = out
	return out
}

func discoverFamilyTokens(fam *Object, tree *Tree, rh *rootHits) []familyToken {
	roots := listStrings(get(fam, "roots"))
	ignore := map[string]bool{}
	for _, x := range iterValue(orEmptyList(get(fam, "ignore"))) {
		ignore[normValue(x)] = true
	}
	exclude := listStrings(get(fam, "exclude"))
	mf, present := fam.Get("min_files")
	minF := minFiles(mf, present)
	candidates := map[string]bool{}
	for _, root := range roots {
		for _, t := range rh.hits(root) {
			candidates[t] = true
		}
	}
	var out []familyToken
	for tok := range candidates {
		if ignore[tok] {
			continue
		}
		excluded := false
		for _, ex := range exclude {
			if strings.Contains(tok, ex) {
				excluded = true
				break
			}
		}
		if excluded {
			continue
		}
		presence := tree.SymFiles[tok]
		if presence >= minF || tree.Structural[tok] {
			out = append(out, familyToken{tok, presence})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].presence != out[j].presence {
			return out[i].presence > out[j].presence
		}
		return out[i].token < out[j].token
	})
	return out
}

func coverageReport(families []*Object, rows []*Object, tree *Tree) Coverage {
	claimed := map[string]bool{}
	for _, r := range rows {
		for _, t := range rowTokens(r) {
			claimed[t] = true
		}
	}
	tokens := make([]string, 0, len(tree.SymFiles)+len(tree.Structural))
	for t := range tree.SymFiles {
		tokens = append(tokens, t)
	}
	for t := range tree.Structural {
		if _, ok := tree.SymFiles[t]; !ok {
			tokens = append(tokens, t)
		}
	}
	rh := &rootHits{tokens: tokens, cache: map[string][]string{}}
	perFamily := []FamilyCoverage{}
	globalCov := map[string]bool{}
	globalOwner := map[string]Value{}
	for _, fam := range families {
		fid := getOr(fam, "id", "?")
		toks := discoverFamilyTokens(fam, tree, rh)
		famCov := 0
		famUnc := []string{}
		for _, t := range toks {
			hit := claimed[t.token]
			if hit {
				famCov++
			} else {
				famUnc = append(famUnc, t.token)
			}
			if _, ok := globalCov[t.token]; !ok {
				globalCov[t.token] = hit
				globalOwner[t.token] = fid
			}
		}
		if len(famUnc) > 40 {
			famUnc = famUnc[:40]
		}
		perFamily = append(perFamily, FamilyCoverage{Family: fid, Discovered: len(toks), Covered: famCov, Uncovered: famUnc})
	}
	total := len(globalCov)
	covered := 0
	uncovered := []UncoveredToken{}
	for _, t := range sortedKeys(globalCov) {
		if globalCov[t] {
			covered++
		} else {
			uncovered = append(uncovered, UncoveredToken{Family: globalOwner[t], Token: t})
		}
	}
	pct := 100.0
	if total > 0 {
		pct = pyRound(100.0*float64(covered)/float64(total), 1)
	}
	return Coverage{Discovered: total, Covered: covered, CoveragePct: PyFloat(pct), CoverageDebt: total - covered, PerFamily: perFamily, Uncovered: uncovered}
}

func rowTokens(r *Object) []string {
	var out []string
	if t := normValue(getOr(r, "canonical", "")); t != "" {
		out = append(out, t)
	}
	if t := normValue(getOr(r, "grounding", "")); t != "" {
		out = append(out, t)
	}
	for _, a := range iterValue(orEmptyList(get(r, "aliases"))) {
		if t := normValue(a); t != "" {
			out = append(out, t)
		}
	}
	return out
}
