package scorecard

import (
	"fmt"
	"sort"
)

// ---------------------------------------------------------------------------
// Hierarchy roll-up.
// ---------------------------------------------------------------------------

type forest struct {
	parentOrder []string
	parent      map[string]string
	children    map[string][]string
	childOrder  []string
	soft        []string
}

func resolveParents(rows []*Object) forest {
	ids := idSet(rows)
	f := forest{parent: map[string]string{}, children: map[string][]string{}}
	for _, r := range rows {
		ridV := get(r, "id")
		p := get(r, "parent")
		if !nonempty(p) {
			continue
		}
		ps := p.(string)
		rid := pyStr(ridV)
		switch {
		case !ids[ps]:
			f.soft = append(f.soft, fmt.Sprintf("%s: parent '%s' resolves to no catalog id", rid, ps))
		case valueEqual(p, ridV):
			f.soft = append(f.soft, rid+": parent points at itself")
		default:
			if _, ok := ridV.(string); !ok {
				continue
			}
			if _, ok := f.parent[rid]; !ok {
				f.parentOrder = append(f.parentOrder, rid)
			}
			f.parent[rid] = ps
		}
	}
	for _, start := range f.parentOrder {
		seen := map[string]bool{}
		cur := start
		for {
			next, ok := f.parent[cur]
			if !ok {
				break
			}
			if seen[cur] {
				f.soft = append(f.soft, start+": parent chain cycles")
				break
			}
			seen[cur] = true
			cur = next
		}
	}
	for _, c := range f.parentOrder {
		p := f.parent[c]
		if _, ok := f.children[p]; !ok {
			f.childOrder = append(f.childOrder, p)
		}
		f.children[p] = append(f.children[p], c)
	}
	for _, p := range f.childOrder {
		sort.Strings(f.children[p])
	}
	return f
}

func subtreeIDs(root string, children map[string][]string) map[string]bool {
	seen := map[string]bool{}
	stack := []string{root}
	for len(stack) > 0 {
		x := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[x] {
			continue
		}
		seen[x] = true
		for _, k := range children[x] {
			if !seen[k] {
				stack = append(stack, k)
			}
		}
	}
	return seen
}

func abstractionRollup(rows []*Object, byid map[string]*Object, rowOwed map[string]int, f forest) []Abstraction {
	recs := []Abstraction{}
	for _, head := range f.childOrder {
		hr, ok := byid[head]
		if !ok {
			continue
		}
		subtree := subtreeIDs(head, f.children)
		mix := make(OrderedCounts, len(Verdicts))
		for i, v := range Verdicts {
			mix[i] = KV{v, 0}
		}
		debt := 0
		worstID, worstRank := head, -1
		for _, n := range sortedSet(subtree) {
			r, ok := byid[n]
			if !ok {
				continue
			}
			v := get(r, "verdict")
			if s, ok := v.(string); ok {
				if rk, ok := verdictRank[s]; ok {
					mix[rk].N++
				}
			}
			debt += rowOwed[n]
			if rk := rankOf(v, len(Verdicts)); rk > worstRank {
				worstRank, worstID = rk, n
			}
		}
		rolled := "undocumented"
		if worstRank >= 0 && worstRank < len(Verdicts) {
			rolled = Verdicts[worstRank]
		}
		declared := get(hr, "verdict")
		depth, cur := 0, head
		for {
			next, ok := f.parent[cur]
			if !ok || depth > len(rows) {
				break
			}
			depth++
			cur = next
		}
		recs = append(recs, Abstraction{
			ID: head, Canonical: get(hr, "canonical"), Family: get(hr, "family"),
			DeclaredVerdict: declared, RolledVerdict: rolled,
			SubtreeSize: len(subtree), ChildCount: len(f.children[head]),
			SubtreeOwed: debt, VerdictMix: mix, Depth: depth,
			Weakest:   Weakest{ID: worstID, Verdict: get(byid[worstID], "verdict")},
			Overclaim: rankOf(declared, len(Verdicts)) < worstRank,
		})
	}
	sort.SliceStable(recs, func(i, j int) bool {
		a, b := recs[i], recs[j]
		if a.SubtreeOwed != b.SubtreeOwed {
			return a.SubtreeOwed > b.SubtreeOwed
		}
		ra, rb := rankOf(a.RolledVerdict, len(Verdicts)), rankOf(b.RolledVerdict, len(Verdicts))
		if ra != rb {
			return ra > rb
		}
		if a.SubtreeSize != b.SubtreeSize {
			return a.SubtreeSize > b.SubtreeSize
		}
		return a.ID < b.ID
	})
	return recs
}

func rollUp(rows []*Object, byid map[string]*Object, rowOwed map[string]int) Rollup {
	f := resolveParents(rows)
	recs := abstractionRollup(rows, byid, rowOwed, f)
	overclaims := []string{}
	roots := 0
	for _, a := range recs {
		if a.Overclaim {
			overclaims = append(overclaims, fmt.Sprintf("%s: abstraction declares '%s' but rolls up to '%s' (weakest: %s = %s)",
				a.ID, pyStr(a.DeclaredVerdict), a.RolledVerdict, a.Weakest.ID, pyStr(a.Weakest.Verdict)))
		}
		if a.Depth == 0 {
			roots++
		}
	}
	nodes := map[string]bool{}
	for k := range f.parent {
		nodes[k] = true
	}
	for k := range f.children {
		nodes[k] = true
	}
	maxDepth := 0
	for n := range nodes {
		d, cur, guard := 0, n, 0
		for {
			next, ok := f.parent[cur]
			if !ok || guard > len(f.parent) {
				break
			}
			d++
			cur = next
			guard++
		}
		maxDepth = max(maxDepth, d)
	}
	return Rollup{Abstractions: recs, Heads: len(recs), Roots: roots, ForestNodes: len(nodes), MaxDepth: maxDepth, Overclaims: overclaims}
}

func kpiHierarchySoft(rows []*Object, byid map[string]*Object) KPI {
	f := resolveParents(rows)
	recs := abstractionRollup(rows, byid, map[string]int{}, f)
	soft := append([]string(nil), f.soft...)
	for _, a := range recs {
		if a.Overclaim {
			soft = append(soft, fmt.Sprintf("%s: rolls up to '%s' but head declares '%s' (weakest descendant %s)",
				a.ID, a.RolledVerdict, pyStr(a.DeclaredVerdict), a.Weakest.ID))
		}
	}
	score := clamp(float64(100 - min(30, 6*len(soft))))
	return softKPI("hierarchy_soft", "honesty", score, "hierarchy resolves + rolls up honestly",
		fmt.Sprintf("%d hierarchy issue(s)", len(soft)), soft)
}
