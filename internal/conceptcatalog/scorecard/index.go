package scorecard

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Index.
// ---------------------------------------------------------------------------

func buildIndex(rows []*Object) *Index {
	type entry struct {
		spellings map[string]bool
		targets   map[string]map[string]bool
	}
	entries := map[string]*entry{}
	for i, r := range rows {
		rid := ridIndexed(r, i)
		canonical, grounding := get(r, "canonical"), get(r, "grounding")
		names := []Value{canonical}
		names = append(names, iterValue(orEmptyList(get(r, "aliases")))...)
		names = append(names, grounding)
		for _, nameV := range names {
			if !nonempty(nameV) {
				continue
			}
			source := "alias"
			if valueEqual(nameV, canonical) {
				source = "canonical"
			} else if valueEqual(nameV, grounding) {
				source = "grounding"
			}
			name := pyStrip(nameV.(string))
			key := normToken(bareName(name))
			if key == "" {
				key = normToken(name)
			}
			if key == "" {
				continue
			}
			e := entries[key]
			if e == nil {
				e = &entry{spellings: map[string]bool{}, targets: map[string]map[string]bool{}}
				entries[key] = e
			}
			e.spellings[name] = true
			if e.targets[rid] == nil {
				e.targets[rid] = map[string]bool{}
			}
			e.targets[rid][source] = true
		}
	}
	keys := make([]string, 0, len(entries))
	for k := range entries {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	idx := &Index{Entries: make([]IndexEntry, 0, len(keys)), Ambiguous: []IndexEntry{}, ByKey: make(OrderedKeyTargets, 0, len(keys))}
	for _, k := range keys {
		e := entries[k]
		targets := sortedKeys(e.targets)
		via := make(OrderedSources, 0, len(targets))
		for _, t := range targets {
			via = append(via, SourcesKV{t, sortedSet(e.targets[t])})
		}
		ie := IndexEntry{Key: k, Spellings: sortedSet(e.spellings), Targets: targets, Via: via, Ambiguous: len(targets) > 1}
		idx.Entries = append(idx.Entries, ie)
		if ie.Ambiguous {
			idx.Ambiguous = append(idx.Ambiguous, ie)
		}
		idx.ByKey = append(idx.ByKey, KeyTargets{k, targets})
	}
	idx.Keys = len(idx.Entries)
	idx.AmbiguousKeys = len(idx.Ambiguous)
	return idx
}

func (idx *Index) byKeyMap() map[string][]string {
	m := make(map[string][]string, len(idx.ByKey))
	for _, kt := range idx.ByKey {
		m[kt.Key] = kt.Targets
	}
	return m
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedSet(m map[string]bool) []string { return sortedKeys(m) }

func indexPairs(idx *Index, already []Pair) []Pair {
	seen := map[[2]string]bool{}
	for _, p := range already {
		seen[[2]string{p.A, p.B}] = true
		seen[[2]string{p.B, p.A}] = true
	}
	emitted := map[[2]string]bool{}
	var out []Pair
	for _, e := range idx.Ambiguous {
		for i, a := range e.Targets {
			for _, b := range e.Targets[i+1:] {
				if seen[[2]string{a, b}] || emitted[[2]string{a, b}] {
					continue
				}
				emitted[[2]string{a, b}] = true
				emitted[[2]string{b, a}] = true
				via := map[string]bool{}
				for _, s := range e.Via.Get(a) {
					via[s] = true
				}
				for _, s := range e.Via.Get(b) {
					via[s] = true
				}
				out = append(out, Pair{a, b, "shared-name", fmt.Sprintf(
					"both answer to the lookup name %s (via %s; %d concepts share it)",
					pyStrString(e.Key), strings.Join(sortedSet(via), "/"), len(e.Targets))})
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].A != out[j].A {
			return out[i].A < out[j].A
		}
		return out[i].B < out[j].B
	})
	return out
}

func unseparatedPairs(sep sepReport, ipairs []Pair, edges edgeSet) []Unseparated {
	out := []Unseparated{}
	for _, p := range sep.pairs {
		if p.State != "mutual" {
			out = append(out, Unseparated{p.A, p.B, p.Kind, p.Why, p.State, p.AToB, p.BToA})
		}
	}
	for _, p := range ipairs {
		ab, ba := edges.has(p.A, p.B), edges.has(p.B, p.A)
		if ab && ba {
			continue
		}
		state := "undrawn"
		if ab || ba {
			state = "one_sided"
		}
		out = append(out, Unseparated{p.A, p.B, p.Kind, p.Why, state, ab, ba})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].A != out[j].A {
			return out[i].A < out[j].A
		}
		return out[i].B < out[j].B
	})
	return out
}

func contrastCount(edges edgeSet, pairs []Pair) int {
	ids := map[string]bool{}
	for e := range edges {
		ids[e[0]] = true
	}
	for _, p := range pairs {
		ids[p.A] = true
		ids[p.B] = true
	}
	return len(ids)
}

func entangledRows(sep sepReport, edges edgeSet, extra []Pair) map[string]string {
	out := map[string]string{}
	visit := func(a, b, why string) {
		for _, side := range [][2]string{{a, b}, {b, a}} {
			me, other := side[0], side[1]
			if _, ok := out[me]; !ok && !edges.has(me, other) {
				out[me] = fmt.Sprintf("names no boundary against its twin %s (%s)", other, why)
			}
		}
	}
	for _, p := range sep.pairs {
		visit(p.A, p.B, p.Why)
	}
	for _, p := range extra {
		visit(p.A, p.B, p.Why)
	}
	return out
}

func expectedVerdict(row *Object, colliding bool, exists func(string) bool, sizes map[string]int, entangled string) (string, string) {
	if !nonempty(get(row, "definition")) {
		return "undocumented", "no definition"
	}
	if colliding {
		return "colliding", "shares a canonical name with another concept"
	}
	hasSibling := familySize(sizes, get(row, "family")) >= 2
	if hasSibling && !nonempty(get(row, "distinction")) {
		return "drifting", "defined but draws no line against its siblings"
	}
	if entangled != "" {
		return "entangled", entangled
	}
	anchor := getOr(row, "glossary_anchor", "")
	if !nonempty(anchor) || !exists(anchor.(string)) {
		return "defined", "boundary drawn but not written in a discoverable doc"
	}
	return "crystal", "defined + distinction + anchored on disk"
}
