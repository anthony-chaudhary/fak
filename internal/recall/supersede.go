package recall

import "sort"

// Supersession resolution (#2624, epic #2618 W5): the structural retirement
// semantics behind a note's author-declared `supersedes: [[old-note]]` edge.
// A correction retires the fact it replaces from the recall working set —
// negative-only and expire-by-default: the retired note's bytes and row
// survive for audit, nothing is hard-deleted, and no timestamp inference or
// "which is newer" judgment runs — the author declares the edge explicitly.

// ResolveSupersession computes which notes a set of supersedes edges withholds
// from the working set. edges maps a superseding note ID to its resolved
// supersession targets; order lists every note ID in index (curation) order —
// the deterministic tie-break axis. An ID absent from order ranks earliest.
//
// The rule, in full (the documented tie-break):
//
//   - A note that is the target of a supersedes edge from a DISTINCT note is
//     withheld. A withheld note's own supersedes edge still retires its
//     predecessor, so a chain C→B→A collapses to its head: only C stays live.
//   - Cycles: in a supersession cycle (a note that can reach itself along
//     supersedes edges), the cycle member LATEST in index order survives —
//     intra-cycle edges pointing at it are inert. An edge from OUTSIDE the
//     cycle still withholds it. The winner is derived from the curation
//     order, never from map iteration, so a fixed store resolves identically
//     on every scan.
//
// Returns withheldBy: target ID → the index-earliest superseder whose edge is
// live. Reachability walks are bounded by visited sets — a cycle terminates,
// never loops.
func ResolveSupersession(edges map[string][]string, order []string) map[string]string {
	pos := make(map[string]int, len(order))
	for i, id := range order {
		if _, ok := pos[id]; !ok {
			pos[id] = i + 1 // 0 is reserved for IDs absent from order
		}
	}

	// reachOf[s] = every ID reachable from s along supersedes edges (s itself
	// only when s sits on a cycle). Bounded by the visited set.
	reachOf := make(map[string]map[string]bool, len(edges))
	for s := range edges {
		seen := map[string]bool{}
		stack := append([]string(nil), edges[s]...)
		for len(stack) > 0 {
			n := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if seen[n] {
				continue
			}
			seen[n] = true
			stack = append(stack, edges[n]...)
		}
		reachOf[s] = seen
	}
	inCycle := func(id string) bool { return reachOf[id][id] }
	sameCycle := func(a, b string) bool {
		return inCycle(a) && inCycle(b) && reachOf[a][b] && reachOf[b][a]
	}
	// cycleSurvives reports whether id is the member of its own cycle latest in
	// index order — the one intra-cycle edges must not retire.
	cycleSurvives := func(id string) bool {
		for m := range reachOf[id] {
			if m != id && sameCycle(id, m) && pos[m] > pos[id] {
				return false
			}
		}
		return true
	}

	// Sources walk in index order (then lexically for any ID off the index) so
	// the withheldBy attribution is deterministic, not map-iteration order.
	sources := append([]string(nil), order...)
	var extra []string
	for s := range edges {
		if _, ok := pos[s]; !ok {
			extra = append(extra, s)
		}
	}
	sort.Strings(extra)
	sources = append(sources, extra...)

	withheldBy := map[string]string{}
	for _, s := range sources {
		for _, t := range edges[s] {
			if t == s {
				continue // a self-supersession is not an edge
			}
			if sameCycle(s, t) && cycleSurvives(t) {
				continue // the cycle's surviving member shrugs off intra-cycle edges
			}
			if _, ok := withheldBy[t]; !ok {
				withheldBy[t] = s
			}
		}
	}
	return withheldBy
}
