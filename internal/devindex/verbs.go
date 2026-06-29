package devindex

// C3 of epic #1287 (#1290): the structured CLI-verb catalog. Today the `fak` verb
// list lives as freeform raw strings in cmd/fak/usage.go — unparseable, and it
// drifts from the main.go dispatch. This file defines the COMMITTED, structured
// verb manifest (name -> synopsis -> owning lane -> doc link) so `fak index verbs
// [<query>]` is robust and `usage.go` can be GENERATED from one source.
//
// Lane note: the manifest + its query function live INSIDE internal/devindex (this
// is the data structure C3 calls for). GENERATING usage.go from it and wiring the
// `fak index verbs` cmd verb are the cmd/ half — out of this package's lane.
//
// Honesty: the manifest is a VIEW the freshness gate (C6 #1293) cross-checks against
// the live main.go switch — a verb in main.go with no manifest entry, or a manifest
// entry naming a lane that no leaf declares, is a drift FINDING, never a silent gap.

import (
	"sort"
	"strings"
)

// Verb is one entry of the structured CLI-verb catalog: the verb name as typed
// (`fak <Name>`), the one-line synopsis shown in usage, optional command aliases
// that route to the same handler (the extra strings in a `case "leaf", "leaves":`),
// the owning lane/leaf the handler's code lives under, and an optional doc-map path
// for the deeper reference. It is the parseable replacement for a raw usage string.
type Verb struct {
	Name     string   `json:"name"`
	Synopsis string   `json:"synopsis"`
	Aliases  []string `json:"aliases,omitempty"`
	Lane     string   `json:"lane,omitempty"`
	Doc      string   `json:"doc,omitempty"`
}

// Spellings returns the verb's canonical name plus every alias — the full set of
// argv[1] tokens that route to this verb. The freshness gate joins on this set so a
// main.go `case "a", "b":` with one manifest entry covering both does not red.
func (v Verb) Spellings() []string {
	out := make([]string, 0, 1+len(v.Aliases))
	out = append(out, v.Name)
	out = append(out, v.Aliases...)
	return out
}

// verbManifest is the committed CLI-verb catalog. It is the ONE source `usage.go`
// is meant to be generated from (the cmd/ half of #1290) and the set the freshness
// gate cross-checks the live main.go switch against (#1293). Keep it sorted by Name
// so the rendered usage and the JSON answer are stable.
//
// This catalogs the user-facing top-level verbs of the queryable self-index family
// and its closest dispatch neighbors; the cmd/ generator pass folds the remainder of
// the main.go switch into the same shape. Each entry's Lane is the leaf the handler
// code lives under, resolvable by LaneForPath, so the join cannot drift off-taxonomy.
var verbManifest = []Verb{
	{Name: "index", Synopsis: "queryable self-index: lane/leaf/docs/claims/verbs (query, don't survey)", Lane: "devindex", Doc: "AGENTS.md"},
	{Name: "run", Synopsis: "run an agent turn (or a recorded trace via --trace) through the kernel", Lane: "cmd"},
	{Name: "commit", Synopsis: "commit staged paths with the lane ship-stamp trailer enforced", Lane: "cmd"},
	{Name: "guard", Synopsis: "wrap an agent harness: deny/repair/quarantine proposed tool calls", Lane: "gateway"},
	{Name: "serve", Synopsis: "run the OpenAI-compatible gateway in front of a local or remote model", Lane: "gateway"},
}

// Verbs returns the structured CLI-verb catalog, sorted by name. It is the read side
// of the C3 manifest; the cmd/ usage generator and `fak index verbs` consume it.
func (c *Catalog) Verbs() []Verb {
	out := make([]Verb, len(verbManifest))
	copy(out, verbManifest)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// VerbByName returns the manifest entry matching the given (case-insensitive) token
// against the verb's canonical name OR any alias, and ok=false when nothing routes.
// The freshness gate uses this to ask "does main.go's case <tok> have a manifest
// entry?" without re-deriving the alias set at the call site.
func (c *Catalog) VerbByName(name string) (Verb, bool) {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return Verb{}, false
	}
	for _, v := range verbManifest {
		for _, sp := range v.Spellings() {
			if strings.ToLower(sp) == n {
				return v, true
			}
		}
	}
	return Verb{}, false
}

// SearchVerbs returns the catalog verbs matching the query, lexically scored (a name
// or alias hit weighs most, then the lane, then the synopsis) and ranked best-first.
// An empty query returns the full catalog in name order — `fak index verbs` with no
// term lists every verb, matching the leaf-search convention.
func (c *Catalog) SearchVerbs(query string) []Verb {
	toks := tokens(query)
	if len(toks) == 0 {
		return c.Verbs()
	}
	type scored struct {
		v Verb
		s int
	}
	var hits []scored
	for _, v := range verbManifest {
		names := strings.ToLower(strings.Join(v.Spellings(), " "))
		lane, syn := strings.ToLower(v.Lane), strings.ToLower(v.Synopsis)
		score := 0
		for _, tk := range toks {
			if strings.Contains(names, tk) {
				score += 3
			}
			if strings.Contains(lane, tk) {
				score += 2
			}
			if strings.Contains(syn, tk) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{v, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].v.Name < hits[j].v.Name
	})
	out := make([]Verb, len(hits))
	for i, h := range hits {
		out[i] = h.v
	}
	return out
}
