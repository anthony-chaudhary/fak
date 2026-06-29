package devindex

// C6 of epic #1287 (#1293): the freshness gate's DRIFT DETECTION. Keep the index
// honest by construction — surface, as named findings, every way the catalog can
// disagree with its live sources:
//
//   - an UndeclaredLeaf: an internal/<X> directory on disk that holds Go files but
//     has no [lanes.trees] lane entry (the same gap internal/hooks.UndeclaredLeaves
//     reds on, recomputed here from the catalog so this tier-1 package stays pure);
//   - a DeadDocLink: an INDEX.md doc-map bullet whose local path no longer resolves
//     on disk (an external http(s) link is not checked here — no network in tier 1);
//   - an UnknownVerb: a `case "<verb>":` in cmd/fak/main.go with no matching entry in
//     the C3 verb manifest (#1290) — the "a verb in main.go with no manifest entry"
//     drift the issue names explicitly.
//
// This file is the DETECTION half (in lane). REDDING THE BUILD on a finding is a CI /
// *_test.go concern that lives outside internal/devindex — out of lane, reported as
// not-yet. A named gap is a finding; a silent gap is the failure this gate kills.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// DriftKind classifies a freshness finding so a caller (a gate test, a scorecard)
// can route or count by category.
type DriftKind string

const (
	// DriftUndeclaredLeaf: an internal/<X> Go package with no dos.toml lane entry.
	DriftUndeclaredLeaf DriftKind = "undeclared-leaf"
	// DriftDeadDocLink: an INDEX.md doc-map entry whose local path is missing.
	DriftDeadDocLink DriftKind = "dead-doc-link"
	// DriftUnknownVerb: a main.go switch case with no C3 verb-manifest entry.
	DriftUnknownVerb DriftKind = "unknown-verb"
)

// Drift is one freshness finding: the kind, the offending token (a leaf name, a doc
// path, or a verb), and a one-line human reason. A non-empty Drift slice from
// CheckFreshness is the signal the (out-of-lane) gate test turns into a red build.
type Drift struct {
	Kind    DriftKind `json:"kind"`
	Subject string    `json:"subject"`
	Reason  string    `json:"reason"`
}

// CheckFreshness compares the loaded catalog against its live sources on disk and
// returns every drift finding, sorted (by kind, then subject) for a stable gate
// message. An empty slice means the index agrees with reality — the green state the
// gate exists to enforce. It reads only the tree under c.Root; no network.
//
// It folds three detectors: undeclared leaves, dead local doc links, and main.go
// verb cases missing from the C3 manifest. A source it cannot read (e.g. no main.go)
// contributes no finding rather than an error — a missing source is the absence of a
// claim, not a drift. The detail accessors below back each fold and are exported so a
// gate can report just the category it cares about.
func (c *Catalog) CheckFreshness() []Drift {
	var out []Drift
	for _, leaf := range c.UndeclaredLeaves() {
		out = append(out, Drift{
			Kind:    DriftUndeclaredLeaf,
			Subject: leaf,
			Reason:  "internal/" + leaf + " holds Go files but has no [lanes.trees] lane entry",
		})
	}
	for _, d := range c.DeadDocLinks() {
		out = append(out, Drift{
			Kind:    DriftDeadDocLink,
			Subject: d.Path,
			Reason:  "doc-map entry " + d.Title + " points at " + d.Path + " which no longer exists",
		})
	}
	for _, verb := range c.UndeclaredVerbs() {
		out = append(out, Drift{
			Kind:    DriftUnknownVerb,
			Subject: verb,
			Reason:  `cmd/fak/main.go case "` + verb + `" has no C3 verb-manifest entry`,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Subject < out[j].Subject
	})
	return out
}

// UndeclaredLeaves returns the names of internal/<X> directories that hold at least
// one .go file but have no declared [lanes.trees] lane. It mirrors
// internal/hooks.UndeclaredLeaves, recomputed from this catalog's already-parsed lane
// set so the tier-1 package need not import the hooks gate. Sorted, deduped.
func (c *Catalog) UndeclaredLeaves() []string {
	declared := map[string]bool{}
	for _, l := range c.Leaves {
		declared[l.Name] = true
	}
	dir := filepath.Join(c.Root, "internal")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var gaps []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := strings.ToLower(e.Name())
		if declared[name] {
			continue
		}
		if !dirHasGoFiles(filepath.Join(dir, e.Name())) {
			continue // not a Go package (testdata/doc dir): not a leaf
		}
		gaps = append(gaps, name)
	}
	sort.Strings(gaps)
	return gaps
}

// dirHasGoFiles reports whether dir directly contains at least one .go file. It
// mirrors internal/hooks.dirHasGoFiles so the undeclared-leaf rule matches the gate's.
func dirHasGoFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
			return true
		}
	}
	return false
}

// DeadDocLinks returns the doc-map entries whose path is a LOCAL repo path (not an
// http(s) URL) that no longer resolves under c.Root. An external URL is left
// unchecked — tier 1 does no network — and an in-page anchor ("#foo") is skipped.
func (c *Catalog) DeadDocLinks() []Doc {
	var dead []Doc
	for _, d := range c.Docs {
		p := strings.TrimSpace(d.Path)
		if p == "" || strings.HasPrefix(p, "#") {
			continue
		}
		if strings.HasPrefix(p, "http://") || strings.HasPrefix(p, "https://") || strings.HasPrefix(p, "mailto:") {
			continue // external / non-file target: not ours to check
		}
		// Strip a trailing #anchor or ?query so the on-disk check sees a real path.
		clean := p
		if i := strings.IndexAny(clean, "#?"); i >= 0 {
			clean = clean[:i]
		}
		if clean == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(c.Root, filepath.FromSlash(clean))); err != nil {
			dead = append(dead, d)
		}
	}
	return dead
}

// mainCaseRE captures the quoted verb tokens of a `case "a", "b":` line in a Go
// switch. We match each quoted string on a line that begins with `case` and ends
// with `:` so a string literal inside a handler body is never mistaken for a verb.
var mainCaseRE = regexp.MustCompile(`"([^"]+)"`)

// UndeclaredVerbs returns the cmd/fak/main.go top-level switch cases that have no
// entry in the C3 verb manifest — the "a verb in main.go with no manifest entry"
// drift (#1293). It parses the dispatch switch out of main.go on disk (read-only);
// a missing main.go yields no findings (absence of a claim, not a drift). Sorted,
// deduped, lowercased.
func (c *Catalog) UndeclaredVerbs() []string {
	b, err := os.ReadFile(filepath.Join(c.Root, "cmd", "fak", "main.go"))
	if err != nil {
		return nil
	}
	known := map[string]bool{}
	for _, v := range verbManifest {
		for _, sp := range v.Spellings() {
			known[strings.ToLower(sp)] = true
		}
	}
	seen := map[string]bool{}
	var out []string
	inSwitch := false
	for _, raw := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(raw)
		// Only scan the top-level os.Args dispatch switch, so a nested switch in a
		// helper does not feed the verb set.
		if !inSwitch {
			if strings.HasPrefix(t, "switch os.Args[1]") {
				inSwitch = true
			}
			continue
		}
		if t == "}" || strings.HasPrefix(t, "default:") {
			break // end of the dispatch switch
		}
		if !strings.HasPrefix(t, "case ") || !strings.HasSuffix(t, ":") {
			continue
		}
		for _, m := range mainCaseRE.FindAllStringSubmatch(t, -1) {
			verb := strings.ToLower(m[1])
			if verb == "" || known[verb] || seen[verb] {
				continue
			}
			seen[verb] = true
			out = append(out, verb)
		}
	}
	sort.Strings(out)
	return out
}
