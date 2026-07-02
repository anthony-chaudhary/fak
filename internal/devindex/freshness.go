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
//     drift the issue names explicitly;
//   - an OrphanNote: a dated note under docs/notes/ that INDEX.md never mentions —
//     the tree->index converse of a dead doc link (INDEX.md's own contract is "if a
//     doc exists, it is reachable from here"). This is the ORPHAN half of the Python
//     reciprocal sync gate (tools/check_index_sync.py) brought into the pure Go view,
//     so the whole self-index drift surface is answerable from one queryable place;
//   - a DeadLLMSLink: a local .md link in llms.txt (the answer-engine index) that no
//     longer resolves — the same dangling check DeadDocLinks does for INDEX.md, applied
//     to the LLM-facing map so a dead link in the index answer engines read is caught.
//
// This file is the DETECTION half (in lane). REDDING THE BUILD on a finding is a CI /
// *_test.go concern that lives outside internal/devindex — out of lane, reported as
// not-yet. A named gap is a finding; a silent gap is the failure this gate kills.

import (
	"io/fs"
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
	// DriftOrphanNote: a dated docs/notes/ note not listed in INDEX.md.
	DriftOrphanNote DriftKind = "orphan-note"
	// DriftDeadLLMSLink: an llms.txt local .md link that no longer resolves on disk.
	DriftDeadLLMSLink DriftKind = "dead-llms-link"
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
// It folds five detectors: undeclared leaves, dead INDEX.md doc links, main.go verb
// cases missing from the C3 manifest, orphaned dated notes, and dead llms.txt links.
// A source it cannot read (e.g. no main.go) contributes no finding rather than an
// error — a missing source is the absence of a claim, not a drift. The detail accessors
// below back each fold and are exported so a gate can report just the category it cares
// about.
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
	for _, note := range c.OrphanNotes() {
		out = append(out, Drift{
			Kind:    DriftOrphanNote,
			Subject: note,
			Reason:  note + " is a dated note under docs/notes/ but is not listed in INDEX.md",
		})
	}
	for _, link := range c.DeadLLMSLinks() {
		out = append(out, Drift{
			Kind:    DriftDeadLLMSLink,
			Subject: link,
			Reason:  "llms.txt links " + link + " which no longer exists on disk",
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
// one .go file but have no declared dos.toml lane. It mirrors
// internal/hooks.UndeclaredLeaves, recomputed from this catalog's already-parsed lane
// set (c.declared — every name in [lanes] AND every [lanes.trees] key, the SAME set
// the authoritative gate builds) so the tier-1 package need not import the hooks gate
// yet reaches the identical verdict (pinned by a live parity test). A leaf declared in
// [lanes] with no explicit tree is declared, NOT drift; counting only [lanes.trees]
// keys would falsely flag it. Sorted, deduped.
func (c *Catalog) UndeclaredLeaves() []string {
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
		if c.declared[name] {
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

// datedNoteRE matches the ISO date a working note carries in its filename, e.g.
// docs/notes/CONCEPT-…-2026-06-30.md. It mirrors the stamp check in the Python
// reciprocal sync gate (tools/check_index_sync.py) so the two agree on what a
// "dated note" is.
var datedNoteRE = regexp.MustCompile(`20\d\d-\d\d-\d\d`)

// isDatedNote reports whether a docs/notes basename is a dated working note that
// INDEX.md is expected to list: a `.md` file (not a README) whose name carries a
// YYYY-MM-DD stamp or starts with the `PLAN-` prefix. Same predicate as the Python
// gate's _is_dated_note, so a note the pre-commit hook flags and one this view
// flags are the same set.
func isDatedNote(base string) bool {
	if base == "README.md" || !strings.HasSuffix(base, ".md") {
		return false
	}
	return datedNoteRE.MatchString(base) || strings.HasPrefix(base, "PLAN-")
}

// OrphanNotes returns the repo-relative paths of dated notes under docs/notes/
// whose basename INDEX.md never mentions — the tree->index converse of
// DeadDocLinks. INDEX.md's own contract is "if a doc exists, it is reachable from
// here", so an unlisted dated note breaks it. The check is a raw-basename substring
// test against INDEX.md's bytes (a note may be reached via prose, not only a link),
// matching tools/check_index_sync.py exactly so the Go view and the Python gate can
// never disagree on an orphan. A missing INDEX.md yields nothing (no map to
// reconcile against). Sorted, deduped; reads only the tree under c.Root — no git,
// no network (tier 1).
func (c *Catalog) OrphanNotes() []string {
	idx, err := os.ReadFile(filepath.Join(c.Root, "INDEX.md"))
	if err != nil {
		return nil
	}
	text := string(idx)
	notesDir := filepath.Join(c.Root, "docs", "notes")
	var out []string
	_ = filepath.WalkDir(notesDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // an unreadable subtree contributes no finding, not an error
		}
		if d.IsDir() {
			return nil
		}
		if !isDatedNote(d.Name()) {
			return nil
		}
		if strings.Contains(text, d.Name()) {
			return nil // referenced somewhere in INDEX.md — reachable, not an orphan
		}
		rel, relErr := filepath.Rel(c.Root, path)
		if relErr != nil {
			rel = path
		}
		out = append(out, filepath.ToSlash(rel))
		return nil
	})
	sort.Strings(out)
	return out
}

// llmsLinkRE captures every markdown link target `](target)`. llms.txt carries inline
// prose links, not just the INDEX.md bullet shape docLineRE matches, so this scans ALL
// links — the same LINK_RE the Python reciprocal gate uses.
var llmsLinkRE = regexp.MustCompile(`\]\(([^)]+)\)`)

// DeadLLMSLinks returns the local .md link targets in llms.txt (the answer-engine
// index) that no longer resolve on disk — the dangling half of the reciprocal sync
// gate applied to the LLM-facing map, which DeadDocLinks (INDEX.md only) does not
// cover. It mirrors tools/check_index_sync.py's link filter exactly: an http(s) /
// mailto / in-page anchor / absolute-path target is skipped, a trailing #anchor or
// ?query is stripped, and only a .md target is checked. Deduped (by cleaned path),
// sorted. A missing llms.txt yields nothing — no map to check. Reads only c.Root; no
// network (tier 1: an external URL is never fetched, only skipped).
func (c *Catalog) DeadLLMSLinks() []string {
	b, err := os.ReadFile(filepath.Join(c.Root, "llms.txt"))
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var dead []string
	for _, m := range llmsLinkRE.FindAllStringSubmatch(string(b), -1) {
		target := strings.TrimSpace(m[1])
		if target == "" {
			continue
		}
		if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") ||
			strings.HasPrefix(target, "mailto:") || strings.HasPrefix(target, "#") ||
			strings.HasPrefix(target, "/") {
			continue // external / anchor / absolute: not a local file this gate owns
		}
		clean := target
		if i := strings.IndexAny(clean, "#?"); i >= 0 {
			clean = clean[:i]
		}
		if clean == "" || !strings.HasSuffix(clean, ".md") || seen[clean] {
			continue
		}
		seen[clean] = true
		if _, err := os.Stat(filepath.Join(c.Root, filepath.FromSlash(clean))); err != nil {
			dead = append(dead, clean)
		}
	}
	sort.Strings(dead)
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
	var out []string
	for _, verb := range mainDispatchVerbs(b) {
		if !known[verb] {
			out = append(out, verb)
		}
	}
	sort.Strings(out)
	return out
}

// mainDispatchVerbs returns the lowercased quoted verb tokens of the top-level
// os.Args[1] dispatch switch in the given main.go bytes (sorted, deduped). It tracks
// brace DEPTH from the `switch os.Args[1] {` line, so a case body that itself contains
// braces — e.g. the Landlock trampoline's `if err != nil { … }` — does not end the
// scan early: a verb whose case sits AFTER such a body is still seen (the bug a naive
// "break on the first `}`" scan had, which silently truncated the verb set at the first
// brace-bearing case). Cases are read only at the switch's own depth; the scan ends at
// that switch's `default:` or its closing brace. Shared by UndeclaredVerbs and the
// freshness test so the detector and its dogfood test cannot disagree.
func mainDispatchVerbs(b []byte) []string {
	seen := map[string]bool{}
	var out []string
	inSwitch := false
	depth := 0
	for _, raw := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(raw)
		if !inSwitch {
			if strings.HasPrefix(t, "switch os.Args[1]") {
				inSwitch = true
				depth = 1 // the `{` that opens the dispatch switch
			}
			continue
		}
		// Only the switch's own depth carries dispatch cases; a nested brace body is
		// skipped. Evaluate the line as a case BEFORE folding in its braces (a case
		// line carries none anyway).
		if depth == 1 {
			if strings.HasPrefix(t, "default:") {
				break // the dispatch switch's default arm — end of the verb set
			}
			if strings.HasPrefix(t, "case ") && strings.HasSuffix(t, ":") {
				for _, m := range mainCaseRE.FindAllStringSubmatch(t, -1) {
					verb := strings.ToLower(m[1])
					if verb != "" && !seen[verb] {
						seen[verb] = true
						out = append(out, verb)
					}
				}
			}
		}
		depth += strings.Count(t, "{") - strings.Count(t, "}")
		if depth <= 0 {
			break // the dispatch switch block closed
		}
	}
	sort.Strings(out)
	return out
}
