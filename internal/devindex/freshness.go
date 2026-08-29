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

// Verdict is a freshness probe's answer, and it carries THREE values rather than two
// (#5962). "I checked and found no drift" and "I could not check" are both non-stale,
// yet only the first may be reported as fresh: a probe that hands back the reassuring
// answer for a detector that never ran is the tool lying to its own operator. Only
// VerdictStale is evidence of drift, so only it may drive a gate or a self-heal.
type Verdict string

const (
	// VerdictFresh: every detector ran and none of them found drift.
	VerdictFresh Verdict = "fresh"
	// VerdictStale: at least one detector PROVED drift. The only actionable verdict.
	VerdictStale Verdict = "stale"
	// VerdictUnknown: nothing proved drift, but at least one detector could not run,
	// so the tree may disagree with the index in a way nothing looked at.
	VerdictUnknown Verdict = "unknown"
)

// Unchecked is one detector that could not run: which detector, the repo-relative
// source it needed, and the read error that stopped it. It is deliberately NOT a
// Drift — an unread source is not evidence that the index disagrees with the tree,
// and folding it into the drift slice would red a build over a check that never
// happened. It is what lets a report be non-stale without being fresh.
type Unchecked struct {
	Detector DriftKind `json:"detector"`
	Source   string    `json:"source"`
	Reason   string    `json:"reason"`
}

// FreshnessReport is the honest form of the freshness fold: the drift findings AND
// the detectors that never got to look. CheckFreshness's []Drift cannot express the
// second, so an empty slice from it means only "no drift was PROVEN" — reading that
// as "the index agrees with reality" is exactly the confusion this report removes.
type FreshnessReport struct {
	Drifts    []Drift     `json:"drifts,omitempty"`
	Unchecked []Unchecked `json:"unchecked,omitempty"`
}

// Verdict folds the report to one word. Stale outranks unknown (proven drift is the
// actionable answer even when some other detector also failed to run) and unknown
// outranks fresh (an unrun detector may be hiding drift nobody looked for).
func (r FreshnessReport) Verdict() Verdict {
	switch {
	case len(r.Drifts) > 0:
		return VerdictStale
	case len(r.Unchecked) > 0:
		return VerdictUnknown
	default:
		return VerdictFresh
	}
}

// Fresh reports whether every detector ran and none found drift — the one state a
// caller may read as "the index is current". It is a method, not a field, so no
// caller can hand out a report claiming a freshness it did not earn.
func (r FreshnessReport) Fresh() bool { return r.Verdict() == VerdictFresh }

// CheckFreshness returns the drift findings only, sorted (by kind, then subject) for
// a stable gate message. An empty slice means NO DRIFT WAS PROVEN — which is not the
// same as "the index agrees with reality", because a detector whose source could not
// be read proves nothing at all. Callers that need that distinction (anything that
// reports a verdict to a human or a machine) must use CheckFreshnessReport and its
// Verdict; callers that only count proven drift, such as a catch-up score, are right
// to use this projection. It reads only the tree under c.Root; no network.
func (c *Catalog) CheckFreshness() []Drift { return c.CheckFreshnessReport().Drifts }

// CheckFreshnessReport compares the loaded catalog against its live sources on disk.
//
// It folds five detectors: undeclared leaves, dead INDEX.md doc links, main.go verb
// cases missing from the C3 manifest, orphaned dated notes, and dead llms.txt links.
// A source it cannot read still contributes no DRIFT finding — a missing source is
// the absence of a claim, not a disagreement — but it is now recorded as an
// Unchecked, so the detector that skipped it can never be mistaken for one that ran
// and found the tree clean (#5962). The detail accessors below back each fold and are
// exported so a gate can report just the category it cares about; they keep their
// drift-only signatures, so a caller that wants the unchecked half asks here.
func (c *Catalog) CheckFreshnessReport() FreshnessReport {
	var rep FreshnessReport
	note := func(u *Unchecked) {
		if u != nil {
			rep.Unchecked = append(rep.Unchecked, *u)
		}
	}

	leaves, leavesUnchecked := c.undeclaredLeaves()
	note(leavesUnchecked)
	for _, leaf := range leaves {
		rep.Drifts = append(rep.Drifts, Drift{
			Kind:    DriftUndeclaredLeaf,
			Subject: leaf,
			Reason:  "internal/" + leaf + " holds Go files but has no [lanes.trees] lane entry",
		})
	}
	for _, d := range c.DeadDocLinks() {
		rep.Drifts = append(rep.Drifts, Drift{
			Kind:    DriftDeadDocLink,
			Subject: d.Path,
			Reason:  "doc-map entry " + d.Title + " points at " + d.Path + " which no longer exists",
		})
	}
	verbs, verbsUnchecked := c.undeclaredVerbs()
	note(verbsUnchecked)
	for _, verb := range verbs {
		rep.Drifts = append(rep.Drifts, Drift{
			Kind:    DriftUnknownVerb,
			Subject: verb,
			Reason:  `cmd/fak/main.go case "` + verb + `" has no C3 verb-manifest entry`,
		})
	}
	notes, notesUnchecked := c.orphanNotes()
	note(notesUnchecked)
	if notesUnchecked != nil && notesUnchecked.Source == "INDEX.md" {
		// DeadDocLinks walks c.Docs, which Load parsed from this same INDEX.md and
		// silently left EMPTY when it could not read it (Load degrades a missing
		// INDEX.md to an empty doc map by design). Without this the doc-link detector
		// would report zero dead links for a doc map it never had — a false green from
		// a check that never happened, which is the whole point of this report.
		note(&Unchecked{Detector: DriftDeadDocLink, Source: notesUnchecked.Source, Reason: notesUnchecked.Reason})
	}
	for _, n := range notes {
		rep.Drifts = append(rep.Drifts, Drift{
			Kind:    DriftOrphanNote,
			Subject: n,
			Reason:  n + " is a dated note under docs/notes/ but is not listed in INDEX.md",
		})
	}
	links, linksUnchecked := c.deadLLMSLinks()
	note(linksUnchecked)
	for _, link := range links {
		rep.Drifts = append(rep.Drifts, Drift{
			Kind:    DriftDeadLLMSLink,
			Subject: link,
			Reason:  "llms.txt links " + link + " which no longer exists on disk",
		})
	}
	sort.SliceStable(rep.Drifts, func(i, j int) bool {
		if rep.Drifts[i].Kind != rep.Drifts[j].Kind {
			return rep.Drifts[i].Kind < rep.Drifts[j].Kind
		}
		return rep.Drifts[i].Subject < rep.Drifts[j].Subject
	})
	sort.SliceStable(rep.Unchecked, func(i, j int) bool {
		if rep.Unchecked[i].Detector != rep.Unchecked[j].Detector {
			return rep.Unchecked[i].Detector < rep.Unchecked[j].Detector
		}
		return rep.Unchecked[i].Source < rep.Unchecked[j].Source
	})
	return rep
}

// UndeclaredLeaves returns the names of internal/<X> directories that hold at least
// one .go file but have no declared dos.toml lane. It mirrors
// internal/hooks.UndeclaredLeaves, recomputed from this catalog's already-parsed lane
// set (c.declared — every name in [lanes] AND every [lanes.trees] key, the SAME set
// the authoritative gate builds) plus the explicit tree resolver, so the tier-1
// package need not import the hooks gate yet reaches the identical verdict (pinned by
// a live parity test). A leaf is owned when its package name is declared or an authored
// tree maps its files to a differently named composite lane. Sorted, deduped.
//
// A tree it cannot read yields no gaps. That is the drift-only view; ask
// CheckFreshnessReport if you need to tell "no gaps" apart from "never looked".
func (c *Catalog) UndeclaredLeaves() []string { g, _ := c.undeclaredLeaves(); return g }

// ExplicitTreeLaneForPath resolves only authored [lanes.trees] exact paths and
// prefixes. Unlike LaneForPath it deliberately has no internal/<package>
// convention fallback, so a live-census parity test can prove that ownership is
// present in the taxonomy rather than merely infer the package name. This is the
// strict read used by the #9326 current-package reconciliation witness.
func (c *Catalog) ExplicitTreeLaneForPath(path string) string {
	p := strings.ToLower(normPath(path))
	if lane, ok := c.exact[p]; ok {
		return lane
	}
	best, owner := "", ""
	for prefix, lane := range c.prefixes {
		if strings.HasPrefix(p, prefix) && len(prefix) > len(best) {
			best, owner = prefix, lane
		}
	}
	return owner
}

// undeclaredLeaves is UndeclaredLeaves plus the reason it could not finish, if any:
// an unreadable internal/ (nothing was scanned at all) or an unreadable package
// directory (a leaf that might hold Go files was skipped). First failure wins — one
// named unchecked source is enough to disqualify a fresh verdict, and the scan keeps
// going so proven drift elsewhere still outranks it.
func (c *Catalog) undeclaredLeaves() ([]string, *Unchecked) {
	dir := filepath.Join(c.Root, "internal")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &Unchecked{Detector: DriftUndeclaredLeaf, Source: "internal", Reason: err.Error()}
	}
	var gaps []string
	var unchecked *Unchecked
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := strings.ToLower(e.Name())
		if c.declared[name] {
			continue
		}
		hasGo, readErr := dirHasGoFiles(filepath.Join(dir, e.Name()))
		if readErr != nil {
			if unchecked == nil {
				unchecked = &Unchecked{Detector: DriftUndeclaredLeaf, Source: "internal/" + e.Name(), Reason: readErr.Error()}
			}
			continue
		}
		if !hasGo {
			continue // not a Go package (testdata/doc dir): not a leaf
		}
		// A composite lane may deliberately own a differently named package
		// (internal/study is part of studyreceipt). Resolve the authored tree before
		// calling the package undeclared; requiring owner == basename would report a
		// gap even though arbitration already has an exact region for every file.
		probe := filepath.ToSlash(filepath.Join("internal", e.Name(), "_lane_ownership.go"))
		if c.ExplicitTreeLaneForPath(probe) != "" {
			continue
		}
		gaps = append(gaps, name)
	}
	sort.Strings(gaps)
	return gaps, unchecked
}

// dirHasGoFiles reports whether dir directly contains at least one .go file. It
// mirrors internal/hooks.dirHasGoFiles so the undeclared-leaf rule matches the gate's.
// The read error is returned rather than folded into a false — "no Go files here" and
// "could not look" are different answers and only the first is a decision (#5962).
func dirHasGoFiles(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, err
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
			return true, nil
		}
	}
	return false, nil
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
func (c *Catalog) OrphanNotes() []string { n, _ := c.orphanNotes(); return n }

// orphanNotes is OrphanNotes plus the reason it could not finish, if any: an
// unreadable INDEX.md (there is no map to reconcile against, so no note can be shown
// reachable) or an unreadable docs/notes subtree (notes that might be orphans were
// never enumerated). The walk continues past a bad subtree so proven orphans
// elsewhere are still reported — stale outranks unknown.
func (c *Catalog) orphanNotes() ([]string, *Unchecked) {
	idx, err := os.ReadFile(filepath.Join(c.Root, "INDEX.md"))
	if err != nil {
		return nil, &Unchecked{Detector: DriftOrphanNote, Source: "INDEX.md", Reason: err.Error()}
	}
	text := string(idx)
	notesDir := filepath.Join(c.Root, "docs", "notes")
	var out []string
	var unchecked *Unchecked
	_ = filepath.WalkDir(notesDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// An unreadable subtree contributes no finding — but it is recorded, so a
			// walk that never saw the notes cannot pass for one that saw them all.
			if unchecked == nil {
				rel, relErr := filepath.Rel(c.Root, path)
				if relErr != nil {
					rel = path
				}
				unchecked = &Unchecked{Detector: DriftOrphanNote, Source: filepath.ToSlash(rel), Reason: err.Error()}
			}
			return nil
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
	return out, unchecked
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
func (c *Catalog) DeadLLMSLinks() []string { l, _ := c.deadLLMSLinks(); return l }

// deadLLMSLinks is DeadLLMSLinks plus the reason it could not finish: an llms.txt
// this process cannot read means the answer-engine map was never scanned, which is
// not the same answer as "every link in it resolves".
func (c *Catalog) deadLLMSLinks() ([]string, *Unchecked) {
	b, err := os.ReadFile(filepath.Join(c.Root, "llms.txt"))
	if err != nil {
		return nil, &Unchecked{Detector: DriftDeadLLMSLink, Source: "llms.txt", Reason: err.Error()}
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
	return dead, nil
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
func (c *Catalog) UndeclaredVerbs() []string { v, _ := c.undeclaredVerbs(); return v }

// undeclaredVerbs is UndeclaredVerbs plus the reason it could not finish. This is the
// detector the issue named: an unreadable cmd/fak/main.go used to read as green, so
// the probe reported "no uncataloged verbs" for a switch it never parsed (#5962).
func (c *Catalog) undeclaredVerbs() ([]string, *Unchecked) {
	b, err := os.ReadFile(filepath.Join(c.Root, "cmd", "fak", "main.go"))
	if err != nil {
		return nil, &Unchecked{Detector: DriftUnknownVerb, Source: "cmd/fak/main.go", Reason: err.Error()}
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
	return out, nil
}

// DispatchVerbs is the exported form of mainDispatchVerbs: the lowercased dispatch
// tokens parsed out of cmd/fak/main.go bytes (sorted, deduped). It exists so the
// pre-push VERB_UNTIERED hygiene gate (internal/hooks/gate_verbtier.go) reads the
// verb set through the SAME parser TestVerbTierCoverageIsTotal uses, rather than a
// second copy — the gate can then never disagree with the CI ratchet it fronts about
// which tokens the switch actually dispatches (epic #2653).
func DispatchVerbs(b []byte) []string { return mainDispatchVerbs(b) }

// mainDispatchVerbs returns the lowercased quoted verb tokens of every top-level
// dispatch switch in the given main.go bytes (sorted, deduped). It recognizes both
// the legacy `switch os.Args[1]` and extracted `switch name` helper form. It tracks
// brace DEPTH from the `switch os.Args[1] {` line, so a case body that itself contains
// braces — e.g. the Landlock trampoline's `if err != nil { … }` — does not end the
// scan early: a verb whose case sits AFTER such a body is still seen (the bug a naive
// "break on the first `}`" scan had, which silently truncated the verb set at the first
// brace-bearing case). Cases are read only at the switch's own depth; the scan ends at
// that switch's `default:` or its closing brace. Shared by UndeclaredVerbs and the
// freshness test so the detector and its dogfood test cannot disagree.
func mainDispatchVerbs(b []byte) []string {
	return dispatchVerbs(b, func(t string) bool {
		return strings.HasPrefix(t, "switch os.Args[1]") || strings.HasPrefix(t, "switch name")
	})
}

// devDispatchVerbs is the same scan against the `fak-dev` artifact, whose dispatch
// switch opens on its own `argv[0]` slice rather than `os.Args[1]`. It is a separate
// opener rather than a third alternative inside mainDispatchVerbs because `cmd/fak`
// ALSO contains a nested `switch argv[0]` that is not a dispatch surface; folding the
// two together would read that nested switch's cases as runtime verbs.
func devDispatchVerbs(b []byte) []string {
	return dispatchVerbs(b, func(t string) bool { return strings.HasPrefix(t, "switch argv[0]") })
}

// dispatchVerbs carries the shared scan; isOpener decides which switch header starts
// a dispatch surface for the artifact being read.
func dispatchVerbs(b []byte, isOpener func(string) bool) []string {
	seen := map[string]bool{}
	var out []string
	inSwitch := false
	depth := 0
	for _, raw := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(raw)
		if !inSwitch {
			if isOpener(t) {
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
				inSwitch = false
				depth = 0
				continue // this dispatch switch ended; keep scanning for extracted helpers
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
			inSwitch = false // this dispatch switch closed; keep scanning for extracted helpers
		}
	}
	sort.Strings(out)
	return out
}
