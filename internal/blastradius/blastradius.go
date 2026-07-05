// Package blastradius is the pure JOIN at the heart of blast-radius containment
// (epic #2712, W3): given a broken package, it computes the AFFECTED SET — the live
// leases and queued issues whose declared tree intersects the broken package's
// DEPENDENCY blast radius (the package plus every package that transitively imports
// it).
//
// Why a separate estimate step. internal/affectedtests already expands a change into
// its dependents (the import-graph blast radius) — but only to pick tests.
// internal/leaseref already knows which trees live workers hold. Nothing joined them,
// so a hold was all-or-nothing guesswork: the fleet could not answer "which in-flight
// agents does this broken tree actually touch?". This package is that join — broken
// tree -> dependents -> the exact leases/issues that intersect — so the fleet can hold
// precisely those and let the disjoint rest run. Scope-hold (W4), single-fixer (W5),
// and the operator card (W7) all consume this affected set; this verb only REPORTS it.
//
// Pure. Estimate takes the import graph, the broken package, the leases, and the
// issues as data and returns the AffectedSet — no I/O, no clock, no graph or lease
// code of its own. It reuses affectedtests.Select for the dependents closure and
// knownbad.TreesIntersect for the tree/glob intersection (the SAME containment rule
// the known-bad ledger W1 uses), so "affected" here means exactly what "matches a
// known-bad tree" means there. The impure shell (cmd/fak/blast.go) gathers the graph
// (go list), the leases (leaseref), and the issue paths.
//
// Namespace contract: the graph nodes, the broken key, and the lease/issue trees must
// all be the same repo-relative form (e.g. "internal/foo") so the intersection is
// meaningful — the shell keys the graph by repo-relative package directory, not by
// import path, for exactly this reason.
package blastradius

import (
	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// Schema is the stable id stamped on the JSON estimate the shell emits.
const Schema = "fak.blast-estimate.v1"

// Lease is one live lease projected into the shape the join reasons over: a lane id
// and the repo-relative tree globs it holds. The shell maps a leaseref.Record
// (ID + TreeGlobs) into this — no lease code lives here.
type Lease struct {
	Lane      string   `json:"lane"`
	TreeGlobs []string `json:"tree_globs"`
}

// Issue is one queued issue projected into the join's shape: an id and the declared
// repo-relative paths it will touch (its "Likely files" pathspec). The shell gathers
// these from the dispatch queue.
type Issue struct {
	ID    string   `json:"id"`
	Paths []string `json:"paths"`
}

// AffectedLease is a lease the join found INSIDE the blast radius, carrying the subset
// of radius trees its globs intersected — the evidence of WHY it is held (what the
// scope-hold and the operator card key on), not a bare boolean.
type AffectedLease struct {
	Lane      string   `json:"lane"`
	TreeGlobs []string `json:"tree_globs"`
	Matched   []string `json:"matched"`
}

// AffectedIssue is the issue-side analogue of AffectedLease.
type AffectedIssue struct {
	ID      string   `json:"id"`
	Paths   []string `json:"paths"`
	Matched []string `json:"matched"`
}

// AffectedSet is the estimate: the dependency blast radius of the broken package, the
// leases and issues that intersect it (the set to HOLD), and the disjoint ones that
// were excluded (the set to let RUN — the witness the done-condition asks for). Every
// slice is non-nil so the JSON always renders arrays, never null.
type AffectedSet struct {
	Broken         string          `json:"broken"`
	Radius         []string        `json:"radius"`
	Leases         []AffectedLease `json:"leases"`
	Issues         []AffectedIssue `json:"issues"`
	ExcludedLeases []string        `json:"excluded_leases"`
	ExcludedIssues []string        `json:"excluded_issues"`
}

// Estimate joins the import graph with the fleet's live work. It expands broken to its
// dependency blast radius — broken plus every package that transitively imports it,
// via affectedtests.Select over the same edges map the shell's go-list fold builds —
// then partitions the leases and issues by whether their declared tree intersects that
// radius (knownbad.TreesIntersect, the W1 containment rule). Affected entries carry the
// matched radius trees; disjoint entries are reported by id in the excluded lists.
//
// broken must be a node key in the graph's namespace (a repo-relative package dir); a
// broken key that is absent from the graph selects just itself, so a lease is affected
// only if it directly touches the broken tree — the conservative, correct floor.
//
// Pure and deterministic: same inputs -> identical AffectedSet.
func Estimate(edges map[string][]string, broken string, leases []Lease, issues []Issue) AffectedSet {
	radius := affectedtests.Select(edges, []string{broken})

	set := AffectedSet{
		Broken:         broken,
		Radius:         radius,
		Leases:         []AffectedLease{},
		Issues:         []AffectedIssue{},
		ExcludedLeases: []string{},
		ExcludedIssues: []string{},
	}

	for _, l := range leases {
		if matched := intersect(l.TreeGlobs, radius); len(matched) > 0 {
			set.Leases = append(set.Leases, AffectedLease{Lane: l.Lane, TreeGlobs: l.TreeGlobs, Matched: matched})
		} else {
			set.ExcludedLeases = append(set.ExcludedLeases, l.Lane)
		}
	}
	for _, is := range issues {
		if matched := intersect(is.Paths, radius); len(matched) > 0 {
			set.Issues = append(set.Issues, AffectedIssue{ID: is.ID, Paths: is.Paths, Matched: matched})
		} else {
			set.ExcludedIssues = append(set.ExcludedIssues, is.ID)
		}
	}
	return set
}

// intersect returns the subset of radius trees that the declared globs intersect, in
// radius order (radius is already sorted by Select, so the result is sorted too).
// Empty means declared is disjoint from the whole radius. The per-tree probe reuses
// knownbad.TreesIntersect — the same directory-prefix containment W1 matches on — and
// keeps the WHY legible: exactly which broken-dependent trees put this lease in the
// hold set. A declared set with no valid tree never matches (nil in, nil out).
func intersect(declared, radius []string) []string {
	if len(declared) == 0 {
		return nil
	}
	var matched []string
	for _, r := range radius {
		if knownbad.TreesIntersect(declared, []string{r}) {
			matched = append(matched, r)
		}
	}
	return matched
}
