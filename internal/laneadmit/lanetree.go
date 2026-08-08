package laneadmit

import (
	"sort"
	"strings"
)

// lanetree.go — the hierarchical half of the lane namespace. laneadmit.go decides admission
// against a FLAT lane vocabulary: every dos.toml `[lanes]` token is an opaque string, two lanes
// are "the same lane" only when the strings are byte-equal, and the whole addressable space is
// however many tokens a human typed into dos.toml (542 at the time of writing). That flat space
// is the concurrency ceiling: `same_lane` serializes, so a lane is a mutex and the fleet can hold
// at most one worker per declared token no matter how disjoint their real edits are.
//
// This file makes the lane name PATH-SHAPED so the space is DERIVED from the tree instead of
// enumerated by hand. `gateway` stays the lane it always was; `gateway/server` and
// `gateway/server.go` become addressable WITHOUT a dos.toml row, deriving their tree from the
// nearest declared ancestor. dos.toml keeps declaring the ~542 roots — the sub-lanes below them
// come for free and track the repo as it grows.
//
// Two invariants make this shippable on a live fleet:
//
//  1. BACKWARD COMPATIBLE BY CONSTRUCTION. Every declared lane is a single `[a-z0-9]+` segment
//     (verified: none of the 542 carries `/`, `.`, `-` or `_`). A lane name with no separator has
//     exactly one segment, so LaneContains degenerates to string equality and LanesConflict
//     degenerates to the `lane == req.Lane` test Decide always ran. Flat verdicts stay
//     byte-identical.
//
//  2. A SUB-LANE IS NEVER MORE PERMISSIVE THAN ITS PARENT. `gateway` contains `gateway/server`,
//     so a worker holding the parent still blocks every child (ConflictLaneAncestry), exclusivity
//     inherits DOWNWARD, and a tree that cannot be derived falls back to the ancestor's coarse
//     tree. Narrowing can only ever be opted INTO, never fallen into.
//
// The file stays pure — no clock, no I/O, no filesystem walk — exactly like Decide. LaneSpace
// takes the path list from a caller that already has one (`git ls-files`, a tree walk) so the
// enumeration is testable and the derivation is the same code the admission path runs.

// LaneSep separates a lane's segments in its canonical, human-facing form. It is `/` because a
// lane IS a path prefix — `gateway/server` names exactly `internal/gateway/server`, so the lane a
// worker types and the tree it gets are the same string modulo the declared root.
const LaneSep = "/"

// laneWireSep encodes LaneSep inside a lease id. It is NOT `/`: a lease id becomes the final
// path segment of `refs/fak/locks/<id>`, and internal/leaseref's validID rejects `/` outright
// (leaseref.go: "no '/', no whitespace [...] keep it one safe segment") precisely so an id cannot
// escape its ref namespace. A `/` here would also collide with git's directory/file ref rule —
// `refs/fak/locks/x` and `refs/fak/locks/x/y` cannot coexist, so leasing a parent lane would make
// every child lane unleasable.
//
// `_` is ref-legal and unused by all 542 declared lanes — but a FILE sub-lane is full of them
// (`cmd/fak/dispatch_wave.go`), and so is every other character validID permits. So a literal
// separator byte inside a segment must be ESCAPED, using laneWireEsc.
const laneWireSep = "_"

// laneWireEsc escapes a literal separator inside a lease-id segment: `-` doubles to `--` and a
// literal `_` becomes `-u`, leaving a lone `_` unambiguously the separator.
//
// Escaping with a DISTINCT character is load-bearing. The obvious scheme — double the separator
// itself, `_` -> `__` — is ambiguous and was measured wrong on this repo: `docs/_config.yml`
// encodes to `docs___config.yml`, and a run of three `_` can be read as (separator)(literal) or
// (literal)(separator). It decoded to `docs_/config.yml`, one of 17 real tracked paths that
// round-tripped to the wrong lane. A separate escape character has no such run.
const laneWireEsc = "-"

// CanonicalLane normalizes a lane name to its canonical hierarchical form: `\` folded to LaneSep,
// `.`/`..`/empty segments dropped, leading and trailing separators trimmed.
//
// It deliberately does NOT lowercase. A lane name is path-shaped, and at file granularity the tail
// IS a real path (`claude/skills/verify/SKILL.md`) that TreeFor turns back into a glob — folding
// its case would produce a tree that matches nothing on a case-sensitive filesystem. Comparison
// is case-insensitive instead (foldLane), so the declared lowercase vocabulary still matches any
// spelling a caller types.
//
// A flat name round-trips unchanged, which is what keeps every existing caller byte-identical.
func CanonicalLane(s string) string {
	segs := LaneSegments(s)
	if len(segs) == 0 {
		return ""
	}
	return strings.Join(segs, LaneSep)
}

// LaneSegments splits a lane into its non-empty segments, dropping the `.` and `..` traversal
// segments so a canonical lane can never escape its declared root. A leading dot is otherwise
// PRESERVED — `.claude` and `.gitignore` are real repo paths and a lane tail must be able to name
// them.
func LaneSegments(lane string) []string {
	var segs []string
	for _, seg := range strings.Split(strings.ReplaceAll(strings.TrimSpace(lane), "\\", LaneSep), LaneSep) {
		switch seg = strings.TrimSpace(seg); seg {
		case "", ".", "..":
			continue
		}
		segs = append(segs, seg)
	}
	return segs
}

// foldLane is the comparison form of a lane: canonical, then case-folded. Every equality test,
// map lookup and containment check goes through it, so `Gateway/Server` and `gateway/server` are
// one lane while the value a caller gets back keeps the case its path had.
func foldLane(s string) string { return strings.ToLower(CanonicalLane(s)) }

// LaneDepth reports how many segments a lane carries: 1 for every flat dos.toml lane, 2 for a
// sub-package lane, 3+ for a file lane under a sub-package. Depth is the multiplier knob — the
// addressable space at depth N is the number of distinct path prefixes of length N.
func LaneDepth(lane string) int { return len(LaneSegments(CanonicalLane(lane))) }

// LaneAncestors returns the canonical lane followed by each proper ancestor, nearest first:
// "gateway/server/handler.go" -> ["gateway/server/handler.go", "gateway/server", "gateway"].
// The nearest-first order is what lets DeclaredAncestor stop at the FIRST hit, which is the most
// specific declaration and therefore the tightest honest tree.
func LaneAncestors(lane string) []string {
	segs := LaneSegments(CanonicalLane(lane))
	out := make([]string, 0, len(segs))
	for i := len(segs); i > 0; i-- {
		out = append(out, strings.Join(segs[:i], LaneSep))
	}
	return out
}

// LaneContains reports whether outer is inner, or a proper ancestor of it. Containment is
// SEGMENT-wise, never raw string prefix: `gate` does not contain `gateway`, and `gateway` does
// contain `gateway/server`. Both sides are canonicalized, so the wire and canonical spellings of
// the same lane compare equal.
func LaneContains(outer, inner string) bool {
	o, i := foldLane(outer), foldLane(inner)
	if o == "" || i == "" {
		return false
	}
	return o == i || strings.HasPrefix(i, o+LaneSep)
}

// LanesConflict reports whether two lane names must serialize against each other. Two lanes
// conflict when either contains the other: a parent's holder blocks every descendant (it may edit
// anywhere beneath), and a child's holder blocks the parent (the parent would sweep the child).
// Disjoint siblings — `gateway/server` and `gateway/router` — do NOT conflict, and that is the
// entire source of the added concurrency.
//
// For two flat lanes this is exactly `a == b`, the test Decide applied before sub-lanes existed.
func LanesConflict(a, b string) bool {
	return LaneContains(a, b) || LaneContains(b, a)
}

// DeclaredAncestor returns the nearest lane at or above lane that dos.toml actually declares, or
// "" when the taxonomy is unloaded or nothing on the chain is declared. This is the resolution
// step every derived rule shares: an undeclared `gateway/server/handler.go` inherits its tree and
// its exclusivity from declared `gateway`.
func (t Taxonomy) DeclaredAncestor(lane string) string {
	if !t.Loaded {
		return ""
	}
	for _, anc := range LaneAncestors(lane) {
		key := strings.ToLower(anc)
		if _, ok := t.Trees[key]; ok {
			return anc
		}
		if t.Exclusive[key] {
			return anc
		}
	}
	return ""
}

// IsExclusive reports whether lane runs alone, INHERITING the flag down the hierarchy: `abi` is
// exclusive (PARTITION.md's frozen, human-owned spine), therefore `abi/registry.go` is too. A
// sub-lane can never quietly escape an exclusive ancestor by naming a narrower unit — that would
// be the one way hierarchy could weaken the partition, so it is closed here.
func (t Taxonomy) IsExclusive(lane string) bool {
	if !t.Loaded {
		return false
	}
	for _, anc := range LaneAncestors(lane) {
		if t.Exclusive[strings.ToLower(anc)] {
			return true
		}
	}
	return false
}

// TreeFor returns the tree globs a lane is admitted against: its own declared `[lanes.trees]` row
// when it has one, else a tree DERIVED by appending the undeclared tail to the nearest declared
// ancestor's directory globs. `gateway` declares `internal/gateway/**`, so:
//
//	gateway/server            -> internal/gateway/server/**
//	gateway/server.go         -> internal/gateway/server.go     (a file lane, no /** suffix)
//	cmd/fak/dispatch_wave.go  -> cmd/fak/dispatch_wave.go
//
// When nothing can be derived — no declared ancestor, or an ancestor whose tree is a bare file
// list with no directory glob to descend into — it returns the ancestor's own coarse tree, or nil
// when there is no ancestor at all. Both fallbacks are conservative: the caller ends up admitting
// against a tree at least as wide as today's, never a narrower one it did not earn.
func (t Taxonomy) TreeFor(lane string) []string {
	lane = CanonicalLane(lane)
	if lane == "" || !t.Loaded {
		return nil
	}
	if own, ok := t.Trees[strings.ToLower(lane)]; ok {
		return cleanGlobs(own)
	}
	anc := t.DeclaredAncestor(lane)
	if anc == "" {
		return nil
	}
	base := cleanGlobs(t.Trees[strings.ToLower(anc)])
	tail := strings.TrimPrefix(lane, anc+LaneSep)
	if tail == "" || tail == lane {
		return base
	}
	var derived []string
	for _, g := range base {
		// Descend only into a glob that is EXPLICITLY directory-shaped (`internal/gateway/**`).
		// A bare entry like release's `version` is a literal path — extensionless, so no
		// filename heuristic can tell it from a directory, but its missing glob suffix can.
		root, isDir := globRoot(g)
		if root == "" || !isDir {
			continue
		}
		path := root + "/" + tail
		if looksLikeFile(tail) {
			derived = append(derived, path)
			continue
		}
		derived = append(derived, path+"/**")
	}
	if len(derived) == 0 {
		return base // conservative: no narrowing earned, keep the ancestor's coarse tree
	}
	return derived
}

// Granularity selects how deep LaneForPath and LaneSpace cut the lane namespace. It is the one
// knob that trades concurrency against blast radius, and the honest choice differs per root — see
// each constant.
type Granularity int

const (
	// GranLeaf is today's behaviour and the DEFAULT: the lane is the declared dos.toml lane and
	// nothing below it. One lane per leaf, 542 of them, one worker each.
	GranLeaf Granularity = iota

	// GranDir cuts at directory boundaries: `gateway/server` for internal/gateway/server/x.go.
	// This is the safe general default for CODE roots. Two workers in sibling directories are
	// separate Go packages, so neither one's half-finished edit can red the other's build — the
	// coupling that makes finer cuts dishonest for code.
	GranDir

	// GranFile cuts at the file: `docs/awesome-caching/README.md`. It is the ceiling of the
	// path-derived space and is honest for roots whose files do NOT co-compile — docs, visuals,
	// examples, testdata — where two workers on two files genuinely cannot break each other. On a
	// Go package it over-promises: disjoint files in one package still share a build.
	GranFile
)

// LaneForPath maps one repo-relative path to the lane that owns it at the requested granularity,
// or "" when no declared lane's tree covers the path. It is the inverse of TreeFor: the declared
// root is matched by longest prefix and the remaining path becomes the lane's tail, so
// LaneForPath and TreeFor round-trip.
func LaneForPath(path string, tax Taxonomy, gran Granularity) string {
	return tax.index().laneFor(path, gran)
}

// LaneSpace enumerates every distinct lane the given paths address at the requested granularity,
// sorted. It is the measurement instrument for the whole idea: run it over `git ls-files` at each
// granularity and the ratio against len(tax.Trees) is the honest, reproducible multiplier on the
// addressable lane count — not a projection.
//
// Ancestors are included: a file lane's parent directories are themselves addressable lanes (a
// worker may want the whole subtree), so the space at GranFile is a superset of GranDir's.
func LaneSpace(paths []string, tax Taxonomy, gran Granularity) []string {
	idx := tax.index()
	seen := map[string]bool{}
	for _, p := range paths {
		lane := idx.laneFor(p, gran)
		if lane == "" {
			continue
		}
		for _, anc := range LaneAncestors(lane) {
			seen[anc] = true
		}
	}
	out := make([]string, 0, len(seen))
	for lane := range seen {
		out = append(out, lane)
	}
	sort.Strings(out)
	return out
}

// laneIndex is the longest-prefix map from a declared tree root back to its lane, built once per
// LaneSpace call so enumerating 12k paths against 542 lanes stays a map probe per path segment
// rather than a scan of every lane.
type laneIndex struct {
	roots map[string]string // "internal/gateway" -> "gateway"
	depth int               // deepest declared root, in segments; bounds the probe loop
}

func (t Taxonomy) index() laneIndex {
	idx := laneIndex{roots: make(map[string]string, len(t.Trees))}
	for lane, globs := range t.Trees {
		for _, g := range globs {
			// Key on the SEGMENT form, not the raw prefix: laneFor probes with LaneSegments
			// output, so `.claude/**` must index as `claude` or the `claude` lane never matches
			// a `.claude/...` path. TreeFor keeps the raw prefix — a derived tree has to name
			// the real on-disk path, dot and all.
			prefix, _ := globRoot(g)
			root := pathKey(prefix)
			if root == "" {
				continue
			}
			// A shorter declared root wins a tie only if nothing claimed it yet: two lanes
			// declaring the same root is a dos.toml bug, and picking deterministically (the
			// lexically smaller lane) keeps the enumeration reproducible.
			if prev, ok := idx.roots[root]; !ok || lane < prev {
				idx.roots[root] = lane
			}
			if n := len(LaneSegments(root)); n > idx.depth {
				idx.depth = n
			}
		}
	}
	return idx
}

// pathKey renders a path in the same segment form lane names use, so a declared tree root and a
// candidate path are compared under one normalization.
func pathKey(s string) string { return foldLane(s) }

// laneFor walks the path's prefixes from the deepest declared root depth downward, so the FIRST
// hit is the longest declared root — `internal/gateway` beats a hypothetical `internal`.
func (idx laneIndex) laneFor(path string, gran Granularity) string {
	// Segments keep their case — the tail becomes part of the lane name and, via TreeFor, part of
	// a glob that must match on disk. Only the ROOT probe is case-folded.
	segs := LaneSegments(path)
	if len(segs) == 0 {
		return ""
	}
	hi := idx.depth
	if hi > len(segs) {
		hi = len(segs)
	}
	for n := hi; n > 0; n-- {
		lane, ok := idx.roots[strings.ToLower(strings.Join(segs[:n], LaneSep))]
		if !ok {
			continue
		}
		tail := segs[n:]
		switch gran {
		case GranDir:
			// Drop the filename: a path's directory is the unit, and a path that IS a directory
			// (no extension on its last segment) keeps all of its tail.
			if len(tail) > 0 && looksLikeFile(tail[len(tail)-1]) {
				tail = tail[:len(tail)-1]
			}
		case GranFile:
			// keep the whole tail, filename included
		default: // GranLeaf
			tail = nil
		}
		if len(tail) == 0 {
			return lane
		}
		return lane + LaneSep + strings.Join(tail, LaneSep)
	}
	return ""
}

// globRoot reduces a tree glob to the plain path prefix it addresses — matching the normalization
// internal/dispatchorder applies before its own overlap test, so a derived tree and a declared one
// are compared under identical geometry — and reports whether the glob was DIRECTORY-shaped
// (`.../**`, `.../*`, or a trailing `/`). Only a directory-shaped glob may be descended into.
// Case is preserved: the root goes straight back into a derived tree that has to match on disk.
func globRoot(g string) (root string, isDir bool) {
	g = strings.TrimSpace(strings.ReplaceAll(g, "\\", "/"))
	g = strings.TrimPrefix(g, "./")
	for {
		switch {
		case strings.HasSuffix(g, "/**"):
			g, isDir = strings.TrimSuffix(g, "/**"), true
		case strings.HasSuffix(g, "/*"):
			g, isDir = strings.TrimSuffix(g, "/*"), true
		case strings.HasSuffix(g, "/"):
			g, isDir = strings.TrimSuffix(g, "/"), true
		default:
			return g, isDir
		}
	}
}

// looksLikeFile reports whether a lane segment or path names a file rather than a directory,
// judged by a dot in its LAST segment. It decides only whether a derived tree gets a `/**` suffix;
// guessing "directory" for an extensionless file is harmless — the glob then covers a path that
// holds nothing, which can only ever over-reserve, never under-reserve.
func looksLikeFile(s string) bool {
	segs := LaneSegments(s)
	if len(segs) == 0 {
		return false
	}
	last := segs[len(segs)-1]
	return strings.Contains(last, ".")
}
