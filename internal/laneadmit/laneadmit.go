package laneadmit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// ReasonCollisionRisk is the closed-vocabulary refusal every surface shares.
// It is dispatchorder's token re-exported so a consumer that only imports
// laneadmit still refuses with the reason dos.toml declares.
const ReasonCollisionRisk = dispatchorder.ReasonCollisionRisk

// Surface names for the execution surfaces that ask for admission. Free-form
// strings are accepted; these constants keep the evidence vocabulary stable.
const (
	SurfaceDispatch = "dispatch"
	SurfaceLoop     = "loop"
	SurfaceManual   = "manual"
)

// Conflict kinds — why a live lease blocks the request.
const (
	ConflictTreeOverlap   = "tree_overlap"   // requested tree geometrically overlaps the lease's tree
	ConflictSameLane      = "same_lane"      // same named lane serializes even on disjoint trees
	ConflictExclusiveLane = "exclusive_lane" // an exclusive lane runs alone against everything
)

// Request is one surface asking "may I act on this lane/tree right now".
type Request struct {
	Surface string   // which surface is asking (SurfaceDispatch / SurfaceLoop / SurfaceManual / ...)
	Lane    string   // named dos.toml lane; "" = a tree-only request
	Tree    []string // requested tree globs; empty falls back to the lane's taxonomy tree
	Holder  string   // requesting holder identity (host:pid, session id, ...)
	LeaseID string   // the lease id the caller would acquire under; a live lease with this id is the caller's own and never conflicts
}

// Lease is one live lease projected out of the shared namespace
// (refs/fak/locks/* via internal/leaseref, or any other store).
type Lease struct {
	ID     string
	Lane   string // named lane when known; "" = unknown (infer with LaneOfLeaseID)
	Tree   []string
	Holder string
}

// Taxonomy is the slice of dos.toml [lanes] the admission decision needs.
type Taxonomy struct {
	Exclusive map[string]bool     // lanes that run alone
	Trees     map[string][]string // lane -> canonical tree globs
	Loaded    bool                // false = no taxonomy available; lane-mode rules are skipped, geometry still applies
}

// Conflict names one live lease that blocks the request, and why.
type Conflict struct {
	LeaseID string   `json:"lease_id"`
	Lane    string   `json:"lane,omitempty"`
	Holder  string   `json:"holder,omitempty"`
	Kind    string   `json:"kind"`
	Tree    []string `json:"tree,omitempty"`
}

// Verdict is the admission decision. Refusals carry the closed-vocabulary
// reason (COLLISION_RISK) plus the conflicting leases as evidence.
type Verdict struct {
	Admit     bool       `json:"admit"`
	Reason    string     `json:"reason,omitempty"`
	Detail    string     `json:"detail,omitempty"`
	Tree      []string   `json:"tree,omitempty"` // the tree the decision was made against (after taxonomy fallback)
	Conflicts []Conflict `json:"conflicts,omitempty"`
}

// Decide is the one admission contract every surface shares: given the live
// lease set and the lane taxonomy, may this request act now? It is pure — no
// clock, no I/O; the caller supplies the state (leaseref live leases, parsed
// dos.toml). Rules, strongest first:
//
//  1. exclusive_lane — an exclusive lane (abi, release, dos, global) runs
//     alone: requesting one conflicts with every live lease, and any live
//     lease on one conflicts with every request.
//  2. same_lane — a named lane serializes: two holders on the same lane
//     conflict even when their trees were narrowed disjoint (the dos
//     arbitrate rule leaseref's geometry-only check never honored).
//  3. tree_overlap — the geometric check dispatch always enforced
//     (dispatchorder.TreesOverlap; an empty tree conservatively overlaps
//     everything).
//
// A live lease whose ID equals req.LeaseID is the caller's own (a renew or
// re-entrant acquire) and never conflicts. The decision is the same contract
// `dos arbitrate` applies fleet-wide; this in-binary twin exists so every fak
// surface can afford to ask it on each act boundary. It shares leaseref's
// honest scope: local visibility, not cross-host atomicity.
func Decide(req Request, live []Lease, tax Taxonomy) Verdict {
	tree := cleanGlobs(req.Tree)
	if len(tree) == 0 && req.Lane != "" && tax.Loaded {
		tree = cleanGlobs(tax.Trees[req.Lane])
	}
	var conflicts []Conflict
	for _, l := range live {
		if l.ID != "" && l.ID == req.LeaseID {
			continue
		}
		lane := l.Lane
		if lane == "" {
			lane = LaneOfLeaseID(l.ID)
		}
		kind := ""
		switch {
		case tax.Loaded && req.Lane != "" && tax.Exclusive[req.Lane]:
			kind = ConflictExclusiveLane
		case tax.Loaded && lane != "" && tax.Exclusive[lane]:
			kind = ConflictExclusiveLane
		case req.Lane != "" && lane == req.Lane:
			kind = ConflictSameLane
		case dispatchorder.TreesOverlap(tree, l.Tree):
			kind = ConflictTreeOverlap
		}
		if kind == "" {
			continue
		}
		conflicts = append(conflicts, Conflict{
			LeaseID: l.ID,
			Lane:    lane,
			Holder:  l.Holder,
			Kind:    kind,
			Tree:    append([]string(nil), l.Tree...),
		})
	}
	if len(conflicts) == 0 {
		return Verdict{Admit: true, Tree: tree}
	}
	sort.Slice(conflicts, func(i, j int) bool { return conflicts[i].LeaseID < conflicts[j].LeaseID })
	first := conflicts[0]
	return Verdict{
		Admit:     false,
		Reason:    ReasonCollisionRisk,
		Detail:    conflictDetail(req, tree, first),
		Tree:      tree,
		Conflicts: conflicts,
	}
}

func conflictDetail(req Request, tree []string, c Conflict) string {
	subject := "tree " + fmt.Sprintf("%v", tree)
	if req.Lane != "" {
		subject = fmt.Sprintf("lane %q tree %v", req.Lane, tree)
	}
	switch c.Kind {
	case ConflictExclusiveLane:
		return fmt.Sprintf("requested %s conflicts with live lease %s (exclusive lane %s runs alone)", subject, c.LeaseID, firstNonEmpty(c.Lane, req.Lane))
	case ConflictSameLane:
		return fmt.Sprintf("requested %s is already held by live lease %s (same lane serializes even on disjoint trees)", subject, c.LeaseID)
	default:
		return fmt.Sprintf("requested %s overlaps live lease %s tree %v", subject, c.LeaseID, c.Tree)
	}
}

// LaneOfLeaseID infers the named lane a lease id was minted for, or "" when
// the id carries none. Two grammars are recognized:
//
//   - the dispatch grammar, grandfathered: "resolve-<lane>" and
//     "resolve-<lane>-<issue#>" (cmd/fak dispatchLaneLeaseID / dispatchIssueLeaseID)
//   - the shared grammar new surfaces mint: "<surface>-lane-<lane>"
//     (e.g. "loop-lane-docs", "coord-lane-gateway")
func LaneOfLeaseID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	if i := strings.Index(id, "-lane-"); i >= 0 {
		return id[i+len("-lane-"):]
	}
	if rest, ok := strings.CutPrefix(id, "resolve-"); ok {
		if i := strings.LastIndexByte(rest, '-'); i > 0 && allDigits(rest[i+1:]) {
			rest = rest[:i]
		}
		return rest
	}
	return ""
}

// LeaseID mints the shared-grammar lease id for a surface acting on a named
// lane ("<surface>-lane-<lane>"), or a scope-token id for a tree-only request
// ("<surface>-<scope>"). LaneOfLeaseID inverts the lane form.
func LeaseID(surface, lane, scope string) string {
	surface = cleanToken(surface)
	if surface == "" {
		surface = "coord"
	}
	if lane = cleanToken(lane); lane != "" {
		return surface + "-lane-" + lane
	}
	if scope = cleanToken(scope); scope != "" {
		return surface + "-" + scope
	}
	return surface
}

// ParseTaxonomy scans dos.toml bytes for the [lanes] exclusive list and the
// [lanes.trees] table — the two slices the admission decision needs. It is a
// deliberately minimal line scanner (the same discipline as the commit-stamp
// hook's reader): bytes in, taxonomy out, no I/O and no TOML dependency.
func ParseTaxonomy(data []byte) Taxonomy {
	tax := Taxonomy{Exclusive: map[string]bool{}, Trees: map[string][]string{}}
	if len(data) == 0 {
		return tax
	}
	tax.Loaded = true
	section, key := "", ""
	for _, raw := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
		line := raw
		if i := strings.IndexByte(line, '#'); i >= 0 { // lane names and globs carry no '#'
			line = line[:i]
		}
		t := strings.TrimSpace(line)
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "[") {
			section, key = strings.Trim(t, "[]"), ""
			continue
		}
		switch section {
		case "lanes":
			if eq := strings.IndexByte(t, '='); eq >= 0 {
				key = strings.TrimSpace(t[:eq])
				t = t[eq+1:]
			}
			if key != "exclusive" {
				continue
			}
			for _, tok := range quotedTokens(t) {
				tax.Exclusive[strings.ToLower(tok)] = true
			}
		case "lanes.trees":
			eq := strings.IndexByte(t, '=')
			if eq < 0 {
				continue
			}
			lane := strings.ToLower(strings.TrimSpace(t[:eq]))
			if lane == "" {
				continue
			}
			tax.Trees[lane] = append(tax.Trees[lane], quotedTokens(t[eq+1:])...)
		}
	}
	return tax
}

func quotedTokens(s string) []string {
	var out []string
	for {
		i := strings.IndexByte(s, '"')
		if i < 0 {
			return out
		}
		j := strings.IndexByte(s[i+1:], '"')
		if j < 0 {
			return out
		}
		if tok := s[i+1 : i+1+j]; tok != "" {
			out = append(out, tok)
		}
		s = s[i+j+2:]
	}
}

func cleanGlobs(globs []string) []string {
	var out []string
	for _, g := range globs {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, g)
		}
	}
	return out
}

func cleanToken(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_':
			b.WriteRune(r)
		default:
			if n := b.Len(); n > 0 && b.String()[n-1] != '-' {
				b.WriteByte('-')
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
