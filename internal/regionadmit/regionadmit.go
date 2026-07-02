package regionadmit

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// ReasonCollisionRisk is the closed refusal token a region-admission refusal
// carries — the same `dos.toml [reasons]` COLLISION_RISK the dispatch loop and
// the dos arbiter already speak, so every surface refuses in one vocabulary.
const ReasonCollisionRisk = dispatchorder.ReasonCollisionRisk

// The rung names a Decision carries so a refusal says WHICH admission rule
// fired, not just that one did. Closed set; evidence, not prose.
const (
	// RungExclusiveRequested: the request names an exclusive lane (abi,
	// release, dos, global) while any lease at all is live — an exclusive
	// lane runs alone.
	RungExclusiveRequested = "exclusive_lane_requested"
	// RungExclusiveLive: a live lease sits on an exclusive lane, which
	// blocks every new region until it clears.
	RungExclusiveLive = "exclusive_lane_live"
	// RungSameLane: the request names the same lane a live lease already
	// holds — a named lane serializes even when the trees look disjoint.
	RungSameLane = "same_lane_live"
	// RungTreeOverlap: the requested tree overlaps a live lease's tree
	// under the same prefix geometry the dispatch fan-out price uses.
	RungTreeOverlap = "tree_overlap"
)

// Taxonomy is the workspace lane data the decision honors: which lanes are
// exclusive (run alone) and each lane's canonical file tree. It is the same
// `dos.toml [lanes]` data the dos arbiter reads, loaded without shelling out.
type Taxonomy struct {
	Exclusive map[string]bool
	Trees     map[string][]string
}

// Request is one surface asking to act on a region: Actor names the would-be
// holder (e.g. "loop:nightly", a session id, host:pid), Lane optionally names
// the dos.toml lane, and Tree is the requested region globs. An empty Tree
// with a named Lane resolves to the lane's canonical tree; empty both ways is
// unknown blast radius and collides conservatively (dispatchorder geometry).
// SelfID names the caller's own lease id so a re-admission (renew/restart of
// the same lease) is not refused against itself.
type Request struct {
	Actor  string
	Lane   string
	Tree   []string
	SelfID string
}

// Lease is the projection of one live lease the decision folds — any source
// (a leaseref record, a dos arbiter row) projects to this shape. Lane may be
// left empty; Decide infers it from Tree when the tree is exactly a declared
// lane tree.
type Lease struct {
	ID     string
	Holder string
	Lane   string
	Tree   []string
}

// Decision is the admission verdict. A refusal carries the closed reason
// (COLLISION_RISK), the rung that fired, a human-readable detail naming the
// conflict, and the conflicting lease itself as evidence.
type Decision struct {
	Admit    bool
	Reason   string
	Rung     string
	Detail   string
	Conflict *Lease
}

// ResolveTree returns the region tree a request means: its explicit Tree, or
// the named lane's canonical tree from the taxonomy. Empty when neither is
// declared — the conservative unknown-blast-radius case.
func ResolveTree(req Request, tax Taxonomy) []string {
	if len(cleanGlobs(req.Tree)) > 0 {
		return cleanGlobs(req.Tree)
	}
	if req.Lane != "" {
		return cleanGlobs(tax.Trees[req.Lane])
	}
	return nil
}

// LaneOf reports the lane whose canonical taxonomy tree exactly matches tree
// (order-insensitive), or "" when no lane owns exactly that tree. This is how
// a lease that recorded only its tree gets its lane semantics back without a
// schema change to the lease record.
func LaneOf(tree []string, tax Taxonomy) string {
	key := treeKey(tree)
	if key == "" {
		return ""
	}
	// Deterministic: on the (unexpected) case of two lanes declaring the same
	// tree, the lexically first lane name wins.
	lanes := make([]string, 0, len(tax.Trees))
	for lane := range tax.Trees {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		if treeKey(tax.Trees[lane]) == key {
			return lane
		}
	}
	return ""
}

// Decide answers "may this actor act on this region right now?" against the
// live lease set and the lane taxonomy. It is pure: state in, verdict out. The
// rules, in order, mirror the dos arbiter's admission contract:
//
//  1. an exclusive lane request admits only when NOTHING else is live;
//  2. a live lease on an exclusive lane refuses every new region;
//  3. a request naming the same lane a live lease holds is refused (a named
//     lane serializes, even on disjoint trees);
//  4. a requested tree overlapping a live lease's tree is refused
//     (dispatchorder.TreesOverlap geometry — empty trees collide
//     conservatively).
//
// A lease whose ID equals req.SelfID is the caller's own and never conflicts.
func Decide(req Request, live []Lease, tax Taxonomy) Decision {
	tree := ResolveTree(req, tax)
	others := make([]Lease, 0, len(live))
	for _, l := range live {
		if req.SelfID != "" && l.ID == req.SelfID {
			continue
		}
		others = append(others, l)
	}
	if tax.Exclusive[req.Lane] && len(others) > 0 {
		c := others[0]
		return refusal(RungExclusiveRequested, &c, fmt.Sprintf(
			"lane %q is exclusive (runs alone) but %d lease(s) are live, e.g. %s",
			req.Lane, len(others), leaseLabel(c)))
	}
	for i := range others {
		l := others[i]
		lane := l.Lane
		if lane == "" {
			lane = LaneOf(l.Tree, tax)
		}
		if tax.Exclusive[lane] {
			return refusal(RungExclusiveLive, &l, fmt.Sprintf(
				"live lease %s holds exclusive lane %q, which blocks every new region until it clears",
				leaseLabel(l), lane))
		}
		if req.Lane != "" && lane == req.Lane {
			return refusal(RungSameLane, &l, fmt.Sprintf(
				"lane %q is already held by live lease %s (a named lane serializes)",
				req.Lane, leaseLabel(l)))
		}
		if dispatchorder.TreesOverlap(tree, l.Tree) {
			return refusal(RungTreeOverlap, &l, fmt.Sprintf(
				"requested tree %v overlaps live lease %s tree %v",
				tree, leaseLabel(l), l.Tree))
		}
	}
	return Decision{Admit: true}
}

func refusal(rung string, conflict *Lease, detail string) Decision {
	return Decision{
		Admit:    false,
		Reason:   ReasonCollisionRisk,
		Rung:     rung,
		Detail:   detail,
		Conflict: conflict,
	}
}

func leaseLabel(l Lease) string {
	if l.Holder == "" {
		return l.ID
	}
	return fmt.Sprintf("%s (holder %s)", l.ID, l.Holder)
}

// LoadTaxonomy reads the workspace's `dos.toml [lanes]` exclusive set and
// `[lanes.trees]` map — the same data `dos doctor --json` reports — with a
// tolerant multi-line-array TOML read and no subprocess. A missing dos.toml
// is an error the caller may treat as "no taxonomy" (geometry-only admission).
func LoadTaxonomy(root string) (Taxonomy, error) {
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		return Taxonomy{}, err
	}
	tax := Taxonomy{Exclusive: map[string]bool{}, Trees: map[string][]string{}}
	section, key := "", ""
	var buf strings.Builder
	depth := 0
	flush := func() {
		if key == "" {
			return
		}
		values := quotedStrings(buf.String())
		switch {
		case section == "lanes" && key == "exclusive":
			for _, lane := range values {
				tax.Exclusive[lane] = true
			}
		case section == "lanes.trees":
			tax.Trees[key] = values
		}
		key = ""
		buf.Reset()
	}
	for _, line := range strings.Split(string(raw), "\n") {
		trimmed := strings.TrimSpace(stripTomlComment(line))
		if depth > 0 {
			buf.WriteString(trimmed)
			buf.WriteString("\n")
			depth += strings.Count(trimmed, "[") - strings.Count(trimmed, "]")
			if depth <= 0 {
				flush()
				depth = 0
			}
			continue
		}
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
			section = strings.TrimSuffix(strings.TrimPrefix(trimmed, "["), "]")
			continue
		}
		k, v, ok := strings.Cut(trimmed, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		buf.WriteString(v)
		buf.WriteString("\n")
		depth = strings.Count(v, "[") - strings.Count(v, "]")
		if depth <= 0 {
			flush()
			depth = 0
		}
	}
	flush()
	return tax, nil
}

// stripTomlComment removes a `#` comment that starts outside a quoted string.
func stripTomlComment(line string) string {
	inString := false
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case '"':
			inString = !inString
		case '#':
			if !inString {
				return line[:i]
			}
		}
	}
	return line
}

// quotedStrings extracts every "..." token from raw, in order.
func quotedStrings(raw string) []string {
	var out []string
	for {
		start := strings.IndexByte(raw, '"')
		if start < 0 {
			return out
		}
		raw = raw[start+1:]
		end := strings.IndexByte(raw, '"')
		if end < 0 {
			return out
		}
		if v := raw[:end]; v != "" {
			out = append(out, v)
		}
		raw = raw[end+1:]
	}
}

func cleanGlobs(tree []string) []string {
	out := make([]string, 0, len(tree))
	for _, g := range tree {
		g = strings.ReplaceAll(strings.TrimSpace(g), "\\", "/")
		if g != "" {
			out = append(out, g)
		}
	}
	return out
}

func treeKey(tree []string) string {
	globs := cleanGlobs(tree)
	if len(globs) == 0 {
		return ""
	}
	sorted := append([]string(nil), globs...)
	sort.Strings(sorted)
	return strings.Join(sorted, "\n")
}
