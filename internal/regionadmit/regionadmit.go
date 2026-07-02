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

// LaneOf reports the lane that owns tree: first by exact, order-insensitive
// tree-set match, then — because a surface may lease a NARROWED region inside
// its lane (a GOAL.md region:, a dispatch --lease-tree) — by containment: the
// single most-specific lane whose canonical tree contains every glob of tree.
// Without the containment rung a narrowed lease silently loses its lane
// semantics (same-lane serialization, exclusive-lane blocking) and only
// geometry protects it. A tree spanning multiple lanes, or contained only by
// a catch-all glob (empty literal prefix, e.g. "**/*"), owns no lane. This is
// how a lease that recorded only its tree gets its lane semantics back
// without a schema change to the lease record.
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
	// Containment rung: the tightest containing lane wins (longest container
	// prefix); the sorted walk makes an exact-specificity tie deterministic.
	best, bestSpec := "", -1
	for _, lane := range lanes {
		if spec, ok := treeContainmentSpec(tree, tax.Trees[lane]); ok && spec > bestSpec {
			best, bestSpec = lane, spec
		}
	}
	return best
}

// treeContainmentSpec reports whether every glob of tree sits under a
// non-catch-all glob of laneTree, and how tight the containment is (the
// shortest container prefix used — higher = more specific). Containment uses
// the same literal-prefix geometry as the overlap check: a glob's prefix is
// its leading path up to the first meta character.
func treeContainmentSpec(tree, laneTree []string) (int, bool) {
	globs := cleanGlobs(tree)
	containers := cleanGlobs(laneTree)
	if len(globs) == 0 || len(containers) == 0 {
		return 0, false
	}
	spec := int(^uint(0) >> 1)
	for _, g := range globs {
		p := globPrefix(g)
		best := -1
		for _, c := range containers {
			cp := globPrefix(c)
			if cp == "" {
				// A catch-all container ("**/*") contains everything and
				// therefore claims nothing — otherwise every lease would
				// classify onto the exclusive global lane and block the world.
				continue
			}
			if (p == cp || strings.HasPrefix(p, cp+"/")) && len(cp) > best {
				best = len(cp)
			}
		}
		if best < 0 {
			return 0, false
		}
		if best < spec {
			spec = best
		}
	}
	return spec, true
}

// globPrefix is the literal leading path of a glob: the part before the first
// meta character (* ? [), trimmed back to the last complete path segment. A
// meta-free glob is its own prefix (a literal file path).
func globPrefix(g string) string {
	i := strings.IndexAny(g, "*?[")
	if i < 0 {
		return strings.TrimSuffix(g, "/")
	}
	p := g[:i]
	if j := strings.LastIndexByte(p, '/'); j >= 0 {
		return p[:j]
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
		stripped, d := tomlScanLine(line)
		trimmed := strings.TrimSpace(stripped)
		if depth > 0 {
			buf.WriteString(trimmed)
			buf.WriteString("\n")
			depth += d
			if depth <= 0 {
				flush()
				depth = 0
			}
			continue
		}
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") && !strings.Contains(trimmed, "=") {
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
		_, depth = tomlScanLine(v)
		if depth <= 0 {
			flush()
			depth = 0
		}
	}
	flush()
	return tax, nil
}

// tomlScanLine strips a `#` comment (quote- and escape-aware) and reports the
// bracket depth delta of the remaining text, counting `[`/`]` only OUTSIDE
// double-quoted strings — a bracket inside a glob value like "docs/x[!a]/**"
// or inside `[reasons]` prose is content, not structure, and counting it
// would silently desync the reader and swallow every later lane. Single-quoted
// TOML strings are not used by the [lanes] tables this reader targets.
func tomlScanLine(line string) (stripped string, depth int) {
	inString, escaped := false, false
	for i := 0; i < len(line); i++ {
		ch := line[i]
		if escaped {
			escaped = false
			continue
		}
		switch ch {
		case '\\':
			if inString {
				escaped = true
			}
		case '"':
			inString = !inString
		case '#':
			if !inString {
				return line[:i], depth
			}
		case '[':
			if !inString {
				depth++
			}
		case ']':
			if !inString {
				depth--
			}
		}
	}
	return line, depth
}

// quotedStrings extracts every "..." token from raw, in order, honoring
// backslash escapes so an escaped quote inside a value cannot desync the scan.
func quotedStrings(raw string) []string {
	var out []string
	var cur strings.Builder
	inString, escaped := false, false
	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		switch {
		case escaped:
			cur.WriteByte(ch)
			escaped = false
		case ch == '\\' && inString:
			escaped = true
		case ch == '"':
			if inString && cur.Len() > 0 {
				out = append(out, cur.String())
			}
			cur.Reset()
			inString = !inString
		case inString:
			cur.WriteByte(ch)
		}
	}
	return out
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
