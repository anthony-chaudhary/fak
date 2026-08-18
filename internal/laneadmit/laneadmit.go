package laneadmit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
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
	// ConflictLaneAncestry is same_lane's hierarchical case: the two lanes are not equal, but one
	// CONTAINS the other (`gateway` vs `gateway/server`), so they still serialize. It is reported
	// as its own kind rather than folded into same_lane because the cure differs — a same_lane
	// holder means "wait", an ancestry holder means "the peer took a coarser lane than it needed;
	// a sibling sub-lane would have been admitted".
	ConflictLaneAncestry = "lane_ancestry"
)

// Request is one surface asking "may I act on this lane/tree right now".
type Request struct {
	Surface string   // which surface is asking (SurfaceDispatch / SurfaceLoop / SurfaceManual / ...)
	Lane    string   // named dos.toml lane; "" = a tree-only request
	Tree    []string // requested tree globs; empty falls back to the lane's taxonomy tree
	Holder  string   // requesting holder identity (host:pid, session id, ...)
	LeaseID string   // the lease id the caller would acquire under; a live lease with this id is the caller's own and never conflicts
	// ReadOnly declares the request writes NOTHING — a provably empty write footprint, the
	// third state distinct from a declared tree (known blast radius) and an ABSENT tree
	// (UNKNOWN blast radius, which still collides conservatively). A read-only request is
	// admitted against every live lease — it cannot clobber a writer and holds nothing a writer
	// can clobber — so a pure read (a `git log`, a `grep`) is never serialized behind an
	// unrelated lane. Default false keeps every existing caller's verdict byte-identical.
	ReadOnly bool
}

// Lease is one live lease projected out of the shared namespace
// (refs/fak/locks/* via internal/leaseref, or any other store).
type Lease struct {
	ID     string
	Lane   string // named lane when known; "" = unknown (infer with LaneOfLeaseID)
	Tree   []string
	Holder string
	// ReadOnly reports the live lease itself writes NOTHING (a held pure read). It blocks no
	// request: a read-only holder cannot be clobbered, so no new act needs to serialize behind
	// it. Default false leaves every existing lease projection blocking exactly as before.
	ReadOnly bool
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
	reqLane := CanonicalLane(req.Lane)
	tree := cleanGlobs(req.Tree)
	if len(tree) == 0 && reqLane != "" && tax.Loaded {
		// TreeFor, not a bare tax.Trees lookup: a DECLARED lane resolves to its own row exactly
		// as before, and an undeclared SUB-lane derives its tree from the nearest declared
		// ancestor instead of falling through to the empty-tree conservative-overlap rule.
		tree = cleanGlobs(tax.TreeFor(reqLane))
	}
	// A provably read-only request writes nothing and therefore contends with nothing — it is
	// admitted against every live lease, ABOVE the empty-tree conservative-overlap rule. This is
	// the tri-state opt-out: an ABSENT tree is unknown blast radius (falls through and collides),
	// a declared ReadOnly is an EMPTY write footprint (never collides).
	if req.ReadOnly {
		return Verdict{Admit: true, Tree: tree}
	}
	var conflicts []Conflict
	for _, l := range live {
		if l.ID != "" && l.ID == req.LeaseID {
			continue
		}
		// A read-only live lease holds nothing writable, so it blocks no request.
		if l.ReadOnly {
			continue
		}
		lane := CanonicalLane(l.Lane)
		if lane == "" {
			lane = LaneOfLeaseID(l.ID)
		}
		kind := ""
		switch {
		// Exclusivity INHERITS down the hierarchy (Taxonomy.IsExclusive): a sub-lane of `abi`
		// cannot escape the serial ABI gate by naming a narrower unit.
		case tax.Loaded && reqLane != "" && tax.IsExclusive(reqLane):
			kind = ConflictExclusiveLane
		case tax.Loaded && lane != "" && tax.IsExclusive(lane):
			kind = ConflictExclusiveLane
		// Case-folded, since CanonicalLane preserves case for the file-lane tail: two spellings of
		// one lane must still report as same_lane rather than falling through to ancestry.
		case reqLane != "" && foldLane(lane) == foldLane(reqLane):
			kind = ConflictSameLane
		// The hierarchical case: unequal lanes where one contains the other still serialize.
		// For two flat lanes LanesConflict is exactly the equality above, so this arm can only
		// fire once a caller opts into a sub-lane.
		case reqLane != "" && lane != "" && LanesConflict(reqLane, lane):
			kind = ConflictLaneAncestry
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
		return fmt.Sprintf("requested %s conflicts with live lease %s (exclusive lane %s runs alone)", subject, c.LeaseID, strmatch.FirstNonEmpty(c.Lane, req.Lane))
	case ConflictSameLane:
		return fmt.Sprintf("requested %s is already held by live lease %s (same lane serializes even on disjoint trees)", subject, c.LeaseID)
	case ConflictLaneAncestry:
		return fmt.Sprintf("requested %s serializes behind live lease %s on lane %q (one lane contains the other; a disjoint sub-lane would be admitted)", subject, c.LeaseID, c.Lane)
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
//
// A SUB-lane travels in the id under laneWireSep, so "loop-lane-docs_notes" decodes back to
// the canonical "docs/notes". A flat id decodes to itself — no declared lane carries `_`, so
// the decode cannot corrupt an existing id.
func LaneOfLeaseID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	if i := strings.Index(id, "-lane-"); i >= 0 {
		return decodeLaneWire(id[i+len("-lane-"):])
	}
	if rest, ok := strings.CutPrefix(id, "resolve-"); ok {
		if i := strings.LastIndexByte(rest, '-'); i > 0 && allDigits(rest[i+1:]) {
			rest = rest[:i]
		}
		return decodeLaneWire(rest)
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
	if lane = encodeLaneWire(lane); lane != "" {
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

// encodeLaneWire renders a canonical lane as the ONE ref-path segment a lease id must be. Each
// segment is scrubbed by cleanToken (the pre-existing rule), a literal `_` inside a segment is
// DOUBLED so it cannot be misread as the separator (`dispatch_wave.go` -> `dispatch__wave.go`),
// and the segments are joined with a single laneWireSep. Finally a `.lock` suffix — the one
// spelling git's check-ref-format rejects outright — is escaped with a trailing separator that
// decodes back to nothing, since CanonicalLane drops a trailing separator.
//
// A flat lane has one segment and no `_`, so its id is byte-identical to the pre-hierarchy form.
func encodeLaneWire(lane string) string {
	segs := LaneSegments(CanonicalLane(lane))
	out := make([]string, 0, len(segs))
	for i, seg := range segs {
		// The FIRST segment is always a declared dos.toml lane (a derived lane is rooted at one by
		// construction), and dos.toml's vocabulary is lowercase — so fold it, and two spellings of
		// a declared lane cannot mint two different refs for one lock. Every segment AFTER it is a
		// real path component whose case must survive: git refs are case-sensitive, so folding the
		// tail would make the id lossless-looking but decode to a lane whose derived tree matches
		// nothing on a case-sensitive filesystem.
		if i == 0 {
			// The FIRST segment is always a declared dos.toml lane and that vocabulary is
			// lowercase, so folding it means two spellings cannot mint two refs for one lock.
			seg = strings.ToLower(seg)
		}
		if tok := escapeLaneSegment(seg); tok != "" {
			out = append(out, tok)
		}
	}
	id := strings.Join(out, laneWireSep)
	if strings.HasSuffix(id, ".lock") {
		id += laneWireSep
	}
	return id
}

// decodeLaneWire inverts encodeLaneWire: `_` is a segment break, `--` is a literal `-`, `-u` is a
// literal `_`. CanonicalLane then drops the trailing separator the `.lock` escape may have added.
//
// An id minted before sub-lanes existed carries neither `_` nor `-` inside its lane part (every
// declared lane is `[a-z0-9]+`), so it decodes to itself and every live lease still resolves to
// the lane it was minted for.
func decodeLaneWire(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == laneWireSep[0]:
			b.WriteString(LaneSep)
		case s[i] == laneWireEsc[0] && i+1 < len(s) && s[i+1] == laneWireEsc[0]:
			b.WriteString(laneWireEsc)
			i++
		case s[i] == laneWireEsc[0] && i+1 < len(s) && s[i+1] == 'u':
			b.WriteString(laneWireSep)
			i++
		case s[i] == laneWireEsc[0] && i+3 < len(s) && s[i+1] == 'x':
			hi, okHi := mathx.HexNibble(s[i+2])
			lo, okLo := mathx.HexNibble(s[i+3])
			if !okHi || !okLo {
				b.WriteByte(s[i])
				continue
			}
			b.WriteByte(hi<<4 | lo)
			i += 3
		default:
			b.WriteByte(s[i])
		}
	}
	return CanonicalLane(b.String())
}

// escapeLaneSegment renders one lane segment into the ref-safe alphabet internal/leaseref's
// validID accepts, REVERSIBLY. It is cleanToken's total twin: where cleanToken collapses an
// unrepresentable byte to `-` (fine for a scope token nobody decodes), this escapes it, because a
// lane recovered from a lease id has to name the same path it was minted from.
//
//   - ->  --        _  ->  -u        any other non-[A-Za-z0-9._] byte  ->  -x<2 hex>
//
// Case is preserved: a segment below the declared root is a real path component, and git refs are
// case-sensitive, so folding it would decode to a lane whose derived tree matches nothing on a
// case-sensitive filesystem.
func escapeLaneSegment(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '.':
			b.WriteByte(c)
		case c == laneWireEsc[0]:
			b.WriteString(laneWireEsc + laneWireEsc)
		case c == laneWireSep[0]:
			b.WriteString(laneWireEsc + "u")
		default:
			b.WriteString(laneWireEsc + "x")
			const hex = "0123456789abcdef"
			b.WriteByte(hex[c>>4])
			b.WriteByte(hex[c&0x0f])
		}
	}
	return b.String()
}

// unhex decodes one lowercase-or-upper hex digit, returning ok=false for a non-hex byte so a
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
