package laneadmit

import (
	"strings"
	"testing"
)

// fixtureTax mirrors the shape of the real dos.toml slice the admission path reads: a handful of
// flat, single-segment lanes with prefix trees, one of them exclusive.
func fixtureTax() Taxonomy {
	return Taxonomy{
		Loaded:    true,
		Exclusive: map[string]bool{"abi": true},
		Trees: map[string][]string{
			"abi":     {"internal/abi/**"},
			"gateway": {"internal/gateway/**"},
			"cmd":     {"cmd/**"},
			"docs":    {"docs/**"},
			"claude":  {".claude/**"},
			"release": {"version", "CHANGELOG.md"},
		},
	}
}

func TestCanonicalLaneAcceptsAllThreeSpellings(t *testing.T) {
	for _, in := range []string{"gateway/server", `gateway\server`, "/gateway//server/", "gateway/./server"} {
		if got := CanonicalLane(in); got != "gateway/server" {
			t.Errorf("CanonicalLane(%q) = %q, want %q", in, got, "gateway/server")
		}
	}
	// Case is PRESERVED (a file tail must survive onto a case-sensitive filesystem) but folded
	// for comparison.
	if got := CanonicalLane("Gateway/Server"); got != "Gateway/Server" {
		t.Errorf("CanonicalLane must preserve case, got %q", got)
	}
	if !LaneContains("gateway", "Gateway/Server") {
		t.Error("lane comparison must be case-insensitive even though the value keeps its case")
	}
	// A flat lane must round-trip untouched — this is what keeps all 542 declared lanes stable.
	for _, flat := range []string{"gateway", "docs", "abi", "cmd"} {
		if got := CanonicalLane(flat); got != flat {
			t.Errorf("CanonicalLane(%q) = %q, want it unchanged", flat, got)
		}
	}
	if got := CanonicalLane("../../etc"); got != "etc" {
		t.Errorf("CanonicalLane must collapse traversal segments, got %q", got)
	}
}

func TestLaneContainsIsSegmentWiseNotStringPrefix(t *testing.T) {
	cases := []struct {
		outer, inner string
		want         bool
	}{
		{"gateway", "gateway/server", true},
		{"gateway", "gateway", true},
		{"gateway/server", "gateway/server/handler.go", true},
		{"gateway", "gatewayx", false},       // the string-prefix trap
		{"gate", "gateway", false},           // ditto
		{"gateway/server", "gateway", false}, // containment is directional
		{"gateway/server", "gateway/router", false},
	}
	for _, c := range cases {
		if got := LaneContains(c.outer, c.inner); got != c.want {
			t.Errorf("LaneContains(%q, %q) = %v, want %v", c.outer, c.inner, got, c.want)
		}
	}
}

// TestLanesConflictDegeneratesToEqualityWhenFlat is the backward-compatibility witness: for two
// single-segment lanes the new rule must be exactly the `lane == req.Lane` test Decide ran before.
func TestLanesConflictDegeneratesToEqualityWhenFlat(t *testing.T) {
	flat := []string{"gateway", "docs", "cmd", "abi", "gatewayx", "gate"}
	for _, a := range flat {
		for _, b := range flat {
			if got, want := LanesConflict(a, b), a == b; got != want {
				t.Errorf("LanesConflict(%q, %q) = %v, want %v (must equal string equality when flat)", a, b, got, want)
			}
		}
	}
}

func TestTreeForDerivesUndeclaredSubLanes(t *testing.T) {
	tax := fixtureTax()
	cases := []struct {
		lane string
		want string
	}{
		{"gateway", "internal/gateway/**"},                  // declared: its own row, unchanged
		{"gateway/server", "internal/gateway/server/**"},    // derived directory sub-lane
		{"gateway/server.go", "internal/gateway/server.go"}, // derived FILE sub-lane, no /** suffix
		{"cmd/fak/dispatch_wave.go", "cmd/fak/dispatch_wave.go"},
		{"claude/skills", ".claude/skills/**"}, // the dotted root must keep its dot in the TREE
	}
	for _, c := range cases {
		got := tax.TreeFor(c.lane)
		if len(got) != 1 || got[0] != c.want {
			t.Errorf("TreeFor(%q) = %v, want [%q]", c.lane, got, c.want)
		}
	}
	// No declared ancestor at all -> no tree, so Decide falls through to the conservative
	// empty-tree overlap rule rather than inventing a narrow one.
	if got := tax.TreeFor("nosuchroot/deep"); got != nil {
		t.Errorf("TreeFor on an undeclared root = %v, want nil", got)
	}
	// An ancestor whose tree is a bare file list has nothing to descend into: fall back to the
	// ancestor's own COARSE tree, never to a narrower guess.
	got := tax.TreeFor("release/notes")
	if len(got) != 2 || got[0] != "version" || got[1] != "CHANGELOG.md" {
		t.Errorf("TreeFor(release/notes) = %v, want the ancestor's coarse tree", got)
	}
}

func TestIsExclusiveInheritsDownward(t *testing.T) {
	tax := fixtureTax()
	for _, lane := range []string{"abi", "abi/registry", "abi/registry/table.go"} {
		if !tax.IsExclusive(lane) {
			t.Errorf("IsExclusive(%q) = false; an exclusive ancestor must not be escapable by naming a narrower unit", lane)
		}
	}
	if tax.IsExclusive("gateway/server") {
		t.Error("IsExclusive must not leak across siblings")
	}
}

func TestDecideAdmitsDisjointSubLanesAndBlocksAncestors(t *testing.T) {
	tax := fixtureTax()
	held := []Lease{{ID: "dispatch-lane-gateway_server", Lane: "gateway/server", Tree: []string{"internal/gateway/server/**"}, Holder: "hostA:1"}}

	// Sibling sub-lane: the whole point. Under the flat vocabulary both were lane "gateway" and
	// the second worker serialized behind the first.
	if v := Decide(Request{Surface: SurfaceDispatch, Lane: "gateway/router", Holder: "hostB:2"}, held, tax); !v.Admit {
		t.Fatalf("disjoint sub-lane must be admitted, got refusal: %s", v.Detail)
	}
	// The parent still blocks: it may edit anywhere beneath the live child.
	v := Decide(Request{Surface: SurfaceDispatch, Lane: "gateway", Holder: "hostB:2"}, held, tax)
	if v.Admit {
		t.Fatal("the parent lane must NOT be admitted while a child lane is held")
	}
	if v.Conflicts[0].Kind != ConflictLaneAncestry {
		t.Errorf("parent/child conflict kind = %q, want %q", v.Conflicts[0].Kind, ConflictLaneAncestry)
	}
	// And a deeper descendant of the held child blocks too.
	if v := Decide(Request{Surface: SurfaceDispatch, Lane: "gateway/server/handler.go", Holder: "hostB:2"}, held, tax); v.Admit {
		t.Fatal("a descendant of a held sub-lane must NOT be admitted")
	}
}

// TestDecideFlatVerdictsUnchanged pins the compatibility contract at the Decide boundary: with
// only flat lanes in play, every arm returns what it returned before sub-lanes existed.
func TestDecideFlatVerdictsUnchanged(t *testing.T) {
	tax := fixtureTax()
	held := []Lease{{ID: "dispatch-lane-gateway", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Holder: "hostA:1"}}

	if v := Decide(Request{Lane: "gateway", Holder: "hostB:2"}, held, tax); v.Admit || v.Conflicts[0].Kind != ConflictSameLane {
		t.Errorf("same flat lane must still refuse with %q, got admit=%v kind=%q", ConflictSameLane, v.Admit, v.Conflicts[0].Kind)
	}
	if v := Decide(Request{Lane: "docs", Holder: "hostB:2"}, held, tax); !v.Admit {
		t.Errorf("a disjoint flat lane must still be admitted, got %s", v.Detail)
	}
	if v := Decide(Request{Lane: "abi", Holder: "hostB:2"}, held, tax); v.Admit || v.Conflicts[0].Kind != ConflictExclusiveLane {
		t.Errorf("an exclusive flat lane must still run alone, got admit=%v", v.Admit)
	}
	// A lane that is only a STRING prefix-sibling must not be treated as RELATED. It is still
	// refused, but on the pre-existing conservative ground: an undeclared lane resolves to no
	// tree, and an empty tree overlaps everything (unchanged from before sub-lanes existed).
	v := Decide(Request{Lane: "gatewayx", Holder: "hostB:2"}, held, tax)
	if v.Admit || v.Conflicts[0].Kind != ConflictTreeOverlap {
		t.Errorf("string-prefix sibling must refuse as %q (unknown tree), got admit=%v kind=%q", ConflictTreeOverlap, v.Admit, v.Conflicts[0].Kind)
	}
}

// TestLeaseIDRoundTripsSubLanesAsOneRefSegment pins the wire encoding against internal/leaseref's
// validID rule: a lease id becomes the last segment of refs/fak/locks/<id>, so it may carry no
// `/` and may not end in `.lock`.
func TestLeaseIDRoundTripsSubLanesAsOneRefSegment(t *testing.T) {
	for _, lane := range []string{"gateway", "gateway/server", "gateway/server.go", "cmd/fak/dispatch_wave.go", "docs/a/b/c.lock"} {
		id := LeaseID(SurfaceDispatch, lane, "")
		if strings.Contains(id, "/") {
			t.Errorf("LeaseID(%q) = %q contains '/'; leaseref.validID rejects it and it would break git's dir/file ref rule", lane, id)
		}
		if strings.HasSuffix(id, ".lock") {
			t.Errorf("LeaseID(%q) = %q ends in .lock; git check-ref-format rejects that ref component", lane, id)
		}
		if got := LaneOfLeaseID(id); got != lane {
			t.Errorf("LaneOfLeaseID(LeaseID(%q)) = %q, want %q", lane, got, lane)
		}
	}
	// The pre-existing flat spelling is untouched, so live leases minted before this change
	// still decode to the same lane.
	if got := LeaseID("loop", "gateway", ""); got != "loop-lane-gateway" {
		t.Errorf("flat LeaseID drifted: %q", got)
	}
	if got := LaneOfLeaseID("resolve-gateway-5854"); got != "gateway" {
		t.Errorf("the grandfathered dispatch grammar must still decode, got %q", got)
	}
}

func TestLaneForPathAndTreeForAreInverses(t *testing.T) {
	tax := fixtureTax()
	cases := []struct {
		path            string
		leaf, dir, file string
	}{
		{"internal/gateway/server/handler.go", "gateway", "gateway/server", "gateway/server/handler.go"},
		{"internal/gateway/server.go", "gateway", "gateway", "gateway/server.go"},
		{"cmd/fak/dispatch_wave.go", "cmd", "cmd/fak", "cmd/fak/dispatch_wave.go"},
		{".claude/skills/verify/SKILL.md", "claude", "claude/skills/verify", "claude/skills/verify/SKILL.md"},
		{"README.md", "", "", ""}, // a root-level file belongs to no declared tree
	}
	for _, c := range cases {
		for gran, want := range map[Granularity]string{GranLeaf: c.leaf, GranDir: c.dir, GranFile: c.file} {
			if got := LaneForPath(c.path, tax, gran); got != want {
				t.Errorf("LaneForPath(%q, gran=%d) = %q, want %q", c.path, gran, got, want)
			}
		}
		if c.file == "" {
			continue
		}
		// The inverse: the file lane's derived tree must name the path it came from.
		if got := tax.TreeFor(c.file); len(got) != 1 || got[0] != c.path {
			t.Errorf("TreeFor(LaneForPath(%q)) = %v, want the original path back", c.path, got)
		}
	}
}

// TestLaneSpaceExpandsWithGranularity is the multiplier witness in miniature: the SAME path set
// addresses strictly more lanes as the cut gets finer, and every coarser lane survives in the
// finer space (a parent stays addressable).
func TestLaneSpaceExpandsWithGranularity(t *testing.T) {
	tax := fixtureTax()
	paths := []string{
		"internal/gateway/server/handler.go",
		"internal/gateway/server/route.go",
		"internal/gateway/router/pick.go",
		"internal/gateway/serve.go",
		"cmd/fak/dispatch_wave.go",
		"cmd/fak/dispatch_tick.go",
		"docs/awesome-caching/README.md",
	}
	leaf := LaneSpace(paths, tax, GranLeaf)
	dir := LaneSpace(paths, tax, GranDir)
	file := LaneSpace(paths, tax, GranFile)

	if len(leaf) != 3 { // gateway, cmd, docs
		t.Errorf("GranLeaf space = %v, want the 3 declared lanes", leaf)
	}
	if !(len(dir) > len(leaf) && len(file) > len(dir)) {
		t.Fatalf("space must strictly widen with granularity: leaf=%d dir=%d file=%d", len(leaf), len(dir), len(file))
	}
	in := func(set []string, want string) bool {
		for _, s := range set {
			if s == want {
				return true
			}
		}
		return false
	}
	for _, coarse := range leaf {
		if !in(dir, coarse) || !in(file, coarse) {
			t.Errorf("coarse lane %q vanished from a finer space; a parent must stay addressable", coarse)
		}
	}
	// 7 paths that shared 3 lanes now address 7 distinct file lanes — the concurrency the flat
	// vocabulary could not express.
	for _, want := range []string{"gateway/server/handler.go", "gateway/router/pick.go", "cmd/fak/dispatch_wave.go"} {
		if !in(file, want) {
			t.Errorf("file lane %q missing from LaneSpace(GranFile) = %v", want, file)
		}
	}
}
