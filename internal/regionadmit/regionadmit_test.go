package regionadmit

import (
	"os"
	"path/filepath"
	"testing"
)

func testTaxonomy() Taxonomy {
	return Taxonomy{
		Exclusive: map[string]bool{"abi": true, "release": true, "global": true},
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"docs":    {"docs/**", "README.md"},
			"cmd":     {"cmd/**"},
			"abi":     {"internal/abi/**"},
			"global":  {"**/*"},
		},
	}
}

func TestDecideAdmitsDisjointRegions(t *testing.T) {
	dec := Decide(
		Request{Actor: "loop:nightly", Lane: "gateway"},
		[]Lease{{ID: "resolve-docs", Holder: "peer", Tree: []string{"docs/**", "README.md"}}},
		testTaxonomy(),
	)
	if !dec.Admit {
		t.Fatalf("disjoint lanes must admit, got refusal: %+v", dec)
	}
}

func TestDecideRefusesTreeOverlap(t *testing.T) {
	dec := Decide(
		Request{Actor: "loop:nightly", Tree: []string{"internal/gateway/http.go"}},
		[]Lease{{ID: "resolve-gateway", Holder: "peer", Tree: []string{"internal/gateway/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("overlapping tree must refuse")
	}
	if dec.Reason != ReasonCollisionRisk {
		t.Fatalf("reason = %q, want %q", dec.Reason, ReasonCollisionRisk)
	}
	if dec.Rung != RungTreeOverlap {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungTreeOverlap)
	}
	if dec.Conflict == nil || dec.Conflict.ID != "resolve-gateway" {
		t.Fatalf("conflict evidence must name the live lease, got %+v", dec.Conflict)
	}
}

func TestDecideSerializesSameLaneOnDisjointLookingTrees(t *testing.T) {
	// The live lease recorded only its tree; lane inference (LaneOf) must give
	// it back its lane semantics, and a same-lane request must refuse even
	// though the REQUESTED explicit tree does not overlap.
	dec := Decide(
		Request{Actor: "session:me", Lane: "gateway", Tree: []string{"docs/gateway.md"}},
		[]Lease{{ID: "resolve-gateway", Holder: "peer", Tree: []string{"internal/gateway/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("same named lane must serialize")
	}
	if dec.Rung != RungSameLane {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungSameLane)
	}
}

func TestDecideExclusiveLaneRequestRunsAlone(t *testing.T) {
	dec := Decide(
		Request{Actor: "op", Lane: "abi"},
		[]Lease{{ID: "resolve-docs", Holder: "peer", Tree: []string{"docs/**", "README.md"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("an exclusive lane request must refuse while any lease is live")
	}
	if dec.Rung != RungExclusiveRequested {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungExclusiveRequested)
	}
	// And with nothing live it admits.
	if dec := Decide(Request{Actor: "op", Lane: "abi"}, nil, testTaxonomy()); !dec.Admit {
		t.Fatalf("exclusive lane with nothing live must admit, got %+v", dec)
	}
}

func TestDecideExclusiveLiveLeaseBlocksEverything(t *testing.T) {
	dec := Decide(
		Request{Actor: "loop:nightly", Lane: "docs"},
		[]Lease{{ID: "release-cut", Holder: "op", Tree: []string{"internal/abi/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("a live lease on an exclusive lane must refuse every new region")
	}
	if dec.Rung != RungExclusiveLive {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungExclusiveLive)
	}
}

func TestDecideSkipsOwnLease(t *testing.T) {
	dec := Decide(
		Request{Actor: "loop:nightly", Lane: "gateway", SelfID: "loop-nightly"},
		[]Lease{{ID: "loop-nightly", Holder: "loop:nightly", Tree: []string{"internal/gateway/**"}}},
		testTaxonomy(),
	)
	if !dec.Admit {
		t.Fatalf("a caller's own lease must not conflict with itself, got %+v", dec)
	}
}

func TestDecideEmptyTreeCollidesConservatively(t *testing.T) {
	dec := Decide(
		Request{Actor: "unknown"},
		[]Lease{{ID: "resolve-docs", Holder: "peer", Tree: []string{"docs/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("an empty request tree is unknown blast radius and must refuse against any live lease")
	}
	if dec.Rung != RungTreeOverlap {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungTreeOverlap)
	}
}

func TestDecideReadOnlyRequestAdmitsAgainstEverything(t *testing.T) {
	// A read-only request writes NOTHING (a provably empty write footprint), distinct from the
	// tree-less unknown-blast-radius case: admitted against a live lease it would otherwise
	// collide with on every rung — an empty region, the same named lane, an exclusive lane.
	live := []Lease{{ID: "resolve-gateway", Holder: "peer", Lane: "gateway", Tree: []string{"internal/gateway/**"}}}
	for _, req := range []Request{
		{Actor: "me", ReadOnly: true},                  // empty region — would conservatively collide
		{Actor: "me", Lane: "gateway", ReadOnly: true}, // same named lane — would serialize
		{Actor: "me", Lane: "release", ReadOnly: true}, // exclusive lane requested — would run alone
	} {
		if dec := Decide(req, live, testTaxonomy()); !dec.Admit {
			t.Fatalf("a read-only request must admit, got %+v for req %+v", dec, req)
		}
	}
}

func TestDecideReadOnlyLeaseBlocksNothing(t *testing.T) {
	// A read-only live lease holds nothing writable, so it blocks no region — not even a live
	// lease sitting on an exclusive lane, which otherwise blocks every new region.
	dec := Decide(
		Request{Actor: "me", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}},
		[]Lease{{ID: "resolve-release", Holder: "peer", Lane: "release", Tree: []string{"VERSION"}, ReadOnly: true}},
		testTaxonomy(),
	)
	if !dec.Admit {
		t.Fatalf("a request must admit against a read-only live lease, got %+v", dec)
	}
}

func TestResolveTreeUsesLaneCanonicalTree(t *testing.T) {
	tree := ResolveTree(Request{Lane: "docs"}, testTaxonomy())
	if len(tree) != 2 || tree[0] != "docs/**" {
		t.Fatalf("ResolveTree = %v, want the docs lane tree", tree)
	}
	explicit := ResolveTree(Request{Lane: "docs", Tree: []string{"docs/notes/**"}}, testTaxonomy())
	if len(explicit) != 1 || explicit[0] != "docs/notes/**" {
		t.Fatalf("an explicit tree must win over the lane tree, got %v", explicit)
	}
}

func TestLaneOfMatchesTreeSetOrderInsensitive(t *testing.T) {
	tax := testTaxonomy()
	if lane := LaneOf([]string{"README.md", "docs/**"}, tax); lane != "docs" {
		t.Fatalf("LaneOf = %q, want docs", lane)
	}
	if lane := LaneOf(nil, tax); lane != "" {
		t.Fatalf("an empty tree owns no lane, got %q", lane)
	}
}

func TestLaneOfClassifiesNarrowedTreesByContainment(t *testing.T) {
	tax := testTaxonomy()
	// A lease narrowed INSIDE its lane keeps its lane semantics.
	if lane := LaneOf([]string{"internal/gateway/http/**"}, tax); lane != "gateway" {
		t.Fatalf("narrowed subtree LaneOf = %q, want gateway", lane)
	}
	// A subset of a lane's multi-glob tree is still that lane's.
	if lane := LaneOf([]string{"docs/**"}, tax); lane != "docs" {
		t.Fatalf("subset tree LaneOf = %q, want docs", lane)
	}
	// A literal file inside a lane tree.
	if lane := LaneOf([]string{"internal/abi/types.go"}, tax); lane != "abi" {
		t.Fatalf("literal file LaneOf = %q, want abi", lane)
	}
	// A tree BROADER than any lane claims nothing.
	if lane := LaneOf([]string{"internal/**"}, tax); lane != "" {
		t.Fatalf("broader-than-lane tree must own no lane, got %q", lane)
	}
	// A tree spanning two lanes claims nothing (geometry still protects it).
	if lane := LaneOf([]string{"internal/gateway/**", "docs/**"}, tax); lane != "" {
		t.Fatalf("multi-lane tree must own no lane, got %q", lane)
	}
	// A catch-all container never claims by containment: with only the global
	// lane containing this path, it stays lane-less rather than exclusive.
	only := Taxonomy{Exclusive: map[string]bool{"global": true}, Trees: map[string][]string{"global": {"**/*"}}}
	if lane := LaneOf([]string{"anything/at/all.go"}, only); lane != "" {
		t.Fatalf("catch-all containment must not classify, got %q", lane)
	}
}

func TestDecideAdmitsDisjointNarrowedSameLaneRegions(t *testing.T) {
	// Two STRICTLY-NARROWED, tree-disjoint sub-regions of ONE named lane are not
	// a real collision: geometry (rung 4) decides, so they co-run instead of
	// serializing on the shared lane name. The live lease's lane is recovered by
	// containment even though its tree is not the lane's canonical tree.
	// Precondition: the live lease's narrowed tree must classify back onto "gateway" so
	// the same-lane rung is actually ENTERED and bypassed via the both-narrowed
	// fall-through -- not skipped because the lane went unrecognized (which would admit
	// for the wrong reason and mask a lane-inference regression).
	if got := LaneOf([]string{"internal/gateway/http/**"}, testTaxonomy()); got != "gateway" {
		t.Fatalf("precondition: live narrowed tree must classify onto gateway, got %q", got)
	}
	dec := Decide(
		Request{Actor: "loop:b", Lane: "gateway", Tree: []string{"internal/gateway/relay/**"}},
		[]Lease{{ID: "loop-a", Holder: "loop:a", Tree: []string{"internal/gateway/http/**"}}},
		testTaxonomy(),
	)
	if !dec.Admit {
		t.Fatalf("two disjoint narrowed sub-regions of one lane must co-admit, got refusal: %+v", dec)
	}
}

func TestDecideSerializesNarrowedAgainstWholeLane(t *testing.T) {
	// A narrowed sub-region vs a WHOLE-lane claim still serializes: the whole-lane
	// tree is not strictly narrower than itself, so the same-lane rung fires.
	dec := Decide(
		Request{Actor: "loop:b", Lane: "cmd", Tree: []string{"cmd/fak/slack_inbound.go"}},
		[]Lease{{ID: "resolve-cmd", Holder: "loop:a", Tree: []string{"cmd/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("a narrowed region vs a whole-lane claim must serialize")
	}
	if dec.Rung != RungSameLane {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungSameLane)
	}
}

func TestDecideRefusesOverlappingNarrowedSameLaneRegions(t *testing.T) {
	// Two narrowed sub-regions of one lane that OVERLAP still refuse — once the
	// same-lane rung defers to geometry, rung 4 catches the overlap and reports
	// it as a tree collision (the more diagnostic evidence).
	dec := Decide(
		Request{Actor: "loop:b", Lane: "cmd", Tree: []string{"cmd/fak/slack/**"}},
		[]Lease{{ID: "resolve-cmd-1", Holder: "loop:a", Tree: []string{"cmd/fak/**"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("two overlapping narrowed sub-regions must refuse")
	}
	if dec.Rung != RungTreeOverlap {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungTreeOverlap)
	}
}

func TestDecideExclusiveSubsetLeaseBlocksEverything(t *testing.T) {
	// The witnessed bypass: a lease on a SUBSET of an exclusive lane's tree
	// must still block every new region.
	dec := Decide(
		Request{Actor: "loop:x", Lane: "cmd"},
		[]Lease{{ID: "abi-edit", Holder: "op", Tree: []string{"internal/abi/types.go"}}},
		testTaxonomy(),
	)
	if dec.Admit {
		t.Fatal("a lease inside an exclusive lane's tree must block every new region")
	}
	if dec.Rung != RungExclusiveLive {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungExclusiveLive)
	}
}

func TestLoadTaxonomyReadsExclusiveAndTrees(t *testing.T) {
	dir := t.TempDir()
	toml := `# workspace
[lanes]
concurrent = [
  "gateway", # a comment with "quotes"
  "docs",
]
exclusive = ["abi", "release"]
autopick = [
  "gateway",
]

[lanes.trees]
gateway = ["internal/gateway/**"]
docs = ["docs/**", "README.md"]
abi = ["internal/abi/**"]

[reasons.COLLISION_RISK]
summary = "not a lane tree"
`
	if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(toml), 0o644); err != nil {
		t.Fatal(err)
	}
	tax, err := LoadTaxonomy(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !tax.Exclusive["abi"] || !tax.Exclusive["release"] || tax.Exclusive["gateway"] {
		t.Fatalf("exclusive = %v, want abi+release only", tax.Exclusive)
	}
	if got := tax.Trees["docs"]; len(got) != 2 || got[0] != "docs/**" || got[1] != "README.md" {
		t.Fatalf("docs tree = %v", got)
	}
	if got := tax.Trees["gateway"]; len(got) != 1 || got[0] != "internal/gateway/**" {
		t.Fatalf("gateway tree = %v", got)
	}
	if _, ok := tax.Trees["summary"]; ok {
		t.Fatal("keys outside [lanes.trees] must not leak into Trees")
	}
	if err := os.Remove(filepath.Join(dir, "dos.toml")); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTaxonomy(dir); err == nil {
		t.Fatal("a missing dos.toml must surface an error, not a silent empty taxonomy")
	}
}

func TestLoadTaxonomyBracketsInsideStringsAreContent(t *testing.T) {
	// The witnessed desync: a char-class glob ("x[!a]/**") or bracketed prose
	// in a quoted value must not change the reader's bracket depth — before
	// the quote-aware scan, one such value silently swallowed every later
	// lane, degrading lane semantics with no error surfaced.
	dir := t.TempDir()
	toml := `[lanes]
concurrent = [
  "weird",
  "gateway",
]
exclusive = ["abi"]

[lanes.trees]
weird = ["docs/x[!a/**"]
gateway = ["internal/gateway/**"]
abi = ["internal/abi/**"]
cmd = ["cmd/**"] # prose with [brackets] and an escaped quote \" inside a comment
`
	if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(toml), 0o644); err != nil {
		t.Fatal(err)
	}
	tax, err := LoadTaxonomy(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := tax.Trees["weird"]; len(got) != 1 || got[0] != "docs/x[!a/**" {
		t.Fatalf("char-class glob tree = %v", got)
	}
	if got := tax.Trees["gateway"]; len(got) != 1 || got[0] != "internal/gateway/**" {
		t.Fatalf("gateway tree lost after a bracketed value: %v", got)
	}
	if got := tax.Trees["cmd"]; len(got) != 1 || got[0] != "cmd/**" {
		t.Fatalf("cmd tree lost after a bracketed comment: %v", got)
	}
	if !tax.Exclusive["abi"] {
		t.Fatalf("exclusive set lost after a bracketed value: %v", tax.Exclusive)
	}
}

func TestLoadTaxonomyOnThisWorkspace(t *testing.T) {
	// The leaf must be able to read the real dos.toml two directories up —
	// the same file the dos arbiter and the dispatch router read.
	tax, err := LoadTaxonomy(filepath.Join("..", ".."))
	if err != nil {
		t.Skipf("workspace dos.toml unavailable: %v", err)
	}
	if !tax.Exclusive["abi"] || !tax.Exclusive["global"] {
		t.Fatalf("workspace exclusive lanes missing abi/global: %v", tax.Exclusive)
	}
	if got := tax.Trees["regionadmit"]; len(got) != 1 || got[0] != "internal/regionadmit/**" {
		t.Fatalf("regionadmit lane tree = %v, want its declared tree", got)
	}
}

func TestHierarchicalLaneResolutionAndAdmission(t *testing.T) {
	tax := testTaxonomy()

	got := ResolveTree(Request{Lane: "gateway/server"}, tax)
	want := []string{"internal/gateway/server/**"}
	if len(got) != 1 || got[0] != want[0] {
		t.Fatalf("ResolveTree(gateway/server) = %v, want %v", got, want)
	}

	exclusive := Decide(
		Request{Actor: "op", Lane: "abi/registry.go"},
		[]Lease{{ID: "docs", Holder: "peer", Tree: []string{"docs/**"}}},
		tax,
	)
	if exclusive.Admit || exclusive.Rung != RungExclusiveRequested {
		t.Fatalf("exclusive descendant = %+v, want %s refusal", exclusive, RungExclusiveRequested)
	}

	parentChild := Decide(
		Request{Actor: "me", Lane: "gateway/server"},
		[]Lease{{ID: "gateway", Holder: "peer", Tree: []string{"internal/gateway/**"}}},
		tax,
	)
	if parentChild.Admit || parentChild.Rung != RungSameLane {
		t.Fatalf("parent/child = %+v, want %s refusal", parentChild, RungSameLane)
	}

	siblings := Decide(
		Request{Actor: "me", Lane: "gateway/server"},
		[]Lease{{ID: "router", Holder: "peer", Tree: []string{"internal/gateway/router/**"}}},
		tax,
	)
	if !siblings.Admit {
		t.Fatalf("disjoint siblings must co-run, got %+v", siblings)
	}
}
