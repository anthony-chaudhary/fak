package laneadmit

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func tax() Taxonomy {
	return Taxonomy{
		Loaded:    true,
		Exclusive: map[string]bool{"abi": true, "release": true, "dos": true, "global": true},
		Trees: map[string][]string{
			"docs":    {"docs/**", "README.md"},
			"gateway": {"internal/gateway/**"},
			"model":   {"internal/model/**"},
			"abi":     {"internal/abi/**"},
		},
	}
}

func TestDecideAdmitsDisjointTrees(t *testing.T) {
	v := Decide(
		Request{Surface: SurfaceLoop, Lane: "docs", Holder: "me"},
		[]Lease{{ID: "resolve-gateway", Tree: []string{"internal/gateway/**"}, Holder: "peer"}},
		tax(),
	)
	if !v.Admit {
		t.Fatalf("disjoint trees must admit, got %+v", v)
	}
	if !reflect.DeepEqual(v.Tree, []string{"docs/**", "README.md"}) {
		t.Fatalf("lane tree must fall back to the taxonomy tree, got %v", v.Tree)
	}
}

func TestDecideRefusesTreeOverlap(t *testing.T) {
	v := Decide(
		Request{Surface: SurfaceLoop, Tree: []string{"internal/gateway/http.go"}, Holder: "me"},
		[]Lease{{ID: "resolve-gateway", Tree: []string{"internal/gateway/**"}, Holder: "peer"}},
		tax(),
	)
	if v.Admit {
		t.Fatal("overlapping tree must refuse")
	}
	if v.Reason != ReasonCollisionRisk {
		t.Fatalf("refusal must carry the closed-vocabulary reason, got %q", v.Reason)
	}
	if len(v.Conflicts) != 1 || v.Conflicts[0].Kind != ConflictTreeOverlap {
		t.Fatalf("want one tree_overlap conflict, got %+v", v.Conflicts)
	}
}

func TestDecideRefusesSameLaneOnDisjointTrees(t *testing.T) {
	// The dos-arbitrate rule geometry alone never honored: a named lane
	// serializes even when the two holders narrowed their trees disjoint.
	v := Decide(
		Request{Surface: SurfaceManual, Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, Holder: "me"},
		[]Lease{{ID: "resolve-gateway", Tree: []string{"internal/gateway/metrics.go"}, Holder: "peer"}},
		tax(),
	)
	if v.Admit {
		t.Fatal("same named lane must serialize even on disjoint trees")
	}
	if v.Conflicts[0].Kind != ConflictSameLane {
		t.Fatalf("want same_lane conflict, got %+v", v.Conflicts)
	}
}

func TestDecideRefusesExclusiveLane(t *testing.T) {
	// Requesting an exclusive lane conflicts with any live lease at all.
	v := Decide(
		Request{Surface: SurfaceManual, Lane: "release", Holder: "me"},
		[]Lease{{ID: "resolve-docs", Tree: []string{"docs/**"}, Holder: "peer"}},
		tax(),
	)
	if v.Admit || v.Conflicts[0].Kind != ConflictExclusiveLane {
		t.Fatalf("exclusive request must refuse against any live lease, got %+v", v)
	}
	// A live lease on an exclusive lane conflicts with any request.
	v = Decide(
		Request{Surface: SurfaceLoop, Lane: "docs", Holder: "me"},
		[]Lease{{ID: "coord-lane-release", Tree: []string{"VERSION"}, Holder: "peer"}},
		tax(),
	)
	if v.Admit || v.Conflicts[0].Kind != ConflictExclusiveLane {
		t.Fatalf("live exclusive lease must refuse every request, got %+v", v)
	}
}

func TestDecideSkipsOwnLease(t *testing.T) {
	v := Decide(
		Request{Surface: SurfaceLoop, Lane: "gateway", LeaseID: "loop-lane-gateway", Holder: "me"},
		[]Lease{{ID: "loop-lane-gateway", Lane: "gateway", Tree: []string{"internal/gateway/**"}, Holder: "me"}},
		tax(),
	)
	if !v.Admit {
		t.Fatalf("a caller's own lease (renew/re-entrant acquire) must not conflict, got %+v", v)
	}
}

func TestDecideEmptyTreeConservativelyCollides(t *testing.T) {
	// dispatchorder.TreesOverlap treats an empty tree as overlapping everything;
	// the seam must keep that conservative floor.
	v := Decide(
		Request{Surface: SurfaceManual, Holder: "me"},
		[]Lease{{ID: "resolve-docs", Tree: []string{"docs/**"}, Holder: "peer"}},
		tax(),
	)
	if v.Admit {
		t.Fatal("a tree-less request must conservatively collide with any live lease")
	}
}

func TestDecideUnloadedTaxonomySkipsLaneModes(t *testing.T) {
	// With no taxonomy, lane-mode rules cannot fire — but geometry still guards.
	v := Decide(
		Request{Surface: SurfaceLoop, Lane: "release", Tree: []string{"internal/loopdrive/**"}, Holder: "me"},
		[]Lease{{ID: "resolve-docs", Tree: []string{"docs/**"}, Holder: "peer"}},
		Taxonomy{},
	)
	if !v.Admit {
		t.Fatalf("unloaded taxonomy + disjoint trees must admit, got %+v", v)
	}
	v = Decide(
		Request{Surface: SurfaceLoop, Tree: []string{"docs/README.md"}, Holder: "me"},
		[]Lease{{ID: "resolve-docs", Tree: []string{"docs/**"}, Holder: "peer"}},
		Taxonomy{},
	)
	if v.Admit {
		t.Fatal("geometry must still refuse without a taxonomy")
	}
}

func TestLaneOfLeaseID(t *testing.T) {
	cases := map[string]string{
		"resolve-gateway":       "gateway", // dispatch lane lease
		"resolve-gateway-1234":  "gateway", // dispatch issue lease
		"loop-lane-docs":        "docs",    // shared grammar: loop drive
		"coord-lane-model":      "model",   // shared grammar: manual session
		"resolve-":              "",
		"session-abcdef":        "",
		"some-opaque-lock-name": "",
		"":                      "",
	}
	for id, want := range cases {
		if got := LaneOfLeaseID(id); got != want {
			t.Errorf("LaneOfLeaseID(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestLeaseIDMintsAndInverts(t *testing.T) {
	id := LeaseID("loop", "gateway", "")
	if id != "loop-lane-gateway" {
		t.Fatalf("LeaseID lane form = %q", id)
	}
	if LaneOfLeaseID(id) != "gateway" {
		t.Fatalf("LaneOfLeaseID must invert LeaseID, got %q", LaneOfLeaseID(id))
	}
	if id := LeaseID("loop", "", "nightly goal"); id != "loop-nightly-goal" {
		t.Fatalf("LeaseID scope form = %q", id)
	}
	if id := LeaseID("", "", ""); id != "coord" {
		t.Fatalf("LeaseID zero form = %q", id)
	}
}

func TestParseTaxonomy(t *testing.T) {
	data := []byte(`
# comment
[lanes]
concurrent = [
  "docs", "gateway", # trailing comment
  "model",
]
exclusive = ["abi", "release",
  "dos", "global"]
autopick = ["docs"]

[lanes.trees]
docs    = ["docs/**", "README.md"] # comment
gateway = ["internal/gateway/**"]

[reasons.COLLISION_RISK]
summary = "not a lane"
`)
	parsed := ParseTaxonomy(data)
	if !parsed.Loaded {
		t.Fatal("taxonomy must load")
	}
	for _, lane := range []string{"abi", "release", "dos", "global"} {
		if !parsed.Exclusive[lane] {
			t.Errorf("exclusive lane %q missing (multi-line list must parse)", lane)
		}
	}
	if parsed.Exclusive["docs"] || parsed.Exclusive["autopick"] {
		t.Errorf("non-exclusive tokens leaked into the exclusive set: %v", parsed.Exclusive)
	}
	if !reflect.DeepEqual(parsed.Trees["docs"], []string{"docs/**", "README.md"}) {
		t.Errorf("docs tree = %v", parsed.Trees["docs"])
	}
	if len(parsed.Trees["gateway"]) != 1 {
		t.Errorf("gateway tree = %v", parsed.Trees["gateway"])
	}
	if _, ok := parsed.Trees["summary"]; ok {
		t.Error("[reasons.*] content must not parse as a lane tree")
	}
	if ParseTaxonomy(nil).Loaded {
		t.Error("empty bytes must report an unloaded taxonomy")
	}
}

func TestParseTaxonomyOnRealDosToml(t *testing.T) {
	// The real dos.toml is the contract this parser exists for; pin the
	// load-bearing entries so a format drift fails here, not in a live refusal.
	b, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Skipf("repo dos.toml unavailable: %v", err)
	}
	parsed := ParseTaxonomy(b)
	if !parsed.Loaded {
		t.Fatal("repo dos.toml must load")
	}
	for _, lane := range []string{"abi", "release", "dos", "global"} {
		if !parsed.Exclusive[lane] {
			t.Errorf("repo dos.toml: exclusive lane %q not parsed", lane)
		}
	}
	if len(parsed.Trees["gateway"]) == 0 || len(parsed.Trees["docs"]) == 0 {
		t.Errorf("repo dos.toml: expected gateway/docs lane trees, got gateway=%v docs=%v",
			parsed.Trees["gateway"], parsed.Trees["docs"])
	}
}
