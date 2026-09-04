package wipattr

import (
	"reflect"
	"testing"
)

func h(file string, edit ...string) Hunk { return Hunk{File: file, Edit: edit} }

// TestAttributeOwnedOrphanShared is the core acceptance proof for #3874: every dirty
// hunk lands in exactly one of OWNED / ORPHAN / SHARED, with the right owner(s).
func TestAttributeOwnedOrphanShared(t *testing.T) {
	dirty := []Hunk{
		h("a.go", "+alpha"), // only sessA has it -> OWNED by sessA
		h("b.go", "+beta"),  // nobody has it     -> ORPHAN
		h("c.go", "+gamma"), // sessA and sessB   -> SHARED
	}
	checkpoints := map[string][]Hunk{
		"sessA": {h("a.go", "+alpha"), h("c.go", "+gamma")},
		"sessB": {h("c.go", "+gamma")},
	}
	got := Attribute(dirty, checkpoints)

	// Totality: one verdict per dirty hunk.
	if len(got) != len(dirty) {
		t.Fatalf("totality broken: got %d attributions for %d hunks", len(got), len(dirty))
	}
	byFile := map[string]Attribution{}
	for _, a := range got {
		byFile[a.File] = a
	}
	if a := byFile["a.go"]; a.State != AttrOwned || a.Owner != "sessA" {
		t.Errorf("a.go: want OWNED by sessA, got %s owner=%q", a.State, a.Owner)
	}
	if a := byFile["b.go"]; a.State != AttrOrphan || a.Owner != "" || len(a.Owners) != 0 {
		t.Errorf("b.go: want ORPHAN with no owner, got %s owner=%q owners=%v", a.State, a.Owner, a.Owners)
	}
	if a := byFile["c.go"]; a.State != AttrShared || !reflect.DeepEqual(a.Owners, []string{"sessA", "sessB"}) {
		t.Errorf("c.go: want SHARED by [sessA sessB], got %s owners=%v", a.State, a.Owners)
	}
}

// TestAttributeIgnoresLineOffsets proves attribution keys on edit content, not on
// line-number position: the same edit recorded at a different @@ offset still matches.
func TestAttributeIgnoresLineOffsets(t *testing.T) {
	// The checkpoint recorded the identical edit payload; only @@ offsets (which
	// ParseHunks discards) would differ in a real diff. Same Edit -> same Signature.
	dirty := []Hunk{h("x.go", "-old", "+new")}
	cps := map[string][]Hunk{"s1": {h("x.go", "-old", "+new")}}
	got := Attribute(dirty, cps)
	if len(got) != 1 || got[0].State != AttrOwned || got[0].Owner != "s1" {
		t.Fatalf("want OWNED by s1, got %+v", got)
	}
}

// TestAttributeDeterministic proves two runs over the same inputs are byte-identical
// and independent of Go's map iteration order.
func TestAttributeDeterministic(t *testing.T) {
	dirty := []Hunk{h("z.go", "+z"), h("a.go", "+a"), h("m.go", "+m")}
	cps := map[string][]Hunk{
		"beta":  {h("m.go", "+m")},
		"alpha": {h("m.go", "+m"), h("a.go", "+a")},
	}
	first := Attribute(dirty, cps)
	for i := 0; i < 20; i++ {
		if got := Attribute(dirty, cps); !reflect.DeepEqual(got, first) {
			t.Fatalf("non-deterministic on run %d:\n first=%+v\n got  =%+v", i, first, got)
		}
	}
	// Sorted by file.
	if first[0].File != "a.go" || first[1].File != "m.go" || first[2].File != "z.go" {
		t.Errorf("not sorted by file: %v %v %v", first[0].File, first[1].File, first[2].File)
	}
	// The SHARED claimants are sorted.
	if m := first[1]; m.State != AttrShared || !reflect.DeepEqual(m.Owners, []string{"alpha", "beta"}) {
		t.Errorf("m.go: want SHARED [alpha beta], got %s %v", m.State, m.Owners)
	}
}

// TestAttributeEmpties covers the fail-safe corners: no dirty hunks, and dirty hunks
// with no checkpoints at all (everything ORPHAN).
func TestAttributeEmpties(t *testing.T) {
	if got := Attribute(nil, map[string][]Hunk{"s": {h("a", "+x")}}); len(got) != 0 {
		t.Errorf("no dirty hunks: want empty, got %+v", got)
	}
	got := Attribute([]Hunk{h("a", "+x"), h("b", "+y")}, nil)
	if len(got) != 2 {
		t.Fatalf("want 2 attributions, got %d", len(got))
	}
	for _, a := range got {
		if a.State != AttrOrphan {
			t.Errorf("%s: want ORPHAN with no checkpoints, got %s", a.File, a.State)
		}
	}
	if orphans := Orphans(got); len(orphans) != 2 {
		t.Errorf("Orphans: want 2, got %d", len(orphans))
	}
}

func TestParseHunks(t *testing.T) {
	diff := `diff --git a/foo.go b/foo.go
index 111..222 100644
--- a/foo.go
+++ b/foo.go
@@ -1,3 +1,4 @@
 context line
-removed
+added one
+added two
diff --git a/bar.go b/bar.go
--- a/bar.go
+++ b/bar.go
@@ -10,2 +10,2 @@ func F() {
-old bar
+new bar
`
	got := ParseHunks(diff)
	want := []Hunk{
		{File: "foo.go", Edit: []string{"-removed", "+added one", "+added two"}},
		{File: "bar.go", Edit: []string{"-old bar", "+new bar"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseHunks mismatch:\n got=%+v\n want=%+v", got, want)
	}
}

// TestParseThenAttribute is the end-to-end pure path: parse a live diff and a
// checkpoint diff, then attribute — the exact composition the cmd shell performs.
func TestParseThenAttribute(t *testing.T) {
	live := `diff --git a/svc.go b/svc.go
--- a/svc.go
+++ b/svc.go
@@ -5,1 +5,2 @@
+ownedLine
diff --git a/lonely.go b/lonely.go
--- a/lonely.go
+++ b/lonely.go
@@ -1,0 +1,1 @@
+orphanLine
`
	// A checkpoint snapshot of svc.go placing the same edit at a different offset.
	cpDiff := `diff --git a/svc.go b/svc.go
--- a/svc.go
+++ b/svc.go
@@ -42,1 +42,2 @@
+ownedLine
`
	got := Attribute(ParseHunks(live), map[string][]Hunk{"sessX": ParseHunks(cpDiff)})
	if len(got) != 2 {
		t.Fatalf("want 2 attributions, got %d: %+v", len(got), got)
	}
	byFile := map[string]Attribution{}
	for _, a := range got {
		byFile[a.File] = a
	}
	if a := byFile["svc.go"]; a.State != AttrOwned || a.Owner != "sessX" {
		t.Errorf("svc.go: want OWNED by sessX, got %s owner=%q", a.State, a.Owner)
	}
	if a := byFile["lonely.go"]; a.State != AttrOrphan {
		t.Errorf("lonely.go: want ORPHAN, got %s", a.State)
	}
}

func TestAttributeInfersOwnershipWhenScopeEmptyAndLaneActive(t *testing.T) {
	dirty := []Hunk{
		h("internal/gateway/server.go", "+freshLine"),
		h("cmd/fak/other.go", "+orphanLine"),
	}
	// Checkpoint has no hunks recorded for either edit
	checkpoints := map[string][]Hunk{
		"sess1": {},
	}

	dosToml := []byte(`
[lanes]
concurrent = ["gateway"]

[lanes.trees]
gateway = ["internal/gateway/**"]
`)

	// sess1 is active in lane "gateway", Scope is empty, dos.toml matches internal/gateway/**
	got := Attribute(dirty, checkpoints,
		WithActiveLane("sess1", "gateway", nil),
		WithDOSToml(dosToml),
	)

	if len(got) != 2 {
		t.Fatalf("expected 2 attributions, got %d", len(got))
	}
	byFile := map[string]Attribution{}
	for _, a := range got {
		byFile[a.File] = a
	}

	// internal/gateway/server.go should be inferred as OWNED by sess1
	gw := byFile["internal/gateway/server.go"]
	if gw.State != AttrOwned || gw.Owner != "sess1" {
		t.Errorf("internal/gateway/server.go: want OWNED by sess1, got %s owner=%q", gw.State, gw.Owner)
	}

	// cmd/fak/other.go is not in the active lane, so it falls back to ORPHAN
	other := byFile["cmd/fak/other.go"]
	if other.State != AttrOrphan {
		t.Errorf("cmd/fak/other.go: want ORPHAN, got %s owner=%q", other.State, other.Owner)
	}
}

func TestAttributeExplicitScopeOverridesOrTakesPrecedence(t *testing.T) {
	dirty := []Hunk{
		h("pkg/foo.go", "+lineA"),
	}
	checkpoints := map[string][]Hunk{"s1": {}}

	got := Attribute(dirty, checkpoints,
		WithSessionScope(SessionScope{
			Session: "s1",
			Scope:   []string{"pkg/foo.go"},
			Active:  true,
		}),
	)
	if len(got) != 1 || got[0].State != AttrOwned || got[0].Owner != "s1" {
		t.Fatalf("want OWNED by s1 via explicit scope, got %+v", got)
	}
}

func TestAttributeInferredLaneSharedWhenMultipleActiveLanesMatch(t *testing.T) {
	dirty := []Hunk{
		h("internal/shared/util.go", "+sharedLine"),
	}
	checkpoints := map[string][]Hunk{"s1": {}, "s2": {}}

	manifest := []string{"internal/shared/**"}
	got := Attribute(dirty, checkpoints,
		WithActiveLane("s1", "shared", manifest),
		WithActiveLane("s2", "shared", manifest),
	)
	if len(got) != 1 || got[0].State != AttrShared {
		t.Fatalf("want SHARED across s1 and s2, got %+v", got)
	}
	if !reflect.DeepEqual(got[0].Owners, []string{"s1", "s2"}) {
		t.Errorf("owners = %v, want [s1 s2]", got[0].Owners)
	}
}
