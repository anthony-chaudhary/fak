package opttarget

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// TestDiscoverDirFindsAnnotatedFixtures proves the scanner harvests exactly the
// annotated consts (alpha, beta), not the plain one (gamma), and lowers each
// annotation into a well-formed OptTarget — the core Phase 1 auto-discovery move.
func TestDiscoverDirFindsAnnotatedFixtures(t *testing.T) {
	got, err := DiscoverDir(filepath.Join("testdata", "discover"))
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("discovered %d targets, want 2 (alpha,beta): %+v", len(got), got)
	}
	// Sorted by Name: alpha then beta.
	alpha, beta := got[0], got[1]
	if alpha.Name != "alpha" || alpha.Metric != "alpha_score" || alpha.Direction != HigherBetter {
		t.Errorf("alpha = %+v, want name=alpha metric=alpha_score dir=higher", alpha)
	}
	if alpha.Site.Const != "Alpha" || alpha.Measurer != "fake" {
		t.Errorf("alpha site/measurer = %+v", alpha)
	}
	if len(alpha.Grammar.Ints) != 3 || alpha.Grammar.Ints[2] != 3 {
		t.Errorf("alpha sweep = %v, want [1 2 3]", alpha.Grammar.Ints)
	}
	if beta.Name != "beta" || beta.Direction != LowerBetter || beta.Site.Const != "Beta" {
		t.Errorf("beta = %+v, want name=beta dir=lower const=Beta", beta)
	}
	if len(beta.Grammar.Ints) != 2 || beta.Grammar.Ints[1] != 20 {
		t.Errorf("beta sweep = %v, want [10 20]", beta.Grammar.Ints)
	}
	for _, tg := range got {
		if tg.Name == "Gamma" || tg.Site.Const == "Gamma" {
			t.Errorf("Gamma was discovered but is unannotated: %+v", tg)
		}
		if err := tg.Validate(); err != nil {
			t.Errorf("discovered target %q does not validate: %v", tg.Name, err)
		}
	}
}

// TestDiscoverDirRejectsMalformed proves a typo'd annotation is an ERROR (caught),
// never a silently dropped or mis-read target.
func TestDiscoverDirRejectsMalformed(t *testing.T) {
	_, err := DiscoverDir(filepath.Join("testdata", "discover_bad"))
	if err == nil {
		t.Fatal("DiscoverDir over a malformed annotation returned nil error, want a refusal")
	}
	if !strings.Contains(err.Error(), "dir=") {
		t.Errorf("error does not name the bad direction: %v", err)
	}
}

// TestDiscoverRealCacheSizeTarget is the live ratchet: the real DefaultCacheSize
// tunable carries a fak:opttarget annotation, so DiscoverDir over the sibling
// rsiloop package MUST find the cache-size target — and its declaration must agree
// with the hand-authored CacheSizeTarget (the same site/metric/direction/measurer/
// sweep). Removing or renaming the annotation turns this red — a target cannot
// silently drop out of the program.
func TestDiscoverRealCacheSizeTarget(t *testing.T) {
	got, err := DiscoverDir(filepath.Join("..", "rsiloop"))
	if err != nil {
		t.Fatalf("discover rsiloop: %v", err)
	}
	var found *OptTarget
	for i := range got {
		if got[i].Name == "lru-cache-size" {
			found = &got[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("the annotated DefaultCacheSize tunable was not discovered; got %d targets: %+v", len(got), got)
	}
	if err := Check(got, []string{"lru-cache-size"}); err != nil {
		t.Errorf("ratchet: %v", err)
	}

	want := CacheSizeTarget([]int{4, 5, 6, 8})
	if found.Metric != want.Metric || found.Direction != want.Direction ||
		found.Measurer != want.Measurer || found.Site.Const != want.Site.Const {
		t.Errorf("discovered cache-size target disagrees with CacheSizeTarget:\n got %+v\n want %+v", *found, want)
	}
	if len(found.Grammar.Ints) != 4 || found.Grammar.Ints[3] != 8 {
		t.Errorf("discovered sweep = %v, want [4 5 6 8]", found.Grammar.Ints)
	}
	if !strings.HasSuffix(found.Site.Path, "tunable.go") {
		t.Errorf("discovered site path = %q, want it to end in tunable.go", found.Site.Path)
	}
}

// TestCheckRatchet exercises the coverage keep-bit directly: a satisfied set
// passes; a missing required target fails with the name surfaced.
func TestCheckRatchet(t *testing.T) {
	disc := []OptTarget{{Name: "alpha"}, {Name: "beta"}}
	if err := Check(disc, []string{"alpha", "beta"}); err != nil {
		t.Errorf("full coverage should pass, got %v", err)
	}
	err := Check(disc, []string{"alpha", "beta", "gamma"})
	if err == nil {
		t.Fatal("missing required target should fail the ratchet")
	}
	if !strings.Contains(err.Error(), "gamma") {
		t.Errorf("ratchet error should name the missing target, got %v", err)
	}
}

// TestMarshalInventoryRoundTrips proves the emitted JSON inventory is valid and
// re-parses to the same targets — the Phase 1 `fak opt discover` payload.
func TestMarshalInventoryRoundTrips(t *testing.T) {
	got, err := DiscoverDir(filepath.Join("testdata", "discover"))
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	b, err := MarshalInventory(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back []OptTarget
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("inventory is not valid JSON: %v\n%s", err, b)
	}
	if len(back) != len(got) {
		t.Fatalf("round-trip len %d != %d", len(back), len(got))
	}
	for i := range got {
		if back[i].Name != got[i].Name || back[i].Site.Const != got[i].Site.Const {
			t.Errorf("round-trip target %d = %+v, want %+v", i, back[i], got[i])
		}
	}
}
