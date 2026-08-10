package selfquery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeRepo lays down the smallest tree devindex.Load accepts: a dos.toml with one
// [lanes.trees] leaf. Load then emits a "leaf:<lane>" dev card, which is all the
// multi-root fan-out test needs to prove attribution.
func writeLaneRepo(t *testing.T, lane, desc string) string {
	t.Helper()
	dir := t.TempDir()
	toml := "[lanes.trees]\n" + lane + " = [\"internal/" + lane + "/**\"]  # " + desc + "\n"
	if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte(toml), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	return abs
}

// TestLoadManyAttributesBothRoots is the ship-alone witness for #3435: a two-root
// load returns cards from BOTH roots, each stamped with the checkout it came from.
func TestLoadManyAttributesBothRoots(t *testing.T) {
	rootA := writeLaneRepo(t, "alpha", "the alpha lane")
	rootB := writeLaneRepo(t, "beta", "the beta lane")

	mc, err := LoadMany([]string{rootA, rootB}, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("LoadMany: %v", err)
	}
	if len(mc.Skipped()) != 0 {
		t.Fatalf("unexpected skips: %v", mc.Skipped())
	}

	cards := mc.Cards(PlaneDev)
	byRoot := map[string]bool{}
	var sawAlpha, sawBeta FeatureCard
	for _, c := range cards {
		if c.Root == "" {
			t.Errorf("card %q has empty Root — not attributed", c.Name)
		}
		byRoot[c.Root] = true
		switch c.Name {
		case "leaf:alpha":
			sawAlpha = c
		case "leaf:beta":
			sawBeta = c
		}
	}
	if sawAlpha.Name == "" || sawAlpha.Root != rootA {
		t.Errorf("leaf:alpha not attributed to rootA: got Root=%q want %q", sawAlpha.Root, rootA)
	}
	if sawBeta.Name == "" || sawBeta.Root != rootB {
		t.Errorf("leaf:beta not attributed to rootB: got Root=%q want %q", sawBeta.Root, rootB)
	}
	if len(byRoot) != 2 {
		t.Errorf("expected cards from 2 distinct roots, got %d: %v", len(byRoot), byRoot)
	}
}

// TestLoadManyQueryRanksUnion proves the ranker runs over the merged union with
// provenance intact — GQL's WHERE/ORDER over the concatenated rows, for free.
func TestLoadManyQueryRanksUnion(t *testing.T) {
	rootA := writeLaneRepo(t, "alpha", "the alpha lane")
	rootB := writeLaneRepo(t, "beta", "the beta lane")
	mc, err := LoadMany([]string{rootA, rootB}, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("LoadMany: %v", err)
	}
	resp, err := mc.Query(Request{Query: "alpha", Plane: PlaneDev, Limit: 5})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(resp.Cards) == 0 {
		t.Fatal("no cards ranked for query 'alpha'")
	}
	top := resp.Cards[0]
	if top.Name != "leaf:alpha" {
		t.Errorf("top card = %q, want leaf:alpha", top.Name)
	}
	if top.Root != rootA {
		t.Errorf("top card Root = %q, want %q", top.Root, rootA)
	}
}

// TestLoadManySkipsBadRoot is the GQL do-not: one unreadable sibling repo is
// skipped-and-reported, never fatal to the whole fan-out.
func TestLoadManySkipsBadRoot(t *testing.T) {
	rootA := writeLaneRepo(t, "alpha", "the alpha lane")
	missing := filepath.Join(t.TempDir(), "no-such-repo")

	mc, err := LoadMany([]string{rootA, missing}, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("LoadMany should tolerate a bad root, got: %v", err)
	}
	if len(mc.Roots()) != 1 || mc.Roots()[0] != rootA {
		t.Errorf("Roots() = %v, want [%s]", mc.Roots(), rootA)
	}
	skipped := mc.Skipped()
	if len(skipped) != 1 {
		t.Fatalf("Skipped() = %v, want exactly the missing root", skipped)
	}
	if !strings.Contains(skipped[0].Error(), "no-such-repo") {
		t.Errorf("skip did not name the bad root: %v", skipped[0])
	}

	// Every root bad -> fatal (nothing to query).
	if _, err := LoadMany([]string{missing}, Options{DevLoader: testDevLoader}); err == nil {
		t.Error("LoadMany with only bad roots should error")
	}
}

// TestSingleRootBackCompat pins the back-compat guarantee: a single-root call
// leaves Root empty and its JSON omits the field, so single-root output and
// SummaryDigest are byte-identical to the pre-#3435 shape.
func TestSingleRootBackCompat(t *testing.T) {
	rootA := writeLaneRepo(t, "alpha", "the alpha lane")

	cat, err := Load(rootA, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	resp, err := cat.Query(Request{Query: "alpha", Plane: PlaneDev, Limit: 5})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	for _, c := range resp.Cards {
		if c.Root != "" {
			t.Errorf("single-root card %q leaked Root=%q", c.Name, c.Root)
		}
		b, _ := json.Marshal(c)
		if strings.Contains(string(b), "\"root\"") {
			t.Errorf("single-root card JSON must omit root: %s", b)
		}
	}

	// A one-root LoadMany returns the same card names as the single-root Load,
	// proving the fan-out is a pure superset (ranking unaffected for one root).
	single := map[string]bool{}
	for _, c := range cat.Cards(PlaneDev) {
		single[c.Name] = true
	}
	mc, err := LoadMany([]string{rootA}, Options{DevLoader: testDevLoader})
	if err != nil {
		t.Fatalf("LoadMany one root: %v", err)
	}
	for _, c := range mc.Cards(PlaneDev) {
		if !single[c.Name] {
			t.Errorf("LoadMany produced card %q absent from single-root Load", c.Name)
		}
	}
}
