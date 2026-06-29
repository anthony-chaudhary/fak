package supportmaturity

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// readClaims loads the repo's real CLAIMS.md — the bound witness the feature roster
// resolves against. It walks up from this test file (located via runtime.Caller) to the
// module root (the directory holding go.mod) and reads CLAIMS.md beside it. The witness is
// the LIVE ledger, not a fixture: that is what makes a rostered feature's rung a binding
// against the honesty ledger rather than a self-report.
func readClaims(t *testing.T) string {
	t.Helper()
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed; cannot locate the test source to find the repo root")
	}
	dir := filepath.Dir(here)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("walked to the filesystem root without finding go.mod")
		}
		dir = parent
	}
	b, err := os.ReadFile(filepath.Join(dir, "CLAIMS.md"))
	if err != nil {
		t.Fatalf("read CLAIMS.md at repo root %s: %v", dir, err)
	}
	return string(b)
}

// TestFromClaimTagTotalAndOrdered pins the register→ladder lowering: every closed tag
// lowers to a VALID rung, the lowering is order-preserving over the ledger's own honesty
// order (STUB < SIMULATED < SHIPPED), and an unrecognized tag floors to M0None.
func TestFromClaimTagTotalAndOrdered(t *testing.T) {
	want := map[ClaimTag]Rung{
		ClaimStub:      M1Fenced,
		ClaimSimulated: M3Runs,
		ClaimShipped:   M4Correct,
	}
	for tag, exp := range want {
		got := FromClaimTag(tag)
		if !got.Valid() {
			t.Fatalf("FromClaimTag(%q) = %v, not a closed M0–M7 rung", tag, got)
		}
		if got != exp {
			t.Fatalf("FromClaimTag(%q) = %s (%s), want %s (%s)", tag, got, got.Label(), exp, exp.Label())
		}
	}
	// Order-preserving over the honesty order.
	if !(FromClaimTag(ClaimStub).Less(FromClaimTag(ClaimSimulated)) &&
		FromClaimTag(ClaimSimulated).Less(FromClaimTag(ClaimShipped))) {
		t.Fatalf("register lowering is not order-preserving: STUB=%s SIMULATED=%s SHIPPED=%s",
			FromClaimTag(ClaimStub), FromClaimTag(ClaimSimulated), FromClaimTag(ClaimShipped))
	}
	// The register band is M1–M4 (the doctrine claim).
	lo, hi := FromClaimTag(ClaimStub), FromClaimTag(ClaimShipped)
	if lo != M1Fenced || hi != M4Correct {
		t.Fatalf("register band = [%s,%s], want [M1,M4]", lo, hi)
	}
	// Unknown tag floors to M0None (the closed-vocabulary guard).
	if got := FromClaimTag(ClaimTag("WIP")); got != M0None {
		t.Fatalf("FromClaimTag(unknown) = %s, want M0None", got)
	}
}

// TestFeatureRosterResolvesToValidRungs is the witness test for #1249: every rostered
// feature resolves to a VALID rung from its bound witness. A feature with a CLAIMS anchor
// must be witnessed (ok==true) and its rung must equal the lowering of the matched line's
// tag (so the rung is READ from the ledger, never asserted); a feature with an empty
// anchor must floor to M0None unwitnessed (the honest gap).
func TestFeatureRosterResolvesToValidRungs(t *testing.T) {
	claims := readClaims(t)
	for _, f := range FeatureRoster {
		rung, tag, ok := FeatureRung(claims, f)
		if !rung.Valid() {
			t.Fatalf("feature %q resolved to %v, not a closed M0–M7 rung", f.ID, rung)
		}
		if f.ClaimAnchor == "" {
			if ok || rung != M0None {
				t.Fatalf("feature %q has no anchor; want (M0None, unwitnessed), got (%s, witnessed=%v)", f.ID, rung, ok)
			}
			continue
		}
		if !ok {
			t.Fatalf("feature %q anchor %q is NOT found in CLAIMS.md — the witness is unbound (stale anchor?)", f.ID, f.ClaimAnchor)
		}
		if rung != FromClaimTag(tag) {
			t.Fatalf("feature %q rung %s does not match its witness tag %q (→ %s)", f.ID, rung, tag, FromClaimTag(tag))
		}
	}
}

// TestFeatureRosterAnchorsUniqueInLedger asserts each non-empty anchor matches EXACTLY ONE
// tagged capability line, so a feature's witness is unambiguous — a substring that matched
// two ledger lines could silently bind the wrong rung.
func TestFeatureRosterAnchorsUniqueInLedger(t *testing.T) {
	claims := readClaims(t)
	lines := strings.Split(claims, "\n")
	for _, f := range FeatureRoster {
		if f.ClaimAnchor == "" {
			continue
		}
		hits := 0
		for _, line := range lines {
			if !strings.Contains(line, f.ClaimAnchor) {
				continue
			}
			if _, ok := claimTagOf(line); ok {
				hits++
			}
		}
		if hits != 1 {
			t.Fatalf("feature %q anchor %q matched %d tagged CLAIMS lines, want exactly 1", f.ID, f.ClaimAnchor, hits)
		}
	}
}

// TestFeatureRosterCoversNamedFamiliesAndVariants pins the issue's roster ask: the three
// seed families (cache / attention / serving) are all present, and the attention family
// carries the five variants the issue names (softmax / linear / MLA / DSA / paged).
func TestFeatureRosterCoversNamedFamiliesAndVariants(t *testing.T) {
	famSeen := map[FeatureFamily]int{}
	for _, f := range FeatureRoster {
		famSeen[f.Family]++
	}
	for _, fam := range FeatureFamilies {
		if famSeen[fam] == 0 {
			t.Fatalf("seed family %q has no rostered feature", fam)
		}
	}
	wantVariants := []string{"attn-softmax", "attn-linear", "attn-mla", "attn-dsa", "attn-paged"}
	have := map[string]bool{}
	for _, f := range FeatureRoster {
		have[f.ID] = true
	}
	for _, id := range wantVariants {
		if !have[id] {
			t.Fatalf("attention variant %q missing from the roster — the issue names softmax/linear/MLA/DSA/paged", id)
		}
	}
}

// TestFeatureRosterWellFormed asserts the roster is a closed, well-formed set: unique
// non-empty ids and names, and every feature declares a family from the closed
// FeatureFamilies set (no off-roster family can slip in).
func TestFeatureRosterWellFormed(t *testing.T) {
	closed := map[FeatureFamily]bool{}
	for _, fam := range FeatureFamilies {
		closed[fam] = true
	}
	seenID, seenName := map[string]bool{}, map[string]bool{}
	for _, f := range FeatureRoster {
		if f.ID == "" || f.Name == "" {
			t.Fatalf("feature has empty id or name: %+v", f)
		}
		if seenID[f.ID] {
			t.Fatalf("duplicate feature id %q", f.ID)
		}
		seenID[f.ID] = true
		if seenName[f.Name] {
			t.Fatalf("duplicate feature name %q", f.Name)
		}
		seenName[f.Name] = true
		if !closed[f.Family] {
			t.Fatalf("feature %q declares family %q outside the closed FeatureFamilies set", f.ID, f.Family)
		}
	}
}

// TestScoreFeaturesFoldsWholeRoster asserts ScoreFeatures returns one scored row per
// feature in roster order, each row's rung/tag agreeing with FeatureRung — the fold the
// higher children (the scorecard, the `fak support` read-out) consume.
func TestScoreFeaturesFoldsWholeRoster(t *testing.T) {
	claims := readClaims(t)
	scored := ScoreFeatures(claims)
	if len(scored) != len(FeatureRoster) {
		t.Fatalf("ScoreFeatures returned %d rows for %d features", len(scored), len(FeatureRoster))
	}
	for i, sf := range scored {
		if sf.ID != FeatureRoster[i].ID {
			t.Fatalf("ScoreFeatures row %d is %q, want roster order %q", i, sf.ID, FeatureRoster[i].ID)
		}
		rung, tag, ok := FeatureRung(claims, FeatureRoster[i])
		if sf.Rung != rung || sf.Tag != tag || sf.Witnessed != ok {
			t.Fatalf("ScoreFeatures row %q = {%s,%q,%v}, want {%s,%q,%v}", sf.ID, sf.Rung, sf.Tag, sf.Witnessed, rung, tag, ok)
		}
		if !sf.Rung.Valid() {
			t.Fatalf("ScoreFeatures row %q has invalid rung %v", sf.ID, sf.Rung)
		}
	}
}
