package main

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// #4321 acceptance clause 2: "the refusal ledger distinguishes lane-lease refusals from
// working-tree co-tenancy refusals in its own category (so the two are never conflated
// in analysis again)."
//
// The conflation was HERE. gardenDispatchSkipReason is the only Go fold that turns a
// dispatch verdict into a named category, and witnessGardenDispatch writes those names
// straight into the loop ledger as `skipped_<bucket>` metric keys. Before this pin the
// fold had exactly one contention bucket, `contended`, holding LANE_BUSY /
// LANE_LEASE_HELD / IN_FLIGHT_DUPLICATE / COLLISION_RISK -- while DIRTY_PATH_COLLISION
// and SAME_ISSUE_WIP matched no arm at all and fell through the default into the
// untyped `refused`. So working-tree co-tenancy was not merely folded IN with lane
// contention, it was invisible: a reader of the ledger could not count it, and the
// residual `refused` bucket mixed it with every other unclassified refusal.
//
// Measured on this repo's own ledger while writing this (both segments,
// .fak/loops.jsonl.001 + .fak/loops.jsonl, 16,514 events): 231 admit/refused events
// carry a working-tree co-tenancy verdict (230 DIRTY_PATH_COLLISION + 1 SAME_ISSUE_WIP)
// against 8 LANE_LEASE_HELD. The class the launcher already stamps as
// `refusal_class: worktree_cotenancy` in evidence_refs had no counterpart on the Go
// side; these tests give it one and keep the two sides bound.

// pythonRefusalClassConst reads a module-level `NAME = "value"` string literal out of
// the dispatch launcher. A missing constant is a hard failure, never a skip: noticing
// that the producer's vocabulary moved is this test's entire job.
func pythonRefusalClassConst(t *testing.T, src []byte, name string) string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` = "([^"]*)"`)
	m := re.FindSubmatch(src)
	if m == nil {
		t.Fatalf("tools/issue_resolve_dispatch.py no longer defines %s -- the refusal-class "+
			"vocabulary this fold mirrors has moved; re-check gardenDispatchCotenancyVerdicts "+
			"before deleting this pin", name)
	}
	return string(m[1])
}

var (
	// (?sm): m so ^ anchors the table to a module-level line (never an indented copy
	// inside a function), s so the lazy body may span lines up to the closing brace.
	pyRefusalClassTableRE = regexp.MustCompile(`(?sm)^_REFUSAL_CLASSES = \{(.*?)\n\}`)
	pyRefusalClassRowRE   = regexp.MustCompile(`"([A-Z_]+)":\s*(REFUSAL_CLASS_[A-Z_]+)`)
)

// readPythonRefusalClasses returns the producer's verdict -> class-VALUE table, e.g.
// {"LANE_LEASE_HELD": "lane_lease", "SAME_ISSUE_WIP": "worktree_cotenancy", ...}.
func readPythonRefusalClasses(t *testing.T) map[string]string {
	t.Helper()
	path := filepath.Join("..", "..", "tools", "issue_resolve_dispatch.py")
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	byConst := map[string]string{
		"REFUSAL_CLASS_LANE_LEASE":         pythonRefusalClassConst(t, src, "REFUSAL_CLASS_LANE_LEASE"),
		"REFUSAL_CLASS_WORKTREE_COTENANCY": pythonRefusalClassConst(t, src, "REFUSAL_CLASS_WORKTREE_COTENANCY"),
	}
	block := pyRefusalClassTableRE.FindSubmatch(src)
	if block == nil {
		t.Fatal("tools/issue_resolve_dispatch.py no longer defines a module-level " +
			"_REFUSAL_CLASSES = {...} table -- the producer side of the #4321 split moved")
	}
	out := map[string]string{}
	for _, row := range pyRefusalClassRowRE.FindAllSubmatch(block[1], -1) {
		verdict, constName := string(row[1]), string(row[2])
		value, ok := byConst[constName]
		if !ok {
			t.Fatalf("_REFUSAL_CLASSES maps %s to unknown class constant %s -- the producer "+
				"grew a THIRD refusal class; give it a Go bucket in gardenDispatchSkipReason "+
				"instead of letting it fall through to the untyped \"refused\"", verdict, constName)
		}
		out[verdict] = value
	}
	if len(out) == 0 {
		t.Fatal("_REFUSAL_CLASSES parsed to zero rows -- the table's shape changed and this " +
			"pin silently stopped checking anything")
	}
	return out
}

// TestRefusalClassVocabularyDoesNotDriftFromTheProducer binds every verdict the Python
// launcher classifies to the Go bucket it must land in. Go cannot import Python, so the
// table is parsed back out of the source -- the same cross-language discipline
// internal/dispatchconservation/producer_vocab_test.go uses for the no-commit classes.
func TestRefusalClassVocabularyDoesNotDriftFromTheProducer(t *testing.T) {
	classes := readPythonRefusalClasses(t)

	laneLease := pythonRefusalClassConst(t, mustReadLauncher(t), "REFUSAL_CLASS_LANE_LEASE")
	cotenancy := pythonRefusalClassConst(t, mustReadLauncher(t), "REFUSAL_CLASS_WORKTREE_COTENANCY")
	wantBucket := map[string]string{
		laneLease: gardenSkipLaneContended,
		cotenancy: gardenSkipWorktreeCotenancy,
	}

	for verdict, class := range classes {
		want, ok := wantBucket[class]
		if !ok {
			t.Fatalf("producer class %q has no Go bucket", class)
		}
		// action="refused" is what the launcher sends with every one of these, so this
		// is the real call shape -- not a synthetic one that could pass while the live
		// path still falls through the default.
		if got := gardenDispatchSkipReason(verdict, "refused"); got != want {
			t.Errorf("gardenDispatchSkipReason(%q, \"refused\") = %q, want %q -- a verdict the "+
				"launcher stamps as refusal_class=%q is bucketed as something else, so the "+
				"ledger's skipped_* keys no longer separate lane-lease from working-tree "+
				"co-tenancy (#4321)", verdict, got, want, class)
		}
	}

	// Both directions: a verdict Go still calls co-tenancy that the producer has since
	// reclassified (or dropped) is drift too, and would over-count the class.
	for verdict := range gardenDispatchCotenancyVerdicts {
		if got := classes[verdict]; got != cotenancy {
			t.Errorf("gardenDispatchCotenancyVerdicts contains %q but the launcher classes it "+
				"as %q (want %q) -- the Go table is stale", verdict, got, cotenancy)
		}
	}
}

func mustReadLauncher(t *testing.T) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "tools", "issue_resolve_dispatch.py")
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return src
}

// TestWorktreeCotenancyIsItsOwnSkipBucket is the regression that would have caught the
// original defect with no Python involved: co-tenancy must be its own named category,
// distinct from lane contention AND from the untyped residual bucket.
func TestWorktreeCotenancyIsItsOwnSkipBucket(t *testing.T) {
	if gardenSkipWorktreeCotenancy == gardenSkipLaneContended {
		t.Fatal("the co-tenancy and lane-contention buckets are the same string -- the two " +
			"refusal families are conflated by construction")
	}
	for _, verdict := range []string{"DIRTY_PATH_COLLISION", "SAME_ISSUE_WIP"} {
		got := gardenDispatchSkipReason(verdict, "refused")
		if got == "refused" || got == "unknown" {
			t.Errorf("%s buckets to the untyped %q -- working-tree co-tenancy is invisible "+
				"in the ledger's skipped_* keys again (#4321)", verdict, got)
		}
		if got != gardenSkipWorktreeCotenancy {
			t.Errorf("gardenDispatchSkipReason(%q) = %q, want %q", verdict, got, gardenSkipWorktreeCotenancy)
		}
	}
	// The lane-lease family keeps its own bucket, and DIRTY_PATH_COLLISION must not have
	// dragged the similarly-spelled COLLISION_RISK across the line with it: both names
	// contain "COLLISION" and they belong to different classes.
	for _, verdict := range []string{"LANE_BUSY", "LANE_LEASE_HELD", "IN_FLIGHT_DUPLICATE", "COLLISION_RISK"} {
		if got := gardenDispatchSkipReason(verdict, "refused"); got != gardenSkipLaneContended {
			t.Errorf("gardenDispatchSkipReason(%q) = %q, want %q", verdict, got, gardenSkipLaneContended)
		}
	}
}

// TestGardenDispatchSkipBucketsAreLedgerSafe pins the property that makes these strings
// usable as ledger metric keys: witnessGardenDispatch emits `skipped_<bucket>` after
// folding hyphens to underscores, so a bucket carrying a space or an underscore-vs-hyphen
// inconsistency would produce a key that no analysis query matches.
func TestGardenDispatchSkipBucketsAreLedgerSafe(t *testing.T) {
	safe := regexp.MustCompile(`^[a-z]+(-[a-z]+)*$`)
	for _, bucket := range []string{gardenSkipLaneContended, gardenSkipWorktreeCotenancy} {
		if !safe.MatchString(bucket) {
			t.Errorf("skip bucket %q is not lowercase-hyphen -- it would not survive the "+
				"skipped_<bucket> metric-key fold in witnessGardenDispatch", bucket)
		}
	}
}
