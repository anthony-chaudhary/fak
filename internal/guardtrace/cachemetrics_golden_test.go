package guardtrace

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// updateGolden regenerates testdata/cache-metrics.golden.json instead of asserting
// against it: `go test ./internal/guardtrace -run TestCacheMetricsGolden -update`. This
// is the explicit -update path #3639 requires — the ONLY sanctioned way the frozen
// shed/reuse numbers change is a deliberate regeneration, never a silent drift.
var updateGolden = flag.Bool("update", false, "regenerate the cache-metrics shed/reuse golden fixture")

// canonicalCacheGoldenPath is the frozen golden the regression pins.
const canonicalCacheGoldenPath = "testdata/cache-metrics.golden.json"

// canonicalFixtureGlob selects the canonical transcript set. Dropping a new
// canon-*.json into testdata and re-running with -update extends the golden.
const canonicalFixtureGlob = "testdata/canon-*.json"

// canonicalCacheMetrics folds every canonical transcript into its shed/reuse row, in a
// deterministic (sorted-by-path) order. It is the single source both the golden assertion
// and the -update regeneration run through, so they can never disagree on ordering.
func canonicalCacheMetrics(t *testing.T) []CacheMetrics {
	t.Helper()
	paths, err := filepath.Glob(canonicalFixtureGlob)
	if err != nil {
		t.Fatalf("glob %s: %v", canonicalFixtureGlob, err)
	}
	if len(paths) == 0 {
		t.Fatalf("no canonical transcripts matched %s", canonicalFixtureGlob)
	}
	sort.Strings(paths)
	out := make([]CacheMetrics, 0, len(paths))
	for _, p := range paths {
		f, err := LoadFixture(p)
		if err != nil {
			t.Fatalf("load canonical fixture %s: %v", p, err)
		}
		out = append(out, f.CacheMetrics())
	}
	return out
}

// marshalCacheGolden renders the frozen rows to the exact bytes the golden holds:
// pretty-printed, trailing newline. Marshaling is deterministic (fixed struct field
// order; ratios pre-rounded in the fold), so equal inputs render byte-identically.
func marshalCacheGolden(t *testing.T, rows []CacheMetrics) []byte {
	t.Helper()
	b, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		t.Fatalf("marshal cache-metrics golden: %v", err)
	}
	return append(b, '\n')
}

// TestCacheMetricsGolden is the regression: the shed/reuse fold of the canonical
// transcript set must match the frozen golden byte-for-byte. A transform that changes a
// shed or reuse number without a matching -update reds here; -update regenerates it.
func TestCacheMetricsGolden(t *testing.T) {
	got := marshalCacheGolden(t, canonicalCacheMetrics(t))

	if *updateGolden {
		if err := os.WriteFile(canonicalCacheGoldenPath, got, 0o644); err != nil {
			t.Fatalf("write golden %s: %v", canonicalCacheGoldenPath, err)
		}
		t.Logf("regenerated %s (%d bytes)", canonicalCacheGoldenPath, len(got))
		return
	}

	want, err := os.ReadFile(canonicalCacheGoldenPath)
	if err != nil {
		t.Fatalf("read golden %s (run: go test ./internal/guardtrace -run TestCacheMetricsGolden -update): %v", canonicalCacheGoldenPath, err)
	}
	if !bytes.Equal(want, got) {
		t.Errorf("cache-metrics golden drift — a shed/reuse number moved without an -update.\n"+
			"regenerate with: go test ./internal/guardtrace -run TestCacheMetricsGolden -update\n\n--- want (golden) ---\n%s\n--- got (current fold) ---\n%s", want, got)
	}
}

// TestCacheMetricsGoldenDetectsDrift is the done-condition witness: it proves the golden
// assertion actually REDS when a shed/reuse number moves. It takes the frozen rows, nudges
// one reuse ratio and one shed count (the two numbers #3639 exists to pin), and asserts
// the rendered bytes diverge from the golden — i.e. TestCacheMetricsGolden would fail on
// exactly that drift. It never writes the golden.
func TestCacheMetricsGoldenDetectsDrift(t *testing.T) {
	golden, err := os.ReadFile(canonicalCacheGoldenPath)
	if err != nil {
		t.Skipf("golden %s absent (run -update first): %v", canonicalCacheGoldenPath, err)
	}
	rows := canonicalCacheMetrics(t)
	if !bytes.Equal(golden, marshalCacheGolden(t, rows)) {
		t.Fatalf("precondition: canonical fold already diverges from golden — regenerate with -update before asserting drift detection")
	}

	for _, drift := range []struct {
		name  string
		apply func(m *CacheMetrics)
	}{
		{"reuse ratio drift", func(m *CacheMetrics) { m.ReuseRatio = round6(m.ReuseRatio + 0.01) }},
		{"shed token drift", func(m *CacheMetrics) { m.ShedTokens++ }},
		{"reused token drift", func(m *CacheMetrics) { m.ReusedTokens++ }},
	} {
		t.Run(drift.name, func(t *testing.T) {
			perturbed := append([]CacheMetrics(nil), rows...)
			drift.apply(&perturbed[len(perturbed)-1])
			if bytes.Equal(golden, marshalCacheGolden(t, perturbed)) {
				t.Errorf("%s did not change the rendered golden — the regression would NOT catch this drift", drift.name)
			}
		})
	}
}

// TestCacheMetricsFoldIsDeterministic guards the byte-compare premise: folding the same
// canonical set twice must render identical bytes (no map iteration, no unrounded float).
func TestCacheMetricsFoldIsDeterministic(t *testing.T) {
	a := marshalCacheGolden(t, canonicalCacheMetrics(t))
	b := marshalCacheGolden(t, canonicalCacheMetrics(t))
	if !bytes.Equal(a, b) {
		t.Errorf("cache-metrics fold is non-deterministic:\n--- run A ---\n%s\n--- run B ---\n%s", a, b)
	}
}
