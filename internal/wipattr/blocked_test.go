package wipattr

import (
	"reflect"
	"strings"
	"testing"
)

// The fixture is the real measured dirty set from the #4320 step-1 pass (see
// docs/dispatch/cmd-lane-split-plan.md), trimmed to the rows that exercise each
// verdict. Using the measured numbers keeps the test honest about what the fold was
// built for: version_modules.go really did carry 87 of 146 dispatch refusals.
func measuredDirty() []Blocker {
	return []Blocker{
		{Path: "cmd/fak/version_modules.go", Set: "cmd/fak", AgeDays: 7.33},
		{Path: "cmd/fak/version_modules_test.go", Set: "cmd/fak", AgeDays: 7.33},
		{Path: "internal/gateway/http.go", Set: "internal/gateway", AgeDays: 3.61},
		{Path: "internal/adjudicator/reversibility.go", Set: "internal/adjudicator", AgeDays: 5.08},
		{Path: "internal/adjudicator/decide.go", Set: "internal/adjudicator", AgeDays: 0.02},
		{Path: "cmd/conceptbench/spine.go", Set: "cmd/conceptbench", AgeDays: 3.02},
	}
}

func measuredBlocks() map[string]int {
	return map[string]int{
		"cmd/fak/version_modules.go":            87,
		"internal/gateway/http.go":              11,
		"internal/adjudicator/reversibility.go": 2,
		"internal/adjudicator/decide.go":        4,
	}
}

func rowFor(t *testing.T, rows []Blocked, path string) Blocked {
	t.Helper()
	for _, r := range rows {
		if r.Path == path {
			return r
		}
	}
	t.Fatalf("no row for %q in %d rows", path, len(rows))
	return Blocked{}
}

// The change-set trap this fold exists to prevent, and the one a per-file mtime
// ranking gets WRONG: internal/adjudicator/reversibility.go is 5.08 days idle on its
// own mtime — comfortably past any staleness threshold — but it is the classifier half
// of a change set whose consumer (decide.go) was edited 30 minutes ago. Landing it
// alone puts a test referencing a not-yet-committed Policy field on the trunk. It must
// come back WAIT, and must NAME the sibling responsible so the verdict is auditable.
func TestRankChangeSetPinsStaleFileLive(t *testing.T) {
	rows := Rank(measuredDirty(), measuredBlocks(), DefaultStaleAfterDays)

	rev := rowFor(t, rows, "internal/adjudicator/reversibility.go")
	if rev.State != BlockWait {
		t.Fatalf("reversibility.go: state = %q, want %q (its set is live via decide.go)", rev.State, BlockWait)
	}
	if rev.AgeDays < DefaultStaleAfterDays {
		t.Fatalf("fixture broken: reversibility.go age %.2f should look stale on its own", rev.AgeDays)
	}
	if rev.SetAgeDays != 0.02 {
		t.Errorf("reversibility.go: set age = %.2f, want 0.02 (decide.go's age)", rev.SetAgeDays)
	}
	if rev.FreshestSibling != "internal/adjudicator/decide.go" {
		t.Errorf("reversibility.go: freshest sibling = %q, want decide.go named", rev.FreshestSibling)
	}
	if !strings.Contains(rev.Reason, "half a change set") {
		t.Errorf("reversibility.go: reason %q should say landing it commits half a change set", rev.Reason)
	}
}

func TestRankLandsStaleBlockingSet(t *testing.T) {
	rows := Rank(measuredDirty(), measuredBlocks(), DefaultStaleAfterDays)

	vm := rowFor(t, rows, "cmd/fak/version_modules.go")
	if vm.State != BlockLand {
		t.Fatalf("version_modules.go: state = %q, want %q", vm.State, BlockLand)
	}
	if vm.Blocks != 87 {
		t.Errorf("version_modules.go: blocks = %d, want 87", vm.Blocks)
	}
	if vm.FreshestSibling != "" {
		t.Errorf("version_modules.go: freshest sibling = %q, want empty (it is the freshest of its set)", vm.FreshestSibling)
	}
	// The whole set is idle, so the test half is landable too — but it blocks nothing
	// on its own, so it must not masquerade as a throughput win.
	vmTest := rowFor(t, rows, "cmd/fak/version_modules_test.go")
	if vmTest.State != BlockIdle {
		t.Errorf("version_modules_test.go: state = %q, want %q (idle, blocks nothing)", vmTest.State, BlockIdle)
	}

	// A fresh single-file set that blocks is a correct transient refusal, not a lever.
	dec := rowFor(t, rows, "internal/adjudicator/decide.go")
	if dec.State != BlockWait {
		t.Errorf("decide.go: state = %q, want %q (peer mid-edit)", dec.State, BlockWait)
	}
	if !strings.Contains(dec.Reason, "refusal is correct") {
		t.Errorf("decide.go: reason %q should name the refusal as correct", dec.Reason)
	}

	// Idle, blocking nothing: hygiene only.
	cb := rowFor(t, rows, "cmd/conceptbench/spine.go")
	if cb.State != BlockIdle {
		t.Errorf("spine.go: state = %q, want %q", cb.State, BlockIdle)
	}
}

// A LAND row must outrank every WAIT row even when the WAIT row blocks far more,
// because a WAIT row cannot be acted on at all — ranking it first would hand the
// operator an item whose only correct action is to skip it.
func TestRankOrdersQueueByActionabilityThenImpact(t *testing.T) {
	rows := Rank([]Blocker{
		{Path: "fresh/blocker.go", Set: "fresh", AgeDays: 0.01},
		{Path: "stale/small.go", Set: "small", AgeDays: 9.0},
		{Path: "stale/big.go", Set: "big", AgeDays: 4.0},
		{Path: "stale/quiet.go", Set: "quiet", AgeDays: 9.0},
		{Path: "fresh/quiet.go", Set: "fresh2", AgeDays: 0.01},
	}, map[string]int{
		"fresh/blocker.go": 500,
		"stale/small.go":   3,
		"stale/big.go":     40,
	}, DefaultStaleAfterDays)

	got := make([]string, len(rows))
	for i, r := range rows {
		got[i] = r.Path
	}
	want := []string{
		"stale/big.go",     // LAND, 40 blocks
		"stale/small.go",   // LAND, 3 blocks
		"fresh/blocker.go", // WAIT — 500 blocks, but unactionable
		"stale/quiet.go",   // IDLE
		"fresh/quiet.go",   // ACTIVE
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("queue order = %v, want %v", got, want)
	}
}

func TestLandableAndBlocksRecovered(t *testing.T) {
	rows := Rank(measuredDirty(), measuredBlocks(), DefaultStaleAfterDays)

	paths := make([]string, 0, len(rows))
	for _, r := range Landable(rows) {
		paths = append(paths, r.Path)
	}
	want := []string{"cmd/fak/version_modules.go", "internal/gateway/http.go"}
	if !reflect.DeepEqual(paths, want) {
		t.Fatalf("landable = %v, want %v", paths, want)
	}

	// 98 = 87 + 11, NOT 104: the adjudicator pair's 2 + 4 blocks are excluded because
	// that set is live. The whole point of the number is that it counts only admissions
	// a sweep can actually recover, so an operator can decide whether to run one.
	if got := BlocksRecovered(rows); got != 87+11 {
		t.Errorf("blocks recovered = %d, want %d (live sets must not be counted)", got, 87+11)
	}
}

// Totality + determinism, the two guarantees attr.go's fold also makes: every input
// path yields exactly one row, and two runs over the same inputs are identical.
func TestRankTotalAndDeterministic(t *testing.T) {
	dirty := measuredDirty()
	first := Rank(dirty, measuredBlocks(), DefaultStaleAfterDays)

	if len(first) != len(dirty) {
		t.Fatalf("rows = %d, want %d (one per input path)", len(first), len(dirty))
	}
	seen := map[string]bool{}
	for _, r := range first {
		if seen[r.Path] {
			t.Errorf("duplicate row for %q", r.Path)
		}
		seen[r.Path] = true
		if r.SetAgeDays > r.AgeDays {
			t.Errorf("%s: set age %.2f > own age %.2f — a set can never be staler than its freshest member",
				r.Path, r.SetAgeDays, r.AgeDays)
		}
		if r.State == "" || r.Reason == "" {
			t.Errorf("%s: state/reason must always be populated, got %q/%q", r.Path, r.State, r.Reason)
		}
	}

	// Reverse the input; the output must be byte-identical.
	reversed := make([]Blocker, 0, len(dirty))
	for i := len(dirty) - 1; i >= 0; i-- {
		reversed = append(reversed, dirty[i])
	}
	if second := Rank(reversed, measuredBlocks(), DefaultStaleAfterDays); !reflect.DeepEqual(first, second) {
		t.Errorf("Rank is input-order dependent:\n first = %+v\nsecond = %+v", first, second)
	}
}

func TestRankEmptySetIsSingleton(t *testing.T) {
	// With no Set key, two unrelated paths must NOT pool: a fresh one cannot pin an
	// unrelated stale one live, or the caller loses every LAND row by omitting Set.
	rows := Rank([]Blocker{
		{Path: "a/stale.go", AgeDays: 6.0},
		{Path: "b/fresh.go", AgeDays: 0.01},
	}, map[string]int{"a/stale.go": 5}, DefaultStaleAfterDays)

	stale := rowFor(t, rows, "a/stale.go")
	if stale.State != BlockLand {
		t.Errorf("a/stale.go: state = %q, want %q (unset Set must be a singleton, not a shared pool)", stale.State, BlockLand)
	}
	if stale.FreshestSibling != "" {
		t.Errorf("a/stale.go: freshest sibling = %q, want empty", stale.FreshestSibling)
	}
}

func TestRankEmptiesAndThresholdFallback(t *testing.T) {
	if rows := Rank(nil, nil, DefaultStaleAfterDays); rows == nil || len(rows) != 0 {
		t.Errorf("Rank(nil,nil) = %v, want empty non-nil", rows)
	}
	if got := Landable(nil); got == nil || len(got) != 0 {
		t.Errorf("Landable(nil) = %v, want empty non-nil", got)
	}
	if got := BlocksRecovered(nil); got != 0 {
		t.Errorf("BlocksRecovered(nil) = %d, want 0", got)
	}

	// A non-positive threshold falls back to the default rather than classifying
	// everything stale (which would recommend landing a file edited seconds ago).
	dirty := []Blocker{{Path: "x.go", AgeDays: 0.5}}
	blocks := map[string]int{"x.go": 9}
	for _, thresh := range []float64{0, -1} {
		rows := Rank(dirty, blocks, thresh)
		if rows[0].State != BlockWait {
			t.Errorf("threshold %.0f: state = %q, want %q (must fall back to the %.0fd default)",
				thresh, rows[0].State, BlockWait, DefaultStaleAfterDays)
		}
	}
}

// A path absent from the ledger map blocks nothing — it must not be treated as
// missing data and skipped, or the fold loses totality.
func TestRankUnknownPathBlocksNothing(t *testing.T) {
	rows := Rank([]Blocker{{Path: "never/refused.go", AgeDays: 30}}, map[string]int{"other.go": 4}, DefaultStaleAfterDays)
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].Blocks != 0 || rows[0].State != BlockIdle {
		t.Errorf("unknown path = %d blocks/%q, want 0/%q", rows[0].Blocks, rows[0].State, BlockIdle)
	}
}
