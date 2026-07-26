package wipattr

import (
	"reflect"
	"testing"
)

// These are the producers' literal sentences, reproduced from
// tools/issue_resolve_dispatch.py:3610 (DIRTY_PATH_COLLISION) and :3736
// (SAME_ISSUE_WIP). If a producer is reworded, this test is the tripwire — the parser
// degrades to zero paths silently, so the shape must be pinned somewhere.
const (
	dirtyOne = "issue #2477 names dirty local path(s) already modified in this " +
		"checkout: cmd/fak/version_modules.go — refusing DIRTY_PATH_COLLISION so a " +
		"new worker cannot overwrite peer WIP; wait for those paths to commit/clear, " +
		"or dispatch a disjoint issue"

	dirtyMany = "issue #4776 names dirty local path(s) already modified in this " +
		"checkout: internal/gateway/http.go, internal/gateway/responses.go, " +
		"cmd/fak/guard.go — refusing DIRTY_PATH_COLLISION so a new worker cannot " +
		"overwrite peer WIP; wait for those paths to commit/clear, or dispatch a " +
		"disjoint issue"

	dirtyTruncated = "issue #4321 names dirty local path(s) already modified in this " +
		"checkout: a.go, b.go (+7 more) — refusing DIRTY_PATH_COLLISION so a new " +
		"worker cannot overwrite peer WIP"

	sameIssue = "issue #2477 has recent same-issue uncommitted WIP (worker claim) " +
		"naming dirty local path(s): cmd/fak/version_modules.go, " +
		"cmd/fak/version_modules_test.go — refusing SAME_ISSUE_WIP so a second " +
		"resolver cannot stack onto unfinished work; continue/commit those paths " +
		"first, or dispatch a disjoint issue"
)

func TestParseBlockedPaths(t *testing.T) {
	cases := []struct {
		name    string
		summary string
		want    []string
	}{
		{"single path", dirtyOne, []string{"cmd/fak/version_modules.go"}},
		{"comma-joined list", dirtyMany, []string{
			"internal/gateway/http.go", "internal/gateway/responses.go", "cmd/fak/guard.go",
		}},
		{"truncation note stripped", dirtyTruncated, []string{"a.go", "b.go"}},
		{"same-issue producer", sameIssue, []string{
			"cmd/fak/version_modules.go", "cmd/fak/version_modules_test.go",
		}},

		// Conservative refusals to guess: a different reason, a reworded clause, or
		// nothing at all must yield no paths rather than a fragment.
		{"unrelated reason", "lane 'cmd' lease held by worker w-7 — refusing LANE_LEASE_HELD", nil},
		{"empty", "", nil},
		{"clause without a list", "issue #9 names dirty local path(s) already modified in this checkout: — refusing DIRTY_PATH_COLLISION", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ParseBlockedPaths(tc.summary); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ParseBlockedPaths() = %v, want %v", got, tc.want)
			}
		})
	}
}

// The hardening case: if a producer's em-dash is ever normalised to ASCII, the final
// path must still parse. Without the tail cut it would glue to "refusing …", get
// dropped as prose, and a single-path refusal would silently count ZERO — losing
// exactly the dominant cluster the ranking depends on.
func TestParseBlockedPathsASCIIDashFallback(t *testing.T) {
	for _, summary := range []string{
		"issue #2477 names dirty local path(s) already modified in this checkout: cmd/fak/version_modules.go -- refusing DIRTY_PATH_COLLISION so a new worker cannot overwrite peer WIP",
		"issue #2477 names dirty local path(s) already modified in this checkout: cmd/fak/version_modules.go refusing DIRTY_PATH_COLLISION",
	} {
		got := ParseBlockedPaths(summary)
		want := []string{"cmd/fak/version_modules.go"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ParseBlockedPaths(%.70s…) = %v, want %v", summary, got, want)
		}
	}
}

func TestCountBlocks(t *testing.T) {
	got := CountBlocks([]string{dirtyOne, dirtyOne, dirtyMany, sameIssue, "", "LANE_LEASE_HELD"})
	want := map[string]int{
		"cmd/fak/version_modules.go":      3, // dirtyOne x2 + sameIssue
		"cmd/fak/version_modules_test.go": 1,
		"internal/gateway/http.go":        1,
		"internal/gateway/responses.go":   1,
		"cmd/fak/guard.go":                1,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("CountBlocks() = %v, want %v", got, want)
	}
}

// The cost unit is "admissions this path refused", so a path named twice by ONE
// summary counts once — otherwise a contract that repeats a path would inflate its
// rank above a path that genuinely blocked more dispatches.
func TestCountBlocksDedupesWithinASummary(t *testing.T) {
	dup := "issue #7 names dirty local path(s) already modified in this checkout: " +
		"x.go, x.go, y.go — refusing DIRTY_PATH_COLLISION"
	got := CountBlocks([]string{dup})
	want := map[string]int{"x.go": 1, "y.go": 1}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("CountBlocks() = %v, want %v", got, want)
	}
}

func TestCountBlocksEmpty(t *testing.T) {
	if got := CountBlocks(nil); got == nil || len(got) != 0 {
		t.Errorf("CountBlocks(nil) = %v, want empty non-nil", got)
	}
}

// End-to-end over the pure layer: ledger summaries in, ranked work queue out. This is
// the fold the cmd shell wires to a real ledger, so it must hold without any I/O.
func TestParseThenRank(t *testing.T) {
	blocks := CountBlocks([]string{dirtyOne, dirtyOne, dirtyOne, dirtyMany})
	rows := Rank([]Blocker{
		{Path: "cmd/fak/version_modules.go", Set: "cmd/fak", AgeDays: 7.33},
		{Path: "internal/gateway/http.go", Set: "internal/gateway", AgeDays: 3.61},
		{Path: "cmd/fak/guard.go", Set: "cmd/fak/guard", AgeDays: 0.02},
	}, blocks, DefaultStaleAfterDays)

	if rows[0].Path != "cmd/fak/version_modules.go" || rows[0].State != BlockLand || rows[0].Blocks != 3 {
		t.Fatalf("top row = %s %s %d blocks, want cmd/fak/version_modules.go LAND 3",
			rows[0].Path, rows[0].State, rows[0].Blocks)
	}
	// guard.go is fresh: it blocks, but the sweep must not offer it.
	guard := rowFor(t, rows, "cmd/fak/guard.go")
	if guard.State != BlockWait {
		t.Errorf("guard.go: state = %q, want %q", guard.State, BlockWait)
	}
	if got := BlocksRecovered(rows); got != 4 {
		t.Errorf("blocks recovered = %d, want 4 (3 + the gateway row's 1)", got)
	}
}
