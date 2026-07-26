package main

import (
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// These tests fence the content probe that keeps `fak wip blocked` from recommending a
// destructive land. The porcelain fixtures are the REAL lines observed on the fak trunk
// on 2026-07-26, when the verb ranked internal/agent/loop.go as WAIT (8 blocked
// admissions) with a pre-#5235 index blob, and rated a pair of staged-deleted gateway
// files whose byte-identical copies were still on disk.

// livePorcelain is the observed status output. Note that role_alternation.go appears
// TWICE — once staged deleted, once untracked — which is exactly the phantom-delete
// shape, and exactly the signal a deduplicated path list destroys.
const livePorcelain = `MM internal/agent/loop.go
MM internal/agent/loop_session.go
 M cmd/fak/version_modules.go
D  internal/gateway/role_alternation.go
D  internal/gateway/role_alternation_test.go
?? internal/gateway/role_alternation.go
?? internal/gateway/role_alternation_test.go
A  cmd/fak/agent_goal_endpoint.go
!! ignored/thing.o
`

// TestParseWipStatusEntriesKeepsPhantomDeleteTwin proves the entry parser preserves the
// duplicate path that carries the phantom-delete signal, while the path list the
// ranking consumes still yields one row per path.
func TestParseWipStatusEntriesKeepsPhantomDeleteTwin(t *testing.T) {
	entries := parseWipStatusEntries(livePorcelain)

	// 8 status lines survive; the ignored (!!) one is dropped.
	if len(entries) != 8 {
		t.Fatalf("entries = %d, want 8: %+v", len(entries), entries)
	}
	var stagedDelete, untracked int
	for _, e := range entries {
		if e.Path == "internal/gateway/role_alternation.go" {
			if e.Untracked {
				untracked++
			} else if e.Index == 'D' {
				stagedDelete++
			}
		}
		if e.Path == "ignored/thing.o" {
			t.Error("ignored (!!) entries must not reach the ranking")
		}
	}
	if stagedDelete != 1 || untracked != 1 {
		t.Fatalf("role_alternation.go: stagedDelete=%d untracked=%d, want 1 and 1 — the twin IS the signal",
			stagedDelete, untracked)
	}

	// The deduplicated view still gives the ranking one row per path: 8 entries over 6
	// distinct paths, because the two phantom-delete twins each collapse.
	paths := parseWipStatusPaths(livePorcelain)
	if len(paths) != 6 {
		t.Fatalf("deduplicated paths = %d, want 6: %v", len(paths), paths)
	}
	seen := map[string]bool{}
	for _, p := range paths {
		if seen[p] {
			t.Errorf("duplicate path %q in the ranking input", p)
		}
		seen[p] = true
	}
}

// TestWipClassifyContentSeparatesResidueFromWork drives the pure classifier over the
// live fixture with the two name-only sets git would have returned, and pins each of
// the four shapes.
func TestWipClassifyContentSeparatesResidueFromWork(t *testing.T) {
	entries := parseWipStatusEntries(livePorcelain)

	// `git diff --name-only HEAD`: loop.go and loop_session.go are ABSENT — their
	// worktrees already equal HEAD, so only their index entries are stale. The staged
	// deletions DO appear (git reports them as deleted); version_modules.go and the
	// staged add appear as real edits.
	diverged := map[string]bool{
		"internal/gateway/role_alternation.go":      true,
		"internal/gateway/role_alternation_test.go": true,
		"cmd/fak/version_modules.go":                true,
		"cmd/fak/agent_goal_endpoint.go":            true,
	}
	// `git diff --name-only @{upstream}`: agent_goal_endpoint.go is ABSENT — its bytes
	// already match the published trunk.
	vsUpstream := map[string]bool{
		"internal/gateway/role_alternation.go":      true,
		"internal/gateway/role_alternation_test.go": true,
		"cmd/fak/version_modules.go":                true,
	}

	got := wipClassifyContent(entries, diverged, vsUpstream, true)
	want := map[string]wipattr.Content{
		"internal/agent/loop.go":                    wipattr.ContentMatchesHEAD,
		"internal/agent/loop_session.go":            wipattr.ContentMatchesHEAD,
		"internal/gateway/role_alternation.go":      wipattr.ContentPhantomDelete,
		"internal/gateway/role_alternation_test.go": wipattr.ContentPhantomDelete,
		"cmd/fak/version_modules.go":                wipattr.ContentDiverged,
		"cmd/fak/agent_goal_endpoint.go":            wipattr.ContentMatchesUpstream,
	}
	for p, w := range want {
		if got[p] != w {
			t.Errorf("%s: content = %s, want %s", p, got[p], w)
		}
	}

	// End to end through the fold. The temp root holds none of these files, so every
	// age is 0 — the FRESHEST possible value, which is what makes this the strong
	// assertion: residue must be refused even when nothing looks abandoned.
	rows := wipattr.Rank(
		wipBlockers(t.TempDir(), parseWipStatusPaths(livePorcelain), time.Now(), got),
		map[string]int{"internal/agent/loop.go": 8, "internal/gateway/role_alternation.go": 21},
		wipattr.DefaultStaleAfterDays)

	state := map[string]wipattr.BlockState{}
	for _, r := range rows {
		state[r.Path] = r.State
	}
	for _, p := range []string{
		"internal/agent/loop.go",
		"internal/gateway/role_alternation.go",
		"cmd/fak/agent_goal_endpoint.go",
	} {
		if state[p] != wipattr.BlockResidue {
			t.Errorf("%s: state = %q, want %q", p, state[p], wipattr.BlockResidue)
		}
	}
	if got := wipattr.ResidueBlocks(rows); got != 8+21 {
		t.Errorf("residue blocks = %d, want %d — the cost must stay visible", got, 8+21)
	}
	if got := wipattr.BlocksRecovered(rows); got != 0 {
		t.Errorf("blocks recovered = %d, want 0 — nothing here is safely landable", got)
	}
}

// TestWipClassifyContentWithoutUpstreamKeepsTheDestructiveShapes pins the partial-probe
// contract: with no tracking branch the landed-upstream shape is simply not claimed,
// but the two shapes that would DELETE work are still caught. Crucially, an absent
// upstream must never be read as "matches upstream" — that misreading recommends
// discarding real, unpublished work.
func TestWipClassifyContentWithoutUpstreamKeepsTheDestructiveShapes(t *testing.T) {
	entries := parseWipStatusEntries(livePorcelain)
	diverged := map[string]bool{
		"internal/gateway/role_alternation.go":      true,
		"internal/gateway/role_alternation_test.go": true,
		"cmd/fak/version_modules.go":                true,
		"cmd/fak/agent_goal_endpoint.go":            true,
	}

	got := wipClassifyContent(entries, diverged, nil, false)

	if got["internal/agent/loop.go"] != wipattr.ContentMatchesHEAD {
		t.Errorf("stale index must still be caught without an upstream, got %s", got["internal/agent/loop.go"])
	}
	if got["internal/gateway/role_alternation.go"] != wipattr.ContentPhantomDelete {
		t.Errorf("phantom delete must still be caught without an upstream, got %s",
			got["internal/gateway/role_alternation.go"])
	}
	// Everything the upstream read would have judged now reads as real work — the safe
	// direction, since the cost is a row that stays in the queue, not one that is lost.
	for _, p := range []string{"cmd/fak/version_modules.go", "cmd/fak/agent_goal_endpoint.go"} {
		if got[p] != wipattr.ContentDiverged {
			t.Errorf("%s: content = %s, want diverged when upstream is unknown", p, got[p])
		}
	}
}

// TestWipNameOnlySetUnquotesAndSkipsBlanks proves the name-only reader spells paths the
// same way parseWipStatusPaths does — if the two disagreed, every quoted path would
// look absent from the diff and therefore be misread as a stale index entry.
func TestWipNameOnlySetUnquotesAndSkipsBlanks(t *testing.T) {
	set := wipNameOnlySet("cmd/fak/a.go\n\"docs/a file with spaces.md\"\n\n")
	want := map[string]bool{"cmd/fak/a.go": true, "docs/a file with spaces.md": true}
	if !reflect.DeepEqual(set, want) {
		t.Errorf("wipNameOnlySet() = %v, want %v", set, want)
	}
	// The spelling must round-trip against the status parser for the same path.
	quoted := parseWipStatusPaths(" M \"docs/a file with spaces.md\"\n")
	if len(quoted) != 1 || !set[quoted[0]] {
		t.Errorf("status path %v does not match the name-only spelling %v", quoted, set)
	}
	if got := wipNameOnlySet(""); len(got) != 0 {
		t.Errorf("wipNameOnlySet(\"\") = %v, want empty", got)
	}
}
