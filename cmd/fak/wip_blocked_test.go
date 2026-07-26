package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// parseWipStatusPaths is the seam where a git-format mistake would silently shrink the
// ranking: a line shape we mis-parse drops that path, and a dropped path reads as
// "nothing blocking here". Every porcelain shape the working tree actually produces is
// pinned.
func TestParseWipStatusPaths(t *testing.T) {
	porcelain := "" +
		" M cmd/fak/version_modules.go\n" + // unstaged modification
		"M  cmd/fak/doctor.go\n" + // staged modification
		"MM internal/a2achan/floor.go\n" + // staged + unstaged
		"A  cmd/fak/guard_allow_scope.go\n" + // added
		"?? cmd/conceptbench/spine_live_test.go\n" + // untracked
		"D  cmd/fak/wip_ticket.go\n" + // deletion (still dirty, unstattable)
		"R  cmd/fak/old_name.go -> cmd/fak/new_name.go\n" + // rename: the NEW path is live
		"!! ignored/thing.go\n" + // ignored: not WIP
		" M cmd/fak/doctor.go\n" + // duplicate: counted once
		"?? untracked_dir/\n" + // a bare directory has no mtime to judge
		"\n"

	want := []string{
		"cmd/fak/version_modules.go",
		"cmd/fak/doctor.go",
		"internal/a2achan/floor.go",
		"cmd/fak/guard_allow_scope.go",
		"cmd/conceptbench/spine_live_test.go",
		"cmd/fak/wip_ticket.go",
		"cmd/fak/new_name.go",
	}
	if got := parseWipStatusPaths(porcelain); !reflect.DeepEqual(got, want) {
		t.Errorf("parseWipStatusPaths() =\n %v\nwant\n %v", got, want)
	}
}

func TestParseWipStatusPathsQuoted(t *testing.T) {
	got := parseWipStatusPaths(" M \"docs/a file with spaces.md\"\n")
	want := []string{"docs/a file with spaces.md"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseWipStatusPaths() = %v, want %v", got, want)
	}
}

func TestParseWipStatusPathsEmpty(t *testing.T) {
	if got := parseWipStatusPaths(""); got == nil || len(got) != 0 {
		t.Errorf("parseWipStatusPaths(\"\") = %v, want empty non-nil", got)
	}
}

// wipBlockers must group by directory (so an implementation and its test share a change
// set) and must never date a file it cannot stat — an unstattable path gets age 0, the
// freshest value, so it can never be recommended for landing.
func TestWipBlockers(t *testing.T) {
	root := t.TempDir()
	now := time.Now()

	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(root, "cmd", "fak", "stale.go")
	if err := os.WriteFile(stale, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	staleTime := now.Add(-6 * 24 * time.Hour)
	if err := os.Chtimes(stale, staleTime, staleTime); err != nil {
		t.Fatal(err)
	}

	// A nil content map is the unprobed caller: every path keeps the pre-Content
	// verdict, which is what this test pins. The four content shapes are exercised in
	// wip_blocked_residue_test.go.
	got := wipBlockers(root, []string{"cmd/fak/stale.go", "cmd/fak/deleted.go"}, now, nil)
	if len(got) != 2 {
		t.Fatalf("blockers = %d, want 2 (totality: one per input path)", len(got))
	}
	if got[0].Set != "cmd/fak" || got[1].Set != "cmd/fak" {
		t.Errorf("sets = %q/%q, want both cmd/fak", got[0].Set, got[1].Set)
	}
	if got[0].AgeDays < 5.9 || got[0].AgeDays > 6.1 {
		t.Errorf("stale.go age = %.2f, want ~6", got[0].AgeDays)
	}
	if got[1].AgeDays != 0 {
		t.Errorf("unstattable path age = %.2f, want 0 (never recommend landing an undateable file)", got[1].AgeDays)
	}

	// End-to-end through the pure fold: the deletion pins the set live, so the
	// genuinely-6-day-old file must come back WAIT rather than LAND.
	rows := wipattr.Rank(got, map[string]int{"cmd/fak/stale.go": 40}, wipattr.DefaultStaleAfterDays)
	if rows[0].Path != "cmd/fak/stale.go" || rows[0].State != wipattr.BlockWait {
		t.Errorf("top row = %s/%s, want cmd/fak/stale.go/%s", rows[0].Path, rows[0].State, wipattr.BlockWait)
	}
	if rows[0].FreshestSibling != "cmd/fak/deleted.go" {
		t.Errorf("freshest sibling = %q, want cmd/fak/deleted.go", rows[0].FreshestSibling)
	}
}

func TestUnquoteWipStatusPath(t *testing.T) {
	cases := map[string]string{
		"cmd/fak/plain.go":     "cmd/fak/plain.go",
		`"docs/with space.md"`: "docs/with space.md",
		`"unterminated`:        "unterminated", // malformed: surface it, don't drop it
		`"docs/tab\there.md"`:  "docs/tab\there.md",
	}
	for in, want := range cases {
		if got := unquoteWipStatusPath(in); got != want {
			t.Errorf("unquoteWipStatusPath(%q) = %q, want %q", in, got, want)
		}
	}
}
