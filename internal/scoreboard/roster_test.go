package scoreboard

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestReportingFamilyRoster pins the canonical roster: the 11-feeder family, in order,
// each carrying a valid FAK_<SURFACE>_CHANNEL key and a fak post/refresh command. This
// is the single enumeration every consumer (notifier, walker #4863, liveness read #4864)
// binds to; pinning it here means a consumer that must match the family has one fixed
// target, and a feeder added or dropped from the roster shows up as a diff here (#4865).
func TestReportingFamilyRoster(t *testing.T) {
	// The family in declaration order — the SSOT prose used to name (scoreboard.go).
	wantOrder := []string{
		"scoreboard", "blockers", "bench", "cachevalue", "capacity",
		"node-usage", "backlog", "dojo", "product", "releases", "steering",
	}

	fam := ReportingFamily()
	if len(fam) != len(wantOrder) {
		t.Fatalf("reporting family size = %d, want %d (the 11-feeder family)", len(fam), len(wantOrder))
	}

	chanKey := regexp.MustCompile(`^FAK_[A-Z0-9_]+_CHANNEL$`)
	seenName := map[string]bool{}
	seenChan := map[string]bool{}
	for i, f := range fam {
		if f.Name != wantOrder[i] {
			t.Errorf("feeder[%d].Name = %q, want %q (roster order is the family's)", i, f.Name, wantOrder[i])
		}
		if seenName[f.Name] {
			t.Errorf("feeder[%d].Name = %q is duplicated in the roster", i, f.Name)
		}
		seenName[f.Name] = true

		if !chanKey.MatchString(f.ChannelEnv) {
			t.Errorf("feeder %q ChannelEnv = %q, want a FAK_<SURFACE>_CHANNEL key", f.Name, f.ChannelEnv)
		}
		if seenChan[f.ChannelEnv] {
			t.Errorf("feeder %q ChannelEnv = %q is duplicated in the roster", f.Name, f.ChannelEnv)
		}
		seenChan[f.ChannelEnv] = true

		if !strings.HasPrefix(f.PostCommand, "fak ") {
			t.Errorf("feeder %q PostCommand = %q, want a `fak ...` verb (the walk's Enter hint)", f.Name, f.PostCommand)
		}
	}
}

// TestReportingFamilyReturnsCopy proves ReportingFamily hands out an immutable view:
// mutating the returned slice must not corrupt the roster the next caller sees.
func TestReportingFamilyReturnsCopy(t *testing.T) {
	got := ReportingFamily()
	if len(got) == 0 {
		t.Fatal("ReportingFamily returned empty roster")
	}
	got[0].Name = "MUTATED"
	if again := ReportingFamily(); again[0].Name != "scoreboard" {
		t.Fatalf("mutating the returned slice leaked into the roster: first feeder now %q", again[0].Name)
	}
}

// TestReportingFamilyIsSoleEnumeration is the no-drift guard the issue calls for: the
// roster must be the ONLY place the family is enumerated. It reads the package's own
// non-test source and asserts (a) the old drift-prone PROSE list ("scoreboard, blockers,
// bench, …") is gone everywhere, (b) the roster file names the whole family, and (c) no
// OTHER non-test file quotes two-or-more members together — a multi-member list is a
// second hardcoded roster drifting in, while a lone incidental mention is left alone.
func TestReportingFamilyIsSoleEnumeration(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	// The exact prose enumeration that used to live in the CICDReportChannel comment —
	// its return is the drift this test exists to catch.
	proseList := "scoreboard, blockers, bench, cachevalue, capacity"

	fam := ReportingFamily()
	const rosterFile = "scoreboard.go"
	rosterSaw := 0

	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		text := string(src)
		if strings.Contains(text, proseList) {
			t.Errorf("%s re-lists the family as prose (%q...): the family is enumerated only by ReportingFamily", name, proseList)
		}

		distinct := 0
		for _, f := range fam {
			if strings.Contains(text, `"`+f.Name+`"`) {
				distinct++
			}
		}
		if name == rosterFile {
			rosterSaw = distinct
			continue
		}
		if distinct >= 2 {
			t.Errorf("%s quotes %d family members together — a second hardcoded family list has drifted in; enumerate the family only via ReportingFamily", name, distinct)
		}
	}

	if rosterSaw != len(fam) {
		t.Errorf("%s names %d/%d family members; the roster must enumerate the whole family in one place", rosterFile, rosterSaw, len(fam))
	}
}
