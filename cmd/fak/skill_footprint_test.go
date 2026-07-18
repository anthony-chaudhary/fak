package main

// Acceptance test for the headless skills-index profile (#3612, epic #3229).
//
// #3234 compacted the resident .claude/skills index to name+description for
// interactive sessions. #3612 adds a `headless` profile that ships the index
// name-only for a single-issue `-p` dispatch worker, keeping the interactive
// (#3234) behavior unchanged. The three acceptance bullets map to the three
// checks below, each asserted by the rendered floor the scorecard reports:
//
//	1. headless floor is name-only and strictly smaller than interactive
//	   (the "asserted by rendered token size" bullet);
//	2. interactive keeps #3234's name+description floor (regression);
//	3. every skill name survives the trim, so a skill is still invocable by
//	   name from a headless worker.

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// footprintJSON is the subset of the `--json` scorecard the acceptance checks read.
type footprintJSON struct {
	Profile          string `json:"profile"`
	SkillCount       int    `json:"skill_count"`
	DescriptionFloor int    `json:"description_floor_bytes"`
	NameFloor        int    `json:"name_floor_bytes"`
	ResidentFloor    int    `json:"resident_floor_bytes"`
	Entries          []struct {
		Name      string `json:"name"`
		NameBytes int    `json:"name_bytes"`
		DescBytes int    `json:"description_bytes"`
	} `json:"entries"`
}

func runFootprintJSON(t *testing.T, argv ...string) footprintJSON {
	t.Helper()
	var out, errb bytes.Buffer
	code := runSkillFootprint(&out, &errb, append([]string{"--json"}, argv...))
	if code != 0 {
		t.Fatalf("runSkillFootprint(%v) exit=%d, stderr=%q", argv, code, errb.String())
	}
	var fp footprintJSON
	if err := json.Unmarshal(out.Bytes(), &fp); err != nil {
		t.Fatalf("parse footprint JSON: %v\n%s", err, out.String())
	}
	return fp
}

// computeSkillFootprint is pure, so a synthetic catalog pins the two floors
// exactly: the name-only floor is the sum of the names, the description floor
// is the sum of the (longer) descriptions, and every name is preserved.
func TestComputeSkillFootprintNameFloorIsStrictlySmaller(t *testing.T) {
	cards := []capindex.CapCard{
		{Ref: capindex.CapRef{Kind: capindex.CapKindSkill, Name: "release"}, Trigger: "cut a versioned release"},
		{Ref: capindex.CapRef{Kind: capindex.CapKindSkill, Name: "verify"}, Trigger: "bind a done-claim to a green test run"},
		{Ref: capindex.CapRef{Kind: capindex.CapKindSkill, Name: "dos-dispatch"}, Trigger: "take a lane through the arbiter"},
	}
	fp := computeSkillFootprint(cards)

	wantName := len("release") + len("verify") + len("dos-dispatch")
	wantDesc := len("cut a versioned release") + len("bind a done-claim to a green test run") + len("take a lane through the arbiter")
	if fp.NameFloor != wantName {
		t.Errorf("NameFloor = %d, want %d", fp.NameFloor, wantName)
	}
	if fp.DescFloor != wantDesc {
		t.Errorf("DescFloor = %d, want %d", fp.DescFloor, wantDesc)
	}
	// Bullet 1: the headless name-only floor is strictly smaller than the
	// interactive name+description floor whenever descriptions are non-empty.
	if fp.NameFloor >= fp.DescFloor {
		t.Errorf("headless NameFloor (%d) must be < interactive DescFloor (%d)", fp.NameFloor, fp.DescFloor)
	}
	// Bullet 3: every skill name survives, so it stays invocable by name.
	if len(fp.Entries) != len(cards) {
		t.Fatalf("entries = %d, want %d", len(fp.Entries), len(cards))
	}
	sumName := 0
	for _, e := range fp.Entries {
		if e.Name == "" || e.NameBytes == 0 {
			t.Errorf("entry has empty name (name=%q bytes=%d) — not invocable by name", e.Name, e.NameBytes)
		}
		if e.NameBytes != len(e.Name) {
			t.Errorf("entry %q: NameBytes=%d, want %d", e.Name, e.NameBytes, len(e.Name))
		}
		sumName += e.NameBytes
	}
	if sumName != fp.NameFloor {
		t.Errorf("sum of entry NameBytes (%d) != NameFloor (%d)", sumName, fp.NameFloor)
	}
}

// Bullet 1, end to end: `--profile headless` ships the name-only floor, which
// is strictly smaller than interactive over the live in-repo skills catalog.
func TestRunSkillFootprintHeadlessProfileIsNameOnly(t *testing.T) {
	fp := runFootprintJSON(t, "--profile", "headless")
	if fp.Profile != "headless" {
		t.Errorf("profile = %q, want headless", fp.Profile)
	}
	if fp.SkillCount == 0 {
		t.Fatal("no skills discovered under .claude/skills — repoRoot() did not resolve the repo")
	}
	// Headless ships the name-only slice.
	if fp.ResidentFloor != fp.NameFloor {
		t.Errorf("headless resident floor = %d, want name floor %d", fp.ResidentFloor, fp.NameFloor)
	}
	// The trim is real: name-only < name+description across the live index.
	if fp.NameFloor >= fp.DescriptionFloor {
		t.Errorf("headless name floor (%d) must be < interactive description floor (%d)", fp.NameFloor, fp.DescriptionFloor)
	}
	// Bullet 3: names survive the trim, so a referenced skill is still invocable.
	for _, e := range fp.Entries {
		if e.Name == "" {
			t.Error("headless entry with empty name — a skill would not be invocable by name")
		}
	}
}

// Bullet 2: interactive keeps #3234's name+description floor, and the default
// (no --profile) is interactive — so existing behavior does not regress.
func TestRunSkillFootprintInteractiveRegression(t *testing.T) {
	interactive := runFootprintJSON(t, "--profile", "interactive")
	if interactive.Profile != "interactive" {
		t.Errorf("profile = %q, want interactive", interactive.Profile)
	}
	if interactive.ResidentFloor != interactive.DescriptionFloor {
		t.Errorf("interactive resident floor = %d, want description floor %d (#3234)", interactive.ResidentFloor, interactive.DescriptionFloor)
	}

	// The default profile must be interactive — a bare `fak skill footprint`
	// keeps the #3234 floor rather than silently trimming to name-only.
	deflt := runFootprintJSON(t)
	if deflt.Profile != "interactive" {
		t.Errorf("default profile = %q, want interactive", deflt.Profile)
	}
	if deflt.ResidentFloor != deflt.DescriptionFloor {
		t.Errorf("default resident floor = %d, want description floor %d", deflt.ResidentFloor, deflt.DescriptionFloor)
	}
}

// An unknown profile is a hard usage error, not a silent fallback that could
// mask a mis-typed profile as one of the two supported floors.
func TestRunSkillFootprintUnknownProfileFails(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSkillFootprint(&out, &errb, []string{"--json", "--profile", "bogus"})
	if code != 2 {
		t.Fatalf("unknown --profile exit = %d, want 2 (stderr=%q)", code, errb.String())
	}
}
