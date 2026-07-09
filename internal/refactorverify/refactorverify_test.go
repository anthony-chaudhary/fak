package refactorverify

import (
	"reflect"
	"testing"
)

// godfile: a god-file before the split — two decls in one file.
const godfile = "package demo\n" +
	"\n" +
	"// Alpha does a thing.\n" +
	"func Alpha() int { return 1 }\n" +
	"\n" +
	"// Beta does another.\n" +
	"func Beta() int { return 2 }\n"

func files(pairs ...string) map[string]string {
	m := map[string]string{}
	for i := 0; i+1 < len(pairs); i += 2 {
		m[pairs[i]] = pairs[i+1]
	}
	return m
}

func TestCleanMotionPreservesDecls(t *testing.T) {
	// Beta moved to a sibling file in the SAME package — nothing dropped, nothing
	// relocated. Beta's new file carries exactly one decl, so it IS over-split.
	before := map[string]map[string]string{"demo": {"demo.go": godfile}}
	after := map[string]map[string]string{"demo": files(
		"demo.go", "package demo\n\n// Alpha does a thing.\nfunc Alpha() int { return 1 }\n",
		"demo_beta.go", "package demo\n\n// Beta does another.\nfunc Beta() int { return 2 }\n",
	)}
	rep := Verify(before, after)
	if len(rep.Dropped) != 0 {
		t.Errorf("dropped = %v, want []", rep.Dropped)
	}
	if len(rep.Relocated) != 0 {
		t.Errorf("relocated = %v, want []", rep.Relocated)
	}
	if len(rep.Oversplit) != 1 || rep.Oversplit[0].File != "demo/demo_beta.go" {
		t.Errorf("oversplit = %v, want [demo/demo_beta.go]", rep.Oversplit)
	}
}

func TestSplitIntoCohesiveFilesNoOversplit(t *testing.T) {
	// A three-decl god-file split into a 1-decl (stays) + 2-decl (new) file: the new
	// file is cohesive, so nothing is flagged.
	god := godfile + "\n// Gamma too.\nfunc Gamma() int { return 3 }\n"
	before := map[string]map[string]string{"demo": {"demo.go": god}}
	after := map[string]map[string]string{"demo": files(
		"demo.go", "package demo\n\nfunc Alpha() int { return 1 }\n",
		"demo_pair.go", "package demo\n\nfunc Beta() int { return 2 }\n\nfunc Gamma() int { return 3 }\n",
	)}
	rep := Verify(before, after)
	if len(rep.Dropped) != 0 {
		t.Errorf("dropped = %v, want []", rep.Dropped)
	}
	if len(rep.Oversplit) != 0 {
		t.Errorf("oversplit = %v, want [] (new file has TWO decls)", rep.Oversplit)
	}
}

func TestDroppedDeclIsTheMissingDefinition(t *testing.T) {
	// Beta vanished entirely — present before, absent everywhere after.
	before := map[string]map[string]string{"demo": {"demo.go": godfile}}
	after := map[string]map[string]string{"demo": {"demo.go": "package demo\n\nfunc Alpha() int { return 1 }\n"}}
	rep := Verify(before, after)
	if len(rep.Relocated) != 0 {
		t.Errorf("relocated = %v, want []", rep.Relocated)
	}
	if len(rep.Dropped) != 1 {
		t.Fatalf("dropped = %v, want 1", rep.Dropped)
	}
	if rep.Dropped[0].Kind != "func" || rep.Dropped[0].Name != "Beta" {
		t.Errorf("dropped[0] = (%s,%s), want (func,Beta)", rep.Dropped[0].Kind, rep.Dropped[0].Name)
	}
}

func TestRelocatedDeclIsNotDropped(t *testing.T) {
	// Beta left package demo and reappeared in package other — RELOCATED, not DROPPED.
	before := map[string]map[string]string{
		"demo":  {"demo.go": godfile},
		"other": {"other.go": "package other\n"},
	}
	after := map[string]map[string]string{
		"demo":  {"demo.go": "package demo\n\nfunc Alpha() int { return 1 }\n"},
		"other": {"other.go": "package other\n\nfunc Beta() int { return 2 }\n"},
	}
	rep := Verify(before, after)
	if len(rep.Dropped) != 0 {
		t.Errorf("dropped = %v, want []", rep.Dropped)
	}
	if len(rep.Relocated) != 1 {
		t.Fatalf("relocated = %v, want 1", rep.Relocated)
	}
	if rep.Relocated[0].Name != "Beta" {
		t.Errorf("relocated[0].name = %q, want Beta", rep.Relocated[0].Name)
	}
	if !reflect.DeepEqual(rep.Relocated[0].To, []string{"other"}) {
		t.Errorf("relocated[0].to = %v, want [other]", rep.Relocated[0].To)
	}
}

func TestTypeAliasConsolidationKeepsLocalNameQuiet(t *testing.T) {
	// A struct is replaced by `type X = pkg.Y`. The local NAME X survives as an alias
	// decl, so decl-level verify stays quiet (no false DROP) — the honest v1 boundary.
	before := map[string]map[string]string{"main": {"a.go": "package main\n\ntype guardInfoSession struct {\n\tRun string\n}\n"}}
	after := map[string]map[string]string{"main": {"a.go": "package main\n\ntype guardInfoSession = guardvars.SessionVars\n"}}
	rep := Verify(before, after)
	if len(rep.Dropped) != 0 {
		t.Errorf("dropped = %v, want []", rep.Dropped)
	}
	if len(rep.Relocated) != 0 {
		t.Errorf("relocated = %v, want []", rep.Relocated)
	}
}

func TestOversplitSkipsMainAndBuildTaggedStubs(t *testing.T) {
	// A new single-decl file is NOT over-split when it is a `func main` entrypoint or a
	// per-OS build-tagged stub.
	before := map[string]map[string]string{}
	after := map[string]map[string]string{
		"cmd/x":      {"main.go": "package main\n\nfunc main() {}\n"},
		"internal/w": {"impl_other.go": "//go:build !windows\n\npackage w\n\nfunc probe() int { return 0 }\n"},
		"internal/y": {"helper.go": "package y\n\nfunc lonely() int { return 1 }\n"},
	}
	rep := Verify(before, after)
	if len(rep.Oversplit) != 1 || rep.Oversplit[0].File != "internal/y/helper.go" {
		t.Errorf("oversplit = %v, want [internal/y/helper.go]", rep.Oversplit)
	}
}

func TestGroupedBlocksAreFootnotedNotClassified(t *testing.T) {
	// A removed grouped const block is counted (grouped_skipped) but never emitted as a
	// DROPPED or RELOCATED claim.
	before := map[string]map[string]string{"demo": {"c.go": "package demo\n\nconst (\n\tA = 1\n\tB = 2\n)\n"}}
	after := map[string]map[string]string{"demo": {"c.go": "package demo\n"}}
	rep := Verify(before, after)
	if len(rep.Dropped) != 0 {
		t.Errorf("dropped = %v, want []", rep.Dropped)
	}
	if len(rep.Relocated) != 0 {
		t.Errorf("relocated = %v, want []", rep.Relocated)
	}
	if rep.GroupedSkipped != 1 {
		t.Errorf("grouped_skipped = %d, want 1", rep.GroupedSkipped)
	}
}

func TestDeclsOfReusesGodsplitFold(t *testing.T) {
	got := DeclsOf(godfile)
	want := []DeclID{{"func", "Alpha"}, {"func", "Beta"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("DeclsOf = %v, want %v", got, want)
	}
}
