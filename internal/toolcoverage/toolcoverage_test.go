package toolcoverage

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAuditLoadBearingCoverageAndDebt(t *testing.T) {
	a := AuditModules([]string{"foo", "bar", "baz"}, map[string]bool{"foo": true}, "the skill calls foo.py and bar.py here")
	if a.TotalModules != 3 || a.Tested != 1 || a.LoadBearing != 2 || a.LoadBearingTested != 1 || a.Debt != 1 {
		t.Fatalf("audit=%+v", a)
	}
	if len(a.LoadBearingUntested) != 1 || a.LoadBearingUntested[0] != "bar" {
		t.Fatalf("untested=%v", a.LoadBearingUntested)
	}
	if a.LoadBearingCoveragePct == nil || *a.LoadBearingCoveragePct != 50.0 {
		t.Fatalf("coverage=%v", a.LoadBearingCoveragePct)
	}
}

func TestAuditUnreferencedUntestedIsNotDebt(t *testing.T) {
	a := AuditModules([]string{"solo"}, map[string]bool{}, "")
	if a.Debt != 0 || a.LoadBearing != 0 || a.LoadBearingCoveragePct != nil {
		t.Fatalf("audit=%+v", a)
	}
}

func TestAuditSubstringDoesNotFalseMatch(t *testing.T) {
	a := AuditModules([]string{"bench_node"}, map[string]bool{}, "we run bench.py in CI")
	if a.LoadBearing != 0 {
		t.Fatalf("bench_node falsely marked load-bearing: %+v", a)
	}
}

func TestBuildPayload(t *testing.T) {
	floor := 90.0
	p := BuildPayload("r", AuditModules([]string{"foo", "bar"}, map[string]bool{}, "foo.py bar.py"), &floor)
	if p.OK || p.Verdict != "BELOW_FLOOR" {
		t.Fatalf("payload=%+v", p)
	}
	p = BuildPayload("r", AuditModules([]string{"foo"}, map[string]bool{"foo": true}, "foo.py"), &floor)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("payload=%+v", p)
	}
	p = BuildPayload("r", AuditModules([]string{"solo"}, map[string]bool{}, ""), &floor)
	if !p.OK || p.Verdict != "NO_LOAD_BEARING_MODULES" {
		t.Fatalf("payload=%+v", p)
	}
}

func TestCollectDiscovery(t *testing.T) {
	root := t.TempDir()
	tools := filepath.Join(root, "tools")
	if err := os.MkdirAll(tools, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(tools, "foo.py"), "x=1")
	mustWrite(t, filepath.Join(tools, "foo_test.py"), "x=1")
	mustWrite(t, filepath.Join(tools, "bar.py"), "x=1")
	mustWrite(t, filepath.Join(tools, "__init__.py"), "")
	skills := filepath.Join(root, ".claude", "skills", "s")
	if err := os.MkdirAll(skills, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(skills, "SKILL.md"), "runs tools/bar.py")
	wf := filepath.Join(root, ".github", "workflows")
	if err := os.MkdirAll(wf, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(wf, "ci.yml"), "python tools/foo.py")

	modules, err := FindModuleStems(tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(modules) != 2 || modules[0] != "bar" || modules[1] != "foo" {
		t.Fatalf("modules=%v", modules)
	}
	tests, err := FindTestStems(tools)
	if err != nil {
		t.Fatal(err)
	}
	if !tests["foo"] {
		t.Fatalf("tests=%v", tests)
	}
	floor := 90.0
	p, err := Collect(root, &floor)
	if err != nil {
		t.Fatal(err)
	}
	if p.Debt != 1 || len(p.LoadBearingUntested) != 1 || p.LoadBearingUntested[0] != "bar" || p.Verdict != "BELOW_FLOOR" {
		t.Fatalf("payload=%+v", p)
	}
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
