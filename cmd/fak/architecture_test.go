package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func TestArchitectureJSON(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"leaf":1,"caller":1,"top":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/leaf/leaf.go", "package leaf\n")
	mustWriteArchitectureFile(t, root, "internal/caller/caller.go", `package caller
import _ "github.com/anthony-chaudhary/fak/internal/leaf"
`)
	mustWriteArchitectureFile(t, root, "internal/top/top.go", `package top
import _ "github.com/anthony-chaudhary/fak/internal/caller"
`)
	var out, errout bytes.Buffer
	if rc := runArchitecture(&out, &errout, []string{"--workspace", root, "--leaf", "leaf", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errout.String())
	}
	var r archreport.Report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != "fak-architecture/1" || len(r.Leaves) != 1 || r.Leaves[0].DeclaredTierName != "primitive" {
		t.Fatalf("%+v", r)
	}
	if got, want := r.Leaves[0].TransitiveDependents, []string{"caller", "top"}; !reflect.DeepEqual(got, want) || r.Leaves[0].BlastRadius != 2 {
		t.Fatalf("transitive dependents=%v blast radius=%d want=%v/2", got, r.Leaves[0].BlastRadius, want)
	}
}
func mustWriteArchitectureFile(t *testing.T, root, path, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestRunArchitectureTextNamesDependentsAndBlastRadius(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"leaf":1,"caller":1,"top":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/leaf/leaf.go", "package leaf\n")
	mustWriteArchitectureFile(t, root, "internal/caller/caller.go", `package caller
import _ "github.com/anthony-chaudhary/fak/internal/leaf"
`)
	mustWriteArchitectureFile(t, root, "internal/top/top.go", `package top
import _ "github.com/anthony-chaudhary/fak/internal/caller"
`)
	var out, errOut bytes.Buffer
	code := runArchitecture(&out, &errOut, []string{"--workspace", root, "--leaf", "leaf"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"dependents=[caller]", "blast-radius=2", "caller: leaf -> caller", "top: leaf -> caller -> top"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output does not name %q: %s", want, out.String())
		}
	}
}

func TestArchitectureTextRendersLateralBiconnectedBlocks(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2}
var tierName=[]string{"zero","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, root, "internal/a/a.go", `package a
import (_ "github.com/anthony-chaudhary/fak/internal/b"; _ "github.com/anthony-chaudhary/fak/internal/c")
`)
	mustWriteArchitectureFile(t, root, "internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	mustWriteArchitectureFile(t, root, "internal/c/c.go", "package c\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"lateral biconnected blocks (single-package resilient):", "tier=foundation-composite members=[a b c] edges=3"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTextRendersLateralArticulationPoints(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2}
var tierName=[]string{"zero","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, root, "internal/a/a.go", `package a
import _ "github.com/anthony-chaudhary/fak/internal/b"
`)
	mustWriteArchitectureFile(t, root, "internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	mustWriteArchitectureFile(t, root, "internal/c/c.go", "package c\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"lateral articulation points (package seams):", "b tier=foundation-composite fragments=[1 1] coupling-pairs=1"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTextRendersLateralBridges(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2}
var tierName=[]string{"zero","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, root, "internal/a/a.go", `package a
import _ "github.com/anthony-chaudhary/fak/internal/b"
`)
	mustWriteArchitectureFile(t, root, "internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	mustWriteArchitectureFile(t, root, "internal/c/c.go", "package c\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"lateral bridges (articulation edges):", "a -> b tier=foundation-composite sides=1/2 coupling-pairs=2", "b -> c tier=foundation-composite sides=2/1 coupling-pairs=2"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTextRendersLateralComponents(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2}
var tierName=[]string{"zero","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, root, "internal/a/a.go", `package a
import _ "github.com/anthony-chaudhary/fak/internal/b"
`)
	mustWriteArchitectureFile(t, root, "internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	mustWriteArchitectureFile(t, root, "internal/c/c.go", "package c\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"lateral components (same-tier coupling):", "foundation-composite   members=3 edges=2 [a b c]"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTextSummarizesTypedEdgeDirections(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"root":1,"low":2,"peer":2,"high":3}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism"}
`)
	for _, leaf := range []string{"root", "peer", "high"} {
		mustWriteArchitectureFile(t, root, "internal/"+leaf+"/"+leaf+".go", "package "+leaf+"\n")
	}
	mustWriteArchitectureFile(t, root, "internal/low/low.go", `package low
import (
 _ "github.com/anthony-chaudhary/fak/internal/root"
 _ "github.com/anthony-chaudhary/fak/internal/peer"
 _ "github.com/anthony-chaudhary/fak/internal/high"
)
`)
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root, "--leaf", "low"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if want := "typed edges: rootward=1 lateral=1 upward=1"; !strings.Contains(out.String(), want) {
		t.Fatalf("output %q missing %q", out.String(), want)
	}
}

func TestArchitectureTextRendersBlastHotspots(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"base":1,"middle":1,"top":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/base/base.go", "package base\n")
	mustWriteArchitectureFile(t, root, "internal/middle/middle.go", `package middle
import _ "github.com/anthony-chaudhary/fak/internal/base"
`)
	mustWriteArchitectureFile(t, root, "internal/top/top.go", `package top
import _ "github.com/anthony-chaudhary/fak/internal/middle"
`)
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"blast hotspots (transitive impact):", "base                   radius=2 max-hops=2", "middle                 radius=1 max-hops=1"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTextNamesStaleDeclarationRecovery(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"healthy":1,"stale":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, root, "internal/healthy/healthy.go", "package healthy\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"diagnostic stale-tier-declaration", "leaf=stale", "create the package or remove its stale tier declaration"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureRecordsUsageAndFoldsWeeks(t *testing.T) {
	root := t.TempDir()
	mustWriteArchitectureFile(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"leaf":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/leaf/leaf.go", "package leaf\n")
	ledger := filepath.Join(t.TempDir(), "usage.jsonl")
	t.Setenv("FAK_ARCHITECTURE_USAGE_FILE", ledger)
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--workspace", root, "--leaf", "leaf", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runArchitecture(&out, &errOut, []string{"--usage", "--json"}); code != 0 {
		t.Fatalf("usage code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Schema string                 `json:"schema"`
		Weeks  []archreport.UsageWeek `json:"weeks"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-architecture-usage-summary/1" || len(got.Weeks) != 1 || got.Weeks[0].Invocations != 1 || got.Weeks[0].Scoped != 1 || got.Weeks[0].JSON != 1 || got.Weeks[0].OK != 1 {
		t.Fatalf("summary=%+v", got)
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), root) || strings.Contains(string(raw), `"leaf"`) {
		t.Fatalf("usage ledger leaked workspace or leaf: %s", raw)
	}
}

func TestArchitectureBaselineWorkspaceDiff(t *testing.T) {
	before, after := t.TempDir(), t.TempDir()
	mustWriteArchitectureFile(t, before, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"leaf":1}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, before, "internal/abi/abi.go", "package abi\n")
	mustWriteArchitectureFile(t, before, "internal/leaf/leaf.go", "package leaf\n")
	mustWriteArchitectureFile(t, after, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"leaf":2,"added":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	mustWriteArchitectureFile(t, after, "internal/abi/abi.go", "package abi\n")
	mustWriteArchitectureFile(t, after, "internal/leaf/leaf.go", `package leaf
import _ "github.com/anthony-chaudhary/fak/internal/added"
`)
	mustWriteArchitectureFile(t, after, "internal/added/added.go", "package added\n")
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--baseline-workspace", before, "--workspace", after, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got archreport.ReportDiff
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != archreport.DiffSchema || got.Changes() != 3 || !reflect.DeepEqual(got.AddedLeaves, []string{"added"}) || len(got.TierChanges) != 1 || !reflect.DeepEqual(got.AddedEdges, []archreport.EdgeChange{{From: "leaf", To: "added"}}) || !reflect.DeepEqual(got.FanInChanges, []archreport.FanInChange{{Leaf: "added", Before: 0, After: 1, Delta: 1}}) {
		t.Fatalf("diff=%+v", got)
	}
}

func TestWriteArchitectureDiffRendersTierGapChanges(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", TierGapChanges: []archreport.TierGapChange{{Leaf: "drift", DeclaredTier: 4, BeforeFloor: 3, AfterFloor: 1, BeforeGap: 1, AfterGap: 3, Delta: 2}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, ""); code != 0 {
		t.Fatalf("code=%d", code)
	}
	if want := "tier-gap drift floor 3 -> 1, gap 1 -> 3 (+2)"; !strings.Contains(out.String(), want) {
		t.Fatalf("output %q missing %q", out.String(), want)
	}
}

func TestWriteArchitectureDiffRendersFanInChanges(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", FanInChanges: []archreport.FanInChange{{Leaf: "shared", Before: 2, After: 5, Delta: 3}, {Leaf: "smaller", Before: 4, After: 2, Delta: -2}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, ""); code != 0 {
		t.Fatalf("code=%d", code)
	}
	for _, want := range []string{"fan-in shared 2 -> 5 (+3)", "fan-in smaller 4 -> 2 (-2)"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestWriteArchitectureDiffEmpty(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean"}, false, ""); code != 0 {
		t.Fatalf("code=%d", code)
	}
	if out.String() != "architecture diff: 0 change(s), verdict=clean\n" {
		t.Fatalf("output=%q", out.String())
	}
}

func TestArchitectureFailOnIntroducedViolations(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolationEdges: []archreport.ViolationEdge{{From: "primitive", FromTierName: "primitive", To: "composite", ToTierName: "foundation-composite", TierDistance: 1}}, IntroducedViolations: []string{"primitive -> composite"}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-violations"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"verdict=regression", "introduced violation primitive(primitive) -> composite(foundation-composite), distance=1", "remediation:", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
	out.Reset()
	if code := writeArchitectureDiff(&out, &errOut, archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", AddedEdges: []archreport.EdgeChange{{From: "a", To: "b"}}}, false, "introduced-violations"); code != 0 {
		t.Fatalf("clean code=%d", code)
	}
}

func TestArchitectureFailOnIntroducedDiagnostics(t *testing.T) {
	diagnostic := archreport.Diagnostic{Kind: archreport.DiagnosticStaleTierDeclaration, Leaf: "gone", Message: "gone is missing", Recovery: "remove its tier row"}
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedDiagnostics: []archreport.Diagnostic{diagnostic}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-diagnostics"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"verdict=regression", "introduced diagnostic stale-tier-declaration leaf=gone", "remove its tier row", "remediation:", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureDiagnosticPolicyIgnoresViolationOnlyRegression(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolationEdges: []archreport.ViolationEdge{{From: "a", To: "b"}}, IntroducedViolations: []string{"a -> b"}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-diagnostics"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIncreasedTierGap(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", TierGapChanges: []archreport.TierGapChange{{Leaf: "drift", BeforeFloor: 3, AfterFloor: 2, BeforeGap: 1, AfterGap: 2, Delta: 1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-tier-gap"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"tier-gap drift", "restore the prior import floor", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureTierGapPolicyIgnoresOtherRegressions(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolationEdges: []archreport.ViolationEdge{{From: "a", To: "b"}}, IntroducedViolations: []string{"a -> b"}, IntroducedDiagnostics: []archreport.Diagnostic{{Kind: "stale", Leaf: "gone"}}, TierGapChanges: []archreport.TierGapChange{{Leaf: "better", Delta: -1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-tier-gap"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIncreasedViolationDistance(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", ViolationDistanceChanges: []archreport.ViolationDistanceChange{{From: "low", To: "high", BeforeDistance: 1, AfterDistance: 2, Delta: 1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-violation-distance"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"violation-distance low -> high 1 -> 2 (+1)", "restore the prior endpoint tiers", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureViolationDistancePolicyIgnoresOtherRegressions(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolationEdges: []archreport.ViolationEdge{{From: "new", To: "edge"}}, ViolationDistanceChanges: []archreport.ViolationDistanceChange{{From: "better", To: "edge", Delta: -1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-violation-distance"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIncreasedBlastRadius(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", BlastRadiusChanges: []archreport.BlastRadiusChange{{Leaf: "shared", Before: 2, After: 5, Delta: 3}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-blast-radius"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"blast-radius shared 2 -> 5 (+3)", "remove/invert the new dependency path", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureBlastRadiusPolicyIgnoresOtherRegressionsAndContractions(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolationEdges: []archreport.ViolationEdge{{From: "new", To: "edge"}}, BlastRadiusChanges: []archreport.BlastRadiusChange{{Leaf: "better", Delta: -2}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-blast-radius"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIntroducedBlastImpacts(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedBlastImpacts: []archreport.BlastImpact{{Source: "shared", Dependent: "new", Path: []string{"shared", "middle", "new"}}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-blast-impacts"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"introduced blast-impact shared -> new path=shared -> middle -> new", "remove/invert each introduced path", "baseline -> workspace"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureBlastImpactPolicyIgnoresResolutions(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", ResolvedBlastImpacts: []archreport.BlastImpact{{Source: "shared", Dependent: "gone", Path: []string{"shared", "gone"}}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-blast-impacts"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	if !strings.Contains(out.String(), "resolved blast-impact shared -> gone") {
		t.Fatalf("output=%q", out.String())
	}
}

func TestArchitectureFailOnIncreasedBlastPathLength(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", BlastPathChanges: []archreport.BlastPathChange{{Source: "shared", Dependent: "target", BeforePath: []string{"shared", "target"}, AfterPath: []string{"shared", "middle", "target"}, BeforeHops: 1, AfterHops: 2, Delta: 1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-blast-path-length"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"blast-path shared -> target hops 1 -> 2 (+1)", "before=shared -> target", "after=shared -> middle -> target", "restore the shorter dependency path"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureBlastPathLengthPolicyIgnoresEqualRerouteAndContraction(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", BlastPathChanges: []archreport.BlastPathChange{{Source: "same", Dependent: "target", Delta: 0}, {Source: "shorter", Dependent: "target", Delta: -1}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "increased-blast-path-length"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIntroducedLateralEdges(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", IntroducedTypedEdges: []archreport.ArchitectureEdge{{From: "left", FromTier: 2, FromTierName: "foundation-composite", To: "right", ToTier: 2, ToTierName: "foundation-composite", Direction: "lateral"}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-lateral-edges"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"typed-edge left(foundation-composite) -> right(foundation-composite)", "direction=lateral", "move the shared seam to a lower tier"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureIntroducedLateralPolicyIgnoresRootwardAndUpward(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, IntroducedTypedEdges: []archreport.ArchitectureEdge{{From: "high", To: "low", Direction: "rootward"}, {From: "low", To: "high", Direction: "upward"}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-lateral-edges"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureFailOnIntroducedLateralCouplings(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedLateralCouplings: []archreport.LateralCoupling{{Tier: 2, TierName: "foundation-composite", Left: "left", Right: "right"}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-lateral-couplings"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"introduced lateral-coupling tier=foundation-composite(2) left <-> right", "remove the lateral bridge", "extract their shared seam downward"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output %q missing %q", out.String(), want)
		}
	}
}

func TestArchitectureLateralCouplingPolicyIgnoresResolutions(t *testing.T) {
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "clean", ResolvedLateralCouplings: []archreport.LateralCoupling{{Tier: 2, Left: "left", Right: "right"}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-lateral-couplings"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureLateralBridgePolicyGatesIntroducedAndIncreased(t *testing.T) {
	tests := []archreport.ReportDiff{
		{IntroducedLateralBridges: []archreport.LateralBridge{{Tier: 2, TierName: "foundation-composite", Left: "a", Right: "b", CouplingPairs: 6}}},
		{LateralBridgeChanges: []archreport.LateralBridgeChange{{Left: "m", Right: "n", BeforeCouplingPairs: 2, AfterCouplingPairs: 6, Delta: 4}}},
	}
	for _, diff := range tests {
		var out, errOut bytes.Buffer
		if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-or-increased-lateral-bridges"); code != 3 {
			t.Fatalf("code=%d output=%s", code, out.String())
		}
		if !strings.Contains(out.String(), "remove the articulation bridge") {
			t.Fatalf("output=%q", out.String())
		}
	}
}

func TestArchitectureLateralBridgePolicyIgnoresResolvedAndDecreased(t *testing.T) {
	diff := archreport.ReportDiff{ResolvedLateralBridges: []archreport.LateralBridge{{Left: "a", Right: "b"}}, LateralBridgeChanges: []archreport.LateralBridgeChange{{Left: "m", Right: "n", Delta: -2}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-or-increased-lateral-bridges"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureLateralArticulationPointPolicyGatesIntroducedAndIncreased(t *testing.T) {
	tests := []archreport.ReportDiff{{IntroducedLateralArticulationPoints: []archreport.LateralArticulationPoint{{Name: "new", CouplingPairs: 3}}}, {LateralArticulationPointChanges: []archreport.LateralArticulationPointChange{{Name: "seam", BeforeCouplingPairs: 1, AfterCouplingPairs: 4, Delta: 3}}}}
	for _, diff := range tests {
		var out, errOut bytes.Buffer
		if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-or-increased-lateral-articulation-points"); code != 3 {
			t.Fatalf("code=%d output=%s", code, out.String())
		}
		if !strings.Contains(out.String(), "remove the package convergence seam") {
			t.Fatalf("output=%q", out.String())
		}
	}
}
func TestArchitectureLateralArticulationPointPolicyIgnoresResolvedAndDecreased(t *testing.T) {
	diff := archreport.ReportDiff{ResolvedLateralArticulationPoints: []archreport.LateralArticulationPoint{{Name: "gone"}}, LateralArticulationPointChanges: []archreport.LateralArticulationPointChange{{Name: "seam", Delta: -2}}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-or-increased-lateral-articulation-points"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureRejectsInvalidFailOn(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--fail-on", "anything"}); code != 2 || !strings.Contains(errOut.String(), "want introduced-violations, introduced-diagnostics, increased-tier-gap, increased-violation-distance, increased-blast-radius, introduced-blast-impacts, increased-blast-path-length, introduced-lateral-edges, introduced-lateral-couplings, introduced-or-increased-lateral-bridges, or introduced-or-increased-lateral-articulation-points") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}
