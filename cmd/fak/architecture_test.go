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
var tier=map[string]int{"leaf":1}
var tierName=[]string{"root","primitive"}
`)
	mustWriteArchitectureFile(t, root, "internal/leaf/leaf.go", "package leaf\n")
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

func TestRunArchitectureTextNamesDependents(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runArchitecture(&out, &errOut, []string{"--leaf", "archreport"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "dependents=") {
		t.Fatalf("output does not name dependents: %s", out.String())
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
	if got.Schema != archreport.DiffSchema || got.Changes() != 3 || !reflect.DeepEqual(got.AddedLeaves, []string{"added"}) || len(got.TierChanges) != 1 || !reflect.DeepEqual(got.AddedEdges, []archreport.EdgeChange{{From: "leaf", To: "added"}}) {
		t.Fatalf("diff=%+v", got)
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
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolations: []string{"primitive -> composite"}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-violations"); code != 3 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
	for _, want := range []string{"verdict=regression", "introduced violation primitive -> composite", "remediation:", "baseline -> workspace"} {
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
	diff := archreport.ReportDiff{Schema: archreport.DiffSchema, Verdict: "regression", IntroducedViolations: []string{"a -> b"}}
	var out, errOut bytes.Buffer
	if code := writeArchitectureDiff(&out, &errOut, diff, false, "introduced-diagnostics"); code != 0 {
		t.Fatalf("code=%d output=%s", code, out.String())
	}
}

func TestArchitectureRejectsInvalidFailOn(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runArchitecture(&out, &errOut, []string{"--fail-on", "anything"}); code != 2 || !strings.Contains(errOut.String(), "want introduced-violations or introduced-diagnostics") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}
