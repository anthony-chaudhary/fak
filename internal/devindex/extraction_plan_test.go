package devindex

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRemainingExtractionReportDisputesFailClosed(t *testing.T) {
	pkg, err := vsLoadCmd(repoRootForSurface(t))
	if err != nil {
		t.Fatal(err)
	}
	report, _, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]string{}
	for _, row := range report.Safe {
		for _, command := range row.Commands {
			seen[command] = "safe"
		}
	}
	for _, row := range report.Excluded {
		if len(row.Reasons) == 0 {
			t.Fatalf("excluded row has no closed reason: %+v", row)
		}
		for _, reason := range row.Reasons {
			if reason.Code == "" || reason.Source == "" || reason.EvidenceCount < 1 {
				t.Fatalf("unattributed reason: %+v", reason)
			}
		}
		for _, command := range row.Commands {
			seen[command] = "excluded"
		}
	}
	if len(seen) != report.Counts.Commands {
		t.Fatalf("partition has %d unique commands, counts says %d", len(seen), report.Counts.Commands)
	}
	for _, command := range []string{"stale-work", "terminal-relief", "score", "trajquery", "signals"} {
		if seen[command] != "excluded" {
			t.Errorf("%s = %q, want fail-closed excluded", command, seen[command])
		}
	}
}

func TestLocalMethodTargetsDoesNotTreatShadowAsPackageType(t *testing.T) {
	pkg := loadExtractionFixture(t, map[string]string{
		"handler.go": `package main
func Handler(){ worker := other{}; worker.Run() }
`,
		"worker.go": "package main\ntype worker struct{}\nfunc (worker) Run(){}\n",
		"other.go":  "package main\ntype other struct{}\nfunc (other) Run(){}\n",
	})
	model := newExtractionModel(pkg)
	closure := model.closure(model.byName["Handler"])
	for _, symbol := range []string{"worker.Run", "other.Run"} {
		found := false
		for _, id := range model.byName[symbol] {
			found = found || closure[id]
		}
		if !found {
			t.Errorf("shadowed selector failed closed without %s in closure", symbol)
		}
	}
}

func TestUnknownTierOverlapFailsClosed(t *testing.T) {
	pkg := loadExtractionFixture(t, map[string]string{
		"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig(); case "unknown-fixture-verb": cmdUnknown() } }
`,
		"config.go":  "package main\nfunc cmdConfig(){ shared() }\n",
		"unknown.go": "package main\nfunc cmdUnknown(){ shared() }\n",
		"shared.go":  "package main\nfunc shared(){}\n",
	})
	report, _, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	row := candidateForCommand(report.Excluded, "config")
	if row == nil || !hasReason(row.Reasons, ReasonUnknownTierOverlap) {
		t.Fatalf("unknown-tier overlap did not fail closed: %+v", report)
	}
}

func TestCandidateDeltasDoNotDoubleCreditSharedImport(t *testing.T) {
	pkg := loadExtractionFixture(t, map[string]string{
		"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig(); case "dormancy": cmdDormancy() } }
`,
		"config.go":   "package main\nimport _ \"github.com/anthony-chaudhary/fak/internal/sharedfixture\"\nfunc cmdConfig(){}\n",
		"dormancy.go": "package main\nimport _ \"github.com/anthony-chaudhary/fak/internal/sharedfixture\"\nfunc cmdDormancy(){}\n",
	})
	report, safeFiles, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	nodes := []ImportNode{
		{ImportPath: "github.com/anthony-chaudhary/fak/cmd/fak", Imports: []string{"github.com/anthony-chaudhary/fak/internal/sharedfixture"}},
		{ImportPath: "github.com/anthony-chaudhary/fak/internal/sharedfixture"},
	}
	projectExtractionDeltas(&report, pkg, nodes, safeFiles)
	for _, command := range []string{"config", "dormancy"} {
		row := candidateForCommand(report.Safe, command)
		if row == nil || row.Delta.DirectImports != 0 {
			t.Fatalf("%s double-credited shared import: %+v", command, row)
		}
	}
	if report.SafeDelta.DirectImports != 1 || report.SafeDelta.Packages != 1 {
		t.Fatalf("safe union did not receive shared import once: %+v", report.SafeDelta)
	}
}

func TestProjectedGraphLossIgnoresInactiveDirectImport(t *testing.T) {
	nodes := []ImportNode{
		{ImportPath: "github.com/anthony-chaudhary/fak/cmd/fak", Imports: []string{"github.com/anthony-chaudhary/fak/internal/live"}},
		{ImportPath: "github.com/anthony-chaudhary/fak/internal/live"},
	}
	direct, packages, internal := projectedGraphLoss(nodes, map[string]bool{"github.com/anthony-chaudhary/fak/internal/windows-only": true})
	if direct != 0 || packages != 0 || internal != 0 {
		t.Fatalf("inactive import credited: %d %d %d", direct, packages, internal)
	}
}

func TestReasonSummaryKeepsCountAndDeterministicWitness(t *testing.T) {
	facts := []ExtractionReason{
		{Code: ReasonSharedDeclaration, Source: "a.go", Symbol: "A"},
		{Code: ReasonSharedDeclaration, Source: "a.go", Symbol: "A"},
		{Code: ReasonSharedDeclaration, Source: "b.go", Symbol: "B"},
		{Code: ReasonSharedDeclaration, Source: "b.go", Symbol: "B"},
		{Code: ReasonSelfExec, Source: "c.go"},
	}
	got := summarizeReasons(facts)
	if len(got) != 2 || got[0].Code != ReasonSelfExec || got[0].EvidenceCount != 1 ||
		got[1].Code != ReasonSharedDeclaration || got[1].Source != "a.go" || got[1].EvidenceCount != 2 {
		t.Fatalf("summary = %+v", got)
	}
}

func TestDeclarationComponentIncludesMethodsTypesGlobalsAndBuildSplitFiles(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig(); case "agent": cmdAgent() } }
func cmdAgent(){ _ = shared }
`,
		"config.go": `package main
type worker struct{ value string }
var shared = "x"
func cmdConfig(){ worker.Run(worker{value: shared}) }
func (worker) Run(){ splitHelper() }
`,
		"config_windows.go": "//go:build windows\n\npackage main\nfunc splitHelper(){}\n",
	}
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	pkg, err := vsLoadPackage(root, "cmd/fak", 1)
	if err != nil {
		t.Fatal(err)
	}
	report, _, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	var config *ExtractionCandidate
	for i := range report.Excluded {
		for _, command := range report.Excluded[i].Commands {
			if command == "config" {
				config = &report.Excluded[i]
			}
		}
	}
	if config == nil {
		t.Fatalf("config not excluded: %+v", report)
	}
	for _, want := range []string{"cmd/fak/config.go", "cmd/fak/config_windows.go"} {
		if !containsString(config.Files, want) {
			t.Errorf("component files %v missing %s", config.Files, want)
		}
	}
	if !hasReason(config.Reasons, ReasonSharedDeclaration) {
		t.Fatalf("global used by runtime did not close component: %+v", config.Reasons)
	}
}

func TestRemainingExtractionReportDeterminism(t *testing.T) {
	pkg, err := vsLoadCmd(repoRootForSurface(t))
	if err != nil {
		t.Fatal(err)
	}
	a, _, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := buildRemainingExtractionReport(pkg)
	if err != nil {
		t.Fatal(err)
	}
	ab, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	bb, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ab, bb) {
		t.Fatal("repeated plans are not byte-identical")
	}
}

func hasReason(reasons []ExtractionReason, want ExtractionReasonCode) bool {
	for _, reason := range reasons {
		if reason.Code == want {
			return true
		}
	}
	return false
}

func loadExtractionFixture(t *testing.T, files map[string]string) *vsPkg {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	pkg, err := vsLoadPackage(root, "cmd/fak", 1)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func candidateForCommand(rows []ExtractionCandidate, command string) *ExtractionCandidate {
	for i := range rows {
		if containsString(rows[i].Commands, command) {
			return &rows[i]
		}
	}
	return nil
}
