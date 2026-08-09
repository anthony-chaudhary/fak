package nativebench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRegistryHasComparisonContractsForInitialNativeCapabilities(t *testing.T) {
	got := All()
	if len(got) < 2 {
		t.Fatalf("contracts=%d, want initial tool-filtering and compression contracts", len(got))
	}
	names := map[string]bool{}
	for _, c := range got {
		names[c.Capability] = true
	}
	for _, name := range []string{"tool_filtering", "context_compression"} {
		if !names[name] {
			t.Errorf("missing %s", name)
		}
	}
}

func TestValidateRequiresTunedNextBestAndWitness(t *testing.T) {
	findings := Validate([]Contract{{Capability: "x", NativePath: "internal/x", Workload: "same", Metrics: []string{"quality"}}})
	text := ""
	for _, f := range findings {
		text += f.Reason + "\n"
	}
	for _, want := range []string{"tuned baseline", "next-best alternative", "witness"} {
		if !strings.Contains(text, want) {
			t.Errorf("findings %q missing %q", text, want)
		}
	}
}

func TestCurrentRegistryIsStructurallyValidApartFromWitnesses(t *testing.T) {
	for _, f := range Validate(All()) {
		if f.Reason != "benchmark witness is missing" {
			t.Errorf("unexpected structural finding for %s: %s", f.Capability, f.Reason)
		}
	}
}

func TestDiscoverNativeLeavesAndCoverage(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "alpha"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "alpha", "alpha.go"), []byte("package alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "testonly"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "testonly", "x_test.go"), []byte("package testonly\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	leaves, err := DiscoverNativeLeaves(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(leaves) != 1 || leaves[0] != "alpha" {
		t.Fatalf("leaves=%v, want [alpha]", leaves)
	}
	report := AuditRoot(root)
	t.Logf("repository native benchmark coverage: %d/%d leaves, %d missing, %d findings", report.Coverage.CoveredLeaves, report.Coverage.NativeLeaves, len(report.Coverage.MissingLeaves), len(report.Findings))
	if report.Coverage.NativeLeaves != 1 || len(report.Coverage.MissingLeaves) != 1 || report.Coverage.MissingLeaves[0] != "alpha" {
		t.Fatalf("coverage=%+v", report.Coverage)
	}
}

func TestAuditRepositoryDiscoversNativeCoverageDebt(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	report := AuditRoot(root)
	t.Logf("repository native benchmark coverage: %d/%d leaves, %d missing, %d findings", report.Coverage.CoveredLeaves, report.Coverage.NativeLeaves, len(report.Coverage.MissingLeaves), len(report.Findings))
	if !report.Coverage.DiscoveryComplete || report.Coverage.NativeLeaves < 100 {
		t.Fatalf("repository discovery did not cover the native leaf inventory: %+v", report.Coverage)
	}
	if report.Coverage.CoveredLeaves != 2 {
		t.Fatalf("covered leaves=%d, want the two initial gateway and ctxmmu leaves", report.Coverage.CoveredLeaves)
	}
	if len(report.Coverage.MissingLeaves) == 0 || report.Complete {
		t.Fatalf("repository-wide debt must remain explicit: %+v", report)
	}
}

func TestValidateRequiresEveryEquivalentIntegrationArm(t *testing.T) {
	contract := Contract{
		Capability: "x", NativePath: "internal/x", Workload: "same", Metrics: []string{"quality"}, Witness: "w.json",
		Alternatives: []Alternative{
			{Name: "tuned", Class: TunedBaseline, Source: "local"},
			{Name: "best", Class: NextBest, Source: "paper"},
		},
		Integrations: []string{"integrated-x"},
	}
	text := ""
	for _, f := range Validate([]Contract{contract}) {
		text += f.Reason + "\n"
	}
	if !strings.Contains(text, `first-class integration "integrated-x" has no comparison arm`) {
		t.Fatalf("findings=%q", text)
	}
	contract.Alternatives = append(contract.Alternatives, Alternative{Name: "fak + integrated-x", Class: FirstClassIntegration, Integration: "integrated-x", Source: "docs/integrations/x.md"})
	for _, f := range Validate([]Contract{contract}) {
		if strings.Contains(f.Reason, "integration") {
			t.Fatalf("unexpected integration finding: %+v", f)
		}
	}
}
