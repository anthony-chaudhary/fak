package nativebench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrefixCacheSweepHasNoFirstClassIntegrationArm(t *testing.T) {
	for _, contract := range All() {
		if contract.Capability != "prefix_cache_budget_sweep" {
			continue
		}
		for _, alternative := range contract.Alternatives {
			if alternative.Class == FirstClassIntegration {
				t.Fatalf("unexpected first-class integration arm %q", alternative.Name)
			}
		}
		return
	}
	t.Fatal("prefix cache sweep contract missing")
}

func TestRegistryHasComparisonContractsForInitialNativeCapabilities(t *testing.T) {
	got := All()
	if len(got) < 2 {
		t.Fatalf("contracts=%d, want caching, memory, adjudication, routing, tool-filtering, compression, prefix-reuse, and tokenization contracts", len(got))
	}
	names := map[string]bool{}
	for _, c := range got {
		names[c.Capability] = true
	}
	for _, name := range []string{"tool_result_caching", "context_memory_management", "policy_adjudication", "model_routing", "tool_filtering", "context_compression", "prefix_kv_reuse", "tokenization"} {
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
	if report.Coverage.NativeLeaves != 1 || len(report.Coverage.MissingLeaves) != 1 || report.Coverage.MissingLeaves[0] != "internal/alpha" {
		t.Fatalf("coverage=%+v", report.Coverage)
	}
}

func TestAuditFindsRepositoryFromNestedWorkingDirectory(t *testing.T) {
	t.Chdir(filepath.Join("..", "..", "cmd", "fak"))
	report := Audit()
	if !report.Coverage.DiscoveryComplete || report.Coverage.NativeLeaves < 100 {
		t.Fatalf("nested audit did not discover repository root: %+v", report)
	}
}

func TestAuditRepositoryDiscoversNativeCoverageDebt(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	report := AuditRoot(root)
	t.Logf("repository native benchmark coverage: %d/%d leaves, %d missing, %d findings", report.Coverage.CoveredLeaves, report.Coverage.NativeLeaves, len(report.Coverage.MissingLeaves), len(report.Findings))
	if !report.Coverage.DiscoveryComplete || report.Coverage.NativeLeaves < 100 {
		t.Fatalf("repository discovery did not cover the native leaf inventory: %+v", report.Coverage)
	}
	if report.Coverage.CoveredLeaves != 61 {
		t.Fatalf("covered leaves=%d, want explicit classified coverage count", report.Coverage.CoveredLeaves)
	}
	if report.Coverage.ClassifiedLeaves != 61 || report.Coverage.UnclassifiedLeaves == 0 {
		t.Fatalf("classification debt is not explicit: %+v", report.Coverage)
	}
	if len(report.Coverage.MissingLeaves) == 0 || report.Complete {
		t.Fatalf("repository-wide debt must remain explicit: %+v", report)
	}
}

func TestValidateClassificationsRejectsUnknownAndMalformedEntries(t *testing.T) {
	contracts := []Contract{{Capability: "known"}}
	classifications := []LeafClassification{
		{Leaf: "internal/x", Disposition: DispositionCapability, Capabilities: []string{"missing"}, Reason: "x"},
		{Leaf: "internal/x", Disposition: DispositionInfrastructure, Capabilities: []string{"known"}},
		{Leaf: "outside", Disposition: "mystery", Reason: "x"},
	}
	findings := validateClassifications(classifications, contracts)
	if len(findings) < 5 {
		t.Fatalf("findings=%+v", findings)
	}
}

func TestClassifiedInfrastructureDoesNotNeedAComparisonContract(t *testing.T) {
	root := t.TempDir()
	for _, leaf := range []string{"gateway", "headroom", "nativebench", "radixkv", "tokenizer"} {
		if err := os.MkdirAll(filepath.Join(root, "internal", leaf), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "internal", leaf, leaf+".go"), []byte("package "+leaf+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	report := AuditRoot(root)
	if report.Coverage.CoveredLeaves != 5 || report.Coverage.UnclassifiedLeaves != 0 {
		t.Fatalf("coverage=%+v", report.Coverage)
	}
	if got := report.Coverage.DispositionCounts[DispositionInfrastructure]; got != 1 {
		t.Fatalf("infrastructure dispositions=%d", got)
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

func TestTokenizationRequiresExternalAndIntegrationArms(t *testing.T) {
	for _, contract := range All() {
		if contract.Capability != "tokenization" {
			continue
		}
		if len(contract.Integrations) != 1 || contract.Integrations[0] != "huggingface/tokenizers" {
			t.Fatalf("tokenization integrations=%v", contract.Integrations)
		}
		var llama, hf bool
		for _, arm := range contract.Alternatives {
			llama = llama || (arm.Class == NextBest && arm.Name == "llama.cpp tokenizer")
			hf = hf || (arm.Class == FirstClassIntegration && arm.Integration == "huggingface/tokenizers")
		}
		if !llama || !hf {
			t.Fatalf("tokenization alternatives=%+v", contract.Alternatives)
		}
		return
	}
	t.Fatal("tokenization contract missing")
}

func TestPrefixReuseRequiresLLMDIntegrationArm(t *testing.T) {
	for _, contract := range All() {
		if contract.Capability != "prefix_kv_reuse" {
			continue
		}
		if len(contract.Integrations) != 1 || contract.Integrations[0] != "llm-d" {
			t.Fatalf("prefix reuse integrations=%v", contract.Integrations)
		}
		for _, arm := range contract.Alternatives {
			if arm.Class == FirstClassIntegration && arm.Integration == "llm-d" {
				return
			}
		}
		t.Fatal("prefix reuse contract lacks fak + llm-d integration arm")
	}
	t.Fatal("prefix_kv_reuse contract missing")
}

func TestContextCompressionRequiresLLMLinguaIntegrationArm(t *testing.T) {
	var found bool
	for _, contract := range All() {
		if contract.Capability != "context_compression" {
			continue
		}
		found = true
		if len(contract.Integrations) != 1 || contract.Integrations[0] != "headroom/lingua" {
			t.Fatalf("integrations=%v, want headroom/lingua", contract.Integrations)
		}
		var arm bool
		for _, alternative := range contract.Alternatives {
			if alternative.Class == FirstClassIntegration && alternative.Integration == "headroom/lingua" {
				arm = true
			}
		}
		if !arm {
			t.Fatal("context compression contract lacks fak + LLMLingua integration arm")
		}
	}
	if !found {
		t.Fatal("context_compression contract missing")
	}
}

func TestWitnessResolutionFromRootAndNestedDirectories(t *testing.T) {
	tempRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempRoot, "go.mod"), []byte("module testroot\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(tempRoot, "docs", "notes"), 0o755); err != nil {
		t.Fatal(err)
	}
	witnessContent := []byte("# Test Witness\n")
	if err := os.WriteFile(filepath.Join(tempRoot, "docs", "notes", "TEST-WITNESS.md"), witnessContent, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(tempRoot, "internal", "nativebench"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(tempRoot, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}

	testContracts := []Contract{
		{
			Capability:   "test_with_existing_witness",
			NativePath:   "internal/nativebench/test.go",
			Workload:     "workload",
			Metrics:      []string{"metric"},
			Alternatives: []Alternative{{Name: "base", Class: TunedBaseline, Source: "src"}, {Name: "alt", Class: NextBest, Source: "src"}},
			Witness:      "../../docs/notes/TEST-WITNESS.md",
		},
		{
			Capability:   "test_with_missing_witness",
			NativePath:   "internal/nativebench/test.go",
			Workload:     "workload",
			Metrics:      []string{"metric"},
			Alternatives: []Alternative{{Name: "base", Class: TunedBaseline, Source: "src"}, {Name: "alt", Class: NextBest, Source: "src"}},
			Witness:      "../../docs/notes/NONEXISTENT-WITNESS.md",
		},
		{
			Capability:   "test_with_blank_witness",
			NativePath:   "internal/nativebench/test.go",
			Workload:     "workload",
			Metrics:      []string{"metric"},
			Alternatives: []Alternative{{Name: "base", Class: TunedBaseline, Source: "src"}, {Name: "alt", Class: NextBest, Source: "src"}},
			Witness:      "",
		},
	}

	// 1. ValidateRoot directly from tempRoot
	findingsFromRoot := ValidateRoot(testContracts, tempRoot)
	if len(findingsFromRoot) != 2 {
		t.Fatalf("expected exactly 2 findings (missing + blank), got %d: %+v", len(findingsFromRoot), findingsFromRoot)
	}
	for _, f := range findingsFromRoot {
		if f.Capability == "test_with_existing_witness" {
			t.Fatalf("existing witness was falsely flagged: %s", f.Reason)
		}
	}

	// 2. Validate from nested directories using chdir
	origWd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(origWd) }()

	for _, dir := range []string{tempRoot, filepath.Join(tempRoot, "cmd", "fak"), filepath.Join(tempRoot, "internal", "nativebench")} {
		if err := os.Chdir(dir); err != nil {
			t.Fatal(err)
		}
		findings := Validate(testContracts)
		if len(findings) != len(findingsFromRoot) {
			t.Fatalf("at %s: expected %d findings, got %d: %+v", dir, len(findingsFromRoot), len(findings), findings)
		}
		for i, f := range findings {
			if f.Capability != findingsFromRoot[i].Capability || !strings.Contains(f.Reason, "witness") {
				t.Fatalf("at %s: mismatch finding %d: got %+v, want %+v", dir, i, f, findingsFromRoot[i])
			}
		}
	}
}

func TestCommittedWitnessesAreNotReportedAsUnreadable(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	moduleRoot, err := moduleRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	findings := ValidateRoot(All(), moduleRoot)
	for _, f := range findings {
		if strings.Contains(f.Reason, "is not readable") {
			t.Errorf("committed witness reported unreadable for %s: %s", f.Capability, f.Reason)
		}
	}
}

func TestAlternativeAndContractTreatmentDistinction(t *testing.T) {
	// 1. First-class integrations default to additive treatments
	intArm := Alternative{
		Name:        "fak + llm-d",
		Class:       FirstClassIntegration,
		Integration: "llm-d",
		Source:      "src",
	}
	if intArm.TreatmentKind() != TreatmentAdditive {
		t.Errorf("integration arm treatment = %q, want %q", intArm.TreatmentKind(), TreatmentAdditive)
	}

	// 2. Baselines and alternative engines default to replacement treatments
	baseArm := Alternative{
		Name:   "llama.cpp baseline",
		Class:  TunedBaseline,
		Source: "src",
	}
	if baseArm.TreatmentKind() != TreatmentReplacement {
		t.Errorf("baseline arm treatment = %q, want %q", baseArm.TreatmentKind(), TreatmentReplacement)
	}

	// 3. Explicit treatment overrides
	explicitArm := Alternative{
		Name:      "custom-layer",
		Class:     NextBest,
		Treatment: TreatmentAdditive,
	}
	if explicitArm.TreatmentKind() != TreatmentAdditive {
		t.Errorf("explicit arm treatment = %q, want %q", explicitArm.TreatmentKind(), TreatmentAdditive)
	}

	// 4. Contract treatment defaults to additive
	contract := Contract{
		Capability: "prefix_kv_reuse",
	}
	if contract.TreatmentKind() != TreatmentAdditive {
		t.Errorf("contract treatment = %q, want %q", contract.TreatmentKind(), TreatmentAdditive)
	}
}

func TestResolveWitnessFromModuleRoot(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	modRoot, err := moduleRoot(root)
	if err != nil {
		t.Fatal(err)
	}

	// Resolving an existing witness relative to module root should return valid path
	target := ResolveWitness(modRoot, "docs/native-inference-goal.md")
	if _, err := os.Stat(target); err != nil {
		t.Fatalf("expected ResolveWitness to resolve docs/native-inference-goal.md: %v", err)
	}
}
