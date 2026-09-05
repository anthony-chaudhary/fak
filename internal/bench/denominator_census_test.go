package bench_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
)

func TestDenominatorCensus_Valid(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain:      []string{"task-alpha", "task-beta", "task-gamma", "task-delta", "task-epsilon"},
		AllowSimulation: false,
	}

	results := []bench.TaskResult{
		{TaskID: "task-alpha", Outcome: bench.OutcomePass},
		{TaskID: "task-beta", Outcome: bench.OutcomePass},
		{TaskID: "task-gamma", Outcome: bench.OutcomeFail},
		{TaskID: "task-delta", Outcome: bench.OutcomeTimeout},
		{TaskID: "task-epsilon", Outcome: bench.OutcomeCrash},
	}

	report, err := bench.VerifyCensusDomain(manifest, results)
	if err != nil {
		t.Fatalf("unexpected census verification error: %v", err)
	}

	if !report.Verified {
		t.Fatal("expected report.Verified to be true")
	}
	if report.NTotal != 5 {
		t.Fatalf("expected NTotal=5, got %d", report.NTotal)
	}
	if report.NPass != 2 {
		t.Fatalf("expected NPass=2, got %d", report.NPass)
	}
	if report.NFail != 1 {
		t.Fatalf("expected NFail=1, got %d", report.NFail)
	}
	if report.NTimeout != 1 {
		t.Fatalf("expected NTimeout=1, got %d", report.NTimeout)
	}
	if report.NCrash != 1 {
		t.Fatalf("expected NCrash=1, got %d", report.NCrash)
	}
	if report.NRefused != 0 {
		t.Fatalf("expected NRefused=0, got %d", report.NRefused)
	}
	if report.NSimulated != 0 {
		t.Fatalf("expected NSimulated=0, got %d", report.NSimulated)
	}

	sumCategorized := report.NPass + report.NFail + report.NTimeout + report.NCrash + report.NRefused
	if sumCategorized != report.NTotal {
		t.Fatalf("categorized sum %d != NTotal %d", sumCategorized, report.NTotal)
	}

	expectedPassRate := 2.0 / 5.0
	if report.PassRate != expectedPassRate {
		t.Fatalf("expected PassRate=%f, got %f", expectedPassRate, report.PassRate)
	}

	if report.DomainProof == "" {
		t.Fatal("expected non-empty DomainProof")
	}

	expectedProof := bench.ComputeDomainProof(manifest.TaskDomain)
	if report.DomainProof != expectedProof {
		t.Fatalf("expected DomainProof=%s, got %s", expectedProof, report.DomainProof)
	}

	// Verify order-independence: shuffled results produce the identical report and domain proof.
	shuffledResults := []bench.TaskResult{
		{TaskID: "task-epsilon", Outcome: bench.OutcomeCrash},
		{TaskID: "task-alpha", Outcome: bench.OutcomePass},
		{TaskID: "task-delta", Outcome: bench.OutcomeTimeout},
		{TaskID: "task-gamma", Outcome: bench.OutcomeFail},
		{TaskID: "task-beta", Outcome: bench.OutcomePass},
	}
	shuffledReport, err := bench.VerifyCensusDomain(manifest, shuffledResults)
	if err != nil {
		t.Fatalf("unexpected error on shuffled results: %v", err)
	}
	if shuffledReport.DomainProof != report.DomainProof {
		t.Fatalf("expected identical DomainProof on shuffled results: %s != %s", shuffledReport.DomainProof, report.DomainProof)
	}
}

func TestDenominatorCensus_MissingTasks(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain: []string{"task-1", "task-2", "task-3"},
	}
	results := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass},
		{TaskID: "task-2", Outcome: bench.OutcomePass},
		// task-3 dropped
	}

	_, err := bench.VerifyCensusDomain(manifest, results)
	if err == nil {
		t.Fatal("expected error on dropped/missing tasks, got nil")
	}
	if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
		t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
	}
}

func TestDenominatorCensus_DuplicateTaskIDs(t *testing.T) {
	t.Run("duplicate in results", func(t *testing.T) {
		manifest := bench.CensusManifest{
			TaskDomain: []string{"task-1", "task-2"},
		}
		results := []bench.TaskResult{
			{TaskID: "task-1", Outcome: bench.OutcomePass},
			{TaskID: "task-1", Outcome: bench.OutcomeFail},
		}

		_, err := bench.VerifyCensusDomain(manifest, results)
		if err == nil {
			t.Fatal("expected error on duplicate task ID in results, got nil")
		}
		if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
			t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
		}
	})

	t.Run("duplicate in manifest domain", func(t *testing.T) {
		manifest := bench.CensusManifest{
			TaskDomain: []string{"task-1", "task-1"},
		}
		results := []bench.TaskResult{
			{TaskID: "task-1", Outcome: bench.OutcomePass},
		}

		_, err := bench.VerifyCensusDomain(manifest, results)
		if err == nil {
			t.Fatal("expected error on duplicate task ID in manifest domain, got nil")
		}
		if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
			t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
		}
	})
}

func TestDenominatorCensus_UnknownTaskIDs(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain: []string{"task-1", "task-2"},
	}
	results := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass},
		{TaskID: "task-ghost", Outcome: bench.OutcomePass},
	}

	_, err := bench.VerifyCensusDomain(manifest, results)
	if err == nil {
		t.Fatal("expected error on unexpected/uncataloged task ID, got nil")
	}
	if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
		t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
	}
}

func TestDenominatorCensus_SimulatedRejected(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain:      []string{"task-1"},
		AllowSimulation: false,
	}
	results := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass, Simulated: true},
	}

	_, err := bench.VerifyCensusDomain(manifest, results)
	if err == nil {
		t.Fatal("expected error when simulation is disallowed, got nil")
	}
	if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
		t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
	}
}

func TestDenominatorCensus_SimulatedAllowed(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain:      []string{"task-1", "task-2"},
		AllowSimulation: true,
	}
	results := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass, Simulated: true},
		{TaskID: "task-2", Outcome: bench.OutcomeRefused, Simulated: false},
	}

	report, err := bench.VerifyCensusDomain(manifest, results)
	if err != nil {
		t.Fatalf("unexpected error when simulation is allowed: %v", err)
	}
	if report.NSimulated != 1 {
		t.Fatalf("expected NSimulated=1, got %d", report.NSimulated)
	}
	if report.NRefused != 1 {
		t.Fatalf("expected NRefused=1, got %d", report.NRefused)
	}
	if report.NPass != 1 {
		t.Fatalf("expected NPass=1, got %d", report.NPass)
	}
	if report.PassRate != 0.5 {
		t.Fatalf("expected PassRate=0.5, got %f", report.PassRate)
	}
}

func TestDenominatorCensus_EmptyManifest(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain: []string{},
	}
	results := []bench.TaskResult{}

	_, err := bench.VerifyCensusDomain(manifest, results)
	if err == nil {
		t.Fatal("expected error on empty manifest task domain, got nil")
	}
	if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
		t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
	}
}

func TestDenominatorCensus_DomainProofMismatch(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain:          []string{"task-1", "task-2"},
		ExpectedDomainProof: "bad-proof-00000000000000000000000000000000000000000000000000000000",
	}
	results := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass},
		{TaskID: "task-2", Outcome: bench.OutcomeFail},
	}

	_, err := bench.VerifyCensusDomain(manifest, results)
	if err == nil {
		t.Fatal("expected error on domain proof mismatch, got nil")
	}
	if !strings.Contains(err.Error(), bench.ReasonUnwitnessedDenominator) {
		t.Fatalf("expected error to contain %s, got %v", bench.ReasonUnwitnessedDenominator, err)
	}
}

func TestDenominatorCensus_PassWitnessRequired(t *testing.T) {
	manifest := bench.CensusManifest{
		TaskDomain:         []string{"task-1"},
		RequirePassWitness: true,
	}
	// Missing witness proof on pass
	resultsMissing := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass, WitnessProof: ""},
	}
	if _, err := bench.VerifyCensusDomain(manifest, resultsMissing); err == nil {
		t.Fatal("expected error when pass witness is required but empty, got nil")
	}

	// Valid witness proof on pass
	resultsValid := []bench.TaskResult{
		{TaskID: "task-1", Outcome: bench.OutcomePass, WitnessProof: "sha256:abc123"},
	}
	report, err := bench.VerifyCensusDomain(manifest, resultsValid)
	if err != nil {
		t.Fatalf("unexpected error with valid witness proof: %v", err)
	}
	if report.NPass != 1 || !report.Verified {
		t.Fatalf("expected verified report with NPass=1, got %+v", report)
	}
}

func TestDenominatorDosReasonRegistered(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	var root string
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			root = dir
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find dos.toml in any parent directory")
		}
		dir = parent
	}

	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	content := string(raw)

	header := "[reasons.UNWITNESSED_DENOMINATOR]"
	if !strings.Contains(content, header) {
		t.Fatalf("dos.toml does not declare %s", header)
	}
}
