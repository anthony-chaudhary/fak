package debtlane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUngatedBenchmarkGateDetection(t *testing.T) {
	tmp := t.TempDir()

	// State 1: Unregistered benchmark -> emits finding "unregistered in benchmark catalog"
	state1Dir := filepath.Join(tmp, "internal", "state1")
	if err := os.MkdirAll(state1Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	state1Code := `package state1

// Worker performs clean unit operations.
type Worker struct{}

// Work returns verified output.
func (w *Worker) Work() int { return 42 }
`
	if err := os.WriteFile(filepath.Join(state1Dir, "worker.go"), []byte(state1Code), 0o644); err != nil {
		t.Fatal(err)
	}
	state1Test := `package state1

import "testing"

func TestWork(t *testing.T) {
	w := &Worker{}
	if got := w.Work(); got != 42 {
		t.Fatalf("expected 42, got %d", got)
	}
}

func BenchmarkWork(b *testing.B) {
	w := &Worker{}
	for i := 0; i < b.N; i++ {
		_ = w.Work()
	}
}
`
	if err := os.WriteFile(filepath.Join(state1Dir, "worker_test.go"), []byte(state1Test), 0o644); err != nil {
		t.Fatal(err)
	}

	lane1 := DebtLane{
		Lane:        "state1",
		UnitOfWork:  "internal/state1",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Integrated:  true,
			Dogfooded:   true,
			Benchmarked: true,
		},
	}

	findings1 := InspectUnitDetectors(&lane1, state1Dir)
	var ungated1 []FindingProvenance
	for _, f := range findings1 {
		if f.Dimension == string(DimUngatedPerformanceBenchmark) {
			ungated1 = append(ungated1, f)
		}
	}
	if len(ungated1) != 1 {
		t.Fatalf("State 1: expected 1 ungated finding, got %d: %+v", len(ungated1), ungated1)
	}
	if !strings.Contains(ungated1[0].Message, "unregistered in benchmark catalog") {
		t.Errorf("State 1: expected message to contain 'unregistered in benchmark catalog', got %q", ungated1[0].Message)
	}

	// State 2: Catalog-only benchmark -> emits finding "is cataloged but lacks declared SLO threshold and executable regression gate reference"
	state2Dir := filepath.Join(tmp, "internal", "state2")
	if err := os.MkdirAll(state2Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	state2Code := `package state2

// Worker performs clean unit operations.
type Worker struct{}

// Work returns verified output.
func (w *Worker) Work() int { return 42 }
`
	if err := os.WriteFile(filepath.Join(state2Dir, "worker.go"), []byte(state2Code), 0o644); err != nil {
		t.Fatal(err)
	}
	state2Test := `package state2

import "testing"

func TestWork(t *testing.T) {
	w := &Worker{}
	if got := w.Work(); got != 42 {
		t.Fatalf("expected 42, got %d", got)
	}
}

// BenchmarkCatalog: BENCHMARK-AUTHORITY.md
func BenchmarkWork(b *testing.B) {
	w := &Worker{}
	for i := 0; i < b.N; i++ {
		_ = w.Work()
	}
}
`
	if err := os.WriteFile(filepath.Join(state2Dir, "worker_test.go"), []byte(state2Test), 0o644); err != nil {
		t.Fatal(err)
	}

	lane2 := DebtLane{
		Lane:        "state2",
		UnitOfWork:  "internal/state2",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Integrated:  true,
			Dogfooded:   true,
			Benchmarked: true,
		},
		Related: RelatedThings{
			BenchmarkWitnesses: []string{"state2"},
		},
	}

	findings2 := InspectUnitDetectors(&lane2, state2Dir)
	var ungated2 []FindingProvenance
	for _, f := range findings2 {
		if f.Dimension == string(DimUngatedPerformanceBenchmark) {
			ungated2 = append(ungated2, f)
		}
	}
	if len(ungated2) != 1 {
		t.Fatalf("State 2: expected 1 ungated finding, got %d: %+v", len(ungated2), ungated2)
	}
	if !strings.Contains(ungated2[0].Message, "is cataloged but lacks declared SLO threshold and executable regression gate reference") {
		t.Errorf("State 2: expected message to contain 'is cataloged but lacks declared SLO threshold and executable regression gate reference', got %q", ungated2[0].Message)
	}

	// State 3: Catalog + threshold + gate -> 0 ungated findings
	state3Dir := filepath.Join(tmp, "internal", "state3")
	if err := os.MkdirAll(state3Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	state3Code := `package state3

// Worker performs clean unit operations.
type Worker struct{}

// Work returns verified output.
func (w *Worker) Work() int { return 42 }
`
	if err := os.WriteFile(filepath.Join(state3Dir, "worker.go"), []byte(state3Code), 0o644); err != nil {
		t.Fatal(err)
	}
	state3Test := `package state3

import "testing"

func TestWork(t *testing.T) {
	w := &Worker{}
	if got := w.Work(); got != 42 {
		t.Fatalf("expected 42, got %d", got)
	}
}

// BenchmarkCatalog: BENCHMARK-AUTHORITY.md#state3
// Threshold: latency <= 100ns
// RegressionGate: go test -run=^$ -bench=BenchmarkWork ./internal/state3
func BenchmarkWork(b *testing.B) {
	w := &Worker{}
	for i := 0; i < b.N; i++ {
		_ = w.Work()
	}
}
`
	if err := os.WriteFile(filepath.Join(state3Dir, "worker_test.go"), []byte(state3Test), 0o644); err != nil {
		t.Fatal(err)
	}

	lane3 := DebtLane{
		Lane:        "state3",
		UnitOfWork:  "internal/state3",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Integrated:  true,
			Dogfooded:   true,
			Benchmarked: true,
		},
		Related: RelatedThings{
			BenchmarkWitnesses: []string{"state3"},
		},
	}

	findings3 := InspectUnitDetectors(&lane3, state3Dir)
	for _, f := range findings3 {
		if f.Dimension == string(DimUngatedPerformanceBenchmark) {
			t.Errorf("State 3: unexpected ungated finding: %+v", f)
		}
	}

	// Verify DimUngatedPerformanceBenchmark is present in BuildCoverageReceipt dimensions
	receipt := BuildCoverageReceipt(tmp, "fak", []DebtLane{lane1, lane2, lane3}, append(findings1, findings2...), 6,
		WithExecutedDimensions(DimUngatedPerformanceBenchmark))
	foundDim := false
	for _, d := range receipt.Dimensions {
		if d == string(DimUngatedPerformanceBenchmark) {
			foundDim = true
			break
		}
	}
	if !foundDim {
		t.Errorf("expected %s in coverage receipt dimensions: %v", DimUngatedPerformanceBenchmark, receipt.Dimensions)
	}

	// Also verify receipt constructed from findings automatically includes DimUngatedPerformanceBenchmark
	receiptAuto := BuildCoverageReceipt(tmp, "fak", []DebtLane{lane1}, findings1, 2)
	foundAutoDim := false
	for _, d := range receiptAuto.Dimensions {
		if d == string(DimUngatedPerformanceBenchmark) {
			foundAutoDim = true
			break
		}
	}
	if !foundAutoDim {
		t.Errorf("expected %s in auto-derived coverage receipt dimensions: %v", DimUngatedPerformanceBenchmark, receiptAuto.Dimensions)
	}
}
