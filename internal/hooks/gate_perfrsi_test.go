package hooks

import (
	"strings"
	"testing"
)

func TestGatePerformanceRSINudge_WarnsOnUnreferencedPerfCode(t *testing.T) {
	cases := []struct {
		name string
		path string
	}{
		{"compute", "internal/compute/fastgemm.go"},
		{"model", "internal/model/attention.go"},
		{"benchmarks", "benchmarks/latency_sweep.go"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := &StagedDiff{
				StagedPaths: []string{tc.path},
				AddedByFile: map[string][]AddedLine{
					tc.path: {{Text: "func OptimizeHotPath() {}"}},
				},
			}
			findings, err := gatePerformanceRSINudge(d)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(findings) != 1 {
				t.Fatalf("expected 1 finding, got %d (%v)", len(findings), findings)
			}
			f := findings[0]
			if f.Gate != "PERFORMANCE_RSI_NUDGE" {
				t.Errorf("gate = %q, want PERFORMANCE_RSI_NUDGE", f.Gate)
			}
			if f.File != tc.path {
				t.Errorf("file = %q, want %q", f.File, tc.path)
			}
			if !f.Advisory {
				t.Errorf("expected advisory finding")
			}
			if !strings.Contains(f.Detail, "performance/optimization code") {
				t.Errorf("finding detail missing context: %q", f.Detail)
			}
		})
	}
}

func TestGatePerformanceRSINudge_SilentOnTrailer(t *testing.T) {
	trailers := []string{
		"performance-rsi: verified by scorecard",
		"Performance-RSI: pass",
		"perfrsi: verified",
		"Perfrsi: validated",
		"ALLOW_NO_PERFRSI_NUDGE=1",
		"allow-no-perfrsi-nudge",
	}

	for _, tr := range trailers {
		t.Run(tr, func(t *testing.T) {
			d := &StagedDiff{
				StagedPaths: []string{"internal/compute/fastgemm.go"},
				AddedByFile: map[string][]AddedLine{
					"internal/compute/fastgemm.go": {
						{Text: "func OptimizeHotPath() {}"},
						{Text: tr},
					},
				},
			}
			findings, err := gatePerformanceRSINudge(d)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(findings) != 0 {
				t.Errorf("expected 0 findings with trailer %q, got %d (%v)", tr, len(findings), findings)
			}
		})
	}
}

func TestGatePerformanceRSINudge_SilentOnEvidenceOrScorecard(t *testing.T) {
	t.Run("staged evidence path", func(t *testing.T) {
		d := &StagedDiff{
			StagedPaths: []string{"internal/compute/fastgemm.go", "internal/perfrsiscore/testdata/complete.json"},
			AddedByFile: map[string][]AddedLine{
				"internal/compute/fastgemm.go": {{Text: "func OptimizeHotPath() {}"}},
			},
		}
		findings, err := gatePerformanceRSINudge(d)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 0 {
			t.Errorf("expected 0 findings when evidence path is staged, got %v", findings)
		}
	})

	t.Run("added reference to perfrsiscore", func(t *testing.T) {
		d := &StagedDiff{
			StagedPaths: []string{"internal/compute/fastgemm.go"},
			AddedByFile: map[string][]AddedLine{
				"internal/compute/fastgemm.go": {
					{Text: "func OptimizeHotPath() {}"},
					{Text: "// benchmark evaluated against perfrsiscore metrics"},
				},
			},
		}
		findings, err := gatePerformanceRSINudge(d)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 0 {
			t.Errorf("expected 0 findings when perfrsiscore referenced, got %v", findings)
		}
	})
}

func TestGatePerformanceRSINudge_SilentOnUnrelatedFiles(t *testing.T) {
	d := &StagedDiff{
		StagedPaths: []string{"pkg/abi/abi.go", "internal/policy/rules.go"},
		AddedByFile: map[string][]AddedLine{
			"pkg/abi/abi.go": {{Text: "type MyType struct{}"}},
		},
	}
	findings, err := gatePerformanceRSINudge(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Errorf("expected 0 findings for unrelated files, got %v", findings)
	}
}

func TestGatePerformanceRSINudge_RegisteredInPreCommitGates(t *testing.T) {
	var found *Gate
	for i, g := range PreCommitGates() {
		if g.Name == "PERFORMANCE_RSI_NUDGE" {
			found = &PreCommitGates()[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("PERFORMANCE_RSI_NUDGE is not registered in PreCommitGates()")
	}
	if found.DefaultMode != "warn" {
		t.Errorf("DefaultMode = %q, want warn", found.DefaultMode)
	}
	if found.ModeEnv != "FLEET_PERFRSI_GUARD" {
		t.Errorf("ModeEnv = %q, want FLEET_PERFRSI_GUARD", found.ModeEnv)
	}
	if found.EscapeEnv != "ALLOW_NO_PERFRSI_NUDGE" {
		t.Errorf("EscapeEnv = %q, want ALLOW_NO_PERFRSI_NUDGE", found.EscapeEnv)
	}
	if found.Check == nil {
		t.Errorf("Check is nil")
	}
}

func TestGatePerformanceRSINudge_CandidatesRecorded(t *testing.T) {
	d := emptyStagedDiff(t.TempDir())
	findings, err := gatePerformanceRSINudge(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected 0 findings on empty diff, got %v", findings)
	}
	n, unit, ok := d.Candidates("PERFORMANCE_RSI_NUDGE")
	if !ok {
		t.Fatalf("candidate denominator was not recorded")
	}
	if n != 0 {
		t.Errorf("candidate count = %d, want 0", n)
	}
	if unit != "touched performance file(s)" {
		t.Errorf("candidate unit = %q, want 'touched performance file(s)'", unit)
	}
}
