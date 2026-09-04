package treedoctor

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestScratchHygieneThresholdLogic(t *testing.T) {
	tests := []struct {
		name         string
		count        int
		threshold    int
		wantExceeded bool
		wantWarning  bool
	}{
		{
			name:         "zero files under threshold",
			count:        0,
			threshold:    DefaultScratchGoFilesThreshold,
			wantExceeded: false,
			wantWarning:  false,
		},
		{
			name:         "boundary at threshold 10000",
			count:        10000,
			threshold:    10000,
			wantExceeded: false,
			wantWarning:  false,
		},
		{
			name:         "just below threshold 9999",
			count:        9999,
			threshold:    10000,
			wantExceeded: false,
			wantWarning:  false,
		},
		{
			name:         "exceeded threshold 10001",
			count:        10001,
			threshold:    10000,
			wantExceeded: true,
			wantWarning:  true,
		},
		{
			name:         "well above threshold 25000",
			count:        25000,
			threshold:    10000,
			wantExceeded: true,
			wantWarning:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rep := BuildScratchHygieneReport(tc.count, tc.threshold)
			if rep.ScratchUntrackedGoFiles != tc.count {
				t.Errorf("ScratchUntrackedGoFiles = %d, want %d", rep.ScratchUntrackedGoFiles, tc.count)
			}
			if rep.Threshold != tc.threshold {
				t.Errorf("Threshold = %d, want %d", rep.Threshold, tc.threshold)
			}
			if rep.Exceeded != tc.wantExceeded {
				t.Errorf("Exceeded = %v, want %v", rep.Exceeded, tc.wantExceeded)
			}
			if tc.wantWarning {
				if rep.Warning == "" {
					t.Errorf("expected non-empty warning when exceeded")
				}
				for _, needle := range []string{
					"_scratch",
					">10,000 untracked .go files",
					"without quarantine",
					"isolate workspace scope",
					"reap scratch",
					"prevent LSP/gopls memory explosion",
				} {
					if !strings.Contains(rep.Warning, needle) {
						t.Errorf("warning missing expected needle %q; got: %s", needle, rep.Warning)
					}
				}
			} else if rep.Warning != "" {
				t.Errorf("expected empty warning when not exceeded, got: %q", rep.Warning)
			}
		})
	}
}

func TestDiagnoseScratchHygieneFilesystem(t *testing.T) {
	root := t.TempDir()
	scratchDir := filepath.Join(root, scratchNamespace)

	// Case 1: No _scratch dir exists yet
	rep := DiagnoseScratchHygiene(root)
	if rep.ScratchUntrackedGoFiles != 0 || rep.Exceeded || rep.Threshold != DefaultScratchGoFilesThreshold {
		t.Fatalf("expected clean report for absent _scratch, got: %+v", rep)
	}

	// Case 2: Create _scratch with various files
	subDir := filepath.Join(scratchDir, "producer-a", "nested")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	gitSubDir := filepath.Join(scratchDir, ".git")
	if err := os.MkdirAll(gitSubDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write 3 .go files in _scratch
	for _, name := range []string{
		filepath.Join(scratchDir, "one.go"),
		filepath.Join(scratchDir, "producer-a", "two.go"),
		filepath.Join(subDir, "three.go"),
	} {
		if err := os.WriteFile(name, []byte("package scratch\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// Write non-.go files and a .go file inside .git (should be skipped)
	if err := os.WriteFile(filepath.Join(scratchDir, "notes.txt"), []byte("text"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scratchDir, "data.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(gitSubDir, "hidden.go"), []byte("package git\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// With default threshold (10,000), 3 files should not exceed threshold
	repDefault := DiagnoseScratchHygiene(root)
	if repDefault.ScratchUntrackedGoFiles != 3 {
		t.Errorf("ScratchUntrackedGoFiles = %d, want 3", repDefault.ScratchUntrackedGoFiles)
	}
	if repDefault.Exceeded {
		t.Errorf("expected Exceeded = false for 3 files with threshold 10,000")
	}

	// With threshold = 2, 3 files should exceed threshold
	repExceeded := DiagnoseScratchHygieneThreshold(root, 2)
	if repExceeded.ScratchUntrackedGoFiles != 3 {
		t.Errorf("ScratchUntrackedGoFiles = %d, want 3", repExceeded.ScratchUntrackedGoFiles)
	}
	if !repExceeded.Exceeded {
		t.Errorf("expected Exceeded = true for 3 files with threshold 2")
	}
	if repExceeded.Warning == "" {
		t.Errorf("expected warning for exceeded scratch hygiene")
	}
}

func TestDiagnoseIntegratesScratchHygiene(t *testing.T) {
	root := t.TempDir()
	now := time.Now()
	run := func(_ context.Context, _ string, _ ...string) (string, int, error) {
		return "", 0, nil
	}

	// Full diagnose populates ScratchHygiene
	rep := Diagnose(context.Background(), run, Options{RepoRoot: root, Now: now})
	if rep.ScratchHygiene.Threshold != DefaultScratchGoFilesThreshold {
		t.Errorf("Threshold = %d, want %d", rep.ScratchHygiene.Threshold, DefaultScratchGoFilesThreshold)
	}
	if rep.ScratchHygiene.Exceeded {
		t.Errorf("unexpected Exceeded in clean temp root")
	}

	// LocksOnly skips ScratchHygiene
	repLocks := Diagnose(context.Background(), run, Options{RepoRoot: root, Now: now, LocksOnly: true})
	if repLocks.ScratchHygiene.Threshold != 0 {
		t.Errorf("LocksOnly should skip ScratchHygiene, got threshold = %d", repLocks.ScratchHygiene.Threshold)
	}
}
