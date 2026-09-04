package treedoctor

import (
	"context"
	"errors"
	"fmt"
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

func TestScratchQuotaEnforcement(t *testing.T) {
	t.Run("small quota threshold boundary", func(t *testing.T) {
		root := t.TempDir()
		// Absent _scratch passes quota
		if err := EnforceScratchQuota(root, 5); err != nil {
			t.Fatalf("expected nil error on absent _scratch, got %v", err)
		}

		scratchDir := filepath.Join(root, scratchNamespace)
		if err := os.MkdirAll(scratchDir, 0o755); err != nil {
			t.Fatal(err)
		}

		// Write 5 files (at quota limit)
		for i := 0; i < 5; i++ {
			p := filepath.Join(scratchDir, fmt.Sprintf("file_%d.txt", i))
			if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		if err := EnforceScratchQuota(root, 5); err != nil {
			t.Fatalf("expected 5 files with quota 5 to pass, got: %v", err)
		}

		rep, err := CheckScratchQuota(root, 5)
		if err != nil {
			t.Fatalf("CheckScratchQuota: %v", err)
		}
		if rep.Exceeded || rep.Count != 5 || rep.Quota != 5 {
			t.Fatalf("unexpected CheckScratchQuota result: %+v", rep)
		}

		// Write 6th file (exceeds quota limit)
		p6 := filepath.Join(scratchDir, "file_5.txt")
		if err := os.WriteFile(p6, []byte("data"), 0o644); err != nil {
			t.Fatal(err)
		}

		err = EnforceScratchQuota(root, 5)
		if err == nil {
			t.Fatalf("expected SCRATCH_QUOTA_EXCEEDED error for 6 files with quota 5")
		}
		if !strings.Contains(err.Error(), ErrCodeScratchQuotaExceeded) {
			t.Errorf("expected error string to contain %q, got: %s", ErrCodeScratchQuotaExceeded, err.Error())
		}
		if !errors.Is(err, ErrScratchQuotaExceeded) {
			t.Errorf("expected errors.Is(err, ErrScratchQuotaExceeded) to be true")
		}

		var quotaErr *ScratchQuotaError
		if !errors.As(err, &quotaErr) {
			t.Fatalf("expected *ScratchQuotaError, got %T: %v", err, err)
		}
		if quotaErr.Code != ErrCodeScratchQuotaExceeded {
			t.Errorf("expected code %q, got %q", ErrCodeScratchQuotaExceeded, quotaErr.Code)
		}
		if quotaErr.Count != 6 {
			t.Errorf("expected count 6, got %d", quotaErr.Count)
		}
		if quotaErr.Quota != 5 {
			t.Errorf("expected quota 5, got %d", quotaErr.Quota)
		}

		repExceeded, err := CheckScratchQuota(root, 5)
		if err != nil {
			t.Fatalf("CheckScratchQuota: %v", err)
		}
		if !repExceeded.Exceeded || repExceeded.Count != 6 || repExceeded.Quota != 5 || repExceeded.Code != ErrCodeScratchQuotaExceeded {
			t.Fatalf("unexpected CheckScratchQuota report: %+v", repExceeded)
		}
	})

	t.Run("default quota files ceiling 1000 with 1001 files", func(t *testing.T) {
		root := t.TempDir()
		scratchDir := filepath.Join(root, scratchNamespace, "bulk")
		if err := os.MkdirAll(scratchDir, 0o755); err != nil {
			t.Fatal(err)
		}

		// Create 1,001 untracked scratch files
		for i := 0; i < 1001; i++ {
			p := filepath.Join(scratchDir, fmt.Sprintf("bulk_%04d.tmp", i))
			if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		// EnforceScratchQuota with 0 (or DefaultScratchQuotaFiles) defaults to 1000
		err := EnforceScratchQuota(root, 0)
		if err == nil {
			t.Fatalf("expected quota exceeded error for 1001 files with default quota")
		}

		var quotaErr *ScratchQuotaError
		if !errors.As(err, &quotaErr) {
			t.Fatalf("expected *ScratchQuotaError, got: %T (%v)", err, err)
		}
		if quotaErr.Code != ErrCodeScratchQuotaExceeded {
			t.Errorf("expected Code %q, got %q", ErrCodeScratchQuotaExceeded, quotaErr.Code)
		}
		if quotaErr.Count != 1001 {
			t.Errorf("expected Count 1001, got %d", quotaErr.Count)
		}
		if quotaErr.Quota != DefaultScratchQuotaFiles {
			t.Errorf("expected Quota %d, got %d", DefaultScratchQuotaFiles, quotaErr.Quota)
		}
	})

	t.Run("git directory files are ignored in count", func(t *testing.T) {
		root := t.TempDir()
		scratchDir := filepath.Join(root, scratchNamespace)
		gitDir := filepath.Join(scratchDir, ".git", "objects")
		if err := os.MkdirAll(gitDir, 0o755); err != nil {
			t.Fatal(err)
		}

		// Write 10 files in .git (should be ignored)
		for i := 0; i < 10; i++ {
			p := filepath.Join(gitDir, fmt.Sprintf("obj_%d", i))
			if err := os.WriteFile(p, []byte("git-data"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		// Write 2 files in regular scratch
		for i := 0; i < 2; i++ {
			p := filepath.Join(scratchDir, fmt.Sprintf("valid_%d.txt", i))
			if err := os.WriteFile(p, []byte("scratch"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		// With quota = 5, total non-git files is 2 <= 5
		if err := EnforceScratchQuota(root, 5); err != nil {
			t.Fatalf("expected git files to be skipped, got error: %v", err)
		}
	})
}

func TestReapSessionScratch(t *testing.T) {
	t.Run("reaps direct and nested session scratch", func(t *testing.T) {
		root := t.TempDir()
		sessDir := filepath.Join(root, scratchNamespace, "sess-run-42", "nested")
		if err := os.MkdirAll(sessDir, 0o755); err != nil {
			t.Fatal(err)
		}
		sessFile := filepath.Join(sessDir, "artifact.log")
		if err := os.WriteFile(sessFile, []byte("session log\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		// Sibling / peer producer should be untouched
		peerDir := filepath.Join(root, scratchNamespace, "peer-producer")
		if err := os.MkdirAll(peerDir, 0o755); err != nil {
			t.Fatal(err)
		}
		peerFile := filepath.Join(peerDir, "retain.txt")
		if err := os.WriteFile(peerFile, []byte("peer data\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ReapSessionScratch(root, "sess-run-42"); err != nil {
			t.Fatalf("ReapSessionScratch: %v", err)
		}

		if _, err := os.Stat(filepath.Join(root, scratchNamespace, "sess-run-42")); !os.IsNotExist(err) {
			t.Fatalf("session scratch directory was not removed: %v", err)
		}

		// Peer file must still exist
		if _, err := os.Stat(peerFile); err != nil {
			t.Fatalf("peer scratch was unexpectedly removed: %v", err)
		}
	})

	t.Run("reaps session located under sessions subdirectory", func(t *testing.T) {
		root := t.TempDir()
		sessDir := filepath.Join(root, scratchNamespace, "sessions", "worker-99")
		if err := os.MkdirAll(sessDir, 0o755); err != nil {
			t.Fatal(err)
		}
		file := filepath.Join(sessDir, "state.json")
		if err := os.WriteFile(file, []byte(`{"status":"ok"}`), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ReapSessionScratch(root, "worker-99"); err != nil {
			t.Fatalf("ReapSessionScratch: %v", err)
		}

		if _, err := os.Stat(sessDir); !os.IsNotExist(err) {
			t.Fatalf("sessions/worker-99 directory still exists: %v", err)
		}
	})

	t.Run("idempotent when session scratch does not exist", func(t *testing.T) {
		root := t.TempDir()
		if err := ReapSessionScratch(root, "nonexistent-sess"); err != nil {
			t.Fatalf("expected nil for nonexistent session scratch, got: %v", err)
		}
	})

	t.Run("automated reap restores quota compliance", func(t *testing.T) {
		root := t.TempDir()
		scratchDir := filepath.Join(root, scratchNamespace)
		// 3 baseline files
		if err := os.MkdirAll(scratchDir, 0o755); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if err := os.WriteFile(filepath.Join(scratchDir, fmt.Sprintf("base_%d.txt", i)), []byte("base"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		// 3 ephemeral session files
		sessDir := filepath.Join(scratchDir, "ephemeral-session")
		if err := os.MkdirAll(sessDir, 0o755); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if err := os.WriteFile(filepath.Join(sessDir, fmt.Sprintf("temp_%d.txt", i)), []byte("tmp"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		// Total is 6 files; with quota = 5, this exceeds quota
		if err := EnforceScratchQuota(root, 5); err == nil {
			t.Fatalf("expected quota exceeded before session reap")
		}

		// Reap the session scratch
		if err := ReapSessionScratch(root, "ephemeral-session"); err != nil {
			t.Fatalf("ReapSessionScratch failed: %v", err)
		}

		// Now 3 files remain; quota 5 is satisfied
		if err := EnforceScratchQuota(root, 5); err != nil {
			t.Fatalf("expected quota to pass after session reap, got: %v", err)
		}
	})

	t.Run("refuses invalid or unsafe session IDs", func(t *testing.T) {
		root := t.TempDir()
		unsafeIDs := []string{
			"",
			"   ",
			".",
			"..",
			"../outside",
			`..\outside`,
			"_scratch",
			"sess*",
			"sess?",
			"[sess]",
			filepath.Join(root, "outside"),
		}
		for _, id := range unsafeIDs {
			if err := ReapSessionScratch(root, id); err == nil {
				t.Errorf("expected error for unsafe session ID %q, got nil", id)
			}
		}
	})
}
