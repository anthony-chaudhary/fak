package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func TestRunAMDStrixValidate_UnknownSelector(t *testing.T) {
	defer amdgpu.ClearPresenceCache()
	origStatus := gitStatusFn
	defer func() { gitStatusFn = origStatus }()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }

	// Seed presence cache with a reachable target to isolate subkernel selector validation
	simTarget := &amdgpu.StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix-devcmd",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		LockupTimeout:  -1,
		LatencyMS:      1.0,
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}
	amdgpu.SavePresenceCache(simTarget)

	t.Run("JSON mode exits 1 and emits failure receipt", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-host", "test-strix-devcmd",
			"-subkernels", "invalid_subkernel_selector",
			"-ablate", "none",
			"-committed-only",
			"-json",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d (stderr: %s)", code, stderr.String())
		}

		var receipt amdgpu.StrixValidationReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\nstdout: %s", err, stdout.String())
		}

		if receipt.Verdict != "FAIL" {
			t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
		}
		if receipt.Verified {
			t.Errorf("receipt.Verified = true, want false")
		}

		foundErr := false
		for _, f := range receipt.Failures {
			if strings.Contains(f, "unknown subkernel selector") && strings.Contains(f, "invalid_subkernel_selector") {
				foundErr = true
				break
			}
		}
		if !foundErr {
			t.Errorf("receipt.Failures missing unknown subkernel selector: %v", receipt.Failures)
		}
	})

	t.Run("text mode exits 1 and prints receipt failures", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-host", "test-strix-devcmd",
			"-subkernels", "invalid_subkernel_selector",
			"-ablate", "none",
			"-committed-only",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d (stderr: %s)", code, stderr.String())
		}

		outStr := stdout.String()
		if !strings.Contains(outStr, "Verdict:     FAIL") {
			t.Errorf("expected stdout to show 'Verdict:     FAIL', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Failures (") {
			t.Errorf("expected stdout to list Failures, got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "unknown subkernel selector") {
			t.Errorf("expected stdout to mention 'unknown subkernel selector', got:\n%s", outStr)
		}
	})
}

func TestRunAMDStrixValidate_RejectsMissingCandidateInputs(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-host", "test-strix-devcmd",
		"-subkernels", "none",
		"-ablate", "none",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite missing candidate archive or --mine input")
	}
	if !strings.Contains(stderr.String(), "no candidate archive or --mine overlay specified") {
		t.Errorf("expected stderr to mention missing candidate inputs, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_InvalidFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixValidate(&stdout, &stderr, []string{"-nonexistent-flag-xyz"})
	if code != 2 {
		t.Fatalf("expected exit code 2 on flag error, got %d", code)
	}
}

func TestRunAMDStrixValidate_RejectsMissingOrAbbreviatedGitTip(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	t.Run("abbreviated git tip is rejected", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-git-tip", "cefecef",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite abbreviated git tip")
		}
		if !strings.Contains(stderr.String(), "abbreviated") {
			t.Errorf("expected stderr to mention abbreviated Git tip, got: %s", stderr.String())
		}
	})

	t.Run("HEAD is rejected as non-immutable tip", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-git-tip", "HEAD",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite HEAD git tip")
		}
		if !strings.Contains(stderr.String(), "Git tip") {
			t.Errorf("expected stderr to mention Git tip error, got: %s", stderr.String())
		}
	})

	t.Run("unresolvable git checkout without explicit tip fails closed", func(t *testing.T) {
		runnerCalled = false
		gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
			return "", errors.New("git not found")
		}

		var stdout, stderr bytes.Buffer
		argv := []string{
			"-candidate-dir", t.TempDir(),
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite unresolvable git tip")
		}
		if !strings.Contains(stderr.String(), "Git tip") {
			t.Errorf("expected stderr to mention Git tip error, got: %s", stderr.String())
		}
	})
}

func TestRunAMDStrixValidate_RejectsMissingCandidateArchive(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "0123456789abcdef0123456789abcdef01234567",
		"-archive", filepath.Join(t.TempDir(), "nonexistent_candidate.json"),
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite missing candidate archive")
	}
	if !strings.Contains(stderr.String(), "missing candidate archive") {
		t.Errorf("expected stderr to mention missing candidate archive, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_RejectsPathTraversalInOverlay(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "0123456789abcdef0123456789abcdef01234567",
		"-overlay", "../escape_root.go",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite overlay path traversal")
	}
	if !strings.Contains(stderr.String(), "traversal") && !strings.Contains(stderr.String(), "escapes") {
		t.Errorf("expected stderr to mention traversal rejection, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_RejectsUnreadableOverlay(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "0123456789abcdef0123456789abcdef01234567",
		"-overlay", "unreadable_nonexistent_file_12345.go",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite unreadable overlay")
	}
	if !strings.Contains(stderr.String(), "overlay file unreadable or missing") {
		t.Errorf("expected stderr to mention unreadable overlay, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_RejectsArchiveDigestMismatch(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	tmpDir := t.TempDir()
	archiveFile := filepath.Join(tmpDir, "archive.json")
	if err := os.WriteFile(archiveFile, []byte(`{"schema":"fak.strix.candidate-archive/v1","base_commit":"0123456789abcdef0123456789abcdef01234567"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "0123456789abcdef0123456789abcdef01234567",
		"-archive", archiveFile,
		"-archive-digest", "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite archive digest mismatch")
	}
	if !strings.Contains(stderr.String(), "disagreement") && !strings.Contains(stderr.String(), "mismatch") {
		t.Errorf("expected stderr to mention digest disagreement, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_BindsCandidateArchiveToRunner(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	tmpDir := t.TempDir()
	overlayRel := "candidate_subkernel.go"
	overlayAbs := filepath.Join(tmpDir, overlayRel)
	if err := os.WriteFile(overlayAbs, []byte("// candidate optimization\npackage main\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	baseCommit := "a0123456789abcdef0123456789abcdef0123456"
	candArchive, err := BuildStrixCandidateArchiveFromPaths(baseCommit, tmpDir, []string{overlayRel})
	if err != nil {
		t.Fatalf("failed to build candidate archive: %v", err)
	}

	var capturedOpts amdgpu.StrixValidationOpts
	var capturedContextTip, capturedContextRef string
	runnerCalled := false

	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		capturedOpts = opts
		sb, _ := amdgpu.SourceBindingFromContext(ctx)
		capturedContextTip = sb.GitTip
		capturedContextRef = sb.GitRef

		target := amdgpu.StrixTarget{
			Mode:         "ssh",
			Host:         opts.Host,
			Reachable:    true,
			CPUModel:     "AMD Ryzen AI MAX+ 395",
			GPUName:      "AMD Radeon 8060S Graphics",
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
			DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
		}
		receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
		receipt.Verdict = "PASS"
		receipt.Verified = true
		receipt.ExecutedCount = 1
		receipt.SelectedCount = 1
		receipt.Subkernels = []amdgpu.StrixSubkernelResult{
			{
				Name:       "argmax",
				Status:     "PASS",
				DurationUS: 400,
				Parity: amdgpu.StrixParityVerdict{
					Passed:                true,
					LogitCosineSimilarity: 0.999999,
				},
			},
		}
		digest, _ := receipt.ComputeDigest()
		receipt.Digest = digest
		return receipt, nil
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-host", "test-strix-devcmd",
		"-git-tip", baseCommit,
		"-candidate-dir", tmpDir,
		"-overlay", overlayRel,
		"-subkernels", "argmax",
		"-ablate", "none",
		"-json",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
	}
	if !runnerCalled {
		t.Fatal("expected runner to be called")
	}

	expectedArchiveDigest := "sha256:" + candArchive.ArchiveSHA256
	if capturedOpts.GitTip != baseCommit {
		t.Errorf("capturedOpts.GitTip = %q, want %q", capturedOpts.GitTip, baseCommit)
	}
	if capturedOpts.GitRef != expectedArchiveDigest {
		t.Errorf("capturedOpts.GitRef = %q, want %q (archive digest must reach internal/amdgpu unchanged)", capturedOpts.GitRef, expectedArchiveDigest)
	}
	if !capturedOpts.RequireSourceBinding {
		t.Errorf("capturedOpts.RequireSourceBinding = false, want true")
	}
	if capturedContextTip != baseCommit {
		t.Errorf("context GitTip = %q, want %q", capturedContextTip, baseCommit)
	}
	if capturedContextRef != expectedArchiveDigest {
		t.Errorf("context GitRef = %q, want %q", capturedContextRef, expectedArchiveDigest)
	}

	var outReceipt amdgpu.StrixValidationReceipt
	if err := json.Unmarshal(stdout.Bytes(), &outReceipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nstdout: %s", err, stdout.String())
	}
	if outReceipt.Verdict != "PASS" || !outReceipt.Verified {
		t.Errorf("expected valid PASS receipt, got %+v", outReceipt)
	}
}

func TestRunAMDStrixValidate_HistoricalOrPartialReceiptFailsClosed(t *testing.T) {
	origRun := runStrixValidationFn
	origStatus := gitStatusFn
	defer func() {
		runStrixValidationFn = origRun
		gitStatusFn = origStatus
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }

	baseCommit := "b0123456789abcdef0123456789abcdef0123456"

	t.Run("receipt with verified=false fails closed", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = false // Unverified
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-committed-only", "-subkernels", "none", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "receipt validation failed") {
			t.Errorf("expected stderr to mention receipt validation failed, got: %s", stderr.String())
		}
	})

	t.Run("receipt with SKIPPED verdict fails closed", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "SKIPPED"
			receipt.Verified = true
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-committed-only", "-subkernels", "none", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
	})

	t.Run("historical v1 receipt with missing GitTip/GitRef provenance fails closed", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, "", "", opts.Command) // Historical v1 without source binding
			receipt.Verdict = "PASS"
			receipt.Verified = true
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-committed-only", "-subkernels", "none", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "historical or unbound receipt") {
			t.Errorf("expected stderr to mention historical or unbound receipt, got: %s", stderr.String())
		}
	})

	t.Run("receipt with failing subkernel fails closed", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:   "argmax",
					Status: "FAIL",
					Error:  "parity violation",
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
	})
}

func TestRunAMDStrixValidate_RejectsPositionalArguments(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-committed-only",
		"unexpected_positional_overlay.go",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite positional overlay arguments")
	}
	if !strings.Contains(stderr.String(), "positional") {
		t.Errorf("expected stderr to mention positional arguments rejected, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_AdmissionTimeoutValidation(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	t.Run("zero admission timeout is rejected", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-committed-only",
			"-admission-timeout", "0",
			"-timeout", "45",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite zero admission timeout")
		}
		if !strings.Contains(stderr.String(), "invalid admission timeout") {
			t.Errorf("expected stderr to mention invalid admission timeout, got: %s", stderr.String())
		}
	})

	t.Run("admission timeout exceeding total timeout is rejected", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-committed-only",
			"-admission-timeout", "50",
			"-timeout", "45",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite admission timeout >= total timeout")
		}
		if !strings.Contains(stderr.String(), "invalid admission timeout") {
			t.Errorf("expected stderr to mention invalid admission timeout, got: %s", stderr.String())
		}
	})
}

func TestRunAMDStrixValidate_RejectsDuplicateOverlayPaths(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()

	tmpDir := t.TempDir()
	overlayFile := filepath.Join(tmpDir, "duplicate_test.go")
	if err := os.WriteFile(overlayFile, []byte("// duplicate test\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return tmpDir, nil
	}

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "a0123456789abcdef0123456789abcdef0123456",
		"-candidate-dir", tmpDir,
		"-mine", "duplicate_test.go",
		"-mine", "./duplicate_test.go",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite duplicate overlay paths")
	}
	if !strings.Contains(stderr.String(), "duplicate overlay path") {
		t.Errorf("expected stderr to mention duplicate overlay path, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_RejectsAbsoluteOverlayPaths(t *testing.T) {
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "a0123456789abcdef0123456789abcdef0123456",
		"-mine", "/etc/passwd",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite absolute overlay path")
	}
	if !strings.Contains(stderr.String(), "absolute overlay path") {
		t.Errorf("expected stderr to mention absolute overlay path, got: %s", stderr.String())
	}
}

type mockSymlinkFileInfo struct {
	os.FileInfo
}

func (m mockSymlinkFileInfo) Mode() os.FileMode {
	return os.ModeSymlink
}

func (m mockSymlinkFileInfo) IsDir() bool {
	return false
}

func TestRunAMDStrixValidate_RejectsSymlinkOverlay(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	origLstat := osLstatFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
		osLstatFn = origLstat
	}()

	tmpDir := t.TempDir()
	targetFile := filepath.Join(tmpDir, "target.go")
	if err := os.WriteFile(targetFile, []byte("// target\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return tmpDir, nil
	}
	osLstatFn = func(name string) (os.FileInfo, error) {
		if strings.HasSuffix(filepath.ToSlash(name), "symlink.go") {
			fi, err := os.Lstat(targetFile)
			if err != nil {
				return nil, err
			}
			return mockSymlinkFileInfo{fi}, nil
		}
		return os.Lstat(name)
	}

	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		return nil, errors.New("runner should not have been called")
	}

	var stdout, stderr bytes.Buffer
	argv := []string{
		"-git-tip", "a0123456789abcdef0123456789abcdef0123456",
		"-candidate-dir", tmpDir,
		"-mine", "symlink.go",
	}

	code := RunAMDStrixValidate(&stdout, &stderr, argv)
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}
	if runnerCalled {
		t.Fatal("runner was called despite symlink overlay")
	}
	if !strings.Contains(stderr.String(), "symlink overlay rejected") {
		t.Errorf("expected stderr to mention symlink overlay rejected, got: %s", stderr.String())
	}
}

func TestRunAMDStrixValidate_CommittedOnlyMode(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	origStatus := gitStatusFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
		gitStatusFn = origStatus
	}()

	tmpDir := t.TempDir()
	baseCommit := "c0123456789abcdef0123456789abcdef0123456"

	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return tmpDir, nil
	}

	t.Run("dirty worktree fails closed in committed-only mode", func(t *testing.T) {
		gitStatusFn = func(ctx context.Context, dir string) (string, error) {
			return " M modified_file.go\n?? untracked.go", nil
		}
		runnerCalled := false
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			return nil, errors.New("runner should not have been called")
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-git-tip", baseCommit,
			"-candidate-dir", tmpDir,
			"-committed-only",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called on dirty worktree in committed-only mode")
		}
		if !strings.Contains(stderr.String(), "worktree is dirty") {
			t.Errorf("expected stderr to mention worktree is dirty, got: %s", stderr.String())
		}
	})

	t.Run("clean worktree builds empty overlay archive and calls runner", func(t *testing.T) {
		gitStatusFn = func(ctx context.Context, dir string) (string, error) {
			return "", nil
		}
		runnerCalled := false
		var capturedOpts amdgpu.StrixValidationOpts
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			capturedOpts = opts
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 350,
					Parity: amdgpu.StrixParityVerdict{
						Passed:                true,
						LogitCosineSimilarity: 0.999999,
					},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-git-tip", baseCommit,
			"-candidate-dir", tmpDir,
			"-committed-only",
			"-subkernels", "argmax",
			"-ablate", "none",
		})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
		}
		if !runnerCalled {
			t.Fatal("expected runner to be called")
		}
		if capturedOpts.GitTip != baseCommit {
			t.Errorf("GitTip = %q, want %q", capturedOpts.GitTip, baseCommit)
		}
		if !strings.HasPrefix(capturedOpts.GitRef, "sha256:") {
			t.Errorf("GitRef = %q, expected sha256: prefix", capturedOpts.GitRef)
		}
	})

	t.Run("both --mine and --committed-only is rejected", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-git-tip", baseCommit,
			"-candidate-dir", tmpDir,
			"-committed-only",
			"-mine", "some_path.go",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "conflicting options") {
			t.Errorf("expected stderr to mention conflicting options, got: %s", stderr.String())
		}
	})
}

func TestRunAMDStrixValidate_HumanOutput_RendersHistoricalNonCredit(t *testing.T) {
	origRun := runStrixValidationFn
	origStatus := gitStatusFn
	defer func() {
		runStrixValidationFn = origRun
		gitStatusFn = origStatus
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }

	baseCommit := "d0123456789abcdef0123456789abcdef0123456"

	t.Run("unverified PASS receipt renders historical/non-credit in human mode", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = false // Unverified
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-git-tip", baseCommit,
			"-committed-only",
			"-subkernels", "none",
			"-ablate", "none",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		outStr := stdout.String()
		if strings.Contains(outStr, "Verdict:     PASS\n") {
			t.Errorf("expected stdout NOT to render current 'Verdict:     PASS', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "PASS (historical/non-credit)") {
			t.Errorf("expected stdout to render 'PASS (historical/non-credit)', got:\n%s", outStr)
		}
	})

	t.Run("valid PASS receipt renders current PASS in human mode", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			target := amdgpu.StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    true,
				CPUModel:     "AMD Ryzen AI MAX+ 395",
				GPUName:      "AMD Radeon 8060S Graphics",
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			}
			receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 400,
					Parity: amdgpu.StrixParityVerdict{
						Passed:                true,
						LogitCosineSimilarity: 0.999999,
					},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-git-tip", baseCommit,
			"-committed-only",
			"-subkernels", "argmax",
			"-ablate", "none",
		})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
		}
		outStr := stdout.String()
		if !strings.Contains(outStr, "Verdict:     PASS\n") {
			t.Errorf("expected stdout to render 'Verdict:     PASS', got:\n%s", outStr)
		}
		if strings.Contains(outStr, "historical/non-credit") {
			t.Errorf("expected stdout not to mention historical/non-credit for valid PASS, got:\n%s", outStr)
		}
	})
}

func TestRunAMDStrixValidate_CanonicalRootAndBaseResolution(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	origStatus := gitStatusFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
		gitStatusFn = origStatus
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }

	mockRoot := "/mock/repo/root"
	mockCommit := "e0123456789abcdef0123456789abcdef0123456"

	revParseCalls := make([]string, 0)
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		call := strings.Join(append([]string{dir}, args...), " ")
		revParseCalls = append(revParseCalls, call)
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return mockRoot, nil
		}
		if len(args) > 1 && args[0] == "--verify" && args[1] == "HEAD^{commit}" {
			return mockCommit, nil
		}
		return "", errors.New("unrecognized rev-parse invocation")
	}

	var capturedOpts amdgpu.StrixValidationOpts
	runnerCalled := false
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		capturedOpts = opts
		target := amdgpu.StrixTarget{
			Mode:         "ssh",
			Host:         opts.Host,
			Reachable:    true,
			CPUModel:     "AMD Ryzen AI MAX+ 395",
			GPUName:      "AMD Radeon 8060S Graphics",
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
			DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
		}
		receipt := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
		receipt.Verdict = "PASS"
		receipt.Verified = true
		receipt.ExecutedCount = 1
		receipt.Subkernels = []amdgpu.StrixSubkernelResult{
			{
				Name:       "argmax",
				Status:     "PASS",
				DurationUS: 400,
				Parity: amdgpu.StrixParityVerdict{
					Passed:                true,
					LogitCosineSimilarity: 0.999999,
				},
			},
		}
		digest, _ := receipt.ComputeDigest()
		receipt.Digest = digest
		return receipt, nil
	}

	var stdout, stderr bytes.Buffer
	code := RunAMDStrixValidate(&stdout, &stderr, []string{
		"-committed-only",
		"-subkernels", "argmax",
		"-ablate", "none",
	})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
	}
	if !runnerCalled {
		t.Fatal("expected runner to be called")
	}
	if capturedOpts.GitTip != mockCommit {
		t.Errorf("capturedOpts.GitTip = %q, want %q", capturedOpts.GitTip, mockCommit)
	}

	foundShowToplevel := false
	foundVerifyCommit := false
	for _, call := range revParseCalls {
		if strings.Contains(call, "--show-toplevel") {
			foundShowToplevel = true
		}
		if strings.Contains(call, "--verify") && strings.Contains(call, "HEAD^{commit}") {
			foundVerifyCommit = true
		}
	}
	if !foundShowToplevel {
		t.Errorf("expected git rev-parse --show-toplevel to be invoked, calls: %v", revParseCalls)
	}
	if !foundVerifyCommit {
		t.Errorf("expected git rev-parse --verify HEAD^{commit} to be invoked, calls: %v", revParseCalls)
	}
}

