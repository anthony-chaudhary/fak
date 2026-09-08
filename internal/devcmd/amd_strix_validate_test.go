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
	defer func() { runStrixValidationFn = origRun }()

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
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-subkernels", "none", "-ablate", "none"})
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
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-subkernels", "none", "-ablate", "none"})
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
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-subkernels", "none", "-ablate", "none"})
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
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-git-tip", baseCommit, "-subkernels", "argmax", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
	})
}

