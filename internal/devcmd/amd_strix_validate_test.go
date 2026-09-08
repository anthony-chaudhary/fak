package devcmd

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func createTestValidReceipt(target amdgpu.StrixTarget, gitRef, gitTip, command string) *amdgpu.StrixValidationReceipt {
	if target.GPUName == "" {
		target.GPUName = "AMD Radeon 8060S Graphics"
	}
	if target.TargetISA == "" {
		target.TargetISA = "gfx1151"
	}
	target.Reachable = true
	if strings.TrimSpace(gitTip) == "" {
		gitTip = "0123456789abcdef0123456789abcdef01234567"
	}
	testHash := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	effectiveRef := gitRef
	if strings.TrimSpace(effectiveRef) == "" || !strings.HasPrefix(effectiveRef, "sha256:") || len(effectiveRef) != 71 {
		effectiveRef = testHash
	}
	if strings.TrimSpace(command) == "" {
		command = "fak-dev amd-strix-validate"
	}

	r := amdgpu.NewStrixValidationReceipt(target, gitRef, gitTip, command)
	r.Verdict = "PASS"
	r.Verified = true
	r.Provenance.SourceArchiveSHA256 = effectiveRef
	r.Provenance.BinarySHA256 = testHash
	r.Provenance.ShaderBundleSHA256 = testHash
	r.Provenance.BuildCommandSHA256 = testHash
	r.Provenance.EngineIdentity = "fak-native/vulkan"
	r.Provenance.CleanupObserved = true
	r.SelectedCount = 1
	r.ExecutedCount = 1
	r.SelectedSubkernels = 1
	r.ExecutedSubkernels = 1

	exit := 0
	deviceIdentity := target.GPUName + "|" + target.TargetISA
	evidence := amdgpu.StrixExecutionEvidence{
		SourceArchiveSHA256: effectiveRef,
		BinarySHA256:        testHash,
		ShaderBundleSHA256:  testHash,
		CommandSHA256:       testHash,
		DeviceIdentity:      deviceIdentity,
		EngineIdentity:      "fak-native/vulkan",
		ArtifactRehashed:    true,
		DeviceTimeoutMS:     60000,
		LeasePathSHA256:     testHash,
		AdmissionWaitMS:     30000,
		Acquired:            true,
		Released:            true,
		AcquireOrdinal:      1,
		ReleaseOrdinal:      2,
		ExitCode:            &exit,
		RawOutputSHA256:     testHash,
		RawOutputBytes:      1,
	}

	r.Subkernels = []amdgpu.StrixSubkernelResult{
		{
			Name:         "argmax",
			Status:       "PASS",
			DurationUS:   350,
			Iterations:   1,
			Parity:       amdgpu.StrixParityVerdict{Passed: true, ArgmaxExact: true},
			ParityEvents: []amdgpu.StrixParityEvent{amdgpu.NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)},
			Evidence:     evidence,
		},
	}
	var b strings.Builder
	for _, s := range r.Subkernels {
		fmt.Fprintf(&b, "subkernel:%s:%s\n", s.Name, s.Evidence.CommandSHA256)
	}
	h := sha256.Sum256([]byte(b.String()))
	r.Provenance.ExecutionManifestSHA256 = "sha256:" + hex.EncodeToString(h[:])
	r.Digest, _ = r.ComputeDigest()
	return r
}

func stubTestCandidateArchive(t *testing.T, expectedTip string) (string, []byte) {
	t.Helper()
	origBuild := buildStrixCandidateArchiveFn
	testBytes := []byte("candidate-test-archive-tar-bytes")
	h := sha256.Sum256(testBytes)
	testDigest := "sha256:" + hex.EncodeToString(h[:])
	buildStrixCandidateArchiveFn = func(ctx context.Context, root, tip string, owned []string) (amdgpu.StrixCandidateArchive, error) {
		if expectedTip != "" && !strings.EqualFold(tip, expectedTip) {
			return amdgpu.StrixCandidateArchive{}, fmt.Errorf("tip mismatch: got %s, want %s", tip, expectedTip)
		}
		return amdgpu.StrixCandidateArchive{
			Bytes:               testBytes,
			SourceArchiveSHA256: testDigest,
		}, nil
	}
	t.Cleanup(func() { buildStrixCandidateArchiveFn = origBuild })
	return testDigest, testBytes
}

func createTestPrebuiltTar(baseCommit string, files map[string][]byte) ([]byte, string) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	if baseCommit != "" {
		baseData := []byte(baseCommit + "\n")
		_ = tw.WriteHeader(&tar.Header{
			Name:     ".strix-base-commit",
			Mode:     0o644,
			Size:     int64(len(baseData)),
			Typeflag: tar.TypeReg,
		})
		_, _ = tw.Write(baseData)
	}
	names := make([]string, 0, len(files))
	for n := range files {
		names = append(names, n)
	}
	for _, n := range names {
		data := files[n]
		_ = tw.WriteHeader(&tar.Header{
			Name:     n,
			Mode:     0o644,
			Size:     int64(len(data)),
			Typeflag: tar.TypeReg,
		})
		_, _ = tw.Write(data)
	}
	_ = tw.Close()
	raw := buf.Bytes()
	h := sha256.Sum256(raw)
	return raw, hex.EncodeToString(h[:])
}

func TestRunAMDStrixValidate_UnknownSelector(t *testing.T) {
	defer amdgpu.ClearPresenceCache()
	origStatus := gitStatusFn
	origGit := gitRevParseFn
	defer func() {
		gitStatusFn = origStatus
		gitRevParseFn = origGit
	}()
	stubTestCandidateArchive(t, "0123456789abcdef0123456789abcdef01234567")
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return "/mock/repo", nil
		}
		return "0123456789abcdef0123456789abcdef01234567", nil
	}

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

	t.Run("nonexistent candidate archive file", func(t *testing.T) {
		runnerCalled = false
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
	})

	t.Run("empty candidate archive file", func(t *testing.T) {
		runnerCalled = false
		emptyFile := filepath.Join(t.TempDir(), "empty.tar")
		if err := os.WriteFile(emptyFile, []byte{}, 0o644); err != nil {
			t.Fatal(err)
		}
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-git-tip", "0123456789abcdef0123456789abcdef01234567",
			"-archive", emptyFile,
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite empty candidate archive")
		}
		if !strings.Contains(stderr.String(), "is empty") {
			t.Errorf("expected stderr to mention archive is empty, got: %s", stderr.String())
		}
	})
}

func TestRunAMDStrixValidate_RejectsPathTraversalInOverlay(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return "/mock/repo", nil
	}

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
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return "/mock/repo", nil
	}

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
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()

	tmpDir := t.TempDir()
	baseCommit := "a0123456789abcdef0123456789abcdef0123456"
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return tmpDir, nil
		}
		return baseCommit, nil
	}

	overlayRel := "candidate_subkernel.go"
	overlayAbs := filepath.Join(tmpDir, overlayRel)
	if err := os.WriteFile(overlayAbs, []byte("// candidate optimization\npackage main\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	stubTestCandidateArchive(t, baseCommit)

	candArchive, err := BuildStrixCandidateArchiveFromPaths(baseCommit, tmpDir, []string{overlayRel})
	if err != nil {
		t.Fatalf("failed to build candidate archive: %v", err)
	}

	var capturedOpts amdgpu.StrixValidationOpts
	var capturedContextTip, capturedContextRef string
	var capturedCandArchive *StrixCandidateArchive
	var capturedAdmissionTimeout time.Duration
	runnerCalled := false

	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runnerCalled = true
		capturedOpts = opts
		sb, _ := amdgpu.SourceBindingFromContext(ctx)
		capturedContextTip = sb.GitTip
		capturedContextRef = sb.GitRef
		if cand, ok := CandidateArchiveFromContext(ctx); ok {
			capturedCandArchive = cand
		}
		if adm, ok := AdmissionTimeoutFromContext(ctx); ok {
			capturedAdmissionTimeout = adm
		}

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
		receipt := createTestValidReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
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
		"-admission-timeout", "15",
		"-timeout", "60",
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
	if string(capturedOpts.CandidateArchive) != string(candArchive.ArchiveBytes) {
		t.Errorf("capturedOpts.CandidateArchive does not match archive bytes")
	}
	if capturedOpts.SourceArchiveSHA256 != expectedArchiveDigest {
		t.Errorf("capturedOpts.SourceArchiveSHA256 = %q, want %q", capturedOpts.SourceArchiveSHA256, expectedArchiveDigest)
	}
	if capturedOpts.AdmissionTimeout != 15*time.Second {
		t.Errorf("capturedOpts.AdmissionTimeout = %v, want 15s", capturedOpts.AdmissionTimeout)
	}
	if capturedContextTip != baseCommit {
		t.Errorf("context GitTip = %q, want %q", capturedContextTip, baseCommit)
	}
	if capturedContextRef != expectedArchiveDigest {
		t.Errorf("context GitRef = %q, want %q", capturedContextRef, expectedArchiveDigest)
	}
	if capturedCandArchive == nil {
		t.Fatal("candidate archive was not propagated in context")
	}
	if capturedCandArchive.ArchiveSHA256 != candArchive.ArchiveSHA256 {
		t.Errorf("capturedCandArchive.ArchiveSHA256 = %q, want %q", capturedCandArchive.ArchiveSHA256, candArchive.ArchiveSHA256)
	}
	if len(capturedCandArchive.ArchiveBytes) == 0 {
		t.Errorf("capturedCandArchive.ArchiveBytes is empty")
	}
	if capturedAdmissionTimeout != 15*time.Second {
		t.Errorf("capturedAdmissionTimeout = %v, want 15s", capturedAdmissionTimeout)
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
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitStatusFn = origStatus
		gitRevParseFn = origGit
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }
	baseCommit := "b0123456789abcdef0123456789abcdef0123456"
	stubTestCandidateArchive(t, baseCommit)
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return "/mock/repo", nil
		}
		return baseCommit, nil
	}

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

	t.Run("zero or negative command timeout is rejected", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-committed-only",
			"-admission-timeout", "5",
			"-timeout", "0",
		})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite zero command timeout")
		}
		if !strings.Contains(stderr.String(), "invalid command timeout") {
			t.Errorf("expected stderr to mention invalid command timeout, got: %s", stderr.String())
		}
	})

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
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		return "/mock/repo", nil
	}

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
		stubTestCandidateArchive(t, baseCommit)
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
			receipt := createTestValidReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
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
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitStatusFn = origStatus
		gitRevParseFn = origGit
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }
	baseCommit := "d0123456789abcdef0123456789abcdef0123456"
	stubTestCandidateArchive(t, baseCommit)
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return "/mock/repo", nil
		}
		return baseCommit, nil
	}

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
			receipt := createTestValidReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
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
	stubTestCandidateArchive(t, mockCommit)

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
		receipt := createTestValidReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
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

func TestRunAMDStrixValidate_RejectsConflictingArchiveAndOverlays(t *testing.T) {
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

	t.Run("archive and mine overlay conflict", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-git-tip", "0123456789abcdef0123456789abcdef01234567",
			"-archive", archiveFile,
			"-mine", "some_file.go",
		}
		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite conflicting options")
		}
		if !strings.Contains(stderr.String(), "conflicting options") {
			t.Errorf("expected stderr to mention conflicting options, got: %s", stderr.String())
		}
	})

	t.Run("archive and committed-only conflict", func(t *testing.T) {
		runnerCalled = false
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-git-tip", "0123456789abcdef0123456789abcdef01234567",
			"-archive", archiveFile,
			"-committed-only",
		}
		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite conflicting options")
		}
		if !strings.Contains(stderr.String(), "conflicting options") {
			t.Errorf("expected stderr to mention conflicting options, got: %s", stderr.String())
		}
	})
}

func TestBuildStrixCandidateArchive_DeterministicTarFormat(t *testing.T) {
	baseCommit := "0123456789abcdef0123456789abcdef01234567"

	t.Run("rejects invalid or abbreviated base git tip", func(t *testing.T) {
		_, err := BuildStrixCandidateArchiveFromPaths("short", ".", []string{"foo.go"})
		if err == nil || !strings.Contains(err.Error(), "40-hex") {
			t.Fatalf("expected error mentioning 40-hex, got: %v", err)
		}
	})

	t.Run("delegates to canonical helper and preserves archive bytes and digest", func(t *testing.T) {
		origBuild := buildStrixCandidateArchiveFn
		defer func() { buildStrixCandidateArchiveFn = origBuild }()
		fakeTar := []byte("tar-bytes-canonical-stream")
		h := sha256.Sum256(fakeTar)
		fakeDigest := "sha256:" + hex.EncodeToString(h[:])
		calledWithRoot, calledWithTip := "", ""
		var calledWithPaths []string
		buildStrixCandidateArchiveFn = func(ctx context.Context, root, tip string, paths []string) (amdgpu.StrixCandidateArchive, error) {
			calledWithRoot = root
			calledWithTip = tip
			calledWithPaths = paths
			return amdgpu.StrixCandidateArchive{
				Bytes:               fakeTar,
				SourceArchiveSHA256: fakeDigest,
			}, nil
		}

		res, err := BuildStrixCandidateArchiveFromPaths(baseCommit, "/some/root", []string{"pkg/foo.go", "internal/bar.go"})
		if err != nil {
			t.Fatalf("BuildStrixCandidateArchiveFromPaths failed: %v", err)
		}
		if calledWithRoot != "/some/root" || calledWithTip != baseCommit || len(calledWithPaths) != 2 {
			t.Errorf("delegation arguments mismatched: root=%q, tip=%q, paths=%v", calledWithRoot, calledWithTip, calledWithPaths)
		}
		if !bytes.Equal(res.ArchiveBytes, fakeTar) {
			t.Errorf("archive bytes mismatch")
		}
		if res.ArchiveSHA256 != strings.TrimPrefix(fakeDigest, "sha256:") {
			t.Errorf("archive sha256 = %q, want %q", res.ArchiveSHA256, strings.TrimPrefix(fakeDigest, "sha256:"))
		}
	})

	t.Run("produces deterministic tar archive on real repository", func(t *testing.T) {
		tempRepo := t.TempDir()
		runGit := func(args ...string) string {
			cmd := exec.Command("git", args...)
			cmd.Dir = tempRepo
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("git %v: %v: %s", args, err, out)
			}
			return strings.TrimSpace(string(out))
		}
		runGit("init")
		runGit("config", "user.email", "test@example.invalid")
		runGit("config", "user.name", "test")
		if err := os.WriteFile(filepath.Join(tempRepo, "base.txt"), []byte("base-content\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runGit("add", "base.txt")
		runGit("commit", "-m", "initial commit")
		tempTip := runGit("rev-parse", "HEAD")

		if err := os.WriteFile(filepath.Join(tempRepo, "overlay.txt"), []byte("overlay-content\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		origBuild := buildStrixCandidateArchiveFn
		defer func() { buildStrixCandidateArchiveFn = origBuild }()
		buildStrixCandidateArchiveFn = amdgpu.BuildStrixCandidateArchive

		a, err := BuildStrixCandidateArchiveFromPaths(tempTip, tempRepo, []string{"overlay.txt"})
		if err != nil {
			t.Fatalf("first archive build failed: %v", err)
		}
		b, err := BuildStrixCandidateArchiveFromPaths(tempTip, tempRepo, []string{"overlay.txt"})
		if err != nil {
			t.Fatalf("second archive build failed: %v", err)
		}
		if a.ArchiveSHA256 != b.ArchiveSHA256 {
			t.Fatalf("archive sha256 non-deterministic: %s != %s", a.ArchiveSHA256, b.ArchiveSHA256)
		}
		if !bytes.Equal(a.ArchiveBytes, b.ArchiveBytes) {
			t.Fatalf("archive bytes non-deterministic")
		}
		if len(a.ArchiveSHA256) != 64 {
			t.Errorf("unexpected archive digest length: %d", len(a.ArchiveSHA256))
		}
	})
}

func TestRunAMDStrixValidate_CurrentV2ValidationInvariants(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	origStatus := gitStatusFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
		gitStatusFn = origStatus
	}()
	gitStatusFn = func(ctx context.Context, dir string) (string, error) { return "", nil }

	baseCommit := "f0123456789abcdef0123456789abcdef0123456"
	stubTestCandidateArchive(t, baseCommit)
	gitRevParseFn = func(ctx context.Context, dir string, args ...string) (string, error) {
		if len(args) > 0 && args[0] == "--show-toplevel" {
			return "/mock/repo", nil
		}
		return baseCommit, nil
	}

	validTarget := amdgpu.StrixTarget{
		Mode:         "ssh",
		Host:         "strix1",
		Reachable:    true,
		CPUModel:     "AMD Ryzen AI MAX+ 395",
		GPUName:      "AMD Radeon 8060S Graphics",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}

	t.Run("missing GitRef archive digest fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, "", opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 300,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, LogitCosineSimilarity: 0.999999},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		// JSON mode
		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "GitRef (archive digest) is empty") {
			t.Errorf("expected stderr to mention empty GitRef, got: %s", stderrJSON.String())
		}

		// Human mode
		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("mismatched GitRef archive digest fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, "sha256:0000000000000000000000000000000000000000000000000000000000000000", opts.GitTip, opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 300,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, LogitCosineSimilarity: 0.999999},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "does not match expected archive digest") {
			t.Errorf("expected stderr to mention digest mismatch, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("missing or mismatched GitTip fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, "", opts.Command)
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 300,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, LogitCosineSimilarity: 0.999999},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("historical v1 schema is rejected for current PASS", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Schema = amdgpu.StrixValidationSchemaV1 // v1 schema
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 300,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, LogitCosineSimilarity: 0.999999},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "historical or non-credit schema") {
			t.Errorf("expected stderr to mention historical schema, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("zero executed subkernels and zero ablations cannot PASS", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = opts.GitRef
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 0
			receipt.Subkernels = nil
			receipt.Ablations = nil
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "missing execution evidence") {
			t.Errorf("expected stderr to mention missing execution evidence, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("valid v2 receipt with passing subkernel succeeds in both JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			return receipt, nil
		}

		// JSON mode
		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 0 {
			t.Fatalf("expected JSON exit code 0, got %d (stderr: %s)", codeJSON, stderrJSON.String())
		}
		var parsed amdgpu.StrixValidationReceipt
		if err := json.Unmarshal(stdoutJSON.Bytes(), &parsed); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v", err)
		}
		if parsed.Verdict != "PASS" || !parsed.Verified {
			t.Errorf("parsed verdict = %s, verified = %v, want PASS/true", parsed.Verdict, parsed.Verified)
		}

		// Human mode
		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 0 {
			t.Fatalf("expected human exit code 0, got %d (stderr: %s)", codeHuman, stderrHuman.String())
		}
		if !strings.Contains(stdoutHuman.String(), "Verdict:     PASS\n") {
			t.Errorf("expected human stdout to show 'Verdict:     PASS', got:\n%s", stdoutHuman.String())
		}
		if strings.Contains(stdoutHuman.String(), "historical/non-credit") {
			t.Errorf("expected human stdout not to mention historical/non-credit for valid PASS, got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("empty --mine overlay path is rejected", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-mine", "", "-subkernels", "argmax", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "empty --mine overlay path rejected") {
			t.Errorf("expected stderr to mention empty --mine overlay path, got: %s", stderr.String())
		}
	})

	t.Run("invalid or regressing ablation arm fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = opts.GitRef
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.Ablations = []amdgpu.StrixAblationResult{
				{
					Feature:      "test_ablation",
					Verdict:      "REGRESSION",
					Speedup:      0.8,
					BaselineArm:  amdgpu.StrixArmResult{Name: "base", LatencyUS: 100},
					CandidateArm: amdgpu.StrixArmResult{Name: "cand", LatencyUS: 125},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "none", "-ablate", "all"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "suffered regression") {
			t.Errorf("expected stderr to mention regression, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "none", "-ablate", "all"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("nil receipt in JSON mode carries candidate archive digest", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			return nil, errors.New("runner exploded")
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		var parsed amdgpu.StrixValidationReceipt
		if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v", err)
		}
		if parsed.Verdict != "FAIL" {
			t.Errorf("verdict = %s, want FAIL", parsed.Verdict)
		}
		if !strings.HasPrefix(parsed.Provenance.GitRef, "sha256:") {
			t.Errorf("expected GitRef to carry sha256 archive digest, got %q", parsed.Provenance.GitRef)
		}
		if parsed.Provenance.GitTip != baseCommit {
			t.Errorf("expected GitTip = %q, got %q", baseCommit, parsed.Provenance.GitTip)
		}
	})

	t.Run("receipt invariant validation failure fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = opts.GitRef
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 350,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, ArgmaxExact: true},
				},
			}
			receipt.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000" // Tampered receipt digest
			return receipt, nil
		}

		// JSON mode
		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "receipt invariant validation failed") {
			t.Errorf("expected stderr to mention invariant validation failure, got: %s", stderrJSON.String())
		}

		// Human mode
		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("receipt with failing parity event fails invariant validation", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := amdgpu.NewStrixValidationReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = opts.GitRef
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 350,
					Parity:     amdgpu.StrixParityVerdict{Passed: false, ArgmaxExact: false}, // failing parity
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "credit eligible") && !strings.Contains(stderrJSON.String(), "parity") && !strings.Contains(stderrJSON.String(), "receipt invariant validation failed") {
			t.Errorf("expected stderr to mention credit eligibility or parity failure, got: %s", stderrJSON.String())
		}
	})

	t.Run("receipt with non-Strix target fails invariant validation", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			nonStrixTarget := validTarget
			nonStrixTarget.GPUName = "NVIDIA GeForce RTX 4090"
			nonStrixTarget.TargetISA = "sm_89"
			receipt := amdgpu.NewStrixValidationReceipt(nonStrixTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = opts.GitRef
			receipt.Verdict = "PASS"
			receipt.Verified = true
			receipt.ExecutedCount = 1
			receipt.Subkernels = []amdgpu.StrixSubkernelResult{
				{
					Name:       "argmax",
					Status:     "PASS",
					DurationUS: 350,
					Parity:     amdgpu.StrixParityVerdict{Passed: true, ArgmaxExact: true},
				},
			}
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "receipt invariant validation failed") {
			t.Errorf("expected stderr to mention invariant validation failure, got: %s", stderrJSON.String())
		}
	})

	t.Run("receipt with missing Provenance.SourceArchiveSHA256 fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = "" // Missing source archive hash
			receipt.Digest, _ = receipt.ComputeDigest()
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "SourceArchiveSHA256 is empty") && !strings.Contains(stderrJSON.String(), "receipt invariant validation failed") && !strings.Contains(stderrJSON.String(), "incomplete immutable source") {
			t.Errorf("expected stderr to mention SourceArchiveSHA256 is empty or invariant validation failed, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("receipt with mismatched Provenance.SourceArchiveSHA256 fails in JSON and human modes", func(t *testing.T) {
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			receipt.Provenance.SourceArchiveSHA256 = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
			receipt.Digest, _ = receipt.ComputeDigest()
			return receipt, nil
		}

		var stdoutJSON, stderrJSON bytes.Buffer
		codeJSON := RunAMDStrixValidate(&stdoutJSON, &stderrJSON, []string{"-committed-only", "-json", "-subkernels", "argmax", "-ablate", "none"})
		if codeJSON != 1 {
			t.Fatalf("expected JSON exit code 1, got %d", codeJSON)
		}
		if !strings.Contains(stderrJSON.String(), "SourceArchiveSHA256") || !strings.Contains(stderrJSON.String(), "does not match") {
			t.Errorf("expected stderr to mention SourceArchiveSHA256 mismatch, got: %s", stderrJSON.String())
		}

		var stdoutHuman, stderrHuman bytes.Buffer
		codeHuman := RunAMDStrixValidate(&stdoutHuman, &stderrHuman, []string{"-committed-only", "-subkernels", "argmax", "-ablate", "none"})
		if codeHuman != 1 {
			t.Fatalf("expected human exit code 1, got %d", codeHuman)
		}
		if !strings.Contains(stdoutHuman.String(), "PASS (historical/non-credit)") {
			t.Errorf("expected human stdout to render PASS (historical/non-credit), got:\n%s", stdoutHuman.String())
		}
	})

	t.Run("runner receives CandidateArchive bytes and AdmissionTimeout from devcmd", func(t *testing.T) {
		var capturedOpts amdgpu.StrixValidationOpts
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			capturedOpts = opts
			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		code := RunAMDStrixValidate(&stdout, &stderr, []string{
			"-committed-only",
			"-subkernels", "argmax",
			"-ablate", "none",
			"-admission-timeout", "12",
			"-timeout", "50",
			"-json",
		})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
		}
		if len(capturedOpts.CandidateArchive) == 0 {
			t.Errorf("expected non-empty CandidateArchive in opts")
		}
		if capturedOpts.AdmissionTimeout != 12*time.Second {
			t.Errorf("captured AdmissionTimeout = %v, want 12s", capturedOpts.AdmissionTimeout)
		}
		if !strings.HasPrefix(capturedOpts.SourceArchiveSHA256, "sha256:") {
			t.Errorf("captured SourceArchiveSHA256 = %q, want sha256: prefix", capturedOpts.SourceArchiveSHA256)
		}
	})
}

func TestRunAMDStrixValidate_PrebuiltTarArchiveBinding(t *testing.T) {
	origRun := runStrixValidationFn
	origGit := gitRevParseFn
	defer func() {
		runStrixValidationFn = origRun
		gitRevParseFn = origGit
	}()

	baseCommit := "c0123456789abcdef0123456789abcdef0123456"
	overlayFiles := map[string][]byte{
		"internal/amdgpu/strix_kernel.go": []byte("package amdgpu\n// candidate kernel\n"),
		"internal/devcmd/test_overlay.go": []byte("package devcmd\n// overlay\n"),
	}

	tarBytes, tarSHA256 := createTestPrebuiltTar(baseCommit, overlayFiles)
	candArchive := &StrixCandidateArchive{
		BaseCommit:    baseCommit,
		ArchiveBytes:  tarBytes,
		ArchiveSHA256: tarSHA256,
	}

	tmpDir := t.TempDir()
	archiveTarPath := filepath.Join(tmpDir, "strix_candidate.tar")
	if err := os.WriteFile(archiveTarPath, candArchive.ArchiveBytes, 0o644); err != nil {
		t.Fatalf("failed to write candidate tar: %v", err)
	}

	validTarget := amdgpu.StrixTarget{
		Mode:         "ssh",
		Host:         "strix1",
		Reachable:    true,
		CPUModel:     "AMD Ryzen AI MAX+ 395",
		GPUName:      "AMD Radeon 8060S Graphics",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}

	t.Run("prebuilt tar archive binds base commit and archive digest to runner", func(t *testing.T) {
		var capturedOpts amdgpu.StrixValidationOpts
		var capturedCandArchive *StrixCandidateArchive
		var capturedAdmissionTimeout time.Duration
		runnerCalled := false

		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			capturedOpts = opts
			if cand, ok := CandidateArchiveFromContext(ctx); ok {
				capturedCandArchive = cand
			}
			if adm, ok := AdmissionTimeoutFromContext(ctx); ok {
				capturedAdmissionTimeout = adm
			}

			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		argv := []string{
			"-archive", archiveTarPath,
			"-host", "strix1",
			"-subkernels", "argmax",
			"-ablate", "none",
			"-admission-timeout", "8",
			"-timeout", "30",
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
			t.Errorf("capturedOpts.GitRef = %q, want %q", capturedOpts.GitRef, expectedArchiveDigest)
		}
		if !capturedOpts.RequireSourceBinding {
			t.Errorf("capturedOpts.RequireSourceBinding = false, want true")
		}
		if string(capturedOpts.CandidateArchive) != string(candArchive.ArchiveBytes) {
			t.Errorf("capturedOpts.CandidateArchive does not match archive bytes")
		}
		if capturedOpts.SourceArchiveSHA256 != expectedArchiveDigest {
			t.Errorf("capturedOpts.SourceArchiveSHA256 = %q, want %q", capturedOpts.SourceArchiveSHA256, expectedArchiveDigest)
		}
		if capturedOpts.AdmissionTimeout != 8*time.Second {
			t.Errorf("capturedOpts.AdmissionTimeout = %v, want 8s", capturedOpts.AdmissionTimeout)
		}
		if capturedAdmissionTimeout != 8*time.Second {
			t.Errorf("capturedAdmissionTimeout = %v, want 8s", capturedAdmissionTimeout)
		}
		if capturedCandArchive == nil {
			t.Fatal("candidate archive missing from context")
		}
		if capturedCandArchive.ArchiveSHA256 != candArchive.ArchiveSHA256 {
			t.Errorf("capturedCandArchive.ArchiveSHA256 = %q, want %q", capturedCandArchive.ArchiveSHA256, candArchive.ArchiveSHA256)
		}
		if len(capturedCandArchive.ArchiveBytes) == 0 {
			t.Errorf("captured candidate archive bytes is empty")
		}
	})

	t.Run("matching archive-digest flag succeeds", func(t *testing.T) {
		runnerCalled := false
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			receipt := createTestValidReceipt(validTarget, opts.GitRef, opts.GitTip, opts.Command)
			return receipt, nil
		}

		var stdout, stderr bytes.Buffer
		argv := []string{
			"-archive", archiveTarPath,
			"-archive-digest", "sha256:" + candArchive.ArchiveSHA256,
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
	})

	t.Run("mismatched archive-digest flag fails closed", func(t *testing.T) {
		runnerCalled := false
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			return nil, nil
		}

		var stdout, stderr bytes.Buffer
		argv := []string{
			"-archive", archiveTarPath,
			"-archive-digest", "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			"-subkernels", "argmax",
			"-ablate", "none",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite mismatched archive digest")
		}
		if !strings.Contains(stderr.String(), "archive/digest disagreement") {
			t.Errorf("expected stderr to mention archive/digest disagreement, got: %s", stderr.String())
		}
	})

	t.Run("mismatched explicit git-tip flag fails closed", func(t *testing.T) {
		runnerCalled := false
		runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runnerCalled = true
			return nil, nil
		}

		var stdout, stderr bytes.Buffer
		argv := []string{
			"-archive", archiveTarPath,
			"-git-tip", "d111111111111111111111111111111111111111",
			"-subkernels", "argmax",
			"-ablate", "none",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d", code)
		}
		if runnerCalled {
			t.Fatal("runner was called despite mismatched git-tip")
		}
		if !strings.Contains(stderr.String(), "does not match GitTip") {
			t.Errorf("expected stderr to mention does not match GitTip, got: %s", stderr.String())
		}
	})
}
