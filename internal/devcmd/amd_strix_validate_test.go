package devcmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

const testStrixTip = "0123456789abcdef0123456789abcdef01234567"

func testSHA256(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])
}

func stubAMDStrixCandidate(t *testing.T, archive []byte) (string, string) {
	t.Helper()
	origBuild, origGit, origStatus := buildStrixCandidateArchiveFn, gitRevParseFn, gitStatusFn
	root := filepath.Join(t.TempDir(), "repo")
	digest := testSHA256(archive)
	gitRevParseFn = func(_ context.Context, _ string, args ...string) (string, error) {
		if len(args) == 1 && args[0] == "--show-toplevel" {
			return root, nil
		}
		if len(args) == 2 && args[0] == "--verify" && args[1] == "HEAD^{commit}" {
			return testStrixTip, nil
		}
		return "", fmt.Errorf("unexpected rev-parse args: %v", args)
	}
	gitStatusFn = func(context.Context, string) (string, error) { return "", nil }
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		return amdgpu.StrixCandidateArchive{Bytes: archive, SourceArchiveSHA256: digest}, nil
	}
	t.Cleanup(func() {
		buildStrixCandidateArchiveFn, gitRevParseFn, gitStatusFn = origBuild, origGit, origStatus
	})
	return filepath.Clean(root), digest
}

func creditableAMDStrixReceipt(t *testing.T, opts amdgpu.StrixValidationOpts) *amdgpu.StrixValidationReceipt {
	t.Helper()
	target := amdgpu.StrixTarget{
		Mode: "ssh", Host: "strix1", Reachable: true,
		CPUModel: "AMD Ryzen AI MAX+ 395", GPUName: "AMD Radeon 8060S Graphics",
		TargetISA: "gfx1151", ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}
	r := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
	hash := testSHA256([]byte("test artifact"))
	exitCode := 0
	evidence := amdgpu.StrixExecutionEvidence{
		SourceArchiveSHA256: opts.SourceArchiveSHA256,
		BinarySHA256:        hash, ShaderBundleSHA256: hash, CommandSHA256: hash,
		DeviceIdentity: target.GPUName + "|" + target.TargetISA,
		EngineIdentity: "fak-native/vulkan", ArtifactRehashed: true,
		DeviceTimeoutMS: 1000, LeasePathSHA256: hash, AdmissionWaitMS: 1,
		Acquired: true, Released: true, AcquireOrdinal: 1, ReleaseOrdinal: 2,
		ExitCode: &exitCode, RawOutputSHA256: hash, RawOutputBytes: 1,
	}
	r.Subkernels = []amdgpu.StrixSubkernelResult{{
		Name: "argmax", Status: "PASS", DurationUS: 1, Iterations: 1,
		ParityEvents: []amdgpu.StrixParityEvent{amdgpu.NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)},
		Evidence:     evidence,
	}}
	r.SelectedCount, r.ExecutedCount = 1, 1
	r.SelectedSubkernels, r.ExecutedSubkernels = 1, 1
	r.Provenance.SourceArchiveSHA256 = opts.SourceArchiveSHA256
	r.Provenance.BinarySHA256 = hash
	r.Provenance.ShaderBundleSHA256 = hash
	r.Provenance.BuildCommandSHA256 = hash
	r.Provenance.EngineIdentity = "fak-native/vulkan"
	r.Provenance.CleanupObserved = true
	manifest := sha256.Sum256([]byte("subkernel:argmax:" + hash + "\n"))
	r.Provenance.ExecutionManifestSHA256 = "sha256:" + hex.EncodeToString(manifest[:])
	r.Verified = true
	r.Digest, _ = r.ComputeDigest()
	if err := r.Validate(); err != nil {
		t.Fatalf("test receipt is invalid: %v", err)
	}
	if !r.CreditEligible() {
		t.Fatal("test receipt is not credit eligible")
	}
	return r
}

func historicalAMDStrixReceipt(t *testing.T, opts amdgpu.StrixValidationOpts) *amdgpu.StrixValidationReceipt {
	t.Helper()
	target := amdgpu.StrixTarget{Mode: "ssh", Host: "strix1", Reachable: true, GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151", ComputeUnits: 40}
	r := amdgpu.NewStrixValidationReceipt(target, opts.GitRef, opts.GitTip, opts.Command)
	r.Schema = amdgpu.StrixValidationSchemaV1
	r.Verified = true
	r.Subkernels = []amdgpu.StrixSubkernelResult{{Name: "argmax", Status: "PASS", DurationUS: 1, Iterations: 1, Parity: amdgpu.StrixParityVerdict{Passed: true, ArgmaxExact: true}}}
	r.Digest, _ = r.ComputeDigest()
	if err := r.Validate(); err != nil {
		t.Fatalf("historical receipt must remain readable: %v", err)
	}
	return r
}

func TestRunAMDStrixValidatePassesCanonicalArchiveFieldsUnchanged(t *testing.T) {
	archive := []byte("canonical candidate archive")
	root, digest := stubAMDStrixCandidate(t, archive)
	origRun, origBuild := runStrixValidationFn, buildStrixCandidateArchiveFn
	defer func() { runStrixValidationFn, buildStrixCandidateArchiveFn = origRun, origBuild }()

	var builtRoot, builtTip string
	var builtMine []string
	buildStrixCandidateArchiveFn = func(_ context.Context, gotRoot, gotTip string, mine []string) (amdgpu.StrixCandidateArchive, error) {
		builtRoot, builtTip, builtMine = gotRoot, gotTip, append([]string(nil), mine...)
		return amdgpu.StrixCandidateArchive{Bytes: archive, SourceArchiveSHA256: digest}, nil
	}
	var captured amdgpu.StrixValidationOpts
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		captured = opts
		return creditableAMDStrixReceipt(t, opts), nil
	}
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixValidate(&stdout, &stderr, []string{
		"--json", "--mine", "internal/amdgpu/strix_validation.go", "--mine", "internal/devcmd/amd_strix_validate.go",
		"--git-tip", testStrixTip, "--subkernels", "argmax", "--ablate", "none", "--timeout", "20", "--admission-timeout", "7",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if builtRoot != root || builtTip != testStrixTip || strings.Join(builtMine, ",") != "internal/amdgpu/strix_validation.go,internal/devcmd/amd_strix_validate.go" {
		t.Fatalf("canonical builder inputs root=%q tip=%q mine=%v", builtRoot, builtTip, builtMine)
	}
	if !bytes.Equal(captured.CandidateArchive, archive) || captured.SourceArchiveSHA256 != digest {
		t.Fatalf("runner candidate changed: bytes=%q digest=%q", captured.CandidateArchive, captured.SourceArchiveSHA256)
	}
	if captured.AdmissionTimeout != 7*time.Second || captured.Timeout != 20*time.Second || captured.AdmissionTimeout <= 0 || captured.AdmissionTimeout >= captured.Timeout {
		t.Fatalf("invalid runner timeout envelope: admission=%s total=%s", captured.AdmissionTimeout, captured.Timeout)
	}
}

func TestRunAMDStrixValidateRejectsInvalidMineBeforeBuilderOrRunner(t *testing.T) {
	archive := []byte("candidate")
	stubAMDStrixCandidate(t, archive)
	origRun, origBuild := runStrixValidationFn, buildStrixCandidateArchiveFn
	defer func() { runStrixValidationFn, buildStrixCandidateArchiveFn = origRun, origBuild }()

	tests := []struct {
		name string
		args []string
	}{
		{"empty", []string{"--mine="}},
		{"windows_drive_absolute", []string{"--mine=C:\\outside.go"}},
		{"windows_unc_absolute", []string{"--mine=\\\\server\\share\\outside.go"}},
		{"posix_absolute", []string{"--mine=/tmp/outside.go"}},
		{"traversal", []string{"--mine=../outside.go"}},
		{"duplicate", []string{"--mine=a.go", "--mine=./a.go"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buildCalls, runCalls := 0, 0
			buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
				buildCalls++
				return amdgpu.StrixCandidateArchive{}, nil
			}
			runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				runCalls++
				return nil, nil
			}
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, tc.args); code != 1 {
				t.Fatalf("exit=%d stderr=%s", code, stderr.String())
			}
			if buildCalls != 0 || runCalls != 0 {
				t.Fatalf("invalid overlay reached builder/runner: build=%d run=%d", buildCalls, runCalls)
			}
		})
	}
}

func TestRunAMDStrixValidateRejectsLegacyArchiveFlags(t *testing.T) {
	for _, arg := range []string{"--archive=x.tar", "--archive-digest=sha256:x", "--overlay=a.go", "--overlay-digest=sha256:x"} {
		t.Run(arg, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, []string{arg}); code != 2 {
				t.Fatalf("legacy flag %q exit=%d stderr=%s", arg, code, stderr.String())
			}
		})
	}
}

func TestRunAMDStrixValidateCommittedOnlyRequiresCleanTree(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	gitStatusFn = func(context.Context, string) (string, error) { return " M internal/devcmd/amd_strix_validate.go", nil }
	buildCalls, runCalls := 0, 0
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		buildCalls++
		return amdgpu.StrixCandidateArchive{}, nil
	}
	runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runCalls++
		return nil, nil
	}
	var stdout, stderr bytes.Buffer
	if code := RunAMDStrixValidate(&stdout, &stderr, []string{"--committed-only"}); code != 1 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if buildCalls != 0 || runCalls != 0 {
		t.Fatalf("dirty committed-only candidate reached builder/runner: build=%d run=%d", buildCalls, runCalls)
	}
}

func TestRunAMDStrixValidateRejectsMalformedOrMismatchedCanonicalArchiveTuple(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runCalls := 0
	runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runCalls++
		return nil, nil
	}
	tests := []struct {
		name    string
		archive amdgpu.StrixCandidateArchive
		want    string
	}{
		{"malformed_digest", amdgpu.StrixCandidateArchive{Bytes: []byte("candidate"), SourceArchiveSHA256: "sha256:ABC"}, "invalid source archive digest"},
		{"digest_mismatch", amdgpu.StrixCandidateArchive{Bytes: []byte("candidate"), SourceArchiveSHA256: testSHA256([]byte("different"))}, "archive/digest disagreement"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
				return tc.archive, nil
			}
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, []string{"--mine=a.go"}); code != 1 || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("exit=%d stderr=%s", code, stderr.String())
			}
		})
	}
	if runCalls != 0 {
		t.Fatalf("invalid canonical archive tuple reached runner %d times", runCalls)
	}
}

func TestRunAMDStrixValidateAdmissionEnvelopeRejectsBeforeBuilder(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	stubBuild := buildStrixCandidateArchiveFn
	buildCalls, runCalls := 0, 0
	buildStrixCandidateArchiveFn = func(ctx context.Context, root, tip string, mine []string) (amdgpu.StrixCandidateArchive, error) {
		buildCalls++
		return stubBuild(ctx, root, tip, mine)
	}
	runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		runCalls++
		return nil, nil
	}
	for _, args := range [][]string{
		{"--timeout=10", "--admission-timeout=0"},
		{"--timeout=10", "--admission-timeout=10"},
		{"--timeout=10", "--admission-timeout=11"},
		{"--timeout=120", "--admission-timeout=61"},
		{"--timeout=121", "--admission-timeout=120"},
	} {
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 1 || !strings.Contains(stderr.String(), "admission timeout") {
			t.Fatalf("args=%v exit=%d stderr=%s", args, code, stderr.String())
		}
	}
	if buildCalls != 0 || runCalls != 0 {
		t.Fatalf("invalid admission envelope reached builder/runner: build=%d run=%d", buildCalls, runCalls)
	}
}

func TestRunAMDStrixValidateV1IsReadableButNonCreditInBothModes(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		return historicalAMDStrixReceipt(t, opts), nil
	}
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"json", []string{"--json", "--committed-only", "--subkernels=argmax", "--ablate=none"}},
		{"human", []string{"--committed-only", "--subkernels=argmax", "--ablate=none"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, tc.args); code != 1 {
				t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
			}
			if !strings.Contains(stderr.String(), "not eligible for current v2") {
				t.Fatalf("missing non-credit reason: %s", stderr.String())
			}
		})
	}
}

func TestRunAMDStrixValidateInvalidReceiptFailsBothOutputModes(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		r := historicalAMDStrixReceipt(t, opts)
		r.Digest = "sha256:tampered"
		return r, nil
	}
	for _, args := range [][]string{{"--json", "--committed-only"}, {"--committed-only"}} {
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 1 || !strings.Contains(stderr.String(), "invariant validation failed") {
			t.Fatalf("args=%v exit=%d stderr=%s", args, code, stderr.String())
		}
	}
}

func TestRunAMDStrixValidateRejectsResealedGitRefMismatchInBothModes(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		r := creditableAMDStrixReceipt(t, opts)
		r.Provenance.GitRef = testSHA256([]byte("different candidate"))
		r.Digest, _ = r.ComputeDigest()
		if err := r.Validate(); err != nil || !r.CreditEligible() {
			t.Fatalf("GitRef mismatch fixture must otherwise be valid and creditable: validate=%v credit=%v", err, r.CreditEligible())
		}
		return r, nil
	}
	for _, args := range [][]string{{"--json", "--committed-only"}, {"--committed-only"}} {
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 1 || !strings.Contains(stderr.String(), "receipt GitRef") {
			t.Fatalf("args=%v exit=%d stderr=%s", args, code, stderr.String())
		}
	}
}
