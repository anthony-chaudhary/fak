package devcmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

type testControllerAuthorityRefusal struct {
	code       string
	recovery   string
	cleanup    *int
	cleanupErr error
}

func (r testControllerAuthorityRefusal) Error() string    { return r.code }
func (r testControllerAuthorityRefusal) Code() string     { return r.code }
func (r testControllerAuthorityRefusal) Recovery() string { return r.recovery }
func (r testControllerAuthorityRefusal) RetryCleanup() error {
	if r.cleanup != nil {
		*r.cleanup++
	}
	return r.cleanupErr
}

type testControllerAuthority struct{ epoch string }

func (a testControllerAuthority) Epoch() string { return a.epoch }

const testStrixTip = "0123456789abcdef0123456789abcdef01234567"

func testSHA256(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])
}

func stubAMDStrixCandidate(t *testing.T, archive []byte) (string, string) {
	t.Helper()
	origBuild, origGit, origStatus, origAuthority := buildStrixCandidateArchiveFn, gitRevParseFn, gitStatusFn, newStrixControllerAuthorityFn
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
	newStrixControllerAuthorityFn = func(context.Context, string) (strixControllerAuthority, error) {
		return testControllerAuthority{epoch: "test-semantic-epoch"}, nil
	}
	t.Cleanup(func() {
		buildStrixCandidateArchiveFn, gitRevParseFn, gitStatusFn = origBuild, origGit, origStatus
		newStrixControllerAuthorityFn = origAuthority
	})
	return filepath.Clean(root), digest
}

func structurallyValidAMDStrixReceipt(t *testing.T, opts amdgpu.StrixValidationOpts) *amdgpu.StrixValidationReceipt {
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
	if r.CreditEligible() {
		t.Fatal("caller-authored test receipt unexpectedly minted physical credit")
	}
	return r
}

func noncreditFullMatrixLikeAMDStrixReceipt(t *testing.T, opts amdgpu.StrixValidationOpts) *amdgpu.StrixValidationReceipt {
	t.Helper()
	r := structurallyValidAMDStrixReceipt(t, opts)
	evidence := r.Subkernels[0].Evidence
	r.Subkernels = append(r.Subkernels, amdgpu.StrixSubkernelResult{
		Name:       "matmul_f32",
		Status:     "PASS",
		DurationUS: 1,
		Iterations: 1,
		ParityEvents: []amdgpu.StrixParityEvent{
			amdgpu.NewCosineMaxAbsParityEvent("fak-native/vulkan", 1, true, 0.999995, 0.999900, 0.0012, 0.01),
		},
		Evidence: evidence,
	})
	r.SelectedCount, r.ExecutedCount = 2, 2
	r.SelectedSubkernels, r.ExecutedSubkernels = 2, 2
	manifest := sha256.Sum256([]byte("subkernel:argmax:" + evidence.CommandSHA256 + "\nsubkernel:matmul_f32:" + evidence.CommandSHA256 + "\n"))
	r.Provenance.ExecutionManifestSHA256 = "sha256:" + hex.EncodeToString(manifest[:])
	r.Digest, _ = r.ComputeDigest()
	if err := r.Validate(); err != nil {
		t.Fatalf("full-matrix-like evidence receipt is invalid: %v", err)
	}
	if r.CreditEligible() {
		t.Fatal("full-matrix-like evidence receipt unexpectedly earned narrow promotion credit")
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
		return structurallyValidAMDStrixReceipt(t, opts), nil
	}
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixValidate(&stdout, &stderr, []string{
		"--evidence-only", "--json", "--mine", "internal/amdgpu/strix_validation.go", "--mine", "internal/devcmd/amd_strix_validate.go",
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
		{"evidence_json", []string{"--evidence-only", "--json", "--committed-only", "--subkernels=argmax", "--ablate=none"}},
		{"evidence_human", []string{"--evidence-only", "--committed-only", "--subkernels=argmax", "--ablate=none"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, tc.args); code != 1 {
				t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
			}
			if !strings.Contains(stderr.String(), "not eligible for current v2") && !strings.Contains(stderr.String(), "not current physical Strix schema") {
				t.Fatalf("missing current-v2 rejection reason: %s", stderr.String())
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
	for _, args := range [][]string{{"--json", "--committed-only"}, {"--committed-only"}, {"--evidence-only", "--json", "--committed-only"}, {"--evidence-only", "--committed-only"}} {
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
		r := structurallyValidAMDStrixReceipt(t, opts)
		r.Provenance.GitRef = testSHA256([]byte("different candidate"))
		r.Digest, _ = r.ComputeDigest()
		if err := r.Validate(); err != nil || r.CreditEligible() {
			t.Fatalf("GitRef mismatch fixture must be structurally valid and non-credit: validate=%v credit=%v", err, r.CreditEligible())
		}
		return r, nil
	}
	for _, args := range [][]string{{"--json", "--committed-only"}, {"--committed-only"}, {"--evidence-only", "--json", "--committed-only"}, {"--evidence-only", "--committed-only"}} {
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 1 || !strings.Contains(stderr.String(), "receipt GitRef") {
			t.Fatalf("args=%v exit=%d stderr=%s", args, code, stderr.String())
		}
	}
}

func TestRunAMDStrixValidateEvidenceOnlyAcceptsValidNoncreditReceipt(t *testing.T) {
	_, digest := stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		return noncreditFullMatrixLikeAMDStrixReceipt(t, opts), nil
	}

	var strictOut, strictErr bytes.Buffer
	if code := RunAMDStrixValidate(&strictOut, &strictErr, []string{"--json", "--committed-only"}); code != 1 {
		t.Fatalf("strict exit=%d stderr=%s stdout=%s", code, strictErr.String(), strictOut.String())
	}
	if !strings.Contains(strictErr.String(), "not eligible for current v2 promotion credit") || !strings.Contains(strictOut.String(), `"promotion_credit_eligible": false`) {
		t.Fatalf("strict mode did not fail closed with marker: stderr=%s stdout=%s", strictErr.String(), strictOut.String())
	}

	var jsonOut, jsonErr bytes.Buffer
	if code := RunAMDStrixValidate(&jsonOut, &jsonErr, []string{"--evidence-only", "--json", "--committed-only"}); code != 0 {
		t.Fatalf("evidence JSON exit=%d stderr=%s stdout=%s", code, jsonErr.String(), jsonOut.String())
	}
	var output struct {
		amdgpu.StrixValidationReceipt
		PromotionCreditEligible bool `json:"promotion_credit_eligible"`
	}
	decoder := json.NewDecoder(&jsonOut)
	if err := decoder.Decode(&output); err != nil {
		t.Fatalf("decode evidence JSON: %v\n%s", err, jsonOut.String())
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		t.Fatalf("evidence JSON has trailing data: err=%v trailing=%v", err, trailing)
	}
	if output.PromotionCreditEligible {
		t.Fatal("full-matrix-like evidence was marked promotion-credit eligible")
	}
	if err := output.StrixValidationReceipt.Validate(); err != nil {
		t.Fatalf("embedded receipt does not validate: %v", err)
	}
	if output.Schema != amdgpu.StrixValidationSchemaV2 || output.Provenance.GitTip != testStrixTip || output.Provenance.GitRef != digest || output.Provenance.SourceArchiveSHA256 != digest {
		t.Fatalf("embedded receipt is not source-bound: schema=%q tip=%q git_ref=%q source=%q", output.Schema, output.Provenance.GitTip, output.Provenance.GitRef, output.Provenance.SourceArchiveSHA256)
	}

	var humanOut, humanErr bytes.Buffer
	if code := RunAMDStrixValidate(&humanOut, &humanErr, []string{"--evidence-only", "--committed-only"}); code != 0 {
		t.Fatalf("evidence human exit=%d stderr=%s stdout=%s", code, humanErr.String(), humanOut.String())
	}
	if !strings.Contains(humanOut.String(), "Verdict:     PASS (valid evidence; promotion non-credit)") || !strings.Contains(humanOut.String(), "promotion_credit_eligible: false") {
		t.Fatalf("evidence human output lacks non-credit marker: %s", humanOut.String())
	}
}

func TestRunAMDStrixValidateEvidenceOnlyRejectsRunErrorAndMarksFailure(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		return noncreditFullMatrixLikeAMDStrixReceipt(t, opts), fmt.Errorf("transport completion failed")
	}
	var stdout, stderr bytes.Buffer
	if code := RunAMDStrixValidate(&stdout, &stderr, []string{"--evidence-only", "--json", "--committed-only"}); code != 1 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stderr.String(), "transport completion failed") || !strings.Contains(stdout.String(), `"promotion_credit_eligible": false`) {
		t.Fatalf("run failure was not preserved and marked non-credit: stderr=%s stdout=%s", stderr.String(), stdout.String())
	}
}

func TestRunAMDStrixValidateEvidenceOnlyRejectsNonPassVerdicts(t *testing.T) {
	stubAMDStrixCandidate(t, []byte("candidate"))
	origRun := runStrixValidationFn
	defer func() { runStrixValidationFn = origRun }()
	for _, verdict := range []string{"FAIL", "SKIPPED"} {
		t.Run(verdict, func(t *testing.T) {
			runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				r := structurallyValidAMDStrixReceipt(t, opts)
				r.Verdict = verdict
				r.Verified = false
				r.Digest, _ = r.ComputeDigest()
				if err := r.Validate(); err != nil {
					t.Fatalf("non-PASS fixture must be integrity-valid: %v", err)
				}
				return r, nil
			}
			var stdout, stderr bytes.Buffer
			if code := RunAMDStrixValidate(&stdout, &stderr, []string{"--evidence-only", "--json", "--committed-only"}); code != 1 {
				t.Fatalf("exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
			}
			if !strings.Contains(stderr.String(), "is not PASS") || !strings.Contains(stdout.String(), `"promotion_credit_eligible": false`) {
				t.Fatalf("non-PASS receipt was not rejected and marked non-credit: stderr=%s stdout=%s", stderr.String(), stdout.String())
			}
		})
	}
}

func TestRunAMDStrixCommandsRequireControllerAuthorityBeforeTransport(t *testing.T) {
	origAuthority, origDiscover, origRun := newStrixControllerAuthorityFn, discoverStrixTargetFn, runStrixValidationFn
	defer func() {
		newStrixControllerAuthorityFn, discoverStrixTargetFn, runStrixValidationFn = origAuthority, origDiscover, origRun
	}()

	root, _ := stubAMDStrixCandidate(t, []byte("candidate"))
	for _, tc := range []struct {
		name     string
		command  func(io.Writer, io.Writer, []string) int
		args     []string
		code     string
		recovery string
	}{
		{"probe_windows", RunAMDStrixProbe, []string{"--candidate-dir", root}, "UNSUPPORTED_VALIDATOR_BINARY", "use a current stamped WSL build"},
		{"probe_stale", RunAMDStrixProbe, []string{"--candidate-dir", root}, "STALE_VALIDATOR_BINARY", ""},
		{"validate_unattested", RunAMDStrixValidate, []string{"--candidate-dir", root, "--mine=a.go"}, "GIT_SNAPSHOT_UNATTESTED", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			discoverCalls, runCalls := 0, 0
			discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
				discoverCalls++
				return nil, fmt.Errorf("transport must not start")
			}
			runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				runCalls++
				return nil, fmt.Errorf("transport must not start")
			}
			var gotRoot string
			cleanupCalls := 0
			cleanupErr := error(nil)
			if tc.name == "probe_windows" {
				cleanupErr = fmt.Errorf("secret snapshot path must not leak")
			}
			newStrixControllerAuthorityFn = func(_ context.Context, resolvedRoot string) (strixControllerAuthority, error) {
				gotRoot = resolvedRoot
				return nil, testControllerAuthorityRefusal{code: tc.code, recovery: tc.recovery, cleanup: &cleanupCalls, cleanupErr: cleanupErr}
			}
			var stdout, stderr bytes.Buffer
			if code := tc.command(&stdout, &stderr, tc.args); code != 1 {
				t.Fatalf("exit=%d stderr=%s", code, stderr.String())
			}
			if gotRoot != filepath.Clean(root) {
				t.Fatalf("authority root=%q want %q", gotRoot, filepath.Clean(root))
			}
			if !strings.Contains(stderr.String(), tc.code) || (tc.recovery != "" && !strings.Contains(stderr.String(), tc.recovery)) {
				t.Fatalf("typed refusal missing from stderr: %s", stderr.String())
			}
			if discoverCalls != 0 || runCalls != 0 {
				t.Fatalf("refusal reached transport: discover=%d run=%d", discoverCalls, runCalls)
			}
			if cleanupCalls != 1 {
				t.Fatalf("refusal cleanup calls=%d want 1", cleanupCalls)
			}
			if cleanupErr != nil && (!strings.Contains(stderr.String(), "snapshot_cleanup=FAILED") || strings.Contains(stderr.String(), "secret snapshot")) {
				t.Fatalf("cleanup outcome was not stable and redacted: %s", stderr.String())
			}
		})
	}

	for _, tc := range []struct {
		name    string
		command func(io.Writer, io.Writer, []string) int
		args    []string
	}{
		{"probe_zero_nil", RunAMDStrixProbe, []string{"--candidate-dir", root}},
		{"validate_zero_nil", RunAMDStrixValidate, []string{"--candidate-dir", root, "--mine=a.go"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			discoverCalls, runCalls := 0, 0
			discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
				discoverCalls++
				return nil, nil
			}
			runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				runCalls++
				return nil, nil
			}
			newStrixControllerAuthorityFn = func(context.Context, string) (strixControllerAuthority, error) {
				return testControllerAuthority{}, nil
			}
			var stdout, stderr bytes.Buffer
			if code := tc.command(&stdout, &stderr, tc.args); code != 1 || !strings.Contains(stderr.String(), "INVALID_CONTROLLER_AUTHORITY") {
				t.Fatalf("exit=%d stderr=%s", code, stderr.String())
			}
			if discoverCalls != 0 || runCalls != 0 {
				t.Fatalf("zero authority reached transport: discover=%d run=%d", discoverCalls, runCalls)
			}
		})
	}

	t.Run("probe_success", func(t *testing.T) {
		authorityCalls, discoverCalls := 0, 0
		newStrixControllerAuthorityFn = func(_ context.Context, gotRoot string) (strixControllerAuthority, error) {
			authorityCalls++
			if gotRoot != filepath.Clean(root) {
				t.Fatalf("authority root=%q want %q", gotRoot, filepath.Clean(root))
			}
			return testControllerAuthority{epoch: "test-semantic-epoch"}, nil
		}
		discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
			discoverCalls++
			return &amdgpu.StrixTarget{Reachable: true}, nil
		}
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixProbe(&stdout, &stderr, []string{"--candidate-dir", root, "--json"}); code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, stderr.String())
		}
		if authorityCalls != 1 || discoverCalls != 1 {
			t.Fatalf("authority=%d discover=%d", authorityCalls, discoverCalls)
		}
	})

	t.Run("validate_success", func(t *testing.T) {
		authorityCalls, runCalls := 0, 0
		newStrixControllerAuthorityFn = func(_ context.Context, gotRoot string) (strixControllerAuthority, error) {
			authorityCalls++
			if gotRoot != filepath.Clean(root) {
				t.Fatalf("authority root=%q want %q", gotRoot, filepath.Clean(root))
			}
			return testControllerAuthority{epoch: "test-semantic-epoch"}, nil
		}
		runStrixValidationFn = func(_ context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			runCalls++
			return structurallyValidAMDStrixReceipt(t, opts), nil
		}
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, []string{"--evidence-only", "--candidate-dir", root, "--mine=a.go", "--json", "--subkernels=argmax", "--ablate=none"}); code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, stderr.String())
		}
		if authorityCalls != 1 || runCalls != 1 {
			t.Fatalf("authority=%d runner=%d", authorityCalls, runCalls)
		}
	})
}
