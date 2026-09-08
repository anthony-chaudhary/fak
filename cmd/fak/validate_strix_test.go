package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func stubStrixCandidateArchive(t *testing.T) {
	t.Helper()
	orig := buildStrixCandidateArchiveFn
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		return amdgpu.StrixCandidateArchive{Bytes: []byte("candidate"), SourceArchiveSHA256: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, nil
	}
	t.Cleanup(func() { buildStrixCandidateArchiveFn = orig })
}

func TestIsGPURelatedValidation(t *testing.T) {
	tests := []struct {
		name     string
		mine     []string
		expected bool
	}{
		{
			name:     "non-gpu changes",
			mine:     []string{"cmd/fak/new_verb.go", "internal/policy/policy.go", "docs/README.md"},
			expected: false,
		},
		{
			name:     "amdgpu package changes",
			mine:     []string{"internal/amdgpu/strixhalo.go"},
			expected: true,
		},
		{
			name:     "compute package changes",
			mine:     []string{"internal/compute/vulkan.go"},
			expected: true,
		},
		{
			name:     "roofline package changes",
			mine:     []string{"internal/roofline/empirical.go"},
			expected: true,
		},
		{
			name:     "acceptance validation changes",
			mine:     []string{"cmd/fak/validate_acceptance.go"},
			expected: true,
		},
		{
			name:     "strix named file changes",
			mine:     []string{"internal/devcmd/amd_strix_validate.go"},
			expected: true,
		},
		{
			name:     "directory boundary internal/model_foo must not match internal/model",
			mine:     []string{"internal/model_foo"},
			expected: false,
		},
		{
			name:     "directory boundary internal/model_foo/file.go must not match internal/model",
			mine:     []string{"internal/model_foo/file.go"},
			expected: false,
		},
		{
			name:     "model package change matches whole directory",
			mine:     []string{"internal/model/llm.go"},
			expected: true,
		},
		{
			name:     "halo keyword in path",
			mine:     []string{"configs/halo_apu.json"},
			expected: true,
		},
		{
			name:     "vulkan keyword in path",
			mine:     []string{"shaders/vulkan_kernel.spv"},
			expected: true,
		},
		{
			name:     "gfx115 keyword in path",
			mine:     []string{"firmware/gfx1151.bin"},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isGPURelatedValidation(tt.mine)
			if got != tt.expected {
				t.Errorf("isGPURelatedValidation(%v) = %v, want %v", tt.mine, got, tt.expected)
			}
		})
	}
}

func TestValidateStrix(t *testing.T) {
	// Invariants on shouldRunStrixValidation
	if !shouldRunStrixValidation(true, nil) {
		t.Errorf("expected explicit strix to run validation")
	}
	if shouldRunStrixValidation(false, []string{"docs/README.md"}) {
		t.Errorf("expected non-gpu paths without explicit flag to skip")
	}
	if !shouldRunStrixValidation(false, []string{"internal/amdgpu/strix_validation.go"}) {
		t.Errorf("expected gpu paths to trigger strix validation check")
	}

	// Execution phase fast-skip on non-GPU changes
	var res validateResult
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, false, "", "", "", []string{"docs/README.md"})
	if err != nil {
		t.Fatalf("unexpected error on non-gpu skip: %v", err)
	}
	foundSkipped := false
	for _, p := range res.SkippedPhases {
		if p == "strix_validation" {
			foundSkipped = true
			break
		}
	}
	if !foundSkipped {
		t.Errorf("expected strix_validation in skipped phases, got %v", res.SkippedPhases)
	}
}

func TestValidateStrixNonGPUChangesSkipCleanly(t *testing.T) {
	var res validateResult
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, false, "", "", "", []string{"docs/README.md", "cmd/fak/new_verb.go"})
	if err != nil {
		t.Fatalf("unexpected error on non-gpu skip: %v", err)
	}
	if !res.OK {
		t.Errorf("expected res.OK to remain true on skip, got false")
	}
	if len(res.Failures) != 0 {
		t.Errorf("expected 0 failures on skip, got %d", len(res.Failures))
	}
	foundSkipped := false
	for _, p := range res.SkippedPhases {
		if p == "strix_validation" {
			foundSkipped = true
			break
		}
	}
	if !foundSkipped {
		t.Errorf("expected strix_validation in skipped phases, got %v", res.SkippedPhases)
	}
}

func TestValidateStrixUnavailableHardwareFailsClosed(t *testing.T) {
	origDiscover := discoverStrixTargetFn
	defer func() { discoverStrixTargetFn = origDiscover }()

	discoverStrixTargetFn = func(ctx context.Context, hostOverride string) (*amdgpu.StrixTarget, error) {
		return nil, errors.New("appliance unreachable in test")
	}

	var res validateResult
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err := executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"strix-host-test",
		"",
		"",
		[]string{"internal/amdgpu/strixhalo.go"},
	)

	if err == nil {
		t.Fatalf("expected non-nil error when hardware is unreachable, got nil")
	}
	if res.OK {
		t.Fatalf("expected res.OK == false when hardware is unreachable, got true")
	}

	foundFailure := false
	const expectedSubstr = "strix hardware validation required for relevant changes but appliance is unreachable (pending hardware evidence)"
	for _, f := range res.Failures {
		if f.Step == "strix-validation" && strings.Contains(f.Detail, expectedSubstr) {
			foundFailure = true
			break
		}
	}
	if !foundFailure {
		t.Errorf("expected failure step 'strix-validation' mentioning %q, got: %+v", expectedSubstr, res.Failures)
	}

	for _, p := range res.SkippedPhases {
		if p == "strix_validation" {
			t.Errorf("relevant GPU change must NOT fail open into skipped phases")
		}
	}
}

func TestValidateStrixNilReceiptPropagatesFailure(t *testing.T) {
	stubStrixCandidateArchive(t)
	origDiscover := discoverStrixTargetFn
	origRun := runStrixValidationFn
	defer func() {
		discoverStrixTargetFn = origDiscover
		runStrixValidationFn = origRun
	}()

	discoverStrixTargetFn = func(ctx context.Context, hostOverride string) (*amdgpu.StrixTarget, error) {
		return &amdgpu.StrixTarget{
			Host:      "strix-target-ok",
			Reachable: true,
			TargetISA: "gfx1151",
		}, nil
	}

	// 1. Nil receipt with error
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		return nil, errors.New("ssh connection dropped during validation")
	}

	var res validateResult
	res.Tip = "0123456789abcdef0123456789abcdef01234567"
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err := executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"",
		[]string{"internal/compute/vulkan.go"},
	)

	if err == nil {
		t.Fatalf("expected error on nil receipt, got nil")
	}
	if res.OK {
		t.Fatalf("expected res.OK == false on nil receipt, got true")
	}
	if len(res.Failures) == 0 || res.Failures[0].Step != "strix-validation" {
		t.Errorf("expected strix-validation failure recorded, got %+v", res.Failures)
	}

	// 2. Nil receipt with nil error
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		return nil, nil
	}
	res = validateResult{OK: true, Tip: "0123456789abcdef0123456789abcdef01234567"}
	recorder = &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err = executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"",
		[]string{"internal/compute/vulkan.go"},
	)

	if err == nil {
		t.Fatalf("expected error on nil receipt with nil valErr, got nil")
	}
	if res.OK {
		t.Fatalf("expected res.OK == false on nil receipt, got true")
	}
}

func TestValidateStrixReceiptValidationFails(t *testing.T) {
	stubStrixCandidateArchive(t)
	origDiscover := discoverStrixTargetFn
	origRun := runStrixValidationFn
	defer func() {
		discoverStrixTargetFn = origDiscover
		runStrixValidationFn = origRun
	}()

	discoverStrixTargetFn = func(ctx context.Context, hostOverride string) (*amdgpu.StrixTarget, error) {
		return &amdgpu.StrixTarget{
			Host:      "strix-target-ok",
			Reachable: true,
			TargetISA: "gfx1151",
		}, nil
	}

	// Receipt fails invariant validation (invalid schema)
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		r := amdgpu.NewStrixValidationReceipt(
			amdgpu.StrixTarget{Host: "strix-target-ok", TargetISA: "gfx1151", Reachable: true, ComputeUnits: 40, DiscoveredAt: time.Now().UTC().Format(time.RFC3339)},
			"ref",
			"tip",
			"fak validate --strix",
		)
		r.Schema = "invalid-schema" // forces receipt.Validate() error
		return r, nil
	}

	var res validateResult
	res.Tip = "0123456789abcdef0123456789abcdef01234567"
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err := executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"",
		[]string{"internal/amdgpu/strixhalo.go"},
	)

	if err == nil {
		t.Fatalf("expected error on invalid receipt invariants, got nil")
	}
	if res.OK {
		t.Fatalf("expected res.OK == false when receipt validation fails, got true")
	}
	if res.StrixValidation == nil || res.StrixValidation.Verdict != "FAIL" {
		t.Errorf("expected StrixValidation verdict FAIL, got %+v", res.StrixValidation)
	}
}

func TestValidateStrixHistoricalV1ReceiptCannotEarnCredit(t *testing.T) {
	stubStrixCandidateArchive(t)
	origDiscover, origRun := discoverStrixTargetFn, runStrixValidationFn
	defer func() { discoverStrixTargetFn, runStrixValidationFn = origDiscover, origRun }()
	target := amdgpu.StrixTarget{Host: "strix-target-ok", GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151", Reachable: true, ComputeUnits: 40, DiscoveredAt: time.Now().UTC().Format(time.RFC3339)}
	discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) { return &target, nil }
	runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		r := amdgpu.NewStrixValidationReceipt(target, "HEAD", "0123456789abcdef0123456789abcdef01234567", "fak validate --strix")
		r.Schema = amdgpu.StrixValidationSchemaV1
		r.Verified = true
		r.Subkernels = []amdgpu.StrixSubkernelResult{{Name: "argmax", Status: "PASS", DurationUS: 1, Iterations: 1, Parity: amdgpu.StrixParityVerdict{Passed: true, ArgmaxExact: true}}}
		digest, err := r.ComputeDigest()
		if err != nil {
			t.Fatal(err)
		}
		r.Digest = digest
		return r, nil
	}
	res := validateResult{OK: true, Tip: "0123456789abcdef0123456789abcdef01234567"}
	recorder := &validateRecorder{ctx: context.Background(), stderr: io.Discard, started: time.Now(), res: &res}
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, false, "", "", "", []string{"internal/amdgpu/strix_receipt.go"})
	if err == nil || res.OK || !strings.Contains(err.Error(), "not credit eligible") {
		t.Fatalf("historical v1 receipt earned current credit: err=%v ok=%v", err, res.OK)
	}
}

func TestValidateStrixReceiptNonPassVerdict(t *testing.T) {
	stubStrixCandidateArchive(t)
	origDiscover := discoverStrixTargetFn
	origRun := runStrixValidationFn
	defer func() {
		discoverStrixTargetFn = origDiscover
		runStrixValidationFn = origRun
	}()

	discoverStrixTargetFn = func(ctx context.Context, hostOverride string) (*amdgpu.StrixTarget, error) {
		return &amdgpu.StrixTarget{
			Host:         "strix-target-ok",
			Reachable:    true,
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
		}, nil
	}

	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		r := amdgpu.NewStrixValidationReceipt(
			amdgpu.StrixTarget{Host: "strix-target-ok", TargetISA: "gfx1151", Reachable: true, ComputeUnits: 40, DiscoveredAt: time.Now().UTC().Format(time.RFC3339)},
			"ref",
			"tip",
			"fak validate --strix",
		)
		r.Verdict = "FAIL"
		r.Verified = false
		r.Failures = []string{"subkernel q4k_matmul failed"}
		digest, _ := r.ComputeDigest()
		r.Digest = digest
		return r, nil
	}

	var res validateResult
	res.Tip = "0123456789abcdef0123456789abcdef01234567"
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err := executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"",
		[]string{"internal/amdgpu/strixhalo.go"},
	)

	if err == nil {
		t.Fatalf("expected error on non-PASS verdict, got nil")
	}
	if res.OK {
		t.Fatalf("expected res.OK == false on non-PASS verdict, got true")
	}
	found := false
	for _, f := range res.Failures {
		if f.Step == "strix-validation" && strings.Contains(f.Detail, "subkernel q4k_matmul failed") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected failure detail mentioning 'subkernel q4k_matmul failed', got %+v", res.Failures)
	}
}

func TestValidateStrixAblationsDefault(t *testing.T) {
	stubStrixCandidateArchive(t)
	origDiscover := discoverStrixTargetFn
	origRun := runStrixValidationFn
	defer func() {
		discoverStrixTargetFn = origDiscover
		runStrixValidationFn = origRun
	}()

	discoverStrixTargetFn = func(ctx context.Context, hostOverride string) (*amdgpu.StrixTarget, error) {
		return &amdgpu.StrixTarget{
			Host:         "strix-target-ok",
			Reachable:    true,
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
			DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	var capturedOpts amdgpu.StrixValidationOpts
	runStrixValidationFn = func(ctx context.Context, opts amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		capturedOpts = opts
		r := amdgpu.NewStrixValidationReceipt(
			amdgpu.StrixTarget{Host: "strix-target-ok", TargetISA: "gfx1151", Reachable: true, ComputeUnits: 40, DiscoveredAt: time.Now().UTC().Format(time.RFC3339)},
			"ref",
			"tip",
			opts.Command,
		)
		r.Verdict = "PASS"
		r.Verified = true
		digest, _ := r.ComputeDigest()
		r.Digest = digest
		return r, nil
	}

	// 1. Default ablateArg == "" -> RunAblations must be false
	var res validateResult
	res.Tip = "0123456789abcdef0123456789abcdef01234567"
	res.OK = true
	recorder := &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}

	err := executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"", // ablateArg empty
		[]string{"internal/amdgpu/strixhalo.go"},
	)
	if err == nil {
		t.Fatalf("empty PASS receipt must fail v2 validation")
	}
	if capturedOpts.RunAblations {
		t.Errorf("expected RunAblations == false when ablateArg == '', got true")
	}
	if !capturedOpts.RequireSourceBinding || capturedOpts.GitTip != "0123456789abcdef0123456789abcdef01234567" || len(capturedOpts.CandidateArchive) == 0 || capturedOpts.SourceArchiveSHA256 == "" {
		t.Fatalf("CLI did not bind the exact candidate overlay: %+v", capturedOpts)
	}
	if capturedOpts.Timeout <= capturedOpts.AdmissionTimeout+time.Minute {
		t.Fatalf("total timeout %s cannot cover build, %s admission, and bounded execution", capturedOpts.Timeout, capturedOpts.AdmissionTimeout)
	}
	defaultTimeout := capturedOpts.Timeout

	// 2. Explicit ablateArg != "" -> RunAblations must be true
	res = validateResult{OK: true, Tip: "0123456789abcdef0123456789abcdef01234567"}
	recorder = &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     &res,
	}
	err = executeStrixValidationPhase(
		context.Background(),
		io.Discard,
		io.Discard,
		&res,
		recorder,
		false,
		"",
		"",
		"all", // ablateArg requested
		[]string{"internal/amdgpu/strixhalo.go"},
	)
	if err == nil {
		t.Fatalf("empty PASS receipt must fail v2 validation with ablations")
	}
	if !capturedOpts.RunAblations {
		t.Errorf("expected RunAblations == true when ablateArg == 'all', got false")
	}
	if len(capturedOpts.Ablations) == 0 {
		t.Errorf("expected Ablations list populated when ablateArg == 'all', got empty")
	}
	if capturedOpts.Timeout <= defaultTimeout {
		t.Errorf("multi-run timeout %s did not grow beyond single-run timeout %s", capturedOpts.Timeout, defaultTimeout)
	}
}
