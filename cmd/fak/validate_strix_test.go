package main

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

type testStrixControllerAuthorityRefusal struct {
	code         string
	recovery     string
	cleanupCalls *int
	cleanupErr   error
}

func (r *testStrixControllerAuthorityRefusal) Error() string {
	return r.code + ": controller authority refused"
}
func (r *testStrixControllerAuthorityRefusal) Code() string     { return r.code }
func (r *testStrixControllerAuthorityRefusal) Recovery() string { return r.recovery }
func (r *testStrixControllerAuthorityRefusal) RetryCleanup() error {
	if r.cleanupCalls != nil {
		*r.cleanupCalls++
	}
	return r.cleanupErr
}

func TestValidateStrixRequiresControllerAuthorityBeforeDiscovery(t *testing.T) {
	tests := []struct {
		name           string
		explicit       bool
		mine           []string
		code           string
		recovery       string
		wantRecovery   bool
		zeroAuthority  bool
		cleanupPending bool
	}{
		{name: "explicit pre-epoch", explicit: true, mine: []string{"docs/README.md"}, code: "STALE_VALIDATOR_BINARY"},
		{name: "automatic unattested", mine: []string{"internal/amdgpu/strix_validation.go"}, code: "GIT_SNAPSHOT_UNATTESTED", cleanupPending: true},
		{name: "explicit native Windows", explicit: true, mine: []string{"docs/README.md"}, code: "UNSUPPORTED_VALIDATOR_BINARY", recovery: "use a current stamped WSL build with /proc/self/exe authority", wantRecovery: true},
		{name: "explicit zero authority with nil error", explicit: true, mine: []string{"docs/README.md"}, code: "CONTROLLER_AUTHORITY_UNAVAILABLE", zeroAuthority: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			origAuthority := newStrixControllerAuthorityFn
			origAuthorityValid := strixControllerAuthorityValidFn
			origDiscover := discoverStrixTargetFn
			origArchive := buildStrixCandidateArchiveFn
			origRun := runStrixValidationFn
			t.Cleanup(func() {
				newStrixControllerAuthorityFn = origAuthority
				strixControllerAuthorityValidFn = origAuthorityValid
				discoverStrixTargetFn = origDiscover
				buildStrixCandidateArchiveFn = origArchive
				runStrixValidationFn = origRun
			})

			const resolvedRoot = "/operator/selected/repository"
			authorityCalls, discoveryCalls, archiveCalls, transportCalls, cleanupCalls := 0, 0, 0, 0, 0
			newStrixControllerAuthorityFn = func(_ context.Context, root string) (amdgpu.StrixControllerAuthority, error) {
				authorityCalls++
				if root != resolvedRoot {
					t.Fatalf("controller authority root = %q, want already-resolved %q", root, resolvedRoot)
				}
				if tt.zeroAuthority {
					return amdgpu.StrixControllerAuthority{}, nil
				}
				refusal := &testStrixControllerAuthorityRefusal{code: tt.code, recovery: tt.recovery, cleanupCalls: &cleanupCalls}
				if tt.cleanupPending {
					refusal.cleanupErr = errors.New("cleanup still pending at /private/snapshot/secret")
				}
				return amdgpu.StrixControllerAuthority{}, refusal
			}
			discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
				discoveryCalls++
				return nil, errors.New("must not discover")
			}
			buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
				archiveCalls++
				return amdgpu.StrixCandidateArchive{}, errors.New("must not archive")
			}
			runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				transportCalls++
				return nil, errors.New("must not transport")
			}

			res := validateResult{OK: true}
			recorder := &validateRecorder{ctx: context.Background(), stderr: io.Discard, started: time.Now(), res: &res}
			err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, resolvedRoot, tt.explicit, "", "", "", tt.mine)
			if err == nil || !strings.Contains(err.Error(), tt.code) {
				t.Fatalf("authority refusal = %v, want typed code %q", err, tt.code)
			}
			if res.OK {
				t.Fatal("controller authority refusal remained creditable")
			}
			if authorityCalls != 1 || discoveryCalls != 0 || archiveCalls != 0 || transportCalls != 0 {
				t.Fatalf("calls authority/discovery/archive/transport = %d/%d/%d/%d, want 1/0/0/0", authorityCalls, discoveryCalls, archiveCalls, transportCalls)
			}
			wantCleanupCalls := 1
			if tt.zeroAuthority {
				wantCleanupCalls = 0
			}
			if cleanupCalls != wantCleanupCalls {
				t.Fatalf("cleanup retries = %d, want %d", cleanupCalls, wantCleanupCalls)
			}
			if tt.cleanupPending {
				var preserved *testStrixControllerAuthorityRefusal
				if !errors.As(err, &preserved) || preserved == nil || !strings.Contains(res.Failures[0].Detail, "cleanup_retry=pending") {
					t.Fatalf("retryable cleanup ownership was not preserved: err=%v failures=%+v", err, res.Failures)
				}
				if strings.Contains(err.Error(), "/private/snapshot/secret") || strings.Contains(res.Failures[0].Detail, "/private/snapshot/secret") {
					t.Fatalf("cleanup error disclosed private snapshot detail: err=%v failures=%+v", err, res.Failures)
				}
			}
			if len(res.Failures) != 1 || res.Failures[0].Step != "strix-controller-authority" || !strings.Contains(res.Failures[0].Detail, tt.code) {
				t.Fatalf("typed controller failure = %+v", res.Failures)
			}
			if strings.Contains(res.Failures[0].Detail, resolvedRoot) {
				t.Fatalf("controller refusal disclosed repository root: %q", res.Failures[0].Detail)
			}
			if strings.Contains(res.Failures[0].Detail, "observed_revision=") == false || strings.Contains(res.Failures[0].Detail, "executable_sha256=") == false {
				t.Fatalf("controller refusal lacks safe evidence fields: %q", res.Failures[0].Detail)
			}
			hasRecovery := strings.Contains(res.Failures[0].Detail, "recovery:")
			if hasRecovery != tt.wantRecovery {
				t.Fatalf("recovery presence = %v, want %v: %q", hasRecovery, tt.wantRecovery, res.Failures[0].Detail)
			}
			if tt.wantRecovery && !strings.Contains(res.Failures[0].Detail, tt.recovery) {
				t.Fatalf("expected recovery %q not in %q", tt.recovery, res.Failures[0].Detail)
			}
		})
	}

	t.Run("runValidate forwards operator-selected root", func(t *testing.T) {
		repo, git := seedGitFixtureRepo(t)
		// Automatic (non --strix) hardware validation is scoped to the public runtime
		// module (internal/validate automaticStrixValidationEligible, 8a78ce093), so the
		// fixture must declare that module for the GPU-path change to reach the
		// controller-authority gate at all; a gitfixture.test module skips the phase.
		commitFiles(t, repo, git, "seed", map[string]string{
			"go.mod":                         "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n",
			"internal/amdgpu/strix_probe.go": "package amdgpu\n\nfunc StrixProbe() {}\n",
		})
		resolvedRoot, err := filepath.Abs(repo)
		if err != nil {
			t.Fatal(err)
		}

		origAuthority := newStrixControllerAuthorityFn
		origDiscover := discoverStrixTargetFn
		origArchive := buildStrixCandidateArchiveFn
		origRun := runStrixValidationFn
		t.Cleanup(func() {
			newStrixControllerAuthorityFn = origAuthority
			discoverStrixTargetFn = origDiscover
			buildStrixCandidateArchiveFn = origArchive
			runStrixValidationFn = origRun
		})

		authorityCalls, discoveryCalls, archiveCalls, transportCalls := 0, 0, 0, 0
		newStrixControllerAuthorityFn = func(_ context.Context, root string) (amdgpu.StrixControllerAuthority, error) {
			authorityCalls++
			if filepath.Clean(root) != filepath.Clean(resolvedRoot) {
				t.Fatalf("runValidate authority root = %q, want resolved --root %q", root, resolvedRoot)
			}
			return amdgpu.StrixControllerAuthority{}, &testStrixControllerAuthorityRefusal{code: "STALE_VALIDATOR_BINARY"}
		}
		discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
			discoveryCalls++
			return nil, errors.New("must not discover")
		}
		buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
			archiveCalls++
			return amdgpu.StrixCandidateArchive{}, errors.New("must not archive")
		}
		runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			transportCalls++
			return nil, errors.New("must not transport")
		}

		res, code, stderr := runValidateJSON(t, []string{
			"--root", repo,
			"--mine", "internal/amdgpu/strix_probe.go",
			"--test-only",
			"--wsl-tests=false",
			"--test-run=^$",
			"--json",
		})
		if code == 0 || res.OK {
			t.Fatalf("runValidate authority refusal remained successful: code=%d stderr=%q result=%+v", code, stderr, res)
		}
		if authorityCalls != 1 || discoveryCalls != 0 || archiveCalls != 0 || transportCalls != 0 {
			t.Fatalf("runValidate calls authority/discovery/archive/transport = %d/%d/%d/%d, want 1/0/0/0", authorityCalls, discoveryCalls, archiveCalls, transportCalls)
		}
		if len(res.Failures) == 0 || res.Failures[len(res.Failures)-1].Step != "strix-controller-authority" {
			t.Fatalf("runValidate typed controller failure = %+v", res.Failures)
		}
	})
}

func stubStrixCandidateArchive(t *testing.T) {
	t.Helper()
	origArchive, origAuthority, origAuthorityValid := buildStrixCandidateArchiveFn, newStrixControllerAuthorityFn, strixControllerAuthorityValidFn
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		return amdgpu.StrixCandidateArchive{Bytes: []byte("candidate"), SourceArchiveSHA256: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, nil
	}
	newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
		return amdgpu.StrixControllerAuthority{}, nil
	}
	strixControllerAuthorityValidFn = func(amdgpu.StrixControllerAuthority) bool { return true }
	t.Cleanup(func() {
		buildStrixCandidateArchiveFn = origArchive
		newStrixControllerAuthorityFn = origAuthority
		strixControllerAuthorityValidFn = origAuthorityValid
	})
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
			name:     "pure-CPU model package change must not demand physical hardware",
			mine:     []string{"internal/model/llm.go"},
			expected: false,
		},
		{
			name:     "pure-CPU model router change must not demand physical hardware",
			mine:     []string{"internal/model/v4_topk_partial.go"},
			expected: false,
		},
		{
			name:     "pure-CPU V4.1 forward change must not demand physical hardware",
			mine:     []string{"internal/model/v41_forward.go"},
			expected: false,
		},
		{
			name:     "model Vulkan backend change still demands physical hardware",
			mine:     []string{"internal/model/vulkan_glm_kda_physical_test.go"},
			expected: true,
		},
		{
			name:     "model Metal backend change still demands physical hardware",
			mine:     []string{"internal/model/metal_decode.go"},
			expected: true,
		},
		{
			name:     "model GPU-direct swap change still demands physical hardware",
			mine:     []string{"internal/model/qwen38_gpudirect_swap.go"},
			expected: true,
		},
		{
			name:     "halo keyword in source path",
			mine:     []string{"internal/amdgpu/halo_apu.go"},
			expected: true,
		},
		{
			name:     "vulkan keyword in shader source path",
			mine:     []string{"shaders/vulkan_kernel.spv"},
			expected: true,
		},
		{
			name:     "gfx115 keyword in path",
			mine:     []string{"firmware/gfx1151.bin"},
			expected: true,
		},
		{
			name:     "docs path mentioning strix and halo must not trigger",
			mine:     []string{"docs/tickets/strix-halo-franchise/INDEX.md"},
			expected: false,
		},
		{
			name:     "docs research path mentioning strix must not trigger",
			mine:     []string{"docs/research/EXTREME-QUANTIZATION-UMA-MOAT-RISK-2026-09-18.md"},
			expected: false,
		},
		{
			name:     "json config path mentioning halo must not trigger",
			mine:     []string{"configs/halo_apu.json"},
			expected: false,
		},
		{
			name:     "source path under gpu root triggers regardless of keyword",
			mine:     []string{"internal/compute/strix/x.go"},
			expected: true,
		},
		{
			name:     "go source path containing gfx115 still triggers",
			mine:     []string{"internal/amdgpu/gfx115_kernels.go"},
			expected: true,
		},
		{
			name:     "assembly path containing strix still triggers",
			mine:     []string{"internal/amdgpu/strix_prefill.s"},
			expected: true,
		},
		{
			name:     "control-plane test file mentioning halo must not trigger",
			mine:     []string{"platform/obs/stack/halo_test.go"},
			expected: false,
		},
		{
			name:     "control-plane source under platform/strix must not trigger",
			mine:     []string{"platform/strix/opencode_runtime.go"},
			expected: false,
		},
		{
			name:     "private factory Halo validator helper must not trigger",
			mine:     []string{"cmd/fak-flow/halo_verify.go"},
			expected: false,
		},
		{
			name:     "private Halo appliance command must not trigger",
			mine:     []string{"cmd/fak-strix/work.go"},
			expected: false,
		},
		{
			name:     "private Ops Halo bridge must not trigger",
			mine:     []string{"cmd/fak-sync/ops_halo_ticket_admission.go"},
			expected: false,
		},
		{
			name:     "test file outside gpu roots mentioning halo must not trigger",
			mine:     []string{"internal/foo/halo_test.go"},
			expected: false,
		},
		{
			name:     "test file under gpu root still triggers via root",
			mine:     []string{"internal/amdgpu/x_test.go"},
			expected: true,
		},
		{
			name:     "public non-test source with vulkan keyword still triggers",
			mine:     []string{"internal/serve/vulkan_backend.go"},
			expected: true,
		},
		{
			name:     "model metal kernel still triggers via model marker",
			mine:     []string{"internal/model/metal_kernel.go"},
			expected: true,
		},
		{
			name:     "compute shader path containing vulkan still triggers",
			mine:     []string{"shaders/vulkan_matmul.comp"},
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

// TestStrixValidationSelectionIsHostScopedForModelStateChanges pins the real
// validator-selection seam for the host-side model changes that were blocked on
// physical Strix hardware (#12382). Asserting on isGPURelatedValidation alone is
// not the defect's acceptance witness: the observable the reporter hit is the
// selection *result* — whether executeStrixValidationPhase skips or reaches
// controller authority. This test therefore drives the phase itself and requires
// that a host-side model path never touches authority, discovery, or the device
// transport, while device-bearing paths and an explicit --strix still do.
func TestStrixValidationSelectionIsHostScopedForModelStateChanges(t *testing.T) {
	// The exact host-side path sets named in #12382, each a real change that
	// passed gofmt/build/vet/its own tests and then failed only because the
	// automatic Strix phase demanded unreachable physical hardware.
	hostOnly := []struct {
		name string
		mine []string
	}{
		{
			// #12342 MTP partial-commit transaction restore.
			name: "mtp transaction rollback",
			mine: []string{
				"internal/model/mtp_transaction.go",
				"internal/model/mtp_transaction_test.go",
			},
		},
		{
			// #12433 host-only test-fixture correction.
			name: "embedding q2k test fixture",
			mine: []string{"internal/model/embedding_q2k_test.go"},
		},
		{
			// #12341/#12346 nonfinite guards in the V4.1 attention path.
			name: "v41 attention nonfinite guards",
			mine: []string{
				"internal/model/v41/v41_sparse_attention.go",
				"internal/model/v41_attention.go",
				"internal/model/v41/v41_sparse_sink_nonfinite_guard_test.go",
				"internal/model/v41_attention_nonfinite_guard_test.go",
			},
		},
		{
			// Mixed host-side model work alongside a non-device package.
			name: "host model plus control plane",
			mine: []string{
				"internal/model/llm.go",
				"internal/model/v4_topk_partial.go",
				"cmd/fak/serve.go",
			},
		},
	}

	// Device-bearing paths must keep their physical requirement.
	deviceBearing := []struct {
		name string
		mine []string
	}{
		{name: "amdgpu package", mine: []string{"internal/amdgpu/strix_validation.go"}},
		{name: "compute package", mine: []string{"internal/compute/vulkan.go"}},
		{name: "model vulkan backend", mine: []string{"internal/model/vulkan_glm_kda_physical_test.go"}},
		{name: "model metal backend", mine: []string{"internal/model/metal_decode.go"}},
		{name: "model gpudirect swap", mine: []string{"internal/model/qwen38_gpudirect_swap.go"}},
	}

	origAuthority := newStrixControllerAuthorityFn
	origAuthorityValid := strixControllerAuthorityValidFn
	origDiscover := discoverStrixTargetFn
	origArchive := buildStrixCandidateArchiveFn
	origRun := runStrixValidationFn
	t.Cleanup(func() {
		newStrixControllerAuthorityFn = origAuthority
		strixControllerAuthorityValidFn = origAuthorityValid
		discoverStrixTargetFn = origDiscover
		buildStrixCandidateArchiveFn = origArchive
		runStrixValidationFn = origRun
	})

	// runPhase drives the real selection seam and reports whether the phase
	// skipped or engaged, plus how many device-plane calls it made.
	runPhase := func(t *testing.T, explicit bool, mine []string) (skipped bool, res validateResult, err error) {
		t.Helper()
		authorityCalls, discoveryCalls, archiveCalls, transportCalls := 0, 0, 0, 0
		newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
			authorityCalls++
			// Refuse so an engaged phase terminates deterministically without
			// any device contact; the call count is the observable.
			return amdgpu.StrixControllerAuthority{}, &testStrixControllerAuthorityRefusal{code: "GIT_SNAPSHOT_UNATTESTED"}
		}
		strixControllerAuthorityValidFn = func(amdgpu.StrixControllerAuthority) bool { return true }
		discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
			discoveryCalls++
			return nil, errors.New("must not discover")
		}
		buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
			archiveCalls++
			return amdgpu.StrixCandidateArchive{}, errors.New("must not archive")
		}
		runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			transportCalls++
			return nil, errors.New("must not transport")
		}

		res = validateResult{OK: true}
		recorder := &validateRecorder{ctx: context.Background(), stderr: io.Discard, started: time.Now(), res: &res}
		err = executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, "", explicit, "", "", "", mine)

		for _, phase := range res.SkippedPhases {
			if phase == "strix_validation" {
				skipped = true
			}
		}
		if skipped && (authorityCalls != 0 || discoveryCalls != 0 || archiveCalls != 0 || transportCalls != 0) {
			t.Fatalf("skipped phase still touched the device plane: authority/discovery/archive/transport = %d/%d/%d/%d",
				authorityCalls, discoveryCalls, archiveCalls, transportCalls)
		}
		if !skipped && authorityCalls == 0 {
			t.Fatal("engaged phase never requested controller authority")
		}
		return skipped, res, err
	}

	for _, tt := range hostOnly {
		t.Run("host only/"+tt.name, func(t *testing.T) {
			if shouldRunStrixValidation(false, tt.mine) {
				t.Fatalf("host-side paths %v were selected for automatic Strix hardware validation", tt.mine)
			}
			skipped, res, err := runPhase(t, false, tt.mine)
			if err != nil {
				t.Fatalf("host-side selection failed instead of skipping: %v", err)
			}
			if !skipped {
				t.Fatalf("strix_validation was not recorded as skipped for %v", tt.mine)
			}
			if !res.OK || len(res.Failures) != 0 {
				t.Fatalf("host-side skip stayed uncreditable: ok=%v failures=%+v", res.OK, res.Failures)
			}
			// Explicit --strix must still reach controller authority for the
			// same paths: host scoping narrows the automatic trigger only and
			// never becomes a bypass of an explicitly demanded device run.
			skippedExplicit, _, errExplicit := runPhase(t, true, tt.mine)
			if skippedExplicit || errExplicit == nil {
				t.Fatalf("explicit --strix was skipped or passed for %v: skipped=%v err=%v", tt.mine, skippedExplicit, errExplicit)
			}
		})
	}

	for _, tt := range deviceBearing {
		t.Run("device bearing/"+tt.name, func(t *testing.T) {
			if !shouldRunStrixValidation(false, tt.mine) {
				t.Fatalf("device paths %v lost their physical Strix requirement", tt.mine)
			}
			skipped, _, _ := runPhase(t, false, tt.mine)
			if skipped {
				t.Fatalf("device paths %v were skipped by the Strix phase", tt.mine)
			}
		})
	}

	t.Run("mixed host and device change still selects device", func(t *testing.T) {
		mine := []string{"internal/model/llm.go", "internal/amdgpu/strix_validation.go"}
		if !shouldRunStrixValidation(false, mine) {
			t.Fatalf("mixed host+device change %v skipped physical validation", mine)
		}
		if skipped, _, _ := runPhase(t, false, mine); skipped {
			t.Fatalf("mixed host+device change %v was skipped", mine)
		}
	})
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
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, "", false, "", "", "", []string{"docs/README.md"})
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
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, "", false, "", "", "", []string{"docs/README.md", "cmd/fak/new_verb.go"})
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
	origDiscover, origAuthority, origAuthorityValid := discoverStrixTargetFn, newStrixControllerAuthorityFn, strixControllerAuthorityValidFn
	defer func() {
		discoverStrixTargetFn = origDiscover
		newStrixControllerAuthorityFn = origAuthority
		strixControllerAuthorityValidFn = origAuthorityValid
	}()
	newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
		return amdgpu.StrixControllerAuthority{}, nil
	}
	strixControllerAuthorityValidFn = func(amdgpu.StrixControllerAuthority) bool { return true }

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
		"",
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
		"",
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
		"",
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
		"",
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
	err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, &res, recorder, "", false, "", "", "", []string{"internal/amdgpu/strix_receipt.go"})
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
		"",
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
		"",
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
		"",
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
