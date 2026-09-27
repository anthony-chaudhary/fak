package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

type testStrixControllerAuthorityRefusal struct {
	code string
}

func (r *testStrixControllerAuthorityRefusal) Error() string {
	return r.code + ": controller authority refused"
}
func (r *testStrixControllerAuthorityRefusal) Code() string        { return r.code }
func (r *testStrixControllerAuthorityRefusal) Recovery() string    { return "" }
func (r *testStrixControllerAuthorityRefusal) RetryCleanup() error { return nil }

// TestValidateStrixRequiresControllerAuthorityBeforeDiscovery drives the live
// entrypoint; the phase-level table lives with the implementation in
// internal/validate/strix_test.go.
func TestValidateStrixRequiresControllerAuthorityBeforeDiscovery(t *testing.T) {
	t.Run("runValidate forwards operator-selected root", func(t *testing.T) {
		repo, git := seedGitFixtureRepo(t)
		commitFiles(t, repo, git, "seed", map[string]string{
			"go.mod":                         cleanGoMod,
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
