package validate

import (
	"context"
	"io"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// Hooks preserves the fak command's existing in-package test seams while the
// implementation lives in this narrower package. The standalone CLI uses Run.
type Hooks struct {
	Phase               func(context.Context, string)
	WSLLookPath         func(string) (string, error)
	WSLCommand          func(context.Context, ...string) ([]byte, error)
	DiscoverStrixTarget func(context.Context, string) (*amdgpu.StrixTarget, error)
	NewStrixAuthority   func(context.Context, string) (amdgpu.StrixControllerAuthority, error)
	StrixAuthorityValid func(amdgpu.StrixControllerAuthority) bool
	RunStrixValidation  func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error)
	BuildStrixCandidate func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error)
}

var hooksMu sync.Mutex

// RunWithHooks runs validation with caller-provided seams for compatibility
// with the monolithic fak command's existing tests.
func RunWithHooks(stdout, stderr io.Writer, argv []string, hooks Hooks) int {
	hooksMu.Lock()
	defer hooksMu.Unlock()

	oldPhase := validatePhaseHook
	oldLookPath := validateWSLLookPath
	oldWSLCommand := validateWSLCommand
	oldDiscover := discoverStrixTargetFn
	oldAuthority := newStrixControllerAuthorityFn
	oldAuthorityValid := strixControllerAuthorityValidFn
	oldRunStrix := runStrixValidationFn
	oldBuildCandidate := buildStrixCandidateArchiveFn
	validateWSLCapabilities.Lock()
	oldByIdentity := validateWSLCapabilities.byIdentity
	oldIdentityByLauncher := validateWSLCapabilities.identityByLauncher
	validateWSLCapabilities.byIdentity = make(map[string]validateWSLCapabilityVerdict)
	validateWSLCapabilities.identityByLauncher = make(map[string]string)
	validateWSLCapabilities.Unlock()

	if hooks.Phase != nil {
		validatePhaseHook = hooks.Phase
	}
	if hooks.WSLLookPath != nil {
		validateWSLLookPath = hooks.WSLLookPath
	}
	if hooks.WSLCommand != nil {
		validateWSLCommand = hooks.WSLCommand
	}
	if hooks.DiscoverStrixTarget != nil {
		discoverStrixTargetFn = hooks.DiscoverStrixTarget
	}
	if hooks.NewStrixAuthority != nil {
		newStrixControllerAuthorityFn = hooks.NewStrixAuthority
	}
	if hooks.StrixAuthorityValid != nil {
		strixControllerAuthorityValidFn = hooks.StrixAuthorityValid
	}
	if hooks.RunStrixValidation != nil {
		runStrixValidationFn = hooks.RunStrixValidation
	}
	if hooks.BuildStrixCandidate != nil {
		buildStrixCandidateArchiveFn = hooks.BuildStrixCandidate
	}
	defer func() {
		validatePhaseHook = oldPhase
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldWSLCommand
		discoverStrixTargetFn = oldDiscover
		newStrixControllerAuthorityFn = oldAuthority
		strixControllerAuthorityValidFn = oldAuthorityValid
		runStrixValidationFn = oldRunStrix
		buildStrixCandidateArchiveFn = oldBuildCandidate
		validateWSLCapabilities.Lock()
		validateWSLCapabilities.byIdentity = oldByIdentity
		validateWSLCapabilities.identityByLauncher = oldIdentityByLauncher
		validateWSLCapabilities.Unlock()
	}()
	return Run(stdout, stderr, argv)
}
