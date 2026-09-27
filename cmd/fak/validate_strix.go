package main

import "github.com/anthony-chaudhary/fak/internal/amdgpu"

var (
	discoverStrixTargetFn           = amdgpu.DiscoverStrixTarget
	newStrixControllerAuthorityFn   = amdgpu.NewStrixControllerAuthority
	strixControllerAuthorityValidFn = func(authority amdgpu.StrixControllerAuthority) bool {
		return authority.Epoch() != "" &&
			authority.ObservedRevision() != "" &&
			authority.ExecutableSHA256() != "" &&
			authority.ExecutableBytes() > 0 &&
			!authority.AdmittedAt().IsZero()
	}
	runStrixValidationFn         = amdgpu.RunStrixValidation
	buildStrixCandidateArchiveFn = amdgpu.BuildStrixCandidateArchive
)
