package modelroute

import "github.com/anthony-chaudhary/fak/pkg/routepolicy"

// The drafter/verifier spec-decode bridge (Plan.SpecRoles, IsSpecEnsemble,
// SpecAccept) lives in pkg/routepolicy.
type (
	SpecRoles  = routepolicy.SpecRoles
	SpecAccept = routepolicy.SpecAccept
	AcceptFunc = routepolicy.AcceptFunc
)
