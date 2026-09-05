package policy

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// Profile represents a declarative permission profile.
type Profile string

const (
	ProfileStandard Profile = "standard"
	ProfileDev      Profile = "dev"
	ProfileProd     Profile = "prod"
	ProfileStrict   Profile = "strict"
	ProfileHardened Profile = "hardened"
	ProfileAudit    Profile = "audit"
)

// ParseProfile parses "standard", "dev", "prod", "strict", "hardened", "audit" (case-insensitive, whitespace-trimmed).
// Returns an error if non-empty and not a known profile.
func ParseProfile(s string) (Profile, error) {
	trimmed := strings.ToLower(strings.TrimSpace(s))
	switch Profile(trimmed) {
	case "":
		return "", nil
	case ProfileStandard:
		return ProfileStandard, nil
	case ProfileDev:
		return ProfileDev, nil
	case ProfileProd:
		return ProfileProd, nil
	case ProfileStrict, ProfileHardened:
		return ProfileStrict, nil
	case ProfileAudit:
		return ProfileAudit, nil
	default:
		return "", fmt.Errorf("unknown permission profile %q (want standard|dev|prod|strict|audit)", s)
	}
}

// ValidateProfile checks if p is a valid profile. An empty profile is allowed (noop).
func ValidateProfile(p Profile) error {
	switch p {
	case "", ProfileStandard, ProfileDev, ProfileProd, ProfileStrict, ProfileHardened, ProfileAudit:
		return nil
	default:
		return fmt.Errorf("unknown permission profile %q (want standard|dev|prod|strict|audit)", p)
	}
}

// Apply applies the profile configuration to rt.
func (p Profile) Apply(rt *Runtime) {
	if rt == nil {
		return
	}
	switch p {
	case ProfileStandard, ProfileDev:
		rt.Adjudicator.Posture = adjudicator.PostureDefaultOpen
		rt.StrictGatedSinks = false
		rt.GatedSinks = nil
		rt.PolicyContext.Posture = abi.PostureDefaultOpen
		rt.PolicyContext.Profile = string(p)
	case ProfileProd, ProfileStrict, ProfileHardened:
		rt.Adjudicator.Posture = adjudicator.PostureFailClosed
		rt.StrictGatedSinks = true
		rt.GatedSinks = map[string]bool{"egress": true, "destructive": true, "exec": true}
		rt.PolicyContext.Posture = abi.PostureFailClosed
		rt.PolicyContext.Profile = string(p)
	case ProfileAudit:
		rt.Adjudicator.Posture = adjudicator.PostureAdmitAndLog
		rt.StrictGatedSinks = false
		rt.PolicyContext.Posture = abi.PostureAdmitAndLog
		rt.PolicyContext.Profile = string(ProfileAudit)
	}
}

// RuntimeForProfile validates the given permission profile and derives a fresh Runtime policy configured with matching posture and sink gates.
func RuntimeForProfile(p Profile) (Runtime, error) {
	if err := ValidateProfile(p); err != nil {
		return Runtime{}, err
	}
	var rt Runtime
	p.Apply(&rt)
	return rt, nil
}
