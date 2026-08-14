package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func resolveAgentWorkProfile(raw string) (syspromptmmu.WorkProfileReadout, error) {
	profile := syspromptmmu.DescribeWorkProfile(raw)
	if !profile.Known {
		return profile, fmt.Errorf("agent: invalid --work-profile %q; supported: %s", raw, strings.Join(syspromptmmu.WorkProfileNames(), ", "))
	}
	return profile, nil
}

func applyAgentWorkProfile(profile syspromptmmu.WorkProfileReadout) (func(), error) {
	previous, hadPrevious := os.LookupEnv(syspromptmmu.WorkProfileEnvVar)
	if err := os.Setenv(syspromptmmu.WorkProfileEnvVar, profile.Profile); err != nil {
		return nil, err
	}
	return func() {
		if hadPrevious {
			_ = os.Setenv(syspromptmmu.WorkProfileEnvVar, previous)
		} else {
			_ = os.Unsetenv(syspromptmmu.WorkProfileEnvVar)
		}
	}, nil
}
