package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func resolveAgentOutputStyle(raw string) (syspromptmmu.StyleReadout, error) {
	style := syspromptmmu.DescribeStyle(raw)
	if !style.Known {
		return style, fmt.Errorf("agent: invalid --output-style %q; supported: %s", raw, strings.Join(syspromptmmu.StyleNames(), ", "))
	}
	return style, nil
}

func applyAgentOutputStyle(style syspromptmmu.StyleReadout) (func(), error) {
	previous, hadPrevious := os.LookupEnv(syspromptmmu.StyleEnvVar)
	if err := os.Setenv(syspromptmmu.StyleEnvVar, style.Style); err != nil {
		return nil, err
	}
	return func() {
		if hadPrevious {
			_ = os.Setenv(syspromptmmu.StyleEnvVar, previous)
		} else {
			_ = os.Unsetenv(syspromptmmu.StyleEnvVar)
		}
	}, nil
}
