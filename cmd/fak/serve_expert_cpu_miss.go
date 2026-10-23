package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// validateServeExpertCPUMiss refuses an unparseable FAK_EXPERT_CPU_MISS before
// any model loads, so a misspelled knob never serves under a guessed mode.
func validateServeExpertCPUMiss(getenv func(string) string) error {
	raw := getenv(model.ExpertCPUMissEnv)
	if _, ok := model.ParseExpertCPUMissMode(raw); !ok {
		return fmt.Errorf("%w: %q, want unset, \"off\", or \"exact\"", model.ErrExpertCPUMissFlag, raw)
	}
	return nil
}
