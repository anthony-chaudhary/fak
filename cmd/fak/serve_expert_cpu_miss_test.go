package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
func TestServeRefusesInvalidExpertCPUMiss(t *testing.T) {
	for raw, wantErr := range map[string]bool{"": false, "off": false, "exact": false, "EXACT": false, "bogus": true} {
		err := validateServeExpertCPUMiss(func(k string) string {
			if k == model.ExpertCPUMissEnv {
				return raw
			}
			return ""
		})
		if wantErr != errors.Is(err, model.ErrExpertCPUMissFlag) || (!wantErr && err != nil) {
			t.Fatalf("%s=%q: err=%v, refuse=%t", model.ExpertCPUMissEnv, raw, err, wantErr)
		}
	}
}
