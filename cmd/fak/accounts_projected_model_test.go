package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// TestProjectedDefaultModelMatchesLaunch binds the persisted-settings default to the launch
// default. The whole point of #3091 is that the two must agree: the account-switched launcher
// pins defaultLaunchModel via --model, and the settings projection seeds the SAME model into a
// fresh seat so a bare `claude --resume` / direct launch under that seat (which pass no --model)
// inherit the same primary. If someone bumps one default, this fails until they bump the other.
func TestProjectedDefaultModelMatchesLaunch(t *testing.T) {
	if accounts.ProjectedDefaultModel != defaultLaunchModel {
		t.Errorf("accounts.ProjectedDefaultModel = %q, defaultLaunchModel = %q; the persisted settings default must equal the launch default (#3091)",
			accounts.ProjectedDefaultModel, defaultLaunchModel)
	}
}
