package fleetaccounts

import (
	"os"
	"testing"
)

// TestMain clears the fleet-wide FAK_SESSIONS_PER_ACCOUNT knob before running the
// package tests so the default-cap assertions observe the committed default rather
// than an ambient fleet override. Tests that exercise the knob set it explicitly
// with t.Setenv.
func TestMain(m *testing.M) {
	os.Unsetenv(SessionsPerAccountEnv)
	os.Exit(m.Run())
}
