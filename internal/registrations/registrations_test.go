package registrations_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/registrations" // the defconfig under test: its blank-imports fire every leaf's init()
)

// TestDefconfigEnrollsAgentDojoSteward guards the WIRING, not the leaf. The
// agentdojo leaf self-registers its ASRSteward in init() (proven by the package's
// own register_test.go) — but an init() only fires if SOMETHING imports the leaf,
// and the kernel never imports a leaf. The defconfig (this package) is the one place
// that turns a leaf on with a single blank-import line. For a stretch the agentdojo
// import was missing here, so the shipped build carried the dynamic AgentDojo gate
// DARK: register_test.go stayed green (it imports the leaf directly) while no real
// build ever ran the steward. This test closes that blind spot by asserting the
// effect at the defconfig boundary — import the defconfig, then require the steward
// to be present in the live population. Drop the agentdojo import from
// registrations.go and this goes red; register_test.go would NOT.
func TestDefconfigEnrollsAgentDojoSteward(t *testing.T) {
	const want = "agentdojo-asr-zero"
	for _, s := range abi.Stewards() {
		if s.Name() == want {
			return
		}
	}
	t.Fatalf("the defconfig did not enroll %q in abi.Stewards() — the agentdojo leaf is not blank-imported in registrations.go, so the dynamic AgentDojo gate ships dark", want)
}

// TestDefconfigEnrollsToolLintSteward guards the same wiring for the static tool
// linter: its init() self-registers the "tool-surface-sound" steward, but that only
// fires if the defconfig blank-imports the leaf. Drop the toollint import from
// registrations.go and this goes red, while the toollint package's own tests stay
// green (they import the leaf directly) — the exact blind spot this asserts at the
// defconfig boundary.
func TestDefconfigEnrollsToolLintSteward(t *testing.T) {
	const want = "tool-surface-sound"
	for _, s := range abi.Stewards() {
		if s.Name() == want {
			return
		}
	}
	t.Fatalf("the defconfig did not enroll %q in abi.Stewards() — the toollint leaf is not blank-imported in registrations.go, so the tool-surface invariant ships dark", want)
}
