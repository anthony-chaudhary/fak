package agentdojo

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // page-out backend the full stack needs
)

// TestInitEnrollsASRStewardAlwaysOn — the load-bearing effect of register.go:
// importing this leaf must enroll the ASRSteward in the always-on population
// (abi.Stewards()), so a steward sweep finds it without any caller naming the
// package. Before register.go the steward self-registered nowhere — it was
// reachable only from tests and cmd/agentdojoredteam — so this is what turns the
// dynamic battery into a wired gate the way the static poison.json check it
// replaced once was.
func TestInitEnrollsASRStewardAlwaysOn(t *testing.T) {
	const want = "agentdojo-asr-zero"
	var found abi.Steward
	for _, s := range abi.Stewards() {
		if s.Name() == want {
			found = s
			break
		}
	}
	if found == nil {
		t.Fatalf("init() did not enroll %q in abi.Stewards() — register.go's RegisterSteward did not fire", want)
	}

	// The enrolled steward must be the real ASR gate: it abstains on the shipped
	// healthy stack (registering a steward that fired on a green build would be a
	// false alarm, not a regression gate).
	violated, witness := found.Check(context.Background())
	if violated {
		t.Fatalf("the enrolled steward fired on the shipped stack: %s", witness)
	}
	if witness != "" {
		t.Fatalf("an abstaining steward must carry no witness, got %q", witness)
	}
}
