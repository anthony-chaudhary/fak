package witness

import (
	"context"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// clean:<pathspec> corroborates a green-tree ship from `git status --porcelain`:
// empty => clean (Confirmed), any output => dirty (Refuted), git missing => Abstain.
func TestCleanClaim(t *testing.T) {
	ctx := context.Background()
	if got := NewWithRunner((&fakeGit{out: "", code: 0}).run, "").Resolve(ctx, nil, "clean:."); got != abi.WitnessConfirmed {
		t.Fatalf("clean tree => Confirmed, got %v", got)
	}
	if got := NewWithRunner((&fakeGit{out: " M internal/foo.go\n", code: 0}).run, "").Resolve(ctx, nil, "clean:."); got != abi.WitnessRefuted {
		t.Fatalf("dirty tree => Refuted, got %v", got)
	}
	if got := NewWithRunner((&fakeGit{err: exec.ErrNotFound}).run, "").Resolve(ctx, nil, "clean:."); got != abi.WitnessAbstain {
		t.Fatalf("git unavailable => Abstain, got %v", got)
	}
}
