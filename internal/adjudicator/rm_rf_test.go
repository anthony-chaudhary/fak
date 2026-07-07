package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRmRfStructuralNoFalsePositive is face (1): the recursive/forced-delete rule
// must NOT fire on the literal text `rm -rf` when it lives inside a quoted
// echo/grep argument, nor on a command whose real word is not `rm`. A regex over
// the raw command string denied these (a false POLICY_BLOCK that reads as an
// agent-chosen end_turn stop under `fak guard -- claude`); the structural matcher
// tokenizes first, so quoted words and `git rm` / `docker run --rm` are never the
// `rm` command word.
func TestRmRfStructuralNoFalsePositive(t *testing.T) {
	benign := []string{
		`echo 'rm -rf /old'`,
		`grep -r 'rm -rf' docs/`,
		`echo "cleanup used to be: rm -rf build"`,
		// The command word is docker, not rm (`--rm` is a docker flag).
		`docker run --rm alpine echo hi`,
		// git rm is exempt by decision: it removes only tracked, git-recoverable files.
		`git rm -rf .`,
		`git rm --cached -r foo`,
		// A wrapper whose real command is echo, not rm (bare `rm` itself is separately
		// confirm-gated by the reversibility rung, so it can't appear as an Allow here).
		`time echo 'rm -rf x'`,
	}
	a := rcePipeAdj(t, defaultRmRfDenyRegex)
	for _, cmd := range benign {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign %q: got %v/%s, want Allow (false POLICY_BLOCK)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestRmRfStructuralCatchesLaunder is face (2): the rule must fire on a real
// recursive OR forced rm at a command boundary regardless of flag order, flag
// spelling, or a wrapping sh -c / sudo / xargs / nohup — the launders the raw
// regex (which inspected only the first flag cluster after `rm`) let slip.
func TestRmRfStructuralCatchesLaunder(t *testing.T) {
	danger := []string{
		`rm -rf /`,
		`rm -fr x`,
		`rm -f x`,                             // force-only: recursive OR force
		`rm -i -rf x`,                         // interactive first, then -rf (flag reorder)
		`rm --recursive --force x`,            // long-flag spelling
		`rm -v -r -f x`,                       // split short flags in later clusters
		`rm -R dir`,                           // capital-R recursive
		`sudo rm -rf /`,                       // sudo unwrap
		`sh -c 'rm -rf /'`,                    // laundered one level deep through sh -c
		`nohup rm -rf x`,                      // transparent wrapper
		`find . -name '*.tmp' | xargs rm -rf`, // xargs wrapper after a pipe
	}
	a := rcePipeAdj(t, defaultRmRfDenyRegex)
	for _, cmd := range danger {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("danger %q: got %v/%s, want Deny/POLICY_BLOCK (laundered rm -rf slipped)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}
