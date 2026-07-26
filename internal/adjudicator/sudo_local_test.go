package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestSudoStructuralNoFalsePositive is face (1): the privilege-escalation rule
// must NOT fire on the literal text `sudo` when it lives inside a quoted
// echo/grep argument, nor — the dominant friction in real remote-GPU bring-up
// trajectories — on a REMOTE escalation carried as an ssh argument, where the
// local command word is ssh and the escalation is governed by the remote host's
// own controls. A regex over the raw command string denied these (a false
// POLICY_BLOCK that reads as an agent-chosen end_turn stop under
// `fak guard -- claude`).
func TestSudoStructuralNoFalsePositive(t *testing.T) {
	benign := []string{
		// The witnessed remote-GPU-bring-up shape: remote escalation on a GPU box.
		`ssh gpu-box 'sudo systemctl restart tritonserver'`,
		`ssh -i ~/.ssh/id_ed25519 gpu-box 'sudo nvidia-smi -pm 1'`,
		// Remote command unquoted: sudo is an ssh ARGUMENT, not a command word.
		`ssh gpu-box sudo reboot`,
		// Compound: local scp then a remote sudo move.
		`scp pkg.deb gpu-box:/tmp/ && ssh gpu-box 'sudo dpkg -i /tmp/pkg.deb'`,
		// Quoted text is never a command word.
		`echo 'sudo make install'`,
		`grep -rn 'sudo' docs/`,
		`echo "the fix needs sudo later"`,
	}
	a := rcePipeAdj(t, defaultSudoDenyRegex)
	for _, cmd := range benign {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign %q: got %v/%s, want Allow (false POLICY_BLOCK)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestSudoStructuralCatchesLaunder is face (2): the rule must fire on a real
// LOCAL escalation at a command boundary regardless of wrapping — env-assigns,
// env, transparent wrappers, sh -c, command substitution, an absolute path —
// plus the doas spelling the raw regex never caught.
func TestSudoStructuralCatchesLaunder(t *testing.T) {
	danger := []string{
		`sudo rm f`,
		`sudo -u root make install`,
		`cd /etc && sudo make install`,
		`FOO=1 sudo id`,
		`env sudo id`,
		`nohup sudo id`,
		`time sudo id`,
		`sh -c 'sudo id'`,
		`echo $(sudo id)`,
		`/usr/bin/sudo id`,
		`command sudo id`,
		`doas id`, // drop-in escalation the raw regex missed
	}
	a := rcePipeAdj(t, defaultSudoDenyRegex)
	for _, cmd := range danger {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("danger %q: got %v/%s, want Deny/POLICY_BLOCK (laundered escalation slipped)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestSudoStructuralOnlyRecognisesShippedSpelling: a policy that ships a
// DIFFERENT sudo spelling keeps the raw-regex path (no structural exemption),
// exactly like the rm_rf and rce_pipe recognisers.
func TestSudoStructuralOnlyRecognisesShippedSpelling(t *testing.T) {
	a := rcePipeAdj(t, `\bsudo\s+rm\b`)
	v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(`echo 'sudo rm x'`)))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("custom-spelling rule must keep the raw-regex path: got %v, want Deny", v.Kind)
	}
}
