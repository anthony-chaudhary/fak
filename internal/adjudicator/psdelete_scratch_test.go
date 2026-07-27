package adjudicator

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// winScratchRoot spells a declared scratchpad root the way it actually arrives on
// Windows — backslash-separated — which is the only host the PowerShell delete rule
// fires on. The drive letter is dropped where the platform's path-list separator is
// ':' (POSIX CI), because a `C:` prefix inside FAK_GUARD_SCRATCHPAD_ROOTS would be
// split there into two bogus roots. The defect under test is the BACKSLASH, not the
// drive letter: both spellings lose every separator to POSIX lexing identically.
// TestWindowsScratchDeleteProverIsSubtractive pins the full drive-letter spelling
// directly against the prover, where no env parsing is involved.
func winScratchRoot() string {
	if os.PathListSeparator == ';' {
		return `C:\agent-scratch\claude`
	}
	return `\agent-scratch\claude`
}

// winOutsidePath is real work: a path that is absolute on the same host and outside
// every declared scratchpad root.
func winOutsidePath() string {
	if os.PathListSeparator == ';' {
		return `C:\work\fak\internal`
	}
	return `\work\fak\internal`
}

// TestWindowsScratchDeleteIsAdmitted is the defect: the scratchpad carve-out that
// commandHasUnsafeRecursiveForcedDelete already grants was UNREACHABLE on Windows,
// because containment was checked against POSIX-lexed tokens, where a backslash is
// an escape. Every Windows path arrived with its separators eaten, resolved under no
// root, and the delete stayed a POLICY_BLOCK — so cleaning up one's own per-session
// scratch directory, which the policy had already decided was safe, was refused.
// Under `fak guard -- claude` that refusal reads as an agent-chosen end_turn.
func TestWindowsScratchDeleteIsAdmitted(t *testing.T) {
	root := winScratchRoot()
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", root)
	for _, tool := range []string{"PowerShell", "shell_command", "functions.shell_command"} {
		t.Run(tool, func(t *testing.T) {
			a := powerShellDeleteAdj(t, tool)
			admitted := []string{
				psDeleteCommand(` ` + root + `\sess\scratch\junk -Recurse -Force`),
				psDeleteCommand(` -LiteralPath ` + root + `\sess\clone -Recurse`),
				psDeleteCommand(` -Path ` + root + `\sess\out -Recurse -Force -Confirm:$false`),
			}
			for _, cmd := range admitted {
				v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
				if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonPolicyBlock {
					t.Errorf("contained scratch cleanup %q stayed a terminal POLICY_BLOCK", cmd)
				}
			}
		})
	}
}

// TestWindowsScratchDeleteEscapesStayDenied is the other face. The relaxation proves
// containment; it must not admit anything that leaves the declared root, and an
// unbounded or unreadable target set must prove nothing at all.
func TestWindowsScratchDeleteEscapesStayDenied(t *testing.T) {
	root := winScratchRoot()
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", root)
	a := powerShellDeleteAdj(t, "PowerShell")
	denied := []string{
		// Real work, outside the scratch root.
		psDeleteCommand(` ` + winOutsidePath() + ` -Recurse -Force`),
		// A sibling that merely shares the root's prefix.
		psDeleteCommand(` ` + root + `-evil\x -Recurse`),
		// Walking back out of the root.
		psDeleteCommand(` ` + root + `\..\..\Roaming -Recurse -Force`),
		// The scratch root itself is too broad a target for the carve-out.
		psDeleteCommand(` ` + root + ` -Recurse -Force`),
		// A glob is an unbounded target set.
		psDeleteCommand(` ` + root + `\* -Recurse -Force`),
		// A list operand: psSegments treats PowerShell's `,` list separator as a
		// statement boundary, so the tail would be dropped from the target set —
		// the prover must decline rather than prove a partial set.
		psDeleteCommand(` -Path ` + root + `\a,` + winOutsidePath() + ` -Recurse`),
		// One contained target does not license an uncontained sibling statement.
		psDeleteCommand(` `+root+`\a -Recurse`) + `; ` + psDeleteCommand(` `+winOutsidePath()+` -Recurse`),
	}
	for _, cmd := range denied {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("escape %q: got %v/%s, want Deny/POLICY_BLOCK", cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestWindowsScratchDeleteNeedsADeclaredRoot pins that the relaxation is driven by
// the operator-declared root and nothing else: with no root declared, the shipped
// refusal is unchanged.
func TestWindowsScratchDeleteNeedsADeclaredRoot(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "")
	a := powerShellDeleteAdj(t, "PowerShell")
	cmd := psDeleteCommand(` ` + winScratchRoot() + `\sess\junk -Recurse -Force`)
	v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("with no declared scratch root: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestForwardSlashScratchDeleteStillAdmitted pins the route that already worked —
// the same delete spelled with forward slashes, which POSIX lexing preserves — so
// the new prover is an ADDITIONAL way to prove containment, not a replacement.
func TestForwardSlashScratchDeleteStillAdmitted(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "/tmp/claude")
	a := powerShellDeleteAdj(t, "PowerShell")
	cmd := psDeleteCommand(` -LiteralPath /tmp/claude/sess/clone -Recurse -Force`)
	v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
	if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonPolicyBlock {
		t.Fatalf("the pre-existing forward-slash carve-out regressed: %q stayed POLICY_BLOCK", cmd)
	}
	escape := psDeleteCommand(` -LiteralPath /tmp/claude-evil/clone -Recurse -Force`)
	if v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(escape))); v.Kind != abi.VerdictDeny {
		t.Fatalf("sibling escape %q = %v, want Deny", escape, v.Kind)
	}
}

// TestWindowsScratchDeleteProverIsSubtractive pins the direction of the change at
// the helper level, with the full drive-letter spelling and injected roots: the
// prover only ever reports "this is provably confined", so a command it cannot read,
// or one that names no delete at all, proves nothing and the shipped deny stands.
func TestWindowsScratchDeleteProverIsSubtractive(t *testing.T) {
	const (
		raw   = `C:\agent-scratch\claude`
		canon = `C:/agent-scratch/claude`
	)
	scratch := []string{canon}
	cases := []struct {
		cmd   string
		prove bool
	}{
		{psDeleteCommand(` ` + raw + `\sess\junk -Recurse -Force`), true},
		{psDeleteCommand(` -LiteralPath ` + raw + `\sess\clone -Recurse`), true},
		{psDeleteCommand(` C:\work\fak -Recurse -Force`), false},
		{psDeleteCommand(` ` + raw + ` -Recurse -Force`), false},   // the root itself
		{psDeleteCommand(` ` + raw + `\* -Recurse -Force`), false}, // unbounded
		{`Get-ChildItem ` + raw, false},                            // no delete statement at all
		{psDeleteCommand(` "` + raw + `\a -Recurse`), false},       // unterminated quote
		{psDeleteCommand(` -Path ` + raw + `\a,` + raw + `\b -Recurse`), false},
	}
	for _, tc := range cases {
		if got := psRemoveItemAllTargetsInScratch(tc.cmd, canon, scratch); got != tc.prove {
			t.Errorf("prover(%q) = %v, want %v", tc.cmd, got, tc.prove)
		}
	}
}
