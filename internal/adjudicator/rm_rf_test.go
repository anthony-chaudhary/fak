package adjudicator

import (
	"context"
	"regexp"
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

func TestRecursiveDeleteScratchContainment(t *testing.T) {
	ws := "/work/fak"
	scratch := []string{"/tmp/claude"}
	remove := "r" + "m"
	cases := []struct {
		name string
		cmd  string
		deny bool
	}{
		{"session child", remove + " -rf /tmp/claude/session-123/clone", false},
		{"relative session child", remove + " -rf ../../tmp/claude/session-123/clone", false},
		{"multiple session children", remove + " -rf /tmp/claude/a /tmp/claude/b", false},
		{"scratch root itself", remove + " -rf /tmp/claude", true},
		{"sibling prefix", remove + " -rf /tmp/claude-evil/clone", true},
		{"mixed inside outside", remove + " -rf /tmp/claude/a /work/fak/real", true},
		{"workspace path", remove + " -rf /work/fak/_scratch/clone", true},
		{"variable target", remove + " -rf $TARGET", true},
		{"glob target", remove + " -rf /tmp/claude/*", true},
		{"no declared root", remove + " -rf /tmp/claude/session-123", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			roots := scratch
			if tc.name == "no declared root" {
				roots = nil
			}
			if got := commandHasUnsafeRecursiveForcedDelete(tc.cmd, ws, roots); got != tc.deny {
				t.Fatalf("unsafe=%v want %v", got, tc.deny)
			}
		})
	}
}

func TestRecursiveDeleteScratchCarveoutEndToEnd(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "/tmp/claude")
	remove := "r" + "m"
	a := rcePipeAdj(t, defaultRmRfDenyRegex)
	allowed := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(remove+" -rf /tmp/claude/session-123/clone")))
	if allowed.Kind == abi.VerdictDeny && allowed.Reason == abi.ReasonPolicyBlock {
		t.Fatalf("contained session cleanup remained terminal POLICY_BLOCK: %v/%s", allowed.Kind, abi.ReasonName(allowed.Reason))
	}
	denied := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(remove+" -rf /tmp/claude-evil/clone")))
	if denied.Kind != abi.VerdictDeny || denied.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("sibling escape = %v/%s, want Deny/POLICY_BLOCK", denied.Kind, abi.ReasonName(denied.Reason))
	}
}

func powerShellDeleteAdj(t *testing.T, tool string) *Adjudicator {
	t.Helper()
	return New(Policy{
		Allow: map[string]bool{tool: true},
		ArgPredicates: []ArgPredicate{{
			Tool: tool, Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(defaultPSDeleteDenyRegex), Reason: abi.ReasonPolicyBlock,
		}},
	})
}

func psDeleteCommand(suffix string) string {
	return "Remove" + "-Item" + suffix
}

func TestPowerShellDeleteRuleIsStructuralForSupportedTools(t *testing.T) {
	tools := []string{"PowerShell", "shell_command", "functions.shell_command"}
	for _, tool := range tools {
		t.Run(tool, func(t *testing.T) {
			a := powerShellDeleteAdj(t, tool)
			danger := []string{
				psDeleteCommand(` -Path C:\scratch -Force`),
				psDeleteCommand(` -LiteralPath C:\scratch -Recurse`),
				psDeleteCommand(` C:\scratch -Confirm:$false -Force`),
			}
			for _, cmd := range danger {
				v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
				if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Errorf("danger %q: got %v/%s, want Deny/POLICY_BLOCK", cmd, v.Kind, abi.ReasonName(v.Reason))
				}
			}

			benign := []string{
				`Write-Output '` + psDeleteCommand(` -Recurse C:\scratch`) + `'`,
				`Write-Output "cleanup docs mention ` + psDeleteCommand(` -Force`) + `"`,
				`$example = '` + psDeleteCommand(` -Recurse C:\scratch`) + `'`,
				`Get-Help ` + psDeleteCommand(` -Force`),
			}
			for _, cmd := range benign {
				v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
				if v.Kind != abi.VerdictAllow {
					t.Errorf("benign %q: got %v/%s, want Allow", cmd, v.Kind, abi.ReasonName(v.Reason))
				}
			}
		})
	}
}

func TestPowerShellDeleteRuleRequiresExactBuiltinRegex(t *testing.T) {
	for _, tool := range []string{"PowerShell", "shell_command", "functions.shell_command"} {
		pr := &ArgPredicate{Tool: tool, Arg: "command", Re: regexp.MustCompile(defaultPSDeleteDenyRegex)}
		if !isRmRfArgRule(pr) {
			t.Errorf("%s built-in delete regex was not structurally classified", tool)
		}
	}
	pr := &ArgPredicate{Tool: "PowerShell", Arg: "command", Re: regexp.MustCompile(`(?i)custom-delete-pattern`)}
	if isRmRfArgRule(pr) {
		t.Fatal("custom PowerShell policy regex must keep raw-regex semantics")
	}
}
