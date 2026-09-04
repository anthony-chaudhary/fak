package adjudicator

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func runAsAdj(t *testing.T, re string) *Adjudicator {
	t.Helper()
	return New(Policy{
		Allow: map[string]bool{"PowerShell": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "PowerShell", Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(re), Reason: abi.ReasonPolicyBlock,
		}},
	})
}

// TestRunAsStructuralNoFalsePositive is face (1) of #2343: the Windows
// privilege-elevation rule must not fire on the PHRASE `Start-Process … -Verb
// RunAs` when it lives inside a quoted argument. Every case below is routine work
// on this very repo — the file that ships the rule is in this checkout, so reading,
// grepping, documenting, and committing about it are ordinary tasks — and each was
// a POLICY_BLOCK, which under `fak guard -- claude` reads as an agent-chosen
// end_turn stop rather than a refusal.
//
// The last two are the self-refuting case: the refusal's own fix text says to print
// the exact command for the operator, and printing it tripped the same rule, so the
// sanctioned recovery could never be taken (the class already closed for -WhatIf and
// `git clean -n`, docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
func TestRunAsStructuralNoFalsePositive(t *testing.T) {
	benign := []string{
		// Grep the policy file that ships this very rule.
		`Select-String -Pattern 'Start-Process -Verb RunAs' -Path cmd\fak\guard-default-policy.json`,
		`Select-String -Pattern "Start-Process -Verb RunAs" -Path C:\work\fak\cmd\fak\guard-default-policy.json`,
		// Document the refusal.
		`git commit -m "docs(guard): explain why Start-Process -Verb RunAs stays refused"`,
		// Read a file whose contents happen to name the phrase.
		`Get-Content .\docs\fak\policy-guide.md | Select-String 'Start-Process -Verb RunAs'`,
		// The refusal's OWN advertised remedy: hand the command to the operator.
		`Write-Output "Run this elevated yourself: Start-Process -Verb RunAs pwsh"`,
		`echo "operator: Start-Process -Verb RunAs cmd.exe"`,
	}
	a := runAsAdj(t, defaultRunAsDenyRegex)
	for _, cmd := range benign {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign %q: got %v/%s, want Allow (false POLICY_BLOCK on a quoted mention)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestRunAsStructuralCatchesRealElevation is face (2): a real elevated launch stays
// denied. Silent privilege escalation from inside an agent loop is the fatal class
// this rule exists for, so the relaxation above must not reach any of these —
// including the launders a command-word walk has to unwrap.
func TestRunAsStructuralCatchesRealElevation(t *testing.T) {
	danger := []string{
		`Start-Process -Verb RunAs powershell`,
		`Start-Process powershell -Verb RunAs`,
		`Start-Process -FilePath cmd.exe -Verb runas`,
		`START-PROCESS -VERB RUNAS cmd`,
		// The colon-bound PowerShell parameter spelling.
		`Start-Process -FilePath cmd -Verb RunAs -ArgumentList '/c whoami'`,
		// A later statement in the line.
		`Get-Process; Start-Process -Verb RunAs cmd`,
		// A later pipeline stage.
		`Get-ChildItem | Start-Process -Verb RunAs cmd`,
		// The call operator in front of a quoted program name.
		`& 'Start-Process' -Verb RunAs cmd`,
		// A nested host payload: the quoted argument of a launcher executes.
		`powershell -Command "Start-Process -Verb RunAs cmd"`,
		`pwsh -Command 'Start-Process -Verb RunAs cmd'`,
		// A Windows-path program spelling: a backslash is a path byte, not an escape.
		`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Start-Process -Verb RunAs cmd"`,
	}
	a := runAsAdj(t, defaultRunAsDenyRegex)
	for _, cmd := range danger {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("danger %q: got %v/%s, want Deny/POLICY_BLOCK (a real elevation slipped)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestRunAsUndecidableKeepsTheDeny pins the fail-CLOSED direction. The structural
// walk is only ever allowed to SUBTRACT a deny, so anything it cannot read — an
// unterminated quote that would swallow the rest of the line into one inert-looking
// token, or a base64 -EncodedCommand payload — must keep the refusal rather than be
// read as a mention.
func TestRunAsUndecidableKeepsTheDeny(t *testing.T) {
	undecidable := []string{
		`Write-Output "Start-Process -Verb RunAs cmd`,
		`powershell -EncodedCommand UwB0AGEAcgB0AA== ; Start-Process -Verb RunAs cmd`,
		// An inert-looking print that would otherwise be admitted, riding the same
		// line as an opaque base64 payload: the payload could be the elevation, so
		// the mention exemption must not carry the whole line.
		`Write-Output "Start-Process -Verb RunAs cmd"; powershell -EncodedCommand UwB0AGEAcgB0AA==`,
	}
	a := runAsAdj(t, defaultRunAsDenyRegex)
	for _, cmd := range undecidable {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny {
			t.Errorf("undecidable %q: got %v, want Deny — an unreadable command must fail closed", cmd, v.Kind)
		}
	}
}

// TestRunAsStructuralOnlyRecognisesShippedSpelling: a policy that ships a DIFFERENT
// elevation spelling keeps the raw-regex path and gets no structural exemption,
// exactly like the rm_rf / rce_pipe / sudo recognisers.
func TestRunAsStructuralOnlyRecognisesShippedSpelling(t *testing.T) {
	a := runAsAdj(t, `(?i)-Verb\s+RunAs`)
	v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(`echo "use -Verb RunAs here"`)))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("custom-spelling rule must keep the raw-regex path: got %v, want Deny", v.Kind)
	}
}

// TestRunAsRecogniserMatchesShippedPolicy binds the recogniser to the file it claims
// to recognise. The structural path is keyed on an EXACT regex string, so a reworded
// rule in guard-default-policy.json would silently drop back to the raw-regex path
// and quietly restore the false positives — a regression with no other symptom. This
// reads the shipped policy and asserts every RunAs rule in it is one this recogniser
// accepts, on the exact four surfaces that carry the rule.
func TestRunAsRecogniserMatchesShippedPolicy(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
			Reason    string `json:"reason"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}
	seen := map[string]bool{}
	for _, r := range manifest.ArgRules {
		if !strings.Contains(r.DenyRegex, "RunAs") {
			continue
		}
		if r.DenyRegex != defaultRunAsDenyRegex {
			t.Errorf("tool %q ships RunAs deny_regex %q, but the structural recogniser is keyed on %q — the rule would fall back to the raw-regex path",
				r.Tool, r.DenyRegex, defaultRunAsDenyRegex)
			continue
		}
		pr := &ArgPredicate{Tool: r.Tool, Arg: r.Arg, Kind: ArgDenyRegex, Re: regexp.MustCompile(r.DenyRegex)}
		if !isRunAsArgRule(pr) {
			t.Errorf("isRunAsArgRule rejects the shipped rule for tool %q arg %q", r.Tool, r.Arg)
		}
		seen[strings.ToLower(r.Tool)] = true
	}
	for _, tool := range []string{"powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command"} {
		if !seen[tool] {
			t.Errorf("shipped policy has no RunAs rule for tool %q — the recogniser's surface list is stale", tool)
		}
	}
}
