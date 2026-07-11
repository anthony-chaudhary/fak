package adjudicator

import (
	"context"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// shellDialectAdj builds an adjudicator carrying ONLY the shipped cross-shell-dialect
// rule (the exact sentinel spelling, so isShellDialectArgRule recognizes it and the
// structural path decides), with the SHELL_DIALECT reason it ships with.
func shellDialectAdj(t *testing.T) *Adjudicator {
	t.Helper()
	return New(Policy{
		Allow: map[string]bool{"Bash": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(defaultShellDialectDenyRegex), Reason: abi.ReasonShellDialect,
		}},
	})
}

// TestShellDialectStructuralNoFalsePositive is face (1): a PowerShell cmdlet NAME
// appearing as an argument or inside quotes is NOT a wrong-shell command — it must
// stay ALLOW. A raw `\bGet-Content\b` over the un-tokenized command denied these (a
// false SHELL_DIALECT that under `fak guard -- claude` reads as an agent-chosen
// end_turn); commandLeadsWithPowerShellCmdlet matches a cmdlet ONLY at a stage's
// resolved command-word position, so a quoted or argument occurrence never fires.
func TestShellDialectStructuralNoFalsePositive(t *testing.T) {
	benign := []string{
		`ls -la`,
		`cat README.md`,
		`git log --oneline | head -5`,
		`grep -R foo .`,
		`head -5 file.txt | wc -l`,
		// The cmdlet name is an ARGUMENT, not the command word.
		`grep Select-Object src/foo.go`,
		`rg -n "Where-Object" docs/`,
		// The cmdlet name is QUOTED text handed to echo — a single token, never a command.
		`echo 'Get-Content is a PowerShell cmdlet'`,
		`echo "to list files use Get-ChildItem"`,
		// A real POSIX pipeline whose stage words are all real binaries.
		`find . -name '*.go' | xargs wc -l`,
	}
	a := shellDialectAdj(t)
	for _, cmd := range benign {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign %q: got %v/%s, want Allow (false SHELL_DIALECT)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestShellDialectStructuralCatchesCmdlets is face (2): a PowerShell cmdlet at the
// leading command-word position of ANY pipeline stage — bare, piped, case-folded,
// or laundered one level through sh -c — must DENY with SHELL_DIALECT, because it
// fails `command not found` (exit 127) in the POSIX Bash tool before doing anything.
func TestShellDialectStructuralCatchesCmdlets(t *testing.T) {
	danger := []string{
		`Get-ChildItem`,                          // bare cmdlet
		`Get-ChildItem | Select-Object -First 5`, // leading stage is a cmdlet
		`get-childitem -Recurse`,                 // PowerShell is case-insensitive
		`Where-Object { $_ -gt 5 }`,              // brace block does not hide the command word
		`ForEach-Object { Write-Host $_ }`,
		`cat access.log | Measure-Object -Line`, // a LATER stage leads with a cmdlet
		`Get-Content foo.txt`,
		`sudo Get-Process`,      // sudo unwrap → cmdlet is still the real word
		`sh -c 'Get-ChildItem'`, // laundered one level through sh -c
		`Test-Path ./foo`,
	}
	a := shellDialectAdj(t)
	for _, cmd := range danger {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonShellDialect {
			t.Errorf("danger %q: got %v/%s, want Deny/SHELL_DIALECT (cmdlet-in-Bash slipped)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestCommandLeadsWithPowerShellCmdletCanon locks the returned canonical spelling
// (drives the refusal hint) and the any-stage semantics of the pure matcher.
func TestCommandLeadsWithPowerShellCmdletCanon(t *testing.T) {
	cases := []struct {
		cmd       string
		wantOK    bool
		wantCanon string
	}{
		{`Get-ChildItem | Select-Object -First 5`, true, "Get-ChildItem"},
		{`cat x | Measure-Object`, true, "Measure-Object"}, // first cmdlet stage wins
		{`get-content x`, true, "Get-Content"},             // case-folded → canonical spelling
		{`ls -la`, false, ""},
		{`grep Get-Content file`, false, ""}, // cmdlet as an argument
	}
	for _, tc := range cases {
		canon, ok := commandLeadsWithPowerShellCmdlet(tc.cmd)
		if ok != tc.wantOK || canon != tc.wantCanon {
			t.Errorf("commandLeadsWithPowerShellCmdlet(%q) = (%q,%v), want (%q,%v)",
				tc.cmd, canon, ok, tc.wantCanon, tc.wantOK)
		}
	}
}

// TestShellDialectRuleRecognized guards against sentinel drift: the exact shipped
// spelling must be recognized (structural path) while a Read-tool or differently
// -spelled rule must NOT be (keeps the raw-regex path). If the const and the shipped
// JSON ever diverge, isShellDialectArgRule stops firing and this fails.
func TestShellDialectRuleRecognized(t *testing.T) {
	good := &ArgPredicate{Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
		Re: regexp.MustCompile(defaultShellDialectDenyRegex)}
	if !isShellDialectArgRule(good) {
		t.Fatal("shipped shell-dialect rule not recognized — const/JSON drift")
	}
	wrongTool := &ArgPredicate{Tool: "Read", Arg: "command", Kind: ArgDenyRegex,
		Re: regexp.MustCompile(defaultShellDialectDenyRegex)}
	if isShellDialectArgRule(wrongTool) {
		t.Error("non-Bash rule should not be recognized as the shell-dialect rule")
	}
	wrongSpelling := &ArgPredicate{Tool: "Bash", Arg: "command", Kind: ArgDenyRegex,
		Re: regexp.MustCompile(`(?i)\bGet-Content\b`)}
	if isShellDialectArgRule(wrongSpelling) {
		t.Error("a differently-spelled rule must keep the raw-regex path, not the structural one")
	}
}
