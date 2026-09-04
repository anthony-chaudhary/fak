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

func tfAdj(t *testing.T, tool, re string) *Adjudicator {
	t.Helper()
	return New(Policy{
		Allow: map[string]bool{tool: true},
		ArgPredicates: []ArgPredicate{{
			Tool: tool, Arg: "command", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(re), Reason: abi.ReasonPolicyBlock,
		}},
	})
}

// TestTerraformSelfRefutingRemedyIsAdmitted is the sharpest case in the whole
// arg_rules table: the refusal's fix text reads "produce the destroy plan for review
// instead: terraform plan -destroy", and `-destroy` matched `\bdestroy\b`, so the
// ONLY recovery the guard offered was blocked by the rule that offered it. An agent
// that followed the instruction verbatim got a second POLICY_BLOCK, which under
// `fak guard -- claude` reads as an agent-chosen end_turn rather than a refusal.
//
// `terraform plan` writes a plan file and changes no infrastructure — that is why
// the policy recommends it — so admitting it costs the floor nothing.
func TestTerraformSelfRefutingRemedyIsAdmitted(t *testing.T) {
	remedies := []string{
		`terraform plan -destroy`,
		`terraform plan -destroy -out=saved.bin`,
		`terraform -chdir=infra plan -destroy`,
		`terraform plan --destroy -no-color`,
	}
	for _, tool := range []string{"Bash", "PowerShell"} {
		re := terraformDestroyDenyRegex
		if tool == "PowerShell" {
			re = terraformDestroyDenyRegexCI
		}
		a := tfAdj(t, tool, re)
		for _, cmd := range remedies {
			v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
			if v.Kind != abi.VerdictAllow {
				t.Errorf("%s %q: got %v/%s, want Allow — this is the refusal's OWN advertised remedy",
					tool, cmd, v.Kind, abi.ReasonName(v.Reason))
			}
		}
	}
}

// TestTerraformMentionsAndReadOnlyAreAdmitted covers the other two false-positive
// shapes: naming the phrase inside a quoted argument (routine on THIS repo, whose
// checkout contains the policy file that ships the rule), and read-only subcommands
// that merely carry the word.
func TestTerraformMentionsAndReadOnlyAreAdmitted(t *testing.T) {
	benign := []string{
		`git commit -m "docs(guard): explain why terraform destroy stays operator-only"`,
		`grep -rn 'terraform destroy' cmd/fak/guard-default-policy.json`,
		`echo "ask the operator to run: terraform destroy -auto-approve"`,
		// Read-only subcommands that name the word.
		`terraform show`,
		`terraform state list  # inspect before a destroy`,
		`terraform validate  # the destroy plan is reviewed separately`,
		`terraform fmt  # tidy the destroy module before review`,
		`terraform -chdir=infra output  # values the destroy would remove`,
	}
	a := tfAdj(t, "Bash", terraformDestroyDenyRegex)
	for _, cmd := range benign {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("benign %q: got %v/%s, want Allow (false POLICY_BLOCK on a mention)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestTerraformRealDestroyStaysDenied is the other face: tearing down live
// infrastructure is the fatal, irreversible class this rule exists for, so the
// relaxation above must not reach any of these. `apply -destroy` is included
// because modern Terraform treats it as exactly equivalent to `destroy`.
func TestTerraformRealDestroyStaysDenied(t *testing.T) {
	danger := []string{
		`terraform destroy`,
		`terraform destroy -auto-approve`,
		`terraform apply -destroy`,
		`terraform apply -destroy -auto-approve`,
		`terraform apply --destroy=true`,
		`terraform -chdir=infra destroy -auto-approve`,
		`terraform.exe destroy`,
		`/usr/local/bin/terraform destroy`,
		// Laundered behind a separator, a pipeline stage, and a nested shell.
		`cd infra && terraform destroy -auto-approve`,
		`echo go; terraform destroy`,
		`sh -c 'terraform destroy -auto-approve'`,
		`bash -c "terraform apply -destroy -auto-approve"`,
	}
	a := tfAdj(t, "Bash", terraformDestroyDenyRegex)
	for _, cmd := range danger {
		v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("danger %q: got %v/%s, want Deny/POLICY_BLOCK (a real teardown slipped)",
				cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestTerraformDestroyPowerShellSurface pins the PowerShell-lexed surface: a
// backslash is a path byte there, not an escape, and a launcher's quoted argument is
// a statement that executes rather than inert text.
func TestTerraformDestroyPowerShellSurface(t *testing.T) {
	a := tfAdj(t, "PowerShell", terraformDestroyDenyRegexCI)
	cases := []struct {
		cmd  string
		want abi.VerdictKind
	}{
		{`Select-String -Pattern 'terraform destroy' -Path cmd\fak\guard-default-policy.json`, abi.VerdictAllow},
		{`terraform plan -destroy`, abi.VerdictAllow},
		{`C:\tools\terraform.exe destroy -auto-approve`, abi.VerdictDeny},
		{`TERRAFORM DESTROY`, abi.VerdictDeny},
		{`powershell -Command "terraform destroy -auto-approve"`, abi.VerdictDeny},
	}
	for _, tc := range cases {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(tc.cmd)))
		if v.Kind != tc.want {
			t.Errorf("%q: got %v, want %v", tc.cmd, v.Kind, tc.want)
		}
	}
}

// TestTerraformSeparatedGlobalFlagValueDoesNotHideTheSubcommand pins the shape that
// slipped past the first command-word decision: a global flag whose value is a
// SEPARATE word (`-chdir infra`) rather than an attached one (`-chdir=infra`). The
// subcommand walk took the first non-flag word, so `infra` was read as the
// subcommand, the real `destroy` behind it was never examined, and a live teardown
// was admitted on every surface. This is the `PowerShell terraform destroy denied`
// case of cmd/fak's TestGuardDefaultPolicyDeniesDangerAllowsBenign.
//
// The benign half is pinned in the same place: stepping over the flag's value must
// not start stepping over ordinary trailing words, or the mention and read-only
// classes this rule deliberately admits would come back as refusals.
func TestTerraformSeparatedGlobalFlagValueDoesNotHideTheSubcommand(t *testing.T) {
	cases := []struct {
		cmd  string
		want abi.VerdictKind
	}{
		{`terraform -chdir infra destroy -auto-approve`, abi.VerdictDeny},
		{`terraform.exe -chdir infra destroy -auto-approve`, abi.VerdictDeny},
		{`terraform -chdir infra apply -destroy -auto-approve`, abi.VerdictDeny},
		{`terraform -chdir infra -input false destroy`, abi.VerdictDeny},
		// The advertised remedy and the read-only subcommands keep working through
		// the same separated spelling.
		{`terraform -chdir infra plan -destroy`, abi.VerdictAllow},
		{`terraform -chdir infra show  # inspect before a destroy`, abi.VerdictAllow},
		// No flag precedes these words, so nothing is stepped over and the walk
		// still stops at the real subcommand.
		{`terraform show  # inspect before a destroy`, abi.VerdictAllow},
		{`terraform state list  # inspect before a destroy`, abi.VerdictAllow},
	}
	for _, tool := range []string{"Bash", "PowerShell"} {
		re := terraformDestroyDenyRegex
		if tool == "PowerShell" {
			re = terraformDestroyDenyRegexCI
		}
		a := tfAdj(t, tool, re)
		for _, tc := range cases {
			v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(tc.cmd)))
			if v.Kind != tc.want {
				t.Errorf("%s %q: got %v/%s, want %v", tool, tc.cmd, v.Kind, abi.ReasonName(v.Reason), tc.want)
			}
		}
	}
}

// TestTerraformUndecidableKeepsTheDeny pins the fail-CLOSED direction: a command the
// walk cannot read must keep the refusal rather than be read as a mention.
func TestTerraformUndecidableKeepsTheDeny(t *testing.T) {
	a := tfAdj(t, "PowerShell", terraformDestroyDenyRegexCI)
	undecidable := []string{
		`Write-Output "terraform destroy`,
		`Write-Output "terraform destroy"; powershell -EncodedCommand dABlAHIAcgA=`,
	}
	for _, cmd := range undecidable {
		v := a.Adjudicate(context.Background(), inlineCall("PowerShell", jsonCmd(cmd)))
		if v.Kind != abi.VerdictDeny {
			t.Errorf("undecidable %q: got %v, want Deny — an unreadable command must fail closed", cmd, v.Kind)
		}
	}
}

// TestTerraformStructuralOnlyRecognisesShippedSpelling: a policy shipping a
// different spelling keeps the raw-regex path, like every other recogniser here.
func TestTerraformStructuralOnlyRecognisesShippedSpelling(t *testing.T) {
	a := tfAdj(t, "Bash", `(?i)terraform.*destroy`)
	v := a.Adjudicate(context.Background(), inlineCall("Bash", jsonCmd(`echo "terraform destroy"`)))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("custom-spelling rule must keep the raw-regex path: got %v, want Deny", v.Kind)
	}
}

// TestTerraformRecogniserMatchesShippedPolicy binds the recogniser to the file it
// claims to recognise. The structural path keys on EXACT regex strings, so a
// reworded rule would silently drop back to the raw-regex path and quietly restore
// the false positives — a regression with no other symptom.
//
// It also asserts the fix text no longer advertises a remedy the rule blocks: the
// recommended `terraform plan -destroy` must now be admitted on every surface that
// ships the rule.
func TestTerraformRecogniserMatchesShippedPolicy(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
			Fix       string `json:"fix"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}
	seen := map[string]bool{}
	fixes := map[string][]string{}
	for _, r := range manifest.ArgRules {
		if !strings.Contains(strings.ToLower(r.DenyRegex), "terraform") {
			continue
		}
		fixes[r.Fix] = append(fixes[r.Fix], r.Tool)
		pr := &ArgPredicate{Tool: r.Tool, Arg: r.Arg, Kind: ArgDenyRegex, Re: regexp.MustCompile(r.DenyRegex)}
		if !isTerraformDestroyArgRule(pr) {
			t.Errorf("tool %q ships terraform deny_regex %q, which the structural recogniser rejects — the rule would fall back to the raw-regex path",
				r.Tool, r.DenyRegex)
			continue
		}
		seen[strings.ToLower(r.Tool)] = true

		// The remedy the fix text names must itself be admitted.
		a := tfAdj(t, r.Tool, r.DenyRegex)
		v := a.Adjudicate(context.Background(), inlineCall(r.Tool, shellCommandArgs(r.Arg, `terraform plan -destroy`)))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("tool %q: the fix text advertises `terraform plan -destroy` but the rule still refuses it (%v/%s)",
				r.Tool, v.Kind, abi.ReasonName(v.Reason))
		}
		if !strings.Contains(r.Fix, "terraform plan -destroy") {
			t.Errorf("tool %q: fix text no longer names the plan remedy this test pins: %q", r.Tool, r.Fix)
		}
	}
	for _, tool := range []string{"bash", "powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command"} {
		if !seen[tool] {
			t.Errorf("shipped policy has no terraform rule for tool %q — the recogniser's surface list is stale", tool)
		}
	}
	// The rule is copied across five surfaces, and the LAST array element has no
	// trailing comma — so a search-and-replace over the other four silently leaves
	// it behind, and one surface then advertises a different boundary than the rest.
	// An agent that hits the rule on shell_command must read the same guidance it
	// would get on Bash.
	if len(fixes) > 1 {
		for fix, tools := range fixes {
			t.Errorf("terraform fix text differs across surfaces %v: %q", tools, fix)
		}
	}
}
