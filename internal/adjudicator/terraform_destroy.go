package adjudicator

import "strings"

// The two shipped spellings of the TERRAFORM destroy deny_regex.
// cmd/fak/guard-default-policy.json ships the rule four times: once for the Bash
// tool (case-sensitive) and once each for the PowerShell, shell_command, and
// functions.shell_command surfaces (case-insensitive). The rule is RECOGNISED by
// these exact strings and then decided STRUCTURALLY, like its neighbours rm_rf
// (#4983), the RCE download-pipe (#1465), the out-of-tree write family, the
// cross-shell-dialect rule (#3941), `sudo` (sudo_local.go), and the Windows
// elevation rule (runas_elevation.go). A policy that ships a different spelling is
// unaffected and keeps the raw-regex path.
//
// The raw regex is `terraform` … anything-not-a-separator … `destroy`, so it fires
// on the WORD `destroy` appearing anywhere after the word `terraform` on a line.
// That over-refuses three ways, all of them routine work:
//
//   - SELF-REFUTING REMEDY. The refusal's own fix text says "produce the destroy
//     plan for review instead: terraform plan -destroy". `-destroy` is preceded by
//     `-`, which is a word boundary, so the advertised recovery matches the rule
//     that advertised it. The redirect could never be taken. That is exactly the
//     class this package already closed for -WhatIf, `git clean -n`, `git push -n`
//     and `git rebase --abort` (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
//     `terraform plan` is read-only: it writes a plan file and changes no
//     infrastructure, which is precisely why the policy recommends it.
//
//   - QUOTED MENTIONS. Documenting or grepping the rule — `git commit -m "docs:
//     why terraform destroy is operator-only"`, or `grep -rn 'terraform destroy'
//     cmd/fak/guard-default-policy.json` — tripped it. The policy file that ships
//     the rule lives in this checkout, so reading and citing it is ordinary work.
//
//   - READ-ONLY SUBCOMMANDS. `terraform show  # inspect before a destroy` reads
//     state and mutates nothing, but carries both words on one line.
//
// A POLICY_BLOCK under `fak guard -- claude` reads as an agent-chosen end_turn
// rather than a refusal, so each of these silently ended a turn.
const (
	terraformDestroyDenyRegex   = `\bterraform(?:\.exe)?\b[^|;&\n]*\bdestroy\b`
	terraformDestroyDenyRegexCI = `(?i)\bterraform(?:\.exe)?\b[^|;&\n]*\bdestroy\b`
)

// isTerraformDestroyArgRule reports whether pr is one of the shipped terraform
// destroy rules on a shell command arg, on one of the four surfaces that ship it.
func isTerraformDestroyArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "bash", "powershell", "shell_command", "functions.shell_command":
	default:
		return false
	}
	switch pr.Re.String() {
	case terraformDestroyDenyRegex, terraformDestroyDenyRegexCI:
		return true
	default:
		return false
	}
}

// commandAppliesTerraformDestroy reports whether cmd actually APPLIES a terraform
// destroy — `terraform` resolved at a statement's command-word position, whose
// SUBCOMMAND tears infrastructure down — as opposed to merely naming the word.
//
// It is used SUBTRACTIVELY: decide.go consults it only once the raw regex has
// already matched, and a false result downgrades that match to an admit. So it can
// never introduce a NEW deny, and every ambiguity resolves to true (keep the deny).
//
// The rule ships on both a POSIX surface (Bash) and a PowerShell one, and
// `shell_command` is either depending on the host, so the command is decided under
// BOTH dialects and the deny is kept if EITHER reads a real destroy. Deciding under
// only one lexer would let the other dialect's quoting hide an invocation.
//
// Known residue, deliberately NOT closed here because closing it would be a
// TIGHTENING rather than a relaxation: a destroy plan saved to a file and applied
// as `terraform apply saved.bin` names no `destroy` anywhere, so the raw regex never
// flagged it and this walk is never consulted for it. That gap predates this change.
func commandAppliesTerraformDestroy(cmd string) bool {
	return posixTerraformDestroys(cmd) || psTerraformDestroys(cmd, 0)
}

// posixTerraformDestroys decides the command under POSIX lexing, reusing the RCE
// walker's source expansion so `sh -c '…'`, `$(…)` and backtick payloads are
// unwrapped and decided as statements in their own right.
func posixTerraformDestroys(cmd string) bool {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rceCommandWord(seg.argv)
			if i < 0 || rceProgramBasename(seg.argv[i]) != "terraform" {
				continue
			}
			if terraformArgvDestroys(seg.argv[i+1:]) {
				return true
			}
		}
	}
	return false
}

// psTerraformDestroys decides the command under PowerShell lexing (backtick escape,
// backslash as a path byte), recursing into a launcher's quoted argument because
// that argument is a statement that executes rather than inert text (#2752).
func psTerraformDestroys(src string, depth int) bool {
	if depth > maxRCEShellSourceDepth {
		return true // too deeply nested to decide: keep the deny
	}
	segs, ok := psSegments(src)
	if !ok {
		return true // unterminated quote: undecidable, keep the deny
	}
	for _, seg := range segs {
		head, rest, ok := psCommandWord(seg)
		if !ok {
			continue
		}
		if head == "terraform" && terraformArgvDestroys(psTokenTexts(rest)) {
			return true
		}
		if !psLivePayloadHeads[head] {
			continue
		}
		for _, tok := range rest {
			if psEncodedPayloadFlag(tok.text) {
				return true // base64 payload we cannot read: keep the deny
			}
		}
		for _, tok := range rest {
			if tok.quoted && psTerraformDestroys(tok.text, depth+1) {
				return true
			}
		}
	}
	return false
}

func psTokenTexts(toks []psToken) []string {
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		out = append(out, t.text)
	}
	return out
}

// terraformArgvDestroys decides a terraform argv — everything AFTER the resolved
// `terraform` command word — by its SUBCOMMAND, which is what actually determines
// whether infrastructure comes down:
//
//	destroy            -> tears down. Denied.
//	apply -destroy     -> exactly equivalent to `destroy` in modern Terraform. Denied.
//	plan [-destroy]    -> writes a plan, changes nothing. Admitted; this is the
//	                      recovery the refusal itself advertises.
//	show / state / output / validate / init / fmt / … -> admitted.
//
// An argv with no subcommand at all (`terraform`, or only global flags) destroys
// nothing and is admitted.
func terraformArgvDestroys(argv []string) bool {
	sub, rest, ok := terraformSubcommand(argv)
	if !ok {
		return false
	}
	switch sub {
	case "destroy":
		return true
	case "apply":
		return terraformArgvHasDestroyFlag(rest)
	default:
		return false
	}
}

// terraformSubcommand returns the first non-flag word of a terraform argv, so a
// global flag before the subcommand (`terraform -chdir=infra destroy`) is stepped
// over rather than mistaken for it.
func terraformSubcommand(argv []string) (string, []string, bool) {
	for i, tok := range argv {
		if strings.HasPrefix(tok, "-") || tok == "" {
			continue
		}
		return strings.ToLower(tok), argv[i+1:], true
	}
	return "", nil, false
}

// terraformArgvHasDestroyFlag reports whether an `apply` argv carries the -destroy
// flag, accepting the single- and double-dash spellings and an explicit `=true`,
// and not counting an explicit negation.
func terraformArgvHasDestroyFlag(argv []string) bool {
	for _, tok := range argv {
		if !strings.HasPrefix(tok, "-") {
			continue
		}
		name, value, hasValue := strings.Cut(strings.ToLower(strings.TrimLeft(tok, "-")), "=")
		if name != "destroy" {
			continue
		}
		if hasValue && (value == "false" || value == "0") {
			continue
		}
		return true
	}
	return false
}
