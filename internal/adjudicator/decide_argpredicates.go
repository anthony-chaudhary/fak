package adjudicator

import (
	"errors"
	"path"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// evalArgPredicates runs every predicate that targets tool against the decoded
// args, returning the first violation's Deny verdict (and true) or a zero verdict
// (and false) if all pass. It is pure and allocation-light: it only iterates the
// (typically tiny) predicate slice and reads scalar arg values.
func evalArgPredicates(preds []ArgPredicate, tool string, args map[string]any) (abi.Verdict, bool, []string) {
	var notes []string
	// note records a violated ADVISORY rule instead of denying: the same bounded
	// string the hard deny would have carried as its witness claim, accumulated so
	// the eventual admit's Meta shows exactly which trial rules the call broke.
	note := func(pr *ArgPredicate, detail string) {
		notes = append(notes, pr.Tool+"."+pr.Arg+" "+detail)
	}
	for i := range preds {
		pr := &preds[i]
		if !strings.EqualFold(pr.Tool, tool) {
			continue // case-insensitive: a "Bash" rule still gates a "bash" call (matches the index)
		}
		val, present := argString(args, pr.Arg)
		switch pr.Kind {
		case ArgAllowExact:
			literal, isString := args[pr.Arg].(string)
			if !isString || literal != pr.Glob {
				if pr.Advisory {
					note(pr, "allow_exact")
					continue
				}
				return argDeny(pr, "allow_exact"), true, notes
			}
		case ArgAllowGlob:
			// Canonicalize before the containment check (#2407): a backslash,
			// redundant dot-segment, env alias, or quote-wrapped spelling of an
			// in-bounds path must not read as an escape, and vice versa. An
			// undecodable value (an unterminated quote) fails closed as
			// MALFORMED rather than sliding past the glob under a raw spelling
			// the canonical form would have caught.
			canon := val
			if present {
				c, ok := canonicalizeArgValue(val)
				if !ok {
					return argMalformed(pr), true, notes
				}
				canon = c
			}
			// Positive requirement: a missing OR out-of-bounds value fails closed.
			if !present || !pathUnderGlob(pr.Glob, canon) {
				if pr.Advisory {
					note(pr, "allow_glob "+pr.Glob)
					continue
				}
				return argDeny(pr, "allow_glob "+pr.Glob), true, notes
			}
		case ArgDenyRegex:
			// The deny_regex families are each decided STRUCTURALLY in their own
			// evalXxxArgRule helper; the first family that claims the rule resolves
			// it, and only an unclaimed rule falls through to the generic
			// canonical-regex match inside evalDenyRegexArgRule.
			if v, denied := evalDenyRegexArgRule(pr, val, present, note); denied {
				return v, true, notes
			}
		case ArgCLIReadOnly:
			if !present {
				return argDeny(pr, "cli_read_only"), true, notes
			}
			if _, _, grammarErr := attenuateCLIGrammar(val); grammarErr != nil && !errors.Is(grammarErr, ErrCLIGrammarNotApplicable) {
				if pr.Advisory {
					note(pr, "cli_read_only")
					continue
				}
				return argDeny(pr, "cli_read_only"), true, notes
			}
		case ArgMaxBytes:
			if present && len(val) > pr.N {
				if pr.Advisory {
					note(pr, "max_bytes "+strconv.Itoa(pr.N))
					continue
				}
				return argDeny(pr, "max_bytes "+strconv.Itoa(pr.N)), true, notes
			}
		}
	}
	return abi.Verdict{}, false, notes
}

// evalDenyRegexArgRule evaluates ONE ArgDenyRegex predicate against the decoded
// arg value. The shipped structural families run first, each decided in its own
// evalXxxArgRule helper; the first family that claims the rule resolves it (a
// hard deny, or an advisory note-and-pass), and only a rule no family claims
// falls through to the generic canonical-regex match. Returns the deny verdict
// (denied=true) or a zero verdict meaning the predicate passed and the caller
// proceeds to the next predicate.
func evalDenyRegexArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool) {
	if v, denied, claimed := evalBuildCacheCleanArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalRCEPipeArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalRmRfArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalOutOfTreeWriteArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalShellDialectArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalSudoArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalRunAsArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalTerraformDestroyArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	if v, denied, claimed := evalDeviceOpArgRule(pr, val, present, note); claimed {
		return v, denied
	}
	// Every OTHER deny_regex rule matches the CANONICAL form (#2407): the raw
	// arg string alone let a backslash, dot-segment, env-alias, or quote-style
	// spelling of the same value slip a rule written against its canonical
	// spelling (the documented #1464/#1466 bypass class). An undecodable value
	// fails closed as MALFORMED instead of matching nothing.
	if present {
		canon, ok := canonicalizeArgValue(val)
		if !ok {
			return argMalformed(pr), true
		}
		if pr.Re != nil && pr.Re.MatchString(canon) {
			if pr.Advisory {
				note(pr, "deny_regex /"+pr.Re.String()+"/")
				return abi.Verdict{}, false
			}
			return argDeny(pr, "deny_regex /"+pr.Re.String()+"/"), true
		}
	}
	return abi.Verdict{}, false
}

// evalCanonicalArgRule is the shared structural decision for a deny_regex family
// that consults the rule's own regular expression once over the CANONICAL
// spelling and then lets a family-specific decider refine the match: when the
// rule is not this family nothing is claimed; a missing arg, a non-matching (or
// decider-exonerated) value, or an ADVISORY rule passes (the advisory is noted);
// a violated hard rule denies.
func evalCanonicalArgRule(applies bool, pr *ArgPredicate, val, detail string, present bool, note func(*ArgPredicate, string), violates func(raw, canon string) bool) (handled bool, verdict abi.Verdict, denied bool) {
	if !applies {
		return false, abi.Verdict{}, false
	}
	if !present {
		return true, abi.Verdict{}, false
	}
	canon, matches, ok := canonicalArgRegexMatch(pr, val)
	if !ok {
		return true, argMalformed(pr), true
	}
	if !matches || !violates(val, canon) {
		return true, abi.Verdict{}, false
	}
	if pr.Advisory {
		note(pr, detail)
		return true, abi.Verdict{}, false
	}
	return true, argDeny(pr, detail), true
}

// evalBuildCacheCleanArgRule claims the build-cache-clean family. Clearing Go's
// build cache is decided by executed command shape rather than raw text: the
// shipped regex is only a selector; quoted examples, grep patterns and commit
// messages remain inert mentions. Returns (verdict, denied, claimed).
func evalBuildCacheCleanArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	if !isBuildCacheCleanArgRule(pr) {
		return abi.Verdict{}, false, false
	}
	if present && commandClearsGoBuildCache(val) {
		if pr.Advisory {
			note(pr, "build_cache_clean go clean -cache")
			return abi.Verdict{}, false, true
		}
		return argDeny(pr, "build_cache_clean go clean -cache"), true, true
	}
	return abi.Verdict{}, false, true
}

// evalRCEPipeArgRule claims the RCE download-pipe family (#1465), decided
// STRUCTURALLY, not by the raw regex. A regex over the un-tokenized command fails
// both ways: it false-positives on quoted echo/grep content (`echo 'curl x | sh'`
// -> a false POLICY_BLOCK, which in `fak guard -- claude` reads as an
// agent-chosen end_turn stop), and it misses a one-character launder
// (`curl x | python3` slips a rule that only names sh/bash).
// commandHasRemotePipeToInterpreter tokenizes the command (skipping quoted
// words), unwraps sh -c / $()/“ sources, and matches a real
// downloader|interpreter pipe at a command boundary across the broadened
// interpreter set. Returns (verdict, denied, claimed).
func evalRCEPipeArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	if !isRCEPipeArgRule(pr) {
		return abi.Verdict{}, false, false
	}
	if present && commandHasRemotePipeToInterpreter(val) {
		if pr.Advisory {
			note(pr, "rce_pipe download|interpreter")
			return abi.Verdict{}, false, true
		}
		return argDeny(pr, "rce_pipe download|interpreter"), true, true
	}
	return abi.Verdict{}, false, true
}

// evalRmRfArgRule claims the recursive/forced-delete family, decided
// STRUCTURALLY for the same two reasons as the RCE pipe rule: the raw regex
// false-positives on quoted text (`echo 'rm -rf /'` -> a false POLICY_BLOCK that
// in `fak guard -- claude` reads as an agent-chosen end_turn stop), and it
// inspects only the FIRST flag cluster after `rm`, so `rm -i -rf`,
// `rm --recursive --force`, and `sh -c 'rm -rf /'` launder past it.
// commandHasUnsafeRecursiveForcedDelete tokenizes the command, resolves the real
// `rm` command word (exempting quoted text and `git rm`), and scans all of rm's
// argv flags for a recursive OR force option. Returns (verdict, denied, claimed).
func evalRmRfArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	if !isRmRfArgRule(pr) {
		return abi.Verdict{}, false, false
	}
	ws, scratch := outOfTreeRoots()
	if present && commandHasUnsafeRecursiveForcedDelete(val, ws, scratch) {
		if pr.Advisory {
			note(pr, "rm_rf recursive/forced delete")
			return abi.Verdict{}, false, true
		}
		return argDeny(pr, "rm_rf recursive/forced delete"), true, true
	}
	return abi.Verdict{}, false, true
}

// evalOutOfTreeWriteArgRule claims the OUT-OF-TREE WRITE family (-o / --output /
// redirect / cp-family `..` rules), decided STRUCTURALLY, not by the raw
// `..`-substring regex, for the same reason as rm_rf/rce_pipe. The raw regex both
// false-DENIES a write that resolves IN-TREE or into the sanctioned harness
// scratchpad (reached via `..`) — a false POLICY_BLOCK that under
// `fak guard -- claude` reads as an agent-chosen end_turn and silently kills the
// turn — and MISSES absolute / `$HOME` escapes. outOfTreeWriteEscapes is
// fail-closed and purely SUBTRACTIVE: it downgrades the deny to an allow ONLY
// when EVERY write destination provably resolves under the workspace root or a
// declared scratchpad root; any escaping, unprovable ($VAR/glob), undecodable,
// or unidentifiable destination — or an unknown workspace root — keeps the deny.
// Gated on the raw match (rawMatches=true) so it never introduces a NEW deny for
// a command the regex would not have flagged. Returns (verdict, denied, claimed).
func evalOutOfTreeWriteArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	handled, verdict, denied := evalCanonicalArgRule(isOutOfTreeWriteArgRule(pr), pr, val, "out_of_tree_write", present, note, func(raw, _ string) bool {
		ws, scratch := outOfTreeRoots()
		return outOfTreeWriteEscapes(raw, ws, scratch, true)
	})
	return verdict, denied, handled
}

// evalShellDialectArgRule claims the CROSS-SHELL-DIALECT family (#3941), decided
// STRUCTURALLY, not by the raw regex, for the same reason as rm_rf/rce_pipe: a
// raw `\bGet-Content\b` false-positives on the cmdlet name as an argument
// (`grep Get-Content f`) or quoted (`echo 'Get-Content'`) — a false POLICY_BLOCK
// that under `fak guard -- claude` reads as an agent-chosen end_turn.
// commandLeadsWithPowerShellCmdlet tokenizes the command (quoted words are never
// a command word), unwraps sh -c / $() / “, and matches a curated cmdlet ONLY
// at a stage's resolved command-word position. The refusal names the recovery
// (PowerShell tool, or the POSIX equivalent) so it is a redirect, not a wall.
// Returns (verdict, denied, claimed).
func evalShellDialectArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	if !isShellDialectArgRule(pr) {
		return abi.Verdict{}, false, false
	}
	if present {
		if cmdlet, ok := commandLeadsWithPowerShellCmdlet(val); ok {
			if pr.Advisory {
				note(pr, "shell_dialect "+cmdlet)
				return abi.Verdict{}, false, true
			}
			return argDeny(pr, "shell_dialect "+cmdlet), true, true
		}
	}
	return abi.Verdict{}, false, true
}

// evalSudoArgRule claims the PRIVILEGE-ESCALATION family (`\bsudo\b`), decided
// STRUCTURALLY, not by the raw regex, for the same reason as rm_rf/rce_pipe: the
// raw regex false-positives on sudo as quoted text (`echo 'sudo make install'`)
// and — the dominant false POLICY_BLOCK in real remote-GPU bring-up trajectories
// — on a REMOTE escalation carried as an ssh argument
// (`ssh gpu-box 'sudo systemctl restart …'`), where the local command word is ssh
// and the escalation is governed by the remote host's own controls, not this
// local floor. commandLocalEscalationWord tokenizes the command, unwraps sh -c /
// $() / “, and matches sudo (or the doas launder the raw regex missed) only at
// a LOCAL resolved command-word position, so a genuine local escalation stays
// denied. Returns (verdict, denied, claimed).
func evalSudoArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	if !isSudoArgRule(pr) {
		return abi.Verdict{}, false, false
	}
	if present {
		if word, ok := commandLocalEscalationWord(val); ok {
			if pr.Advisory {
				note(pr, "sudo_local "+word)
				return abi.Verdict{}, false, true
			}
			return argDeny(pr, "sudo_local "+word), true, true
		}
	}
	return abi.Verdict{}, false, true
}

// evalRunAsArgRule claims the WINDOWS privilege-elevation family
// (`Start-Process … -Verb RunAs`), decided STRUCTURALLY for the same reason as
// its POSIX twin — closing the asymmetry sudo_local.go itself named, where the
// POSIX escalation word got a command-word decision while this one kept the raw
// regex (#2343). The raw regex false-positives on the phrase as quoted text,
// which on THIS repo is routine work: grepping the policy that ships the rule
// (`Select-String -Pattern 'Start-Process -Verb RunAs' …`) and committing a doc
// about it were both POLICY_BLOCKs. Worse, the refusal's own fix text says to
// PRINT the command for the operator, and printing it tripped the same rule —
// the self-refuting-remedy class already closed for -WhatIf and `git clean -n`.
// commandInvokesRunAsElevation tokenizes with PowerShell's lexical rules
// (backtick escapes, backslash is a path byte), resolves each statement's
// command word, and requires Start-Process AT that position carrying -Verb RunAs
// in the same argv — unwrapping nested host payloads so a real elevation stays
// denied. Gated on the raw match and purely SUBTRACTIVE, so it never introduces
// a new deny; every ambiguity (unterminated quote, -EncodedCommand) keeps the
// deny. Returns (verdict, denied, claimed).
func evalRunAsArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	handled, verdict, denied := evalCanonicalArgRule(isRunAsArgRule(pr), pr, val, "runas_elevation", present, note, func(raw, canon string) bool {
		return commandInvokesRunAsElevation(raw) || commandInvokesRunAsElevation(canon)
	})
	return verdict, denied, handled
}

// evalTerraformDestroyArgRule claims the TERRAFORM DESTROY family, decided
// STRUCTURALLY by SUBCOMMAND, because the raw regex is `terraform` … `destroy` on
// one line and so refuses the recovery it itself advertises: its own fix text
// says "produce the destroy plan for review instead: terraform plan -destroy",
// and `-destroy` matches `\bdestroy\b`, so the advertised escape is blocked by
// the rule advertising it. That is the self-refuting-remedy class already closed
// for -WhatIf, `git clean -n` and `git rebase --abort`. It also refused quoted
// mentions (documenting or grepping the policy file that ships the rule, which
// lives in this checkout) and read-only subcommands (`terraform show  # inspect
// before a destroy`). commandAppliesTerraformDestroy resolves `terraform` at a
// command-word position under BOTH POSIX and PowerShell lexing and denies only a
// real teardown — `destroy`, or `apply -destroy`, which is its exact equivalent
// — while admitting `plan -destroy` and every read-only subcommand. Gated on the
// raw match and purely SUBTRACTIVE, so it never introduces a new deny; every
// ambiguity keeps the deny. Returns (verdict, denied, claimed).
func evalTerraformDestroyArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	handled, verdict, denied := evalCanonicalArgRule(isTerraformDestroyArgRule(pr), pr, val, "terraform_destroy", present, note, func(raw, canon string) bool {
		return commandAppliesTerraformDestroy(raw) || commandAppliesTerraformDestroy(canon)
	})
	return verdict, denied, handled
}

// evalDeviceOpArgRule claims the DISK/DEVICE VERB family (#5429) — the POSIX
// raw-device rule and its PowerShell volume/disk sibling — decided STRUCTURALLY
// for the same reason as the other families (the POSIX and the Windows elevation
// rules are one family decided in two). Both were RAW pattern matches with no
// decider, so the match WAS the verdict: naming one of these verbs anywhere on a
// shell command line was refused, and the refusal then reported a documentation
// edit as an attempted device operation. A quoted grep pattern over the policy
// file that ships the rule, a commit message explaining why the rule exists, and
// a here-doc body writing that explanation into a doc were all POLICY_BLOCKs —
// and under `fak guard -- claude` a POLICY_BLOCK reads as an agent-chosen
// end_turn, so the caller never got a second look.
// commandPerformsDeviceOperation resolves each segment's command word under BOTH
// POSIX and PowerShell lexing, checks redirect targets for a raw block device,
// and admits a mention only against an inert-head ALLOW-LIST — an evaluator, a
// launcher, an unresolvable command word or an unparseable quote all keep the
// deny. Gated on the raw match and purely SUBTRACTIVE, so it never introduces a
// new deny; a real device operation stays refused on every surface it ships on.
// Returns (verdict, denied, claimed).
func evalDeviceOpArgRule(pr *ArgPredicate, val string, present bool, note func(*ArgPredicate, string)) (abi.Verdict, bool, bool) {
	handled, verdict, denied := evalCanonicalArgRule(isDeviceOpArgRule(pr), pr, val, "device_op", present, note, func(raw, canon string) bool {
		return commandPerformsDeviceOperation(raw) || commandPerformsDeviceOperation(canon)
	})
	return verdict, denied, handled
}

// canonicalArgRegexMatch decodes a rule argument once and reports whether the
// rule's regular expression matches that canonical spelling. The ok result is
// false only when canonicalization fails, so callers can preserve their shared
// fail-closed MALFORMED path while applying distinct structural decisions.
func canonicalArgRegexMatch(pr *ArgPredicate, val string) (canon string, matches, ok bool) {
	canon, ok = canonicalizeArgValue(val)
	if !ok {
		return "", false, false
	}
	return canon, pr.Re != nil && pr.Re.MatchString(canon), true
}

// argDeny builds the bounded-disclosure Deny for a violated arg predicate. The
// witness Claim names the offending tool.arg and the bound it broke — never the
// arg value (which may be sensitive) nor the rest of the policy.
//
// Every caller passes a detail whose LEADING atom is a source literal naming the
// structural rule that fired ("rm_rf recursive/forced delete", "allow_glob "+glob,
// "deny_regex /"+re+"/"). That atom is the rung's identity, and until #5863 it
// reached the journal only as a substring of the free-text Claim — so a consumer
// had to prose-parse to tell the recursive-delete rung from the out-of-tree-write
// rung, both of which land on ("monitor", "POLICY_BLOCK"). abi.DenyRuleID
// promotes it onto Meta as a closed-vocabulary id. It discloses nothing new: the
// atom is already in the Claim, and abi.DenyRuleID admits only the literals the
// floor itself declares, so the trailing detail (which may name a policy glob or
// regex) can never ride along.
func argDeny(pr *ArgPredicate, detail string) abi.Verdict {
	reason := pr.Reason
	if reason == abi.ReasonNone {
		reason = abi.ReasonPolicyBlock
	}
	v := abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  reason,
		By:      "monitor",
		Payload: abi.WitnessPayload{Claim: pr.Tool + "." + pr.Arg + " " + detail},
	}
	// The rule's sanctioned alternative rides the verdict so the refusal is a
	// redirect, not a dead end (same bounded-disclosure budget: static manifest
	// text, never the arg value).
	if pr.Fix != "" {
		v.Meta = map[string]string{"fix": pr.Fix}
	}
	if rule, ok := abi.DenyRuleID(detail); ok {
		v.Meta = withDenyRule(v.Meta, rule)
	}
	return v
}

// withDenyRule stamps a validated rule id onto a verdict's Meta, allocating the
// map only when the verdict carried none. Callers pass ONLY an id abi.DenyRuleID
// has already admitted.
func withDenyRule(meta map[string]string, rule string) map[string]string {
	if meta == nil {
		meta = map[string]string{}
	}
	meta[abi.MetaDenyRule] = rule
	return meta
}

// denyRule returns a Meta map carrying just the rung's rule id, for a deny site
// that has no other bounded context to attach. An unregistered id yields nil, so
// a rung that forgets to declare its vocabulary entry emits nothing rather than
// free text.
func denyRule(rule string) map[string]string {
	id, ok := abi.DenyRuleID(rule)
	if !ok {
		return nil
	}
	return map[string]string{abi.MetaDenyRule: id}
}

// argMalformed builds the fail-closed Deny for an arg value the canonicalizer
// (#2407) could not decode (the one remaining case after #2771: an unterminated
// quote — a value that opens ' or " and never closes it). Always MALFORMED,
// never pr.Reason: the failure is that the rung could not tell what the value
// canonically IS, not that a specific rule matched it, so it must not be
// softened by that rule's own Advisory declaration.
//
// The Claim NAMES the concrete decode failure (not a bare "undecodable arg
// value") and the verdict carries a bounded Meta["fix"], so the refusal reaches
// the agent as a specific, actionable note through the unified remedy seam
// (#2749) rather than a silent MALFORMED that reads as a broken tool (#2771).
// Both stay within the deny channel's disclosure budget: static text naming the
// arg and the fix, never the arg value itself.
func argMalformed(pr *ArgPredicate) abi.Verdict {
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonMalformed,
		By:      "monitor",
		Payload: abi.WitnessPayload{Claim: pr.Tool + "." + pr.Arg + " has an unterminated quote (a value that opens ' or \" must close it)"},
		Meta: withDenyRule(map[string]string{
			"fix": "close the quote in " + pr.Arg + ", or drop the leading quote if the value is not meant to be quote-wrapped, then retry",
		}, abi.DenyRuleArgMalformed),
	}
}

// argString returns the string form of args[key] and whether the key was present.
// A scalar (string / number / bool) renders to its natural string; an
// object / array / null is treated as absent — arg predicates target scalar
// values, and a non-scalar where a path or command is expected fails the
// positive (ArgAllowGlob) requirement, which is the fail-closed outcome.
func argString(args map[string]any, key string) (string, bool) {
	v, ok := args[key]
	if !ok {
		return "", false
	}
	switch s := v.(type) {
	case string:
		return s, true
	case float64:
		return strconv.FormatFloat(s, 'g', -1, 64), true
	case bool:
		if s {
			return "true", true
		}
		return "false", true
	default:
		return "", false
	}
}

// pathUnderGlob reports whether value is a path admitted by an allow-glob. Two
// forms, slash-normalized and path-cleaned first so a backslash or "./" cannot
// dodge the check:
//   - "<dir>/**": CONTAINMENT — value must resolve to <dir> itself or inside it;
//     a "../" escape (which path.Clean folds out) fails. This is the
//     "allow write_file only under ./out/**" form. A bare "**" / "./**" admits
//     any relative path that does not escape the working root.
//   - otherwise: a single path.Match (single-segment '*'/'?' wildcards).
func pathUnderGlob(glob, value string) bool {
	norm := func(s string) string { return path.Clean(strings.ReplaceAll(s, `\`, "/")) }
	v := norm(value)
	if strings.HasSuffix(glob, "/**") || glob == "**" {
		dir := norm(strings.TrimSuffix(glob, "**")) // "./out/**" -> "out"; "**" -> "."
		if dir == "." || dir == "/" {               // "**", "./**": any non-escaping relative path
			return v != ".." && !strings.HasPrefix(v, "../") && !strings.HasPrefix(v, "/")
		}
		return v == dir || strings.HasPrefix(v, dir+"/")
	}
	ok, err := path.Match(norm(glob), v)
	return err == nil && ok
}
