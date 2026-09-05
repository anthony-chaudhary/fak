// denyrules.go — the CLOSED policy-RUNG vocabulary: WHICH rule refused.
//
// reasons.go gives a refusal its CLASS (POLICY_BLOCK, SELF_MODIFY, ...). That is
// the right granularity for a caller deciding what to do next, and the wrong one
// for a fleet operator asking WHY a cluster of workers died: measured over
// .dispatch-runs/guard-audit/*.jsonl, a single (By, Reason) pair —
// ("monitor", "POLICY_BLOCK") — covers the recursive-delete rung, the
// out-of-tree-write rung and the raw deny-regex rung at once, and
// ("gitgate", "POLICY_BLOCK") covers seven distinct trunk-discipline laws. The
// rung identity IS present on the wire, but only as the leading words of a
// free-text witness Claim that runs up to 447 characters and shares no grammar
// across rungs, so routing on it means a bespoke prose parser per family.
//
// A DenyRuleID is that missing routing key: a stable, closed-vocabulary id for
// the rung that refused, carried on Verdict.Meta[MetaDenyRule] by the refusing
// rung and recorded as its own field by the decision journal.
//
// DISCLOSURE. This vocabulary is the whole safety argument for the field. Every
// id below is a compile-time literal MINTED BY THE FLOOR — never a byte of tool
// argument, path, host, or command text. DenyRuleID is a SET-MEMBERSHIP test, not
// a character filter: a value that is not already in the set is REJECTED whole,
// never trimmed, filtered, or truncated into the set. So the field's value space
// is exactly the finite list in coreDenyRules, and no caller — a rung, a
// model-controlled Meta map, a future contributor — can put anything else there.
// That is strictly stronger than a scrub-and-bound defense, which is what leaked
// an env-assignment value into args_label before #5863: a filter that DROPS the
// offending characters silently fuses what is left, whereas a membership test has
// no partial-credit outcome at all.
//
// ADDITIVE. A new rung adds its id here in the same change that mints it; until
// it does, its refusals simply carry no deny_rule (fail-closed and lossless — the
// existing witness Claim is untouched). Never add an id that is derived from
// input rather than authored in source.
package abi

import "strings"

// MetaDenyRule is the Verdict.Meta key a refusing rung stamps with its
// DenyRuleID. Meta is the established seam for bounded refusal context (it
// already carries "fix" and "claim"), so this needs no ABI signature change.
const MetaDenyRule = "deny_rule"

// maxDenyRuleIDLen bounds the canonicalization input so a pathological value
// never allocates. It is far above the longest real id (commit_by_explicit_path,
// 24 bytes) and below anything a secret is likely to fit in — but it is only a
// cheap pre-filter: membership below is the actual gate.
const maxDenyRuleIDLen = 64

// Arg-predicate rung ids (internal/adjudicator/decide_argpredicates.go). Each is
// the leading atom of the bounded witness detail that rung already emits.
const (
	DenyRuleAllowGlob        = "allow_glob"
	DenyRuleDenyRegex        = "deny_regex"
	DenyRuleMaxBytes         = "max_bytes"
	DenyRuleCLIReadOnly      = "cli_read_only"
	DenyRuleArgMalformed     = "arg_malformed"
	DenyRuleRCEPipe          = "rce_pipe"
	DenyRuleRmRf             = "rm_rf"
	DenyRuleOutOfTreeWrite   = "out_of_tree_write"
	DenyRuleShellDialect     = "shell_dialect"
	DenyRuleSudoLocal        = "sudo_local"
	DenyRuleRunAsElevation   = "runas_elevation"
	DenyRuleTerraformDestroy = "terraform_destroy"
	DenyRuleDeviceOp         = "device_op"
)

// Self-modify rung ids (internal/adjudicator/decide.go). All three cite
// SELF_MODIFY and disclose only the offending glob, so today they are
// indistinguishable on the wire even though they are three different rungs
// reached by three different call shapes.
const (
	DenyRuleSelfModifyPath      = "self_modify_path"      // a write-shaped tool's path arg matched a guarded glob
	DenyRuleSelfModifyCommand   = "self_modify_command"   // a shell command's write verb targeted a guarded glob
	DenyRuleSelfModifySynthTool = "self_modify_synthtool" // an exec of an agent-authored script reached a guarded glob
	DenyRuleCredentialPathBlock = "credential_path_block" // a tool call targeted a guarded credential or host-config path
)

// Trunk-discipline law ids (internal/gitgate). Every gitgate law is AUTHORED as
// "<law-id>[ refused]: <prose>", so the law's own leading atom is its id.
const (
	DenyRuleSkipHooks            = "skip_hooks"
	DenyRuleSkipSigning          = "skip_signing"
	DenyRuleCommitByPath         = "commit_by_explicit_path"
	DenyRuleRemoteRef            = "remote_ref"
	DenyRuleTagForce             = "tag_force"
	DenyRuleTagDelete            = "tag_delete"
	DenyRuleAutostash            = "autostash"
	DenyRuleResetHard            = "reset_hard"
	DenyRuleCleanForce           = "clean_force"
	DenyRuleOffTrunk             = "off_trunk"
	DenyRuleHistoryRewrite       = "history_rewrite"
	DenyRulePushMirror           = "push_mirror"
	DenyRulePushPrune            = "push_prune"
	DenyRuleNeverAmendShared     = "never_amend_shared"
	DenyRuleUnattributablePush   = "unattributable_push"
	DenyRuleUnscopedStash        = "unscoped_stash"
	DenyRuleWholeTreeDiscard     = "whole_tree_discard"
	DenyRuleSweepGuard           = "sweep_guard"
	DenyRulePullRebase           = "pull_rebase"
	DenyRuleForcePush            = "force_push"
	DenyRuleAmend                = "amend"
	DenyRuleCollectiveCommit     = "collective_commit"
	DenyRuleSpuriousStagedDelete = "spurious_staged_deletion"
)

// coreDenyRules is the closed set. A value reaches a journal row only by being
// IN this map — see the disclosure note at the top of the file.
var coreDenyRules = map[string]bool{
	DenyRuleAllowGlob:        true,
	DenyRuleDenyRegex:        true,
	DenyRuleMaxBytes:         true,
	DenyRuleCLIReadOnly:      true,
	DenyRuleArgMalformed:     true,
	DenyRuleRCEPipe:          true,
	DenyRuleRmRf:             true,
	DenyRuleOutOfTreeWrite:   true,
	DenyRuleShellDialect:     true,
	DenyRuleSudoLocal:        true,
	DenyRuleRunAsElevation:   true,
	DenyRuleTerraformDestroy: true,
	DenyRuleDeviceOp:         true,

	DenyRuleSelfModifyPath:      true,
	DenyRuleSelfModifyCommand:   true,
	DenyRuleSelfModifySynthTool: true,
	DenyRuleCredentialPathBlock: true,

	DenyRuleSkipHooks:            true,
	DenyRuleSkipSigning:          true,
	DenyRuleCommitByPath:         true,
	DenyRuleRemoteRef:            true,
	DenyRuleTagForce:             true,
	DenyRuleTagDelete:            true,
	DenyRuleAutostash:            true,
	DenyRuleResetHard:            true,
	DenyRuleCleanForce:           true,
	DenyRuleOffTrunk:             true,
	DenyRuleHistoryRewrite:       true,
	DenyRulePushMirror:           true,
	DenyRulePushPrune:            true,
	DenyRuleNeverAmendShared:     true,
	DenyRuleUnattributablePush:   true,
	DenyRuleUnscopedStash:        true,
	DenyRuleWholeTreeDiscard:     true,
	DenyRuleSweepGuard:           true,
	DenyRulePullRebase:           true,
	DenyRuleForcePush:            true,
	DenyRuleAmend:                true,
	DenyRuleCollectiveCommit:     true,
	DenyRuleSpuriousStagedDelete: true,
}

// DenyRuleID canonicalizes a rung's authored rule label and reports whether the
// result is in the closed vocabulary. Canonicalization accepts the two spellings
// rungs actually author — the arg-predicate family's snake_case atom and
// gitgate's hyphenated law prefix with its "refused"/":" punctuation — and folds
// them to one on-the-wire form.
//
// It is deliberately TOTAL and REJECTING: any input that does not canonicalize
// onto a member returns ("", false). Nothing is filtered into the set, so no
// input byte can survive in the returned value. Callers MUST honor the bool and
// emit nothing on false.
func DenyRuleID(raw string) (string, bool) {
	s := strings.TrimSpace(raw)
	// A rung hands us its whole authored label — an arg-predicate detail
	// ("rm_rf recursive/forced delete") or a gitgate law
	// ("skip-hooks refused: <prose>"). Take the first whitespace-delimited word and
	// drop a trailing colon. Both steps are NARROWING only: they can never lengthen
	// the candidate or splice two tokens together.
	if i := strings.IndexAny(s, " \t\r\n"); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSuffix(s, ":")
	if s == "" || len(s) > maxDenyRuleIDLen {
		return "", false
	}
	s = strings.ToLower(strings.ReplaceAll(s, "-", "_"))
	if !coreDenyRules[s] {
		return "", false
	}
	return s, true
}

// DenyRuleIDs returns the closed vocabulary as a sorted slice — the value space a
// consumer may switch on, and the lint surface for a rung adding a law.
func DenyRuleIDs() []string {
	out := make([]string, 0, len(coreDenyRules))
	for id := range coreDenyRules {
		out = append(out, id)
	}
	sortStrings(out)
	return out
}
