// denyrules.go — the CLOSED policy-RUNG vocabulary: WHICH rule refused.
//
// reasons.go names a refusal's CLASS; that is too coarse for a fleet operator
// asking WHY a cluster of workers died — one (By, Reason) pair covers several
// distinct rungs, and the rung identity rides only in the leading words of a
// free-text witness Claim. A DenyRuleID is the missing routing key: a stable,
// closed-vocabulary id for the refusing rung, carried on Verdict.Meta[MetaDenyRule]
// and recorded as its own field by the decision journal.
//
// Safety: every id is a compile-time literal minted by the floor, never a byte
// of tool argument, path, host, or command text. DenyRuleID is a set-membership
// test, not a character filter: a value not already in the set is rejected whole,
// never trimmed or truncated into it — no partial-credit outcome, unlike the
// scrub-and-bound filter that leaked an env-assignment value into args_label
// before #5863.
//
// Additive: a new rung adds its id here in the same change that mints it; until
// then its refusals carry no deny_rule (fail-closed, lossless). Never add an id
// derived from input rather than authored in source.
package abi

import "strings"

// MetaDenyRule is the Verdict.Meta key a refusing rung stamps with its DenyRuleID;
// Meta is the established seam for bounded refusal context, so this needs no ABI signature change.
const MetaDenyRule = "deny_rule"

// maxDenyRuleIDLen is a cheap pre-filter so a pathological value never allocates;
// it is far above the longest real id and below anything a secret fits in, but membership is the actual gate.
const maxDenyRuleIDLen = 64

// Arg-predicate rung ids (internal/adjudicator/decide_argpredicates.go); each is the leading atom of that rung's witness detail.
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

// Self-modify rung ids (internal/adjudicator/decide.go). All three cite SELF_MODIFY and disclose
// only the offending glob, so they are indistinguishable on the wire; these ids separate the rungs.
const (
	DenyRuleSelfModifyPath      = "self_modify_path"      // a write-shaped tool's path arg matched a guarded glob
	DenyRuleSelfModifyCommand   = "self_modify_command"   // a shell command's write verb targeted a guarded glob
	DenyRuleSelfModifySynthTool = "self_modify_synthtool" // an exec of an agent-authored script reached a guarded glob
	DenyRuleCredentialPathBlock = "credential_path_block" // a tool call targeted a guarded credential or host-config path
)

// Trunk-discipline law ids (internal/gitgate); every law is authored as "<law-id> refused: <prose>", so its leading atom is its id.
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

// coreDenyRules is the closed set: a value reaches a journal row only by being in this map.
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

// DenyRuleID canonicalizes a rung's authored rule label — the arg-predicate
// family's snake_case atom or gitgate's hyphenated law prefix with its
// "refused"/":" punctuation — to one on-the-wire form, and reports whether the
// result is in the closed vocabulary. It is deliberately rejecting: any input
// that does not canonicalize onto a member returns ("", false), so no input
// byte survives in the returned value. Callers MUST honor the bool and emit nothing on false.
func DenyRuleID(raw string) (string, bool) {
	s := strings.TrimSpace(raw)
	// Take the first whitespace-delimited word and drop a trailing colon; both
	// steps are narrowing only — they can never lengthen the candidate or splice
	// two tokens together.
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
