package dispatchtick

// Witness-sweep semantics for finished dispatch workers, ported from
// tools/issue_resolve_dispatch.py (#1324 proposal #2 + the #1396 pick-held-invariant
// rung). A finished worker's slot is graded through `dos commit-audit` into a claim
// (CLAIM_WITNESSED / CLAIM_UNWITNESSED / CLAIM_NO_COMMIT) and, for a no-commit exit,
// a structured reason classified from the log tail. Only the two RE-BLOCKABLE guard
// refusals (self_modify / policy_block) hold their issue out of the next pick: a
// re-dispatch would hit the same guard identically, so re-storming it burns budget
// for zero commits. An auth wall re-probes after the time cooldown; a banner no-op
// is owned by the backend-health gate. This file is the pure half — the runs-dir
// walk, git/dos subprocesses, and sidecar writes live in the cmd/fak shell.

import (
	"fmt"
	"regexp"
	"strings"
)

// Claim verdicts a finished worker slot grades into — the .witness sidecar vocabulary.
const (
	ClaimWitnessed   = "CLAIM_WITNESSED"
	ClaimUnwitnessed = "CLAIM_UNWITNESSED"
	ClaimNoCommit    = "CLAIM_NO_COMMIT"
)

// WitnessOK is the DOS witness rung a truly-resolved commit must clear — the same
// non-forgeable keep-bit the closure audit grades against.
const WitnessOK = "diff-witnessed"

// Why a FINISHED worker landed no resolving commit (the .witness `reason` field).
const (
	NoCommitSelfModify  = "self_modify"
	NoCommitPolicyBlock = "policy_block"
	NoCommitAuthWall    = "auth_wall"
	NoCommitOffTrunk    = "off_trunk"
	NoCommitBannerNoop  = "banner_noop"
	NoCommitUnknown     = "unknown"
)

// WitnessSidecarSuffix marks a worker slot as audited-once: a commit's diff (so its
// verdict) is immutable, so a slot is graded exactly one time.
const WitnessSidecarSuffix = ".witness"

// WitnessTailBytes bounds how much of a (possibly multi-MB) worker log the no-commit
// classifier inspects — the guard summary and final turn live at the end.
const WitnessTailBytes = 4096

// StubLogMaxBytes is the banner-no-op size floor shared with the live-lane reap: a
// genuinely live worker streams kilobytes within seconds, so a log at or under this
// floor carrying only the startup banner is a terminal no-op (#1275).
const StubLogMaxBytes = 512

var (
	capBannerRE  = regexp.MustCompile(`(?i)hit your[\w\s]*limit|limit\s+exhausted`)
	glmWallRE    = regexp.MustCompile(`(?i)Limit Exhausted|limit will reset at|usage limit reached`)
	noopBannerRE = regexp.MustCompile(`(?i)>\s*build\s*[·:]`)
)

// WitnessRecord is one finished worker slot's graded verdict — the row the sweep
// appends to the payload buckets and writes as the .witness sidecar.
type WitnessRecord struct {
	Issue   int
	Log     string
	SHA     string
	Claim   string
	Verdict string
	Witness string
	Reason  string
}

// Map renders the record in the exact sidecar shape the Python dispatcher writes:
// explicit nulls for an absent sha/verdict/witness, and a reason key only on a
// no-commit record, so every existing sidecar reader parses both dialects.
func (r WitnessRecord) Map() map[string]any {
	out := map[string]any{
		"issue":   r.Issue,
		"log":     r.Log,
		"sha":     nil,
		"claim":   r.Claim,
		"verdict": nil,
		"witness": nil,
	}
	if r.SHA != "" {
		out["sha"] = r.SHA
	}
	if r.Verdict != "" {
		out["verdict"] = r.Verdict
	}
	if r.Witness != "" {
		out["witness"] = r.Witness
	}
	if r.Claim == ClaimNoCommit {
		out["reason"] = r.Reason
	}
	return out
}

// ClassifyNoCommitReason classifies why a finished worker landed no resolving commit
// from the log TAIL (last WitnessTailBytes) and total log size, so the witness records
// a STRUCTURED reason instead of an opaque CLAIM_NO_COMMIT. size < 0 means the log
// could not be stat'd — the banner-no-op floor then fails open to unknown, exactly
// like the Python classifier's OSError branch. Pure + fail-open: no recognized
// signature -> unknown, never a false positive.
func ClassifyNoCommitReason(tail string, size int64) string {
	switch {
	case strings.Contains(tail, "SELF_MODIFY"):
		return NoCommitSelfModify
	case strings.Contains(tail, "POLICY_BLOCK"):
		return NoCommitPolicyBlock
	case capBannerRE.MatchString(tail) || glmWallRE.MatchString(tail):
		return NoCommitAuthWall
	case strings.Contains(tail, "OFF_TRUNK"):
		return NoCommitOffTrunk
	case size >= 0 && size <= StubLogMaxBytes && noopBannerRE.MatchString(tail):
		return NoCommitBannerNoop
	default:
		return NoCommitUnknown
	}
}

// HeldNoCommitIssues folds this tick's witness records into the issue numbers the
// picker must HOLD: a slot that exited CLAIM_NO_COMMIT for a re-blockable structural
// reason (self_modify / policy_block) re-blocks identically on re-dispatch, so the
// pick skips it this tick instead of re-storming the same un-landable drain (#1396).
func HeldNoCommitIssues(records []WitnessRecord) map[int]bool {
	held := map[int]bool{}
	for _, rec := range records {
		if rec.Claim != ClaimNoCommit {
			continue
		}
		if rec.Reason == NoCommitSelfModify || rec.Reason == NoCommitPolicyBlock {
			held[rec.Issue] = true
		}
	}
	return held
}

// SubjectCitesIssue reports whether a commit subject names `#<issue>` at a word
// boundary — the same binding key the closure audit uses. RE2 has no lookbehind, so
// the Python `(?<![\w-])#N\b` is expressed as an explicit leading boundary: a glued
// `x#1324` or `-#1324` token binds nothing, a normal `(#1324)` binds.
func SubjectCitesIssue(subject string, issue int) bool {
	if strings.TrimSpace(subject) == "" {
		return false
	}
	re := regexp.MustCompile(fmt.Sprintf(`(^|[^\w-])#%d\b`, issue))
	return re.MatchString(subject)
}

// FirstResolvingSHA scans a newest-first `git log --pretty=format:%H<US>%s` stream
// for the first commit whose subject cites #issue — the commit THIS worker landed
// for its assigned issue. Empty when no subject cites it (the worker landed nothing,
// or committed a wrong-issue subject, so the slot claims nothing).
func FirstResolvingSHA(logLines string, issue int) string {
	for _, line := range strings.Split(logLines, "\n") {
		sha, subject, ok := strings.Cut(line, "\x1f")
		if !ok {
			continue
		}
		sha = strings.TrimSpace(sha)
		if sha != "" && SubjectCitesIssue(subject, issue) {
			return sha
		}
	}
	return ""
}

// CommitWitnessed grades a `dos commit-audit` row into the slot keep-bit: true only
// on verdict OK AND a diff-witness — a subject-only claim never counts as productive.
func CommitWitnessed(verdict, witness string) bool {
	return strings.EqualFold(strings.TrimSpace(verdict), "OK") && strings.TrimSpace(witness) == WitnessOK
}
