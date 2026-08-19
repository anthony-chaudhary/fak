package dispatchtick

// Witness-sweep semantics for finished dispatch workers, ported from
// tools/issue_resolve_dispatch.py (#1324 proposal #2 + the #1396 pick-held-invariant
// rung). A finished worker's slot is graded through `dos commit-audit` into a claim
// (CLAIM_WITNESSED / CLAIM_UNWITNESSED / CLAIM_NO_COMMIT) and, for a no-commit exit,
// a structured reason classified from the log tail. Only the two RE-BLOCKABLE guard
// refusals (self_modify / policy_block) hold their issue out of the next pick: a
// re-dispatch would hit the same guard identically, so re-storming it burns budget
// for zero commits. A usage/rate/unknown-model wall is model-switchable (Layer-2
// re-dispatch downgrades onto the next chain model); a genuine auth wall re-probes
// after the time cooldown; a banner no-op is owned by the backend-health gate. This
// file is the pure half — the runs-dir
// walk, git/dos subprocesses, and sidecar writes live in the cmd/fak shell.

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
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

// Test-run claim verdicts (#3838) — the additive rung that binds a "done" claim to a
// green test run of the resolving commit's changed package, not just its diff shape.
// diff-witnessed proves the diff did the KIND of thing the subject claimed; it says
// NOTHING about whether that change passes its own tests. GREEN = affected tests ran
// and passed; RED = they ran and failed (a diff-witnessed commit can still be RED —
// that is exactly the gap this rung closes); UNRUN = no test was run (no resolving
// commit, no test-bearing changed package, the runner is disabled, or it faulted).
// UNRUN is a VALID, surfaced state — the rung never fabricates a pass it did not see.
const (
	ClaimTestGreen = "CLAIM_TEST_GREEN"
	ClaimTestRed   = "CLAIM_TEST_RED"
	ClaimTestUnrun = "CLAIM_TEST_UNRUN"
)

// GradeTestRun folds an affected-package test run into the #3838 test-run claim. It is
// the pure grader the shell's runner and the `verify` skill both bind through: `ran`
// is whether a test actually executed (false -> UNRUN, so a runner that never fired,
// found no test-bearing package, or faulted can NEVER masquerade as a pass), and
// `passed` is its exit verdict (only meaningful when ran). Fail-safe by construction:
// the only path to GREEN is ran && passed.
func GradeTestRun(ran, passed bool) string {
	switch {
	case !ran:
		return ClaimTestUnrun
	case passed:
		return ClaimTestGreen
	default:
		return ClaimTestRed
	}
}

// Why a FINISHED worker landed no resolving commit (the .witness `reason` field).
//
// The model-switchable trio (usage_cap / model_unknown / rate_limit) is distinct from
// the guard refusals AND from a genuine auth wall: a switch to a DIFFERENT model can
// clear them — a different weekly bucket, an entitled model id, a clear capacity pool —
// so Layer-2 re-dispatch downgrades onto the next chain model instead of re-storming the
// same walled one. auth_wall is now a GENUINE login/credit/access wall that a model
// switch cannot fix (only a human /login or a billing/entitlement change), corrected
// from its former mislabel where its regexes actually only caught usage caps.
const (
	NoCommitSelfModify   = "self_modify"
	NoCommitPolicyBlock  = "policy_block"
	NoCommitAuthWall     = "auth_wall"
	NoCommitUsageCap     = "usage_cap"
	NoCommitModelUnknown = "model_unknown"
	NoCommitRateLimit    = "rate_limit"
	NoCommitOffTrunk     = "off_trunk"
	NoCommitBannerNoop   = "banner_noop"
	NoCommitUnknown      = "unknown"
)

// WitnessSidecarSuffix marks a worker slot as audited-once: a commit's diff (so its
// verdict) is immutable, so a slot is graded exactly one time.
const WitnessSidecarSuffix = ".witness"

// WitnessTailBytes bounds how much of a (possibly multi-MB) worker log the no-commit
// classifier inspects — the guard summary and final turn live at the end.
const WitnessTailBytes = 16 << 10

// StubLogMaxBytes is the banner-no-op size floor shared with the live-lane reap: a
// genuinely live worker streams kilobytes within seconds, so a log at or under this
// floor carrying only the startup banner is a terminal no-op (#1275).
const StubLogMaxBytes = 512

var (
	capBannerRE  = regexp.MustCompile(`(?i)hit your[\w\s]*limit|limit\s+exhausted|account cooled by a live usage cap`)
	glmWallRE    = regexp.MustCompile(`(?i)Limit Exhausted|limit will reset at|usage limit reached`)
	noopBannerRE = regexp.MustCompile(`(?i)>\s*build\s*[·:]`)
)

// WitnessRecord is one finished worker slot's graded verdict — the row the sweep
// appends to the payload buckets and writes as the .witness sidecar.
type WitnessRecord struct {
	// SessionID and RegistrationID bind the independently graded attempt to a durable session identity.
	SessionID      string
	RegistrationID string
	Issue          int
	Log            string
	SHA            string
	Claim          string
	Verdict        string
	Witness        string
	Reason         string
	// Model is the primary model the finished slot was PINNED to (Layer 5b), scraped
	// from the worker's .model sidecar. Empty when the slot ran on the seat/agent
	// default (no --model pin) — the historical floor. Layer-2 downgrade re-dispatch
	// keys off this + Reason: a model-switchable no-commit exit whose ladder head was
	// Model advances to the NEXT chain model instead of re-storming the same walled one.
	Model string
	Speed string
	// Zone is the placement rung the slot was served from (device / fleet / vendor),
	// scraped from the .zone sidecar. Empty when the rung is not recorded — an unpinned
	// seat-default slot, a tick with no roster, or a model the roster does not bind — and
	// deliberately NOT defaulted to the device rung, since over-reporting self-hosting is
	// the error an operator would act on. See AttributeZone / FoldZoneShare.
	Zone string
	// TestClaim is the #3838 test-run rung: CLAIM_TEST_GREEN / CLAIM_TEST_RED /
	// CLAIM_TEST_UNRUN for the resolving commit's affected-package tests, recorded
	// ALONGSIDE (never replacing) the diff-shape Verdict/Witness. Empty on a no-commit
	// slot (no resolving commit -> no test rung at all), so a no-commit sidecar stays
	// byte-identical to before this rung.
	TestClaim string
	// FootprintClaim is additive advisory evidence comparing the resolving commit's
	// changed paths with the worker's declared lease tree (#4599). Empty means the
	// footprint could not be graded (including an absent/empty lease tree).
	FootprintClaim     string
	OutOfLanePathCount int
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
	if r.SessionID != "" {
		out["session_id"] = r.SessionID
	}
	if r.RegistrationID != "" {
		out["registration_id"] = r.RegistrationID
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
	// The model key is emitted ONLY when the slot was pinned to a non-default model, so
	// an unconfigured fleet's sidecar stays byte-identical to before Layer 5b (#the
	// model-switch program) — a floor worker (model=="") writes no model key at all.
	if r.Model != "" {
		out["model"] = r.Model
	}
	if r.Speed != "" {
		out["speed"] = r.Speed
	}
	// The zone key rides the same rule as the model key: emitted ONLY when the slot's rung
	// was actually attributed, so a fleet with no roster writes a sidecar byte-identical to
	// before this seam and no reader ever sees a rung that was assumed rather than resolved.
	if r.Zone != "" {
		out["zone"] = r.Zone
	}
	// The #3838 test-run rung is emitted ONLY when a test claim was graded (a resolving
	// commit was found and the runner produced GREEN/RED/UNRUN). A no-commit slot leaves
	// it empty, so its sidecar stays byte-identical to before this rung.
	if r.TestClaim != "" {
		out["test_claim"] = r.TestClaim
	}
	if r.FootprintClaim != "" {
		out["footprint_claim"] = r.FootprintClaim
		out["out_of_lane_path_count"] = r.OutOfLanePathCount
	}
	return out
}

// ClassifyNoCommitReason classifies why a finished worker landed no resolving commit
// from the log TAIL (last WitnessTailBytes) and total log size, so the witness records
// a STRUCTURED reason instead of an opaque CLAIM_NO_COMMIT. size < 0 means the log
// could not be stat'd — the banner-no-op floor then fails open to unknown, exactly
// like the Python classifier's OSError branch. Pure + fail-open: no recognized
// signature -> unknown, never a false positive.
//
// Precedence mirrors sessionsignals.TerminalFailure (AUTH > LIMIT > API_ERR) with the
// guard refusals first and the model-unknown class inserted between a usage cap and a
// transient rate-limit: a genuine login/credit/access wall (auth_wall) is checked
// BEFORE the usage cap so a credit wall is never mislabeled a switchable cap, and the
// local capBannerRE/glmWallRE stay (they catch the GLM "limit will reset at" banner
// that sessionsignals.IsLimitError does not) — sessionsignals only WIDENS coverage.
func ClassifyNoCommitReason(tail string, size int64) string {
	switch {
	case strings.Contains(tail, "SELF_MODIFY"):
		return NoCommitSelfModify
	case strings.Contains(tail, "POLICY_BLOCK"):
		return NoCommitPolicyBlock
	case sessionsignals.NeedsLoginPrompt(tail) || sessionsignals.IsAuthError(tail):
		return NoCommitAuthWall
	case capBannerRE.MatchString(tail) || glmWallRE.MatchString(tail) || sessionsignals.IsLimitError(tail):
		return NoCommitUsageCap
	case sessionsignals.UnknownModel(tail):
		return NoCommitModelUnknown
	case sessionsignals.IsAPIError(tail):
		return NoCommitRateLimit
	case strings.Contains(tail, "OFF_TRUNK"):
		return NoCommitOffTrunk
	case size >= 0 && size <= StubLogMaxBytes && noopBannerRE.MatchString(tail):
		return NoCommitBannerNoop
	default:
		return NoCommitUnknown
	}
}

// ModelSwitchableReason reports whether a no-commit reason is one a switch to a
// DIFFERENT model can address — a usage/weekly cap (a different model has its own
// bucket), an unknown/unentitled model id, or a transient rate-limit/overload. A
// genuine auth wall, a guard refusal, an off-trunk refusal, or a banner no-op is NOT
// model-switchable, so Layer-2 re-dispatch skips them.
func ModelSwitchableReason(reason string) bool {
	switch reason {
	case NoCommitUsageCap, NoCommitModelUnknown, NoCommitRateLimit:
		return true
	}
	return false
}

// NextDowngradeModel returns the next model to re-dispatch on after a model-switchable
// no-commit exit whose slot ran on `current`, walking `chain` (the ordered downgrade
// ladder). A seat-default slot (current == "") advances to the head of the chain: its
// walled model was the seat's own default, so the first EXPLICIT chain model is the first
// genuine switch. A pinned slot advances to the entry AFTER its model. ok is false when the
// ladder is exhausted (current is the last chain entry, or the chain is empty) — a further
// switch would just re-offer a model already walled, so the caller must STOP, not loop.
func NextDowngradeModel(current string, chain []string) (string, bool) {
	if len(chain) == 0 {
		return "", false
	}
	current = strings.ToLower(strings.TrimSpace(current))
	if current == "" {
		return chain[0], true
	}
	for i, m := range chain {
		if strings.ToLower(strings.TrimSpace(m)) == current {
			if i+1 < len(chain) {
				return chain[i+1], true
			}
			return "", false // last rung: ladder exhausted
		}
	}
	// The walled model is not on the ladder at all — the head is a genuine different model.
	return chain[0], true
}

// ModelDowngradeReDispatch maps each issue whose last finished slot exited CLAIM_NO_COMMIT
// with a MODEL-SWITCHABLE reason (usage_cap / model_unknown / rate_limit) to the NEXT model
// to re-dispatch it on, walking the downgrade chain from the slot's own .model. This is the
// pure half of Layer-2 in-tick re-dispatch: a switchable wall is not HELD (unlike a guard
// refusal) but re-dispatching it on the SAME walled model just walls again, so it advances
// one rung down the ladder. An issue whose ladder is exhausted is omitted — the caller lets
// the normal pick/cooldown handle it rather than re-storming a model that will wall.
func ModelDowngradeReDispatch(records []WitnessRecord, chain []string) map[int]string {
	out := map[int]string{}
	for _, rec := range records {
		if rec.Claim != ClaimNoCommit || !ModelSwitchableReason(rec.Reason) {
			continue
		}
		if next, ok := NextDowngradeModel(rec.Model, chain); ok {
			out[rec.Issue] = next
		}
	}
	return out
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
