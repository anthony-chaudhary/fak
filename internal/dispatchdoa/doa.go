// Package dispatchdoa detects DOA (dead-on-arrival) dispatch spawns: a worker that
// the dispatcher DID spawn but that died before it ever began work.
//
// Invariant: dispatch DOA classification is fail-closed and deterministic.
//
// # Why this exists (#5868)
//
// Between 2026-07-28 and 2026-08-03 the dispatcher spawned every worker with an
// argv flag the `fak.exe` on PATH did not define. Go's flag package rejected the
// argv and exited before the guard launched the agent, so not one turn was served.
// Re-derived from this repo's own retained .dispatch-runs corpus (2386 resolve-*.log
// records, 2026-07-16..08-07):
//
//	07-16..07-27  1717 runs    0 DOA    0.0%
//	2026-07-28      34 runs   30 DOA   88.2%
//	2026-07-29      55 runs   55 DOA  100.0%
//	2026-07-30      43 runs   43 DOA  100.0%
//	2026-07-31      53 runs   53 DOA  100.0%
//	2026-08-01      61 runs   61 DOA  100.0%
//	2026-08-02      56 runs   56 DOA  100.0%
//	2026-08-03      80 runs   52 DOA   65.0%
//	08-04..08-07   287 runs    0 DOA    0.0%
//
// 350 dead worker-units — 91.6% of the 382 spawned across that 7-day window — and
// nothing surfaced it. Every one still wrote a .witness carrying
// `claim: CLAIM_NO_COMMIT, reason: unknown` (verified: 350/350), and `unknown` is
// already the fleet's largest no-commit bucket, so a TOTAL outage was
// indistinguishable from ordinary background noise. `fak dispatch status`, the
// surface an operator actually reads, showed "0 live worker(s)" throughout — which
// reads exactly like an idle fleet.
//
// # The signal, and why it cannot confuse a fast legitimate run
//
// The discriminator is the GUARD'S AGENT-LAUNCH BANNER — the
// `fak guard <ver> — kernel-adjudicated: <agent argv>` line cmd/fak/guard_banner.go
// prints once the guard has parsed its flags, resolved the agent command and is
// about to hand it the wire. It is a RECORDED FIELD in the worker log, not a timing
// or size heuristic, and it is emitted BEFORE the agent's first turn. So a worker
// that legitimately starts, finds nothing to do and exits two seconds later STILL
// carries it; a worker that died at argv parse CANNOT.
//
// Measured over the whole 2386-record corpus, the separation is total:
//
//	spawn header + launch banner, no flag-parse error : 2036 records (every healthy run)
//	spawn header, NO launch banner, flag-parse error  :  350 records (every outage run)
//	any other combination                             :    0 records
//
// The size gate corroborates it with a wide margin rather than carrying the
// decision: the LARGEST banner-less log is 1595 bytes, the SMALLEST log carrying a
// banner is 6094 bytes — a 4.5 KiB gap with nothing in it. StubMaxBytes sits at
// 4096, ~2.5x above every observed DOA log and ~1.5x below every observed real one.
//
// Both must hold, so neither can misfire alone: a hypothetical `--quiet` worker
// (banner suppressed) is still over the size floor, and a genuinely tiny real log
// still carries the banner. Fail-open throughout — an unstat-able log, a log with no
// spawn header, or any launch marker at all yields NOT DOA. The detector never
// invents a death it did not see.
//
// # Not keyed to one flag
//
// The next drift will be a different flag, a missing binary or a bad working
// directory. The SHAPE — "the dispatcher spawned it, it wrote a stub, it never
// reached launch" — is what is detected; the literal `flag provided but not
// defined` string is one recognized CAUSE among several, and a shape match with no
// recognized signature is reported honestly as CauseUnrecognized rather than
// dropped.
package dispatchdoa

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// SpawnHeaderPrefix is the one-line header tools/issue_resolve_dispatch.py
// (spawn_issue_worker) flushes into the worker log BEFORE exec, so a later reader
// can tell "the dispatcher never ran the child" from "the child ran and wrote
// nothing". Requiring it is what makes this detector answer a question about a
// SPAWNED worker: a log that never carried it is some other artifact, and is never
// classified DOA.
const SpawnHeaderPrefix = "# fak-spawn "

// StubMaxBytes is the size ceiling for a DOA-shaped log. A worker that reached
// launch has already printed the guard's multi-KiB startup report, so it clears this
// by a wide margin: on the #5868 corpus the largest DOA log was 1595 bytes and the
// smallest launched log 6094 bytes. Set well inside that empty gap — high enough
// that a bigger stub (a longer usage block, a stack of exec diagnostics) is still
// caught, low enough that no real run can reach it.
//
// Deliberately NOT dispatchtick.StubLogMaxBytes (512): that floor exists for the
// banner-noop class and is BELOW every one of the 350 DOA logs, which is one reason
// the outage slipped past the existing stub check.
const StubMaxBytes = 4096

// HeadBytes bounds how much of a log the classifier inspects. A DOA log is a stub by
// construction, so its head is the whole file; the bound only protects a caller that
// hands over more than it promised.
const HeadBytes = 8192

// The cause vocabulary: WHY a spawned worker never reached launch. Closed set, most
// specific first (see Classify). CauseUnrecognized is a first-class member, not a
// failure: a new drift mode shows up as a rising unrecognized share instead of being
// silently misfiled into a known cause.
const (
	// CauseFlagParse: the binary's flag parser rejected an argv the dispatcher
	// passed — dispatcher and binary have drifted. This is #5868 itself.
	CauseFlagParse = "flag_parse"
	// CauseExecFailure: the OS could not run the image at all (missing binary,
	// wrong architecture, no execute permission).
	CauseExecFailure = "exec_failure"
	// CauseWorkingDir: the spawn cwd did not exist or was not usable.
	CauseWorkingDir = "working_dir"
	// CauseUsageError: the binary printed its usage block and exited without a more
	// specific diagnostic (an unknown subcommand, a missing required argument).
	CauseUsageError      = "usage_error"
	CauseAuthMissing     = "auth_missing"
	CauseAuthExpired     = "auth_expired"
	CauseAuthMismatched  = "auth_mismatched"
	CauseGatewayRejected = "gateway_rejected"
	CauseAuthInvalid     = "auth_invalid"
	CauseProcessStart    = "process_start"
	CauseImmediateExit   = "immediate_exit"
	CauseMissingEvidence = "missing_evidence"
	// CauseUnrecognized: the DOA shape holds — spawned, stub log, never launched —
	// but no known signature matched. Kept honest rather than folded away.
	CauseUnrecognized = "unrecognized"
)

var causeNextActions = map[string]string{
	CauseFlagParse:       "repair the worker argv contract, then rerun one canary",
	CauseExecFailure:     "repair the worker runtime/executable path before retrying",
	CauseWorkingDir:      "repair the declared worker directory before retrying",
	CauseUsageError:      "inspect the usage diagnostic and repair the launch contract",
	CauseAuthMissing:     "select a seat with a credential, then rerun account/read with refreshToken=true",
	CauseAuthExpired:     "refresh or re-login the selected account before retrying",
	CauseAuthMismatched:  "repair the selected-seat credential provenance before retrying",
	CauseGatewayRejected: "re-login the selected account and recheck the exact guarded gateway route before retrying",
	CauseAuthInvalid:     "refresh or re-login the selected account, then rerun account/read with refreshToken=true",
	CauseProcessStart:    "repair the executable/path/process-start failure before retrying",
	CauseImmediateExit:   "inspect the first post-launch error and repair the worker route",
	CauseMissingEvidence: "repair launch evidence capture; do not infer success from an empty log",
	CauseUnrecognized:    "inspect the sampled log and add a stable typed classifier for the novel signature",
}

var (
	// launchMarkerRE is the positive proof that the worker BEGAN WORK. Any one of
	// these means NOT DOA, whatever the size:
	//   - "kernel-adjudicated:" — the guard's agent-launch banner (cmd/fak/guard_banner.go),
	//     printed once flags parse and the agent argv resolves, before the first turn;
	//   - "fak-turn " — the per-turn economy trace the guard emits per adjudicated turn;
	//   - "gateway    :" / "gateway http" — the guard's startup report gateway row, which
	//     only prints once the in-process gateway is listening.
	// Independent markers, so suppressing any one (e.g. a future --quiet worker) does
	// not by itself manufacture a DOA.
	launchMarkerRE = regexp.MustCompile(`kernel-adjudicated:|(?m)^\s*fak-turn |(?mi)^\s*gateway\s*[:]|gateway http`)

	// flagParseRE is Go's flag package rejecting an argv it does not define, plus the
	// equivalent from the other parsers a worker command can reach. The #5868 literal
	// is the FIRST alternative, not the only one.
	flagParseRE = regexp.MustCompile(`flag provided but not defined` +
		`|flag needs an argument` +
		`|(?i)unknown (?:flag|option|shorthand flag)` +
		`|(?i)unrecognized (?:option|argument)` +
		`|(?i)no such option`)

	// execFailureRE is the OS refusing to run the image.
	execFailureRE = regexp.MustCompile(`(?i)executable file not found` +
		`|exec format error` +
		`|(?i)is not recognized as an internal or external command` +
		`|(?i)command not found` +
		`|(?i)cannot execute binary file` +
		`|(?i)permission denied.*exec|exec.*permission denied` +
		`|fork/exec`)

	// workingDirRE is the spawn cwd being absent or unusable.
	workingDirRE = regexp.MustCompile(`(?i)chdir .*: (?:no such file or directory|the system cannot find)` +
		`|(?i)the directory name is invalid` +
		`|(?i)not a git repository` +
		`|(?i)getwd: no such file or directory`)

	// usageRE is a bare usage block — the weakest signature, checked last.
	usageRE             = regexp.MustCompile(`(?mi)^usage: `)
	authMissingRE       = regexp.MustCompile(`(?i)(\bauth_missing\b|credential_reason"?\s*[:=]\s*"?missing\b|no account home|no credential|not logged in|login required)`)
	authExpiredRE       = regexp.MustCompile(`(?i)(\bauth_expired\b|credential_reason"?\s*[:=]\s*"?expired\b|credential (?:is )?expired|token expired|expired token)`)
	authMismatchedRE    = regexp.MustCompile(`(?i)(\bauth_mismatched\b|credential_reason"?\s*[:=]\s*"?mismatched\b|credential (?:is )?mismatched|different account home|credential provenance mismatch)`)
	authGatewayRejectRE = regexp.MustCompile(`(?i)(\b(?:auth_)?gateway_rejected\b|credential_reason"?\s*[:=]\s*"?gateway_rejected\b|invalid_refresh_token|invalid refresh token|401 unauthorized|gateway rejected the credential|upstream rejected the credential)`)
	authInvalidRE       = regexp.MustCompile(`(?i)(auth_invalid|authentication failed|unauthorized)`)
	processStartRE      = regexp.MustCompile(`(?i)(executable file not found|failed to start|process start|createprocess)`)
	immediateExitRE     = regexp.MustCompile(`(?i)(exited abnormally|nonzero_exit|immediate exit)`)
)

// Verdict is one run's classification.
type Verdict struct {
	// DOA is true only when the full shape held: spawned, stub-sized, no launch marker.
	DOA bool
	// Cause is one of the Cause* constants when DOA, and "" otherwise.
	Cause string
	// Signature is a scrubbed fingerprint for an unrecognized cause.
	Signature string
}

// Invariant: dispatch DOA classification is fail-closed and deterministic.
// Any malformed log, unstat-able file, missing spawn header, or observed
// launch marker yields a non-DOA verdict to prevent false positive attributions.
//
// Guard: files lacking the required SpawnHeaderPrefix or exceeding StubMaxBytes
// are rejected early fail-closed to avoid unnecessary scans of healthy runs.
//
// Classify grades one worker log into a DOA verdict from its HEAD (the first
// HeadBytes; a DOA log is a stub so that is all of it) and its total size in bytes.
// size < 0 means the log could not be stat'd.
//
// PURE and FAIL-OPEN. Every gate must pass affirmatively for a DOA verdict:
//
//  1. size must be known and at/under StubMaxBytes — a launched worker's startup
//     report alone clears this by ~1.5x, so an over-floor log is never DOA and is
//     never even read;
//  2. head must carry SpawnHeaderPrefix — proof the DISPATCHER spawned this worker,
//     so a foreign, truncated or hand-written artifact is never graded;
//  3. head must carry NO launch marker — the discriminator. A run that reached the
//     guard's agent-launch banner (or a turn trace, or a live gateway row) BEGAN
//     WORK, and a legitimately instant run still prints it, so a fast real run can
//     never be labelled DOA.
//
// Any doubt returns {false, ""}. A false negative costs one missed row in a
// breakdown; a false positive would blame the dispatcher for a healthy worker's
// choice not to commit, which is the more expensive error here.
func Classify(head string, size int64) Verdict {
	if size < 0 || size > StubMaxBytes {
		return Verdict{}
	}
	if len(head) > HeadBytes {
		head = head[:HeadBytes]
	}
	if !strings.HasPrefix(strings.TrimLeft(head, "\r\n"), SpawnHeaderPrefix) {
		return Verdict{}
	}
	if launchMarkerRE.MatchString(head) {
		return Verdict{}
	}
	// Past every gate: the dispatcher spawned it, it wrote a stub, it never launched.
	// Precedence runs most-specific first so a flag-parse death (which ALSO prints a
	// usage block) is never demoted to the generic usage_error class.
	if cause := CredentialCause(head); cause != "" {
		return Verdict{DOA: true, Cause: cause}
	}
	switch {
	case immediateExitRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseImmediateExit}
	case flagParseRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseFlagParse}
	case execFailureRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseExecFailure}
	case processStartRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseProcessStart}
	case workingDirRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseWorkingDir}
	case missingEvidence(head):
		return Verdict{DOA: true, Cause: CauseMissingEvidence}
	case usageRE.MatchString(head):
		return Verdict{DOA: true, Cause: CauseUsageError}
	default:
		return Verdict{DOA: true, Cause: CauseUnrecognized, Signature: unknownSignature(head)}
	}
}

// CredentialCause returns the scrubbed selected-seat credential cause carried by
// a prelaunch refusal or gateway 401. An empty result means the text is not a
// recognized credential failure. Callers may expose the returned token, never the
// source text, so diagnostics do not echo credential material.
func CredentialCause(text string) string {
	switch {
	case authMissingRE.MatchString(text):
		return CauseAuthMissing
	case authExpiredRE.MatchString(text):
		return CauseAuthExpired
	case authMismatchedRE.MatchString(text):
		return CauseAuthMismatched
	case authGatewayRejectRE.MatchString(text):
		return CauseGatewayRejected
	case authInvalidRE.MatchString(text):
		return CauseAuthInvalid
	default:
		return ""
	}
}

func missingEvidence(head string) bool {
	_, body, found := strings.Cut(head, "\n")
	return !found || strings.TrimSpace(body) == ""
}

// unknownSignature identifies a novel startup failure without echoing credentials,
// paths, prompts, or other potentially sensitive log content into status JSON.
func unknownSignature(head string) string {
	_, body, _ := strings.Cut(head, "\n")
	body = strings.TrimSpace(body)
	if body == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(body))
	return fmt.Sprintf("sha256:%x", sum[:8])
}

// Verdict rungs for a folded window. A fleet whose spawns are dying is not a
// degraded fleet, it is a stopped one, so the ladder is deliberately short.
const (
	// StatusClear: no DOA spawn in the window.
	StatusClear = "clear"
	// StatusWarn: at least one DOA spawn. The healthy baseline is exactly zero
	// (0 of 2036 records across the 22 non-outage days of the #5868 corpus), so a
	// single DOA is already off-baseline and worth an operator's eye.
	StatusWarn = "warn"
	// StatusAlarm: DOA spawns dominate the window — dispatcher and binary have
	// drifted and fleet throughput is zero by construction.
	StatusAlarm = "alarm"
)

// AlarmMinDOA and AlarmRate are the alarm rung. The rate condition is what makes it
// mean "the fleet is stopped" rather than "a worker died"; the count condition keeps
// a one-run window (1 of 1 = 100%) from alarming on a single fluke.
//
// Calibrated against the outage: day one, 2026-07-28, was 30 DOA of 34 runs = 88.2%,
// which clears both conditions comfortably — this rung WOULD have fired on day one
// rather than day six. Every one of the six following days was 65-100%.
const (
	AlarmMinDOA = 3
	AlarmRate   = 0.5
)

// Run is one finished worker slot handed to Fold: its log name (for the operator's
// sample list) and its Verdict.
type Run struct {
	Log     string
	Verdict Verdict
}

// Report is the folded spawn-health view of a window.
type Report struct {
	// Runs is the DENOMINATOR: every finished worker slot considered in the window,
	// DOA or not. Reported so a rate can never be read without its base.
	Runs int
	// DOA is how many of them never reached launch.
	DOA int
	// Rate is DOA/Runs, or 0 when the window is empty.
	Rate float64
	// Causes counts DOA runs by Cause.
	Causes map[string]int
	// NextActions gives one stable remediation for every observed cause.
	NextActions map[string]string
	// Diagnostics maps unrecognized log paths to scrubbed failure fingerprints.
	Diagnostics map[string]string
	// Status is StatusClear / StatusWarn / StatusAlarm.
	Status string
	// Sample names up to SampleMax DOA logs, oldest-first by name, so an operator can
	// open the evidence instead of taking the count on faith.
	Sample []string
}

// SampleMax caps the named-evidence list; the count and rate are always exact.
const SampleMax = 5

// Fold folds finished worker slots into the spawn-health report. Pure over its
// input: no clock, no I/O. An empty window folds to a clear, zero-rate report.
func Fold(runs []Run) Report {
	rep := Report{Runs: len(runs), Status: StatusClear}
	var doaLogs []string
	for _, r := range runs {
		if !r.Verdict.DOA {
			continue
		}
		rep.DOA++
		if rep.Causes == nil {
			rep.Causes = map[string]int{}
		}
		cause := r.Verdict.Cause
		if cause == "" {
			cause = CauseUnrecognized
		}
		rep.Causes[cause]++
		if rep.NextActions == nil {
			rep.NextActions = map[string]string{}
		}
		rep.NextActions[cause] = causeNextActions[cause]
		if cause == CauseUnrecognized && r.Log != "" && r.Verdict.Signature != "" {
			if rep.Diagnostics == nil {
				rep.Diagnostics = map[string]string{}
			}
			rep.Diagnostics[r.Log] = r.Verdict.Signature
		}
		if r.Log != "" {
			doaLogs = append(doaLogs, r.Log)
		}
	}
	if rep.Runs > 0 {
		rep.Rate = float64(rep.DOA) / float64(rep.Runs)
	}
	switch {
	case rep.DOA == 0:
		rep.Status = StatusClear
	case rep.DOA >= AlarmMinDOA && rep.Rate >= AlarmRate:
		rep.Status = StatusAlarm
	default:
		rep.Status = StatusWarn
	}
	sort.Strings(doaLogs)
	if len(doaLogs) > SampleMax {
		doaLogs = doaLogs[:SampleMax]
	}
	rep.Sample = doaLogs
	return rep
}

// TopCauses renders the cause histogram as a deterministic "cause=N" list, biggest
// first then alphabetical, for a one-line operator readout.
func (r Report) TopCauses() []string {
	if len(r.Causes) == 0 {
		return nil
	}
	keys := make([]string, 0, len(r.Causes))
	for k := range r.Causes {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if r.Causes[keys[i]] != r.Causes[keys[j]] {
			return r.Causes[keys[i]] > r.Causes[keys[j]]
		}
		return keys[i] < keys[j]
	})
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+itoa(r.Causes[k]))
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
