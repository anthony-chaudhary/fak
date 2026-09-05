package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/walkfiles"
)

const codexLoopSchema = "fak.sessions.codex_loop.v1"
const codexLoopRecentSchema = "fak.sessions.codex_loop_recent.v1"
const codexLoopHookOverrideEnv = "FAK_ALLOW_DIRECT_CODEX_CONTINUE"
const codexLoopHookHardenedEnv = "FAK_CODEX_SUBMIT_HARDENED"
const codexLoopHookAuditJournalEnv = "FAK_AUDIT_JOURNAL"
const codexLoopHookTimeoutReason = "codex_hardened_submit_timeout"
const codexLoopHookDefaultBudget = 500 * time.Millisecond
const codexLoopLaunchMaxBytes int64 = 4 << 20
const codexLoopLaunchPrefixBytes int64 = 256 << 10

// These two seams are variables so the hook's hard failure modes can be injected
// without making the general transcript diagnoser artificial. They are only mutated
// by the serial CodexLoopHook witness tests.
var codexLoopHookBudget = codexLoopHookDefaultBudget
var codexLoopHookDiagnose = probeCodexLoopProvider

type codexLoopDiagnosis struct {
	Schema            string                 `json:"schema"`
	Path              string                 `json:"path"`
	SessionID         string                 `json:"session_id,omitempty"`
	ParentSessionID   string                 `json:"parent_session_id,omitempty"`
	Originator        string                 `json:"originator,omitempty"`
	CLI               string                 `json:"cli_version,omitempty"`
	ModelProvider     string                 `json:"model_provider,omitempty"`
	WorkingDir        string                 `json:"working_directory,omitempty"`
	GuardWitnessed    bool                   `json:"guard_witnessed,omitempty"`
	GitCommit         string                 `json:"git_commit,omitempty"`
	GitBranch         string                 `json:"git_branch,omitempty"`
	StartedAt         string                 `json:"started_at,omitempty"`
	LastEventAt       string                 `json:"last_event_at,omitempty"`
	FinalStatus       string                 `json:"final_status,omitempty"`
	FinalTokensUsed   int64                  `json:"final_tokens_used,omitempty"`
	FinalTimeSeconds  int64                  `json:"final_time_seconds,omitempty"`
	TurnAborted       bool                   `json:"turn_aborted,omitempty"`
	AbortReason       string                 `json:"abort_reason,omitempty"`
	AbortDurationMS   int64                  `json:"abort_duration_ms,omitempty"`
	ToolCalls         int                    `json:"tool_calls"`
	ToolOutputs       int                    `json:"tool_outputs"`
	LastTokenTotal    int64                  `json:"last_token_total,omitempty"`
	LastTokenInput    int64                  `json:"last_token_input,omitempty"`
	LastTokenOutput   int64                  `json:"last_token_output,omitempty"`
	RepeatedOutcomes  []codexRepeatedOutcome `json:"repeated_outcomes,omitempty"`
	LivelockNotices   []codexLivelockNotice  `json:"livelock_notices,omitempty"`
	Verdict           string                 `json:"verdict"`
	Reason            string                 `json:"reason,omitempty"`
	NextAction        string                 `json:"next_action,omitempty"`
	ObservabilityGaps []string               `json:"observability_gaps,omitempty"`
	abruptlyEnded     bool
}

type codexLoopLaunchScan struct {
	BytesRead int64
	Truncated bool
}

type codexLoopRecentReport struct {
	Schema             string                 `json:"schema"`
	CodexHome          string                 `json:"codex_home"`
	SinceHours         float64                `json:"since_hours,omitempty"`
	Limit              int                    `json:"limit"`
	Scanned            int                    `json:"scanned"`
	LoopCount          int                    `json:"loop_count"`
	ActionCount        int                    `json:"action_count"`
	OKCount            int                    `json:"ok_count"`
	ProviderCounts     map[string]int         `json:"provider_counts,omitempty"`
	UnguardedCount     int                    `json:"unguarded_count,omitempty"`
	GuardedLoopCount   int                    `json:"guarded_loop_count,omitempty"`
	UnguardedLoopCount int                    `json:"unguarded_loop_count,omitempty"`
	UnknownLoopCount   int                    `json:"unknown_loop_count,omitempty"`
	ToolCalls          int                    `json:"tool_calls"`
	ToolOutputs        int                    `json:"tool_outputs"`
	LastTokenTotalSum  int64                  `json:"last_token_total_sum,omitempty"`
	Verdict            string                 `json:"verdict"`
	Reason             string                 `json:"reason,omitempty"`
	NextAction         string                 `json:"next_action,omitempty"`
	TopRepeated        []codexRepeatedOutcome `json:"top_repeated,omitempty"`
	Diagnoses          []codexLoopDiagnosis   `json:"diagnoses"`
}

type codexLoopHookInput struct {
	SessionID      string `json:"session_id"`
	ThreadID       string `json:"thread_id"`
	ConversationID string `json:"conversation_id"`
	HookEventName  string `json:"hook_event_name"`
	TurnID         string `json:"turn_id"`
	Prompt         string `json:"prompt"`
}

type codexLoopHookBlock struct {
	Decision string `json:"decision"`
	Reason   string `json:"reason"`
}

type codexLoopHookOutput struct {
	Decision           string                       `json:"decision,omitempty"`
	Reason             string                       `json:"reason,omitempty"`
	Continue           *bool                        `json:"continue,omitempty"`
	SystemMessage      string                       `json:"systemMessage,omitempty"`
	HookSpecificOutput *codexLoopHookSpecificOutput `json:"hookSpecificOutput,omitempty"`
}

type codexLoopHookSpecificOutput struct {
	HookEventName     string `json:"hookEventName"`
	AdditionalContext string `json:"additionalContext"`
}

func (o codexLoopHookOutput) additionalContext() string {
	if o.HookSpecificOutput == nil {
		return ""
	}
	return o.HookSpecificOutput.AdditionalContext
}

type codexWorkflowDefaultWitness struct {
	Schema         string `json:"schema"`
	SessionID      string `json:"session_id"`
	FirstPromptAt  string `json:"first_prompt_at"`
	Classification string `json:"classification"`
	Decision       string `json:"decision"`
	Reason         string `json:"reason"`
}

type codexLoopHookBlockWitness struct {
	Timestamp     string `json:"timestamp"`
	Event         string `json:"event"`
	SessionID     string `json:"session_id"`
	ModelProvider string `json:"model_provider"`
	Reason        string `json:"reason"`
}

type codexRepeatedOutcome struct {
	Tool             string   `json:"tool"`
	OutputDigest     string   `json:"output_digest"`
	OutputExcerpt    string   `json:"output_excerpt,omitempty"`
	Count            int      `json:"count"`
	LongestRun       int      `json:"longest_run"`
	FirstTimestamp   string   `json:"first_timestamp,omitempty"`
	LastTimestamp    string   `json:"last_timestamp,omitempty"`
	TokenTotal       int64    `json:"token_total,omitempty"`
	TokenEvents      int      `json:"token_events,omitempty"`
	ArgsDigestCount  int      `json:"args_digest_count,omitempty"`
	FirstArgsDigest  string   `json:"first_args_digest,omitempty"`
	OtherArgsDigests []string `json:"other_args_digests,omitempty"`
}

type codexLivelockNotice struct {
	RepeatedCall   string `json:"repeated_call"`
	Approach       string `json:"approach,omitempty"`
	Count          int    `json:"count"`
	MinRepeat      int    `json:"min_repeat,omitempty"`
	MaxRepeat      int    `json:"max_repeat,omitempty"`
	FirstTimestamp string `json:"first_timestamp,omitempty"`
	LastTimestamp  string `json:"last_timestamp,omitempty"`
}

type codexPendingToolCall struct {
	Tool       string
	ArgsDigest string
	Timestamp  string
}

type codexOutcomeKey struct {
	Tool         string
	OutputDigest string
}

type codexOutcomeAccum struct {
	out            codexRepeatedOutcome
	argsDigests    map[string]bool
	currentRun     int
	latestTokenHit bool
}

func sessionsCodexLoop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "archive" {
		return runSessionsCodexLoopArchive(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("sessions codex-loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	project := addDogfoodProjectFlags(fs)
	sessionID := fs.String("session", "", "Codex session id to find under --codex-home/sessions (default: $CODEX_THREAD_ID when set)")
	path := fs.String("path", "", "explicit Codex session JSONL path")
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	recent := fs.Bool("recent", false, "scan recent Codex session JSONL files instead of one path")
	sinceHours := fs.Float64("since-hours", 24, "with --recent, only scan files modified within N hours (0 = all)")
	limit := fs.Int("limit", 20, "with --recent, cap sessions scanned after newest-first sorting")
	asJSON := fs.Bool("json", false, "emit a machine-readable diagnosis")
	failOn := fs.String("fail-on", "none", "exit 1 when verdict/posture reaches this threshold: none|loop|action|unguarded (action also fails LOOP)")
	syncIssues := fs.Bool("sync-issues", false, "with --recent, fold LOOP classes into deduplicated GitHub issues (one per tool/output-digest class); output becomes the issue plan/result")
	issueLive := fs.Bool("live", false, "with --sync-issues, create/update issues with gh (else dry-run plan)")
	issueFetchExisting := fs.Bool("fetch-existing", false, "with --sync-issues, dry-run but query gh to classify create vs update")
	issueRepo := fs.String("repo", "", "with --sync-issues, owner/repo for gh (default: current repo)")
	issueMilestone := fs.String("milestone", dogfoodissues.DefaultMilestone, "with --sync-issues, milestone title to assign to created/updated issues")
	issueLimit := fs.Int("issue-limit", 300, "with --sync-issues, existing-issue scan limit for live/fetch modes")
	issueExistingJSON := fs.String("existing-json", "", "with --sync-issues, fixture list of existing gh issues for dry-run tests")
	var issueLabels stringList
	fs.Var(&issueLabels, "label", "with --sync-issues, extra label for newly-created issues; repeatable")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak sessions codex-loop [--session ID | --path FILE | --recent] [--codex-home DIR] [--json] [--fail-on none|loop|action|unguarded] [--sync-issues [--live]]")
		fs.PrintDefaults()
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if !*recent && (*syncIssues || *issueLive || *issueFetchExisting) {
		fmt.Fprintln(stderr, "fak sessions codex-loop: --sync-issues/--live/--fetch-existing require --recent")
		return 2
	}
	if *recent {
		if strings.TrimSpace(*path) != "" || strings.TrimSpace(*sessionID) != "" {
			fmt.Fprintln(stderr, "fak sessions codex-loop: --recent cannot be combined with --path or --session")
			return 2
		}
		r, err := diagnoseRecentCodexLoops(*codexHome, *sinceHours, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
			return 1
		}
		if *syncIssues {
			return runCodexLoopSyncIssues(stdout, stderr, r, *asJSON, codexLoopIssueOptions{
				Live:            *issueLive,
				FetchExisting:   *issueFetchExisting,
				Repo:            strings.TrimSpace(*issueRepo),
				Milestone:       strings.TrimSpace(*issueMilestone),
				ExistingJSON:    strings.TrimSpace(*issueExistingJSON),
				Limit:           *issueLimit,
				Labels:          []string(issueLabels),
				ParentIssue:     *project.parent,
				ProjectBaseline: project.baseline(), CompletionStandard: *project.standard, TargetEnvelope: *project.target, WitnessedEnvelope: *project.witnessed,
			})
		}
		gateCode, ok := codexLoopFailOnRecentExitCode(r, *failOn)
		if !ok {
			return codexLoopInvalidFailOn(stderr, *failOn)
		}
		return finishCodexLoopReport(stdout, stderr, *asJSON, r,
			func() string { return renderCodexLoopRecentReport(r) },
			gateCode, *failOn, r.Verdict, codexLoopRecentGateReason(r, *failOn))
	}
	if strings.TrimSpace(*path) != "" && strings.TrimSpace(*sessionID) != "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop: use only one of --path or --session")
		return 2
	}
	resolved, fh, code, ok := openCodexLoopSessionReported(stderr, *codexHome, *sessionID, *path,
		"fak sessions codex-loop: %v\n", "fak sessions codex-loop: open %s: %v\n", 2, 1)
	if !ok {
		return code
	}
	defer fh.Close()
	d, err := diagnoseCodexLoop(fh, resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
		return 1
	}
	// The provider name in session_meta is operator-controlled. Bind the plain
	// diagnostic path to the same durable guard-launch witness used by the hook,
	// recent-session report, and pre-spawn launcher gate before classifying it.
	bindCodexGuardWitness(&d, *codexHome, d.SessionID)
	gateCode, ok := codexLoopFailOnDiagnosisExitCode(d, *failOn)
	if !ok {
		return codexLoopInvalidFailOn(stderr, *failOn)
	}
	return finishCodexLoopReport(stdout, stderr, *asJSON, d,
		func() string { return renderCodexLoopDiagnosis(d) },
		gateCode, *failOn, d.Verdict, codexLoopDiagnosisGateReason(d, *failOn))
}

// codexLoopInvalidFailOn refuses an unrecognized --fail-on spelling. The --recent fold and
// the single-session diagnosis accept the same four gate levels, so they name the same four
// in the same usage line and exit 2 alike: a gate the command cannot understand must never
// quietly degrade into "none" on one path and refuse on the other.
func codexLoopInvalidFailOn(stderr io.Writer, failOn string) int {
	fmt.Fprintf(stderr, "fak sessions codex-loop: invalid --fail-on %q (want none, loop, action, or unguarded)\n", failOn)
	return 2
}

// finishCodexLoopReport publishes one codex-loop report and returns the command's exit code.
// The --recent fold and the single-session diagnosis publish identically: the payload as JSON
// or as rendered text on stdout, and then -- whenever the --fail-on gate refused -- the typed
// REFUSE line on stderr naming the gate, the verdict and the reason. Publishing through one
// path is what stops a refusal from being announced on the text surface and swallowed on the
// JSON one (or the reverse). render is a thunk so the JSON form never pays to build a text
// report it discards.
func finishCodexLoopReport(stdout, stderr io.Writer, asJSON bool, payload any, render func() string, gateCode int, failOn, verdict, reason string) int {
	if asJSON {
		if code := encodeJSONOrFail(stdout, stderr, payload, "fak sessions codex-loop"); code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, render())
	}
	if gateCode != 0 {
		fmt.Fprintf(stderr, "fak sessions codex-loop: gate REFUSE fail-on=%s verdict=%s reason=%s\n", codexLoopFailOnName(failOn), verdict, reason)
	}
	return gateCode
}

// sessionsCodexLoopHook is the opt-in turn-boundary seam for #3023 and #9234.
// Guarded children use it for scoped first-prompt context; hardened mode additionally
// reuses transcript diagnosis to block direct providers with the advisory gate's typed
// reason. Ordinary direct Codex prompts return before reading stdin or the transcript.
type codexLoopHookRunResult struct {
	code   int
	stdout []byte
	stderr []byte
}

// sessionsCodexLoopHook keeps the untrusted filesystem/diagnosis work behind a
// bounded, buffered boundary. Nothing reaches Codex until a complete, parseable block
// decision exists. A timeout, panic, or ordinary fail-open result therefore returns 0
// with zero output bytes, regardless of what the inner path attempted to report.
func sessionsCodexLoopHook(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	if codexSubmitIsUsageSummary(argv) {
		return emitCodexSubmitUsageSummary(stdout, stderr, codexSubmitHomeArg(argv))
	}
	outcome := "allow"
	defer func() {
		if err := appendCodexSubmitUsage(argv, outcome); err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop-hook: append usage ledger: %v (continuing)\n", err)
		}
	}()
	diagnose := codexLoopHookDiagnose
	budget := codexLoopHookBudget
	resultc := make(chan codexLoopHookRunResult, 1)
	go func() {
		result := codexLoopHookRunResult{}
		defer func() {
			if recover() != nil {
				result = codexLoopHookRunResult{}
			}
			resultc <- result
		}()

		var out, errOut bytes.Buffer
		result.code = sessionsCodexLoopHookUnbounded(&out, &errOut, stdin, argv, diagnose)
		result.stdout = append([]byte(nil), out.Bytes()...)
		result.stderr = append([]byte(nil), errOut.Bytes()...)
	}()

	if budget <= 0 {
		budget = codexLoopHookDefaultBudget
	}
	timer := time.NewTimer(budget)
	defer timer.Stop()

	select {
	case result := <-resultc:
		if result.code != 0 {
			if len(result.stderr) > 0 {
				_, _ = stderr.Write(result.stderr)
			}
			return result.code
		}
		if len(result.stdout) == 0 {
			return 0
		}
		var output codexLoopHookOutput
		if err := json.Unmarshal(result.stdout, &output); err != nil {
			return 0
		}
		if output.Decision != "block" && output.additionalContext() == "" {
			return 0
		}
		if output.Decision == "block" {
			outcome = "block"
		}
		if _, err := stdout.Write(result.stdout); err != nil {
			return 1
		}
		return 0
	case <-timer.C:
		if codexLoopHookTimeoutMustBlock(argv) {
			outcome = "block"
			reason := codexLoopHookTimeoutReason + ": hardened mode could not finish the bounded direct-session diagnosis, so the prompt remains blocked; retry or use the intentional --allow-direct override"
			if err := json.NewEncoder(stdout).Encode(codexLoopHookOutput{Decision: "block", Reason: reason}); err != nil {
				return 1
			}
		}
		return 0
	}
}

func codexLoopHookTimeoutMustBlock(argv []string) bool {
	if codexLoopHookOverrideEnabled(os.Getenv(guardActiveEnv)) {
		return false
	}
	if codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookOverrideEnv)) || codexLoopHookBoolFlagEnabled(argv, "allow-direct") {
		return false
	}
	return codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookHardenedEnv)) || codexLoopHookBoolFlagEnabled(argv, "hardened")
}

func codexLoopHookBoolFlagEnabled(argv []string, name string) bool {
	flagName := "--" + name
	for _, arg := range argv {
		if arg == flagName {
			return true
		}
		if !strings.HasPrefix(arg, flagName+"=") {
			continue
		}
		enabled, err := strconv.ParseBool(strings.TrimPrefix(arg, flagName+"="))
		return err == nil && enabled
	}
	return false
}

func codexSessionArtifactPath(codexHome, sessionID, dir, invalidMessage string) (string, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return "", err
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" || filepath.Base(sessionID) != sessionID {
		return "", errors.New(invalidMessage)
	}
	return filepath.Join(home, dir, sessionID+".json"), nil
}

func codexGuardWitnessPath(codexHome, sessionID string) (string, error) {
	return codexSessionArtifactPath(codexHome, sessionID, "fak-guarded-sessions", "invalid Codex session id for guard witness")
}

func writeCodexGuardWitness(codexHome, sessionID string) error {
	path, err := codexGuardWitnessPath(codexHome, sessionID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	body, err := json.Marshal(struct {
		Schema    string `json:"schema"`
		SessionID string `json:"session_id"`
		GuardedAt string `json:"guarded_at"`
	}{"fak.codex_guard_witness.v1", sessionID, time.Now().UTC().Format(time.RFC3339Nano)})
	if err != nil {
		return err
	}
	body = append(body, '\n')
	return os.WriteFile(path, body, 0o600)
}

func codexWorkflowDefaultWitnessPath(codexHome, sessionID string) (string, error) {
	return codexSessionArtifactPath(codexHome, sessionID, "fak-workflow-defaults", "invalid Codex session id for workflow-default witness")
}

func writeCodexWorkflowDefaultWitness(codexHome string, witness codexWorkflowDefaultWitness) error {
	path, err := codexWorkflowDefaultWitnessPath(codexHome, witness.SessionID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	body, err := json.Marshal(witness)
	if err != nil {
		return err
	}
	body = append(body, '\n')
	return os.WriteFile(path, body, 0o600)
}

func codexWorkflowDefaultWitnessExists(codexHome, sessionID string) bool {
	path, err := codexWorkflowDefaultWitnessPath(codexHome, sessionID)
	if err != nil {
		return false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var witness codexWorkflowDefaultWitness
	return json.Unmarshal(raw, &witness) == nil && witness.Schema == "fak.codex_workflow_default.v1" && witness.SessionID == sessionID
}

const codexWorkflowDefaultContext = `FAK guarded-workflow default: this is the first prompt in the session. After the initial evidence-gathering step, use fak's ultracode-style workflow generation when the work is meaningfully multi-step, parallelizable, or unattended. Start with the smallest working spine, then generate the guarded workflow through fak orchestration/workflow surfaces; keep lane leases, independent witnesses, dogfood telemetry, and observable outcomes. Stay direct for a trivial one-step answer or edit.`

func codexWorkflowDefaultOutput(in codexLoopHookInput, codexHome string) (codexLoopHookOutput, bool) {
	if strings.TrimSpace(in.SessionID) == "" || strings.TrimSpace(in.Prompt) == "" || codexWorkflowDefaultWitnessExists(codexHome, in.SessionID) {
		return codexLoopHookOutput{}, false
	}
	classification := "consider-workflow"
	reason := "first guarded prompt; workflow generation is the default for reasonable multi-step work"
	if len(strings.Fields(in.Prompt)) <= 4 {
		classification = "likely-direct"
		reason = "first guarded prompt is short; preserve the direct path unless initial inspection reveals multi-step work"
	}
	witness := codexWorkflowDefaultWitness{
		Schema: "fak.codex_workflow_default.v1", SessionID: in.SessionID,
		FirstPromptAt: time.Now().UTC().Format(time.RFC3339Nano), Classification: classification,
		Decision: "inject", Reason: reason,
	}
	if err := writeCodexWorkflowDefaultWitness(codexHome, witness); err != nil {
		return codexLoopHookOutput{}, false
	}
	cont := true
	return codexLoopHookOutput{
		Continue: &cont, SystemMessage: "fak: guarded workflow default armed (" + classification + ")",
		HookSpecificOutput: &codexLoopHookSpecificOutput{
			HookEventName: "UserPromptSubmit", AdditionalContext: codexWorkflowDefaultContext,
		},
	}, true
}

func codexGuardWitnessExists(codexHome, sessionID string) bool {
	path, err := codexGuardWitnessPath(codexHome, sessionID)
	if err != nil {
		return false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var witness struct {
		Schema    string `json:"schema"`
		SessionID string `json:"session_id"`
	}
	return json.Unmarshal(raw, &witness) == nil && witness.Schema == "fak.codex_guard_witness.v1" && witness.SessionID == sessionID
}

// bindCodexGuardWitness binds the durable guard-launch witness to a freshly parsed
// diagnosis and RE-CLASSIFIES it. diagnoseCodexLoop must classify from the transcript
// alone, before any caller has looked the witness up, so the guarded/unguarded branch of
// classifyCodexLoopDiagnosis always ran with GuardWitnessed=false: a session that DID
// enter through `fak guard` was handed the direct-provider next action ("launch future
// Codex sessions through `fak codex`") — advice it had already followed, and that no
// relaunch could ever clear. The three classifier-owned fields are reset first so the
// re-run REPLACES the stale verdict instead of appending a second copy of its gaps.
func bindCodexGuardWitness(d *codexLoopDiagnosis, codexHome, sessionID string) {
	d.GuardWitnessed = codexGuardWitnessExists(codexHome, sessionID)
	d.Verdict, d.Reason, d.NextAction, d.ObservabilityGaps = "OK", "", "", nil
	classifyCodexLoopDiagnosis(d)
}

func codexLoopFirstNonEmpty(values ...string) string {
	return firstTrimmedOr("", values...)
}

func codexLoopHookOverrideInstruction() string {
	if runtime.GOOS == "windows" {
		return "set `$env:" + codexLoopHookOverrideEnv + "=1` in PowerShell"
	}
	return "prefix the Codex command with `" + codexLoopHookOverrideEnv + "=1`"
}

func resolveCodexLoopSessionPathReported(stderr io.Writer, codexHome, sessionID, path, format string) (string, bool) {
	resolved, err := resolveCodexLoopSessionPath(codexHome, sessionID, path)
	if err != nil {
		fmt.Fprintf(stderr, format, err)
		return "", false
	}
	return resolved, true
}

func openCodexLoopSessionReported(stderr io.Writer, codexHome, sessionID, path, resolveFormat, openFormat string, resolveCode, openCode int) (string, *os.File, int, bool) {
	resolved, ok := resolveCodexLoopSessionPathReported(stderr, codexHome, sessionID, path, resolveFormat)
	if !ok {
		return "", nil, resolveCode, false
	}
	fh, err := os.Open(resolved)
	if err != nil {
		fmt.Fprintf(stderr, openFormat, resolved, err)
		return "", nil, openCode, false
	}
	return resolved, fh, 0, true
}

func emitCodexWorkflowDefault(stdout, stderr io.Writer, in *codexLoopHookInput, codexHome, sessionID string) int {
	in.SessionID = sessionID
	if output, ok := codexWorkflowDefaultOutput(*in, codexHome); ok {
		if err := json.NewEncoder(stdout).Encode(output); err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop-hook: encode workflow default: %v\n", err)
			return 1
		}
	}
	return 0
}

func sessionsCodexLoopHookUnbounded(stdout, stderr io.Writer, stdin io.Reader, argv []string, diagnose func(io.Reader, string) (codexLoopDiagnosis, error)) int {
	fs := flag.NewFlagSet("sessions codex-loop-hook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	allowDirect := fs.Bool("allow-direct", false, "explicitly allow this intentional direct-provider continuation")
	hardened := fs.Bool("hardened", false, "block an unguarded direct-provider continuation")
	_ = fs.Bool("usage-summary", false, "emit weekly UserPromptSubmit invocation counts")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: fak sessions codex-loop-hook [--codex-home DIR] [--hardened] [--allow-direct] (or set %s=1 / %s=1)\n", codexLoopHookHardenedEnv, codexLoopHookOverrideEnv)
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	guardActive := codexLoopHookOverrideEnabled(os.Getenv(guardActiveEnv))
	hardenedActive := *hardened || codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookHardenedEnv))
	if !guardActive && !hardenedActive {
		return 0
	}
	if !guardActive && (*allowDirect || codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookOverrideEnv))) {
		return 0
	}
	var in codexLoopHookInput
	if err := json.NewDecoder(io.LimitReader(stdin, 1<<20)).Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: unreadable hook payload (allowing turn; recovery: relaunch with `fak codex` or inspect the Codex hook payload): %v\n", err)
		return 0
	}
	sessionID := codexLoopFirstNonEmpty(in.SessionID, in.ThreadID, in.ConversationID, os.Getenv("CODEX_THREAD_ID"))
	if sessionID != "" {
		if restorativeNote, ok := consumeCodexPendingRestoration(*codexHome, sessionID); ok {
			if guardActive {
				if err := writeCodexGuardWitness(*codexHome, sessionID); err != nil {
					fmt.Fprintf(stderr, "fak sessions codex-loop-hook: persist guard witness: %v (allowing turn)\n", err)
				}
			}
			cont := true
			out := codexLoopHookOutput{
				Continue:      &cont,
				SystemMessage: "fak: restorative invariant restoration",
				HookSpecificOutput: &codexLoopHookSpecificOutput{
					HookEventName:     "UserPromptSubmit",
					AdditionalContext: restorativeNote,
				},
			}
			if err := json.NewEncoder(stdout).Encode(out); err != nil {
				fmt.Fprintf(stderr, "fak sessions codex-loop-hook: encode restoration: %v\n", err)
				return 1
			}
			return 0
		}
	}
	if sessionID != "" && guardActive {
		if err := writeCodexGuardWitness(*codexHome, sessionID); err != nil {
			fmt.Fprintf(stderr, "fak sessions codex-loop-hook: persist guard witness: %v (allowing turn)\n", err)
		}
		return emitCodexWorkflowDefault(stdout, stderr, &in, *codexHome, sessionID)
	}
	if sessionID == "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop-hook: hook payload and CODEX_THREAD_ID have no session identifier (allowing turn; recovery: relaunch with `fak codex` or set CODEX_THREAD_ID)")
		return 0
	}

	resolved, fh, code, ok := openCodexLoopSessionReported(stderr, *codexHome, sessionID, "",
		"fak sessions codex-loop-hook: %v (allowing turn; recovery: verify --codex-home/CODEX_HOME and relaunch with `fak codex`)\n",
		"fak sessions codex-loop-hook: open %s: %v (allowing turn; recovery: verify the transcript is readable and relaunch with `fak codex`)\n", 0, 0)
	if !ok {
		return code
	}
	// Snapshot and close before diagnosis. An injected/slow/panicking diagnose path
	// must never retain a Windows file handle after the outer budget allows the turn:
	// handing fh straight to diagnose leaks the handle on exactly those two paths (a
	// panic skips the close, and a diagnose slower than the budget still holds it long
	// after sessionsCodexLoopHook has returned 0), and Windows refuses to unlink a file
	// with a live handle, so the stranded handle wedges whoever rotates or deletes the
	// transcript next. A deferred close would only cover the panic; closing here covers
	// both, because the handle is already gone before diagnose is entered.
	//
	// The snapshot is bounded by the same codexLoopLaunchMaxBytes ceiling the launch
	// scan uses, which is also probeCodexLoopProvider's max scanner token: any
	// session_meta record the streaming read could have parsed as the leading record
	// still fits. A transcript whose session_meta lies past that bound reports an empty
	// model_provider, which codexLoopDiagnosisUnguarded treats as "not unguarded" — the
	// same fail-open answer every other unreadable-transcript path here gives.
	snapshot, readErr := io.ReadAll(io.LimitReader(fh, codexLoopLaunchMaxBytes))
	closeErr := fh.Close()
	if readErr != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: read %s: %v (allowing turn; recovery: verify the transcript is readable and relaunch with `fak codex`)\n", resolved, readErr)
		return 0
	}
	d, diagnoseErr := diagnose(bytes.NewReader(snapshot), resolved)
	if diagnoseErr != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: diagnose: %v (allowing turn; recovery: inspect the rollout JSONL and relaunch with `fak codex`)\n", diagnoseErr)
		return 0
	}
	bindCodexGuardWitness(&d, *codexHome, sessionID)
	if closeErr != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: close %s: %v\n", resolved, closeErr)
	}
	if !codexLoopDiagnosisUnguarded(d) {
		return emitCodexWorkflowDefault(stdout, stderr, &in, *codexHome, sessionID)
	}

	reason := codexLoopDiagnosisGateReason(d, "unguarded") +
		": this active Codex session uses model_provider=" + strings.TrimSpace(d.ModelProvider) +
		", so fak cannot enforce the guard before the next turn. Relaunch with `fak codex`" +
		" or `fak guard -- codex`. For an intentional direct session, pass `--allow-direct`" +
		" to the hook or " + codexLoopHookOverrideInstruction() + " and resubmit."
	if err := json.NewEncoder(stdout).Encode(codexLoopHookOutput{Decision: "block", Reason: reason}); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: encode block: %v (turn not blocked; recovery: relaunch with `fak codex`)\n", err)
		return 1
	}
	if err := appendCodexLoopHookBlockWitness(os.Getenv(codexLoopHookAuditJournalEnv), codexLoopHookBlockWitness{
		Timestamp:     time.Now().UTC().Format(time.RFC3339Nano),
		Event:         "codex_continuation_guard_block",
		SessionID:     sessionID,
		ModelProvider: strings.TrimSpace(d.ModelProvider),
		Reason:        codexLoopDiagnosisGateReason(d, "unguarded"),
	}); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: append block witness: %v\n", err)
	}
	return 0
}

func appendCodexLoopHookBlockWitness(path string, witness codexLoopHookBlockWitness) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	line, err := json.Marshal(witness)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	if _, err = fh.Write(line); err != nil {
		_ = fh.Close()
		return err
	}
	return fh.Close()
}

func codexLoopHookOverrideEnabled(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func codexLoopFailOnRecentExitCode(r codexLoopRecentReport, failOn string) (int, bool) {
	if codexLoopFailOnName(failOn) == "unguarded" {
		if r.UnguardedCount > 0 {
			return 1, true
		}
		return 0, true
	}
	return codexLoopFailOnExitCode(r.Verdict, failOn)
}

func codexLoopFailOnDiagnosisExitCode(d codexLoopDiagnosis, failOn string) (int, bool) {
	if codexLoopFailOnName(failOn) == "unguarded" {
		if codexLoopDiagnosisUnguarded(d) {
			return 1, true
		}
		return 0, true
	}
	return codexLoopFailOnExitCode(d.Verdict, failOn)
}

func codexLoopRecentGateReason(r codexLoopRecentReport, failOn string) string {
	if codexLoopFailOnName(failOn) == "unguarded" && r.UnguardedCount > 0 {
		return fmt.Sprintf("recent_codex_sessions_include_%d_unguarded_provider_session(s)", r.UnguardedCount)
	}
	return r.Reason
}

func codexLoopDiagnosisGateReason(d codexLoopDiagnosis, failOn string) string {
	if codexLoopFailOnName(failOn) == "unguarded" && codexLoopDiagnosisUnguarded(d) {
		return "codex_session_bypassed_fak_guard"
	}
	return d.Reason
}

func codexLoopDiagnosisUnguarded(d codexLoopDiagnosis) bool {
	return strings.TrimSpace(d.ModelProvider) != "" && !d.GuardWitnessed
}

// codexLoopGuardedLaunchGate distinguishes a loop that a guarded launch can
// remediate from one that already happened behind fak. A direct-provider loop is
// evidence to enter through fak guard, not a reason to make that entrypoint
// unreachable. Guarded or unknown-route loops still trip the configured threshold.
func codexLoopGuardedLaunchGate(r codexLoopRecentReport, failOn string) (exitCode, remediationCount int, ok bool) {
	threshold, ok := codexLoopFailOnRank(failOn)
	if !ok {
		return 2, 0, false
	}
	if threshold == 0 {
		return 0, 0, true
	}
	for _, d := range r.Diagnoses {
		// A rollout that ends on an unmatched tool call was killed or truncated while
		// work was in flight. It is crash evidence, not proof that a repetition loop
		// completed. Keep the advisory diagnosis intact, but never poison the next
		// guarded launch with it. (#4212)
		if d.abruptlyEnded {
			continue
		}
		if codexLoopVerdictRank(d.Verdict) < threshold {
			continue
		}
		if !codexLoopDiagnosisUnguarded(d) {
			return 1, 0, true
		}
		remediationCount++
	}
	return 0, remediationCount, true
}

func codexLoopFailOnExitCode(verdict, failOn string) (int, bool) {
	threshold, ok := codexLoopFailOnRank(failOn)
	if !ok {
		return 2, false
	}
	if threshold == 0 {
		return 0, true
	}
	rank := codexLoopVerdictRank(verdict)
	if rank >= threshold {
		return 1, true
	}
	return 0, true
}

func codexLoopFailOnRank(failOn string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(failOn)) {
	case "", "none", "off", "false", "0":
		return 0, true
	case "action", "any":
		return 1, true
	case "loop":
		return 2, true
	default:
		return 0, false
	}
}

func codexLoopFailOnName(failOn string) string {
	switch strings.ToLower(strings.TrimSpace(failOn)) {
	case "unguarded", "direct":
		return "unguarded"
	}
	switch rank, ok := codexLoopFailOnRank(failOn); {
	case !ok:
		return strings.TrimSpace(failOn)
	case rank == 1:
		return "action"
	case rank == 2:
		return "loop"
	default:
		return "none"
	}
}

func codexLoopVerdictRank(verdict string) int {
	switch strings.ToUpper(strings.TrimSpace(verdict)) {
	case "LOOP":
		return 2
	case "ACTION":
		return 1
	default:
		return 0
	}
}

func diagnoseRecentCodexLoops(codexHome string, sinceHours float64, limit int) (codexLoopRecentReport, error) {
	return diagnoseRecentCodexLoopsWith(codexHome, sinceHours, limit, diagnoseCodexLoopPath)
}

func diagnoseNewestCodexLoopForLaunch(codexHome string, sinceHours float64, workingDir string) (codexLoopRecentReport, codexLoopLaunchScan, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return codexLoopRecentReport{}, codexLoopLaunchScan{}, err
	}
	// Launch admission is repository-local. A LOOP in an unrelated checkout must
	// not make every `fakc` invocation on the machine unreachable. Inspect a
	// bounded newest-first window and select the newest rollout whose session cwd
	// is this cwd (or an ancestor/descendant, so launching from a repo subdir works).
	paths, err := discoverRecentCodexLoopSessionPaths(home, sinceHours, 20)
	if err != nil {
		return codexLoopRecentReport{}, codexLoopLaunchScan{}, err
	}
	for _, path := range paths {
		d, scan, diagnoseErr := diagnoseCodexLoopLaunchPath(path)
		if diagnoseErr != nil {
			return codexLoopRecentReport{}, scan, fmt.Errorf("diagnose %s: %w", path, diagnoseErr)
		}
		if !codexWorkingDirsOverlap(workingDir, d.WorkingDir) {
			continue
		}
		bindCodexGuardWitness(&d, home, d.SessionID)
		rep := codexLaunchReport(home, sinceHours, d)
		return rep, scan, nil
	}
	return codexLoopRecentReport{
		Schema: codexLoopRecentSchema, CodexHome: home, SinceHours: sinceHours, Limit: 1, Verdict: "OK",
	}, codexLoopLaunchScan{}, nil
}

func codexLaunchReport(home string, sinceHours float64, d codexLoopDiagnosis) codexLoopRecentReport {
	r := codexLoopRecentReport{
		Schema: codexLoopRecentSchema, CodexHome: home, SinceHours: sinceHours, Limit: 1, Scanned: 1,
		Verdict: d.Verdict, Reason: d.Reason, NextAction: d.NextAction, Diagnoses: []codexLoopDiagnosis{d},
		ToolCalls: d.ToolCalls, ToolOutputs: d.ToolOutputs, LastTokenTotalSum: d.LastTokenTotal,
	}
	if provider := strings.TrimSpace(d.ModelProvider); provider != "" {
		r.ProviderCounts = map[string]int{provider: 1}
		if codexLoopDiagnosisUnguarded(d) {
			r.UnguardedCount = 1
		}
	}
	switch d.Verdict {
	case "LOOP":
		r.LoopCount = 1
		switch {
		case d.GuardWitnessed:
			r.GuardedLoopCount = 1
		case codexLoopDiagnosisUnguarded(d):
			r.UnguardedLoopCount = 1
		default:
			r.UnknownLoopCount = 1
		}
	case "ACTION":
		r.ActionCount = 1
	default:
		r.OKCount = 1
	}
	if top, ok := codexTopLoopDrivingOutcome(d.RepeatedOutcomes); ok {
		r.TopRepeated = []codexRepeatedOutcome{top}
	}
	return r
}

func codexWorkingDirsOverlap(a, b string) bool {
	a = filepath.Clean(strings.TrimSpace(a))
	b = filepath.Clean(strings.TrimSpace(b))
	// Empty cwd is legacy transcript/test data. Preserve the historical global
	// fallback when either side cannot be scoped; modern Codex session_meta always
	// carries cwd, so unrelated current repositories are isolated.
	if a == "." || b == "." || a == "" || b == "" {
		return true
	}
	relAB, errAB := filepath.Rel(a, b)
	relBA, errBA := filepath.Rel(b, a)
	inside := func(rel string, err error) bool {
		return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	return inside(relAB, errAB) || inside(relBA, errBA)
}

func diagnoseRecentCodexLoopsWith(codexHome string, sinceHours float64, limit int, diagnosePath func(string) (codexLoopDiagnosis, error)) (codexLoopRecentReport, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return codexLoopRecentReport{}, err
	}
	paths, err := discoverRecentCodexLoopSessionPaths(home, sinceHours, limit)
	if err != nil {
		return codexLoopRecentReport{}, err
	}
	r := codexLoopRecentReport{
		Schema:         codexLoopRecentSchema,
		CodexHome:      home,
		SinceHours:     sinceHours,
		Limit:          normalizedCodexLoopLimit(limit),
		Verdict:        "OK",
		ProviderCounts: map[string]int{},
		Diagnoses:      make([]codexLoopDiagnosis, 0, len(paths)),
	}
	for _, path := range paths {
		d, err := diagnosePath(path)
		if err != nil {
			return r, fmt.Errorf("diagnose %s: %w", path, err)
		}
		bindCodexGuardWitness(&d, home, d.SessionID)
		r.Diagnoses = append(r.Diagnoses, d)
		r.Scanned++
		r.ToolCalls += d.ToolCalls
		r.ToolOutputs += d.ToolOutputs
		r.LastTokenTotalSum += d.LastTokenTotal
		if provider := strings.TrimSpace(d.ModelProvider); provider != "" {
			r.ProviderCounts[provider]++
			if codexLoopDiagnosisUnguarded(d) {
				r.UnguardedCount++
			}
		}
		switch d.Verdict {
		case "LOOP":
			r.LoopCount++
			switch {
			case d.GuardWitnessed:
				r.GuardedLoopCount++
			case codexLoopDiagnosisUnguarded(d):
				r.UnguardedLoopCount++
			default:
				r.UnknownLoopCount++
			}
		case "ACTION":
			r.ActionCount++
		default:
			r.OKCount++
		}
		if len(d.RepeatedOutcomes) > 0 {
			r.TopRepeated = append(r.TopRepeated, d.RepeatedOutcomes[0])
		}
	}
	sort.Slice(r.TopRepeated, func(i, j int) bool {
		if r.TopRepeated[i].TokenTotal != r.TopRepeated[j].TokenTotal {
			return r.TopRepeated[i].TokenTotal > r.TopRepeated[j].TokenTotal
		}
		if r.TopRepeated[i].Count != r.TopRepeated[j].Count {
			return r.TopRepeated[i].Count > r.TopRepeated[j].Count
		}
		return r.TopRepeated[i].Tool < r.TopRepeated[j].Tool
	})
	if len(r.TopRepeated) > 5 {
		r.TopRepeated = r.TopRepeated[:5]
	}
	if len(r.ProviderCounts) == 0 {
		r.ProviderCounts = nil
	}
	switch {
	case r.LoopCount > 0:
		r.Verdict = "LOOP"
		r.Reason = "recent_codex_sessions_have_repeated_tool_outputs"
	case r.ActionCount > 0:
		r.Verdict = "ACTION"
		r.Reason = "recent_codex_sessions_have_livelock_advisories"
	default:
		r.Verdict = "OK"
	}
	if r.GuardedLoopCount+r.UnknownLoopCount > 0 {
		r.NextAction = "inspect the guarded or unknown-route LOOP rollout and add or tighten a hard fuse for its top tool/result class"
	} else if r.UnguardedCount > 0 {
		r.NextAction = "launch future Codex sessions through `fak codex` or `fak guard -- codex`; direct Codex sessions cannot use the gateway's repeated-result fuse"
	} else if r.Verdict == "LOOP" {
		r.NextAction = "inspect the top repeated outcome and add or tighten a hard fuse for that tool/result class"
	}
	return r, nil
}

func diagnoseCodexLoopPath(path string) (codexLoopDiagnosis, error) {
	fh, err := os.Open(path)
	if err != nil {
		return codexLoopDiagnosis{}, fmt.Errorf("open: %w", err)
	}
	d, diagnoseErr := diagnoseCodexLoop(fh, path)
	closeErr := fh.Close()
	if diagnoseErr != nil {
		return d, diagnoseErr
	}
	if closeErr != nil {
		return d, fmt.Errorf("close: %w", closeErr)
	}
	return d, nil
}

func diagnoseCodexLoopLaunchPath(path string) (codexLoopDiagnosis, codexLoopLaunchScan, error) {
	fh, err := os.Open(path)
	if err != nil {
		return codexLoopDiagnosis{}, codexLoopLaunchScan{}, fmt.Errorf("open: %w", err)
	}
	snapshot, scan, readErr := readCodexLoopLaunchSnapshot(fh)
	closeErr := fh.Close()
	if readErr != nil {
		return codexLoopDiagnosis{}, scan, readErr
	}
	if closeErr != nil {
		return codexLoopDiagnosis{}, scan, fmt.Errorf("close: %w", closeErr)
	}
	d, err := diagnoseCodexLoop(bytes.NewReader(snapshot), path)
	return d, scan, err
}

func readCodexLoopLaunchSnapshot(fh *os.File) ([]byte, codexLoopLaunchScan, error) {
	info, err := fh.Stat()
	if err != nil {
		return nil, codexLoopLaunchScan{}, fmt.Errorf("stat: %w", err)
	}
	size := info.Size()
	if size <= codexLoopLaunchMaxBytes {
		b, err := io.ReadAll(io.LimitReader(fh, codexLoopLaunchMaxBytes))
		return b, codexLoopLaunchScan{BytesRead: int64(len(b))}, err
	}

	prefixLen := codexLoopLaunchPrefixBytes
	if prefixLen > codexLoopLaunchMaxBytes {
		prefixLen = codexLoopLaunchMaxBytes
	}
	suffixLen := codexLoopLaunchMaxBytes - prefixLen
	prefix := make([]byte, prefixLen)
	nPrefix, prefixErr := fh.ReadAt(prefix, 0)
	if prefixErr != nil && !errors.Is(prefixErr, io.EOF) {
		return nil, codexLoopLaunchScan{}, fmt.Errorf("read prefix: %w", prefixErr)
	}
	suffix := make([]byte, suffixLen)
	nSuffix, suffixErr := fh.ReadAt(suffix, size-suffixLen)
	if suffixErr != nil && !errors.Is(suffixErr, io.EOF) {
		return nil, codexLoopLaunchScan{}, fmt.Errorf("read suffix: %w", suffixErr)
	}
	snapshot := make([]byte, 0, nPrefix+1+nSuffix)
	snapshot = append(snapshot, prefix[:nPrefix]...)
	snapshot = append(snapshot, '\n')
	snapshot = append(snapshot, suffix[:nSuffix]...)
	return snapshot, codexLoopLaunchScan{
		BytesRead: int64(nPrefix + nSuffix),
		Truncated: true,
	}, nil
}

func discoverRecentCodexLoopSessionPaths(home string, sinceHours float64, limit int) ([]string, error) {
	root := filepath.Join(home, "sessions")
	var cutoff time.Time
	if sinceHours > 0 {
		cutoff = time.Now().Add(-time.Duration(sinceHours * float64(time.Hour)))
	}
	matches, walkErr := collectCodexLoopSessionFiles(root, func(_ string) bool { return true })
	if errors.Is(walkErr, os.ErrNotExist) {
		return nil, nil
	}
	if walkErr != nil {
		return nil, fmt.Errorf("walk %s: %w", root, walkErr)
	}
	if !cutoff.IsZero() {
		kept := matches[:0]
		for _, m := range matches {
			if !m.mtime.Before(cutoff) {
				kept = append(kept, m)
			}
		}
		matches = kept
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].mtime.After(matches[j].mtime) })
	limit = normalizedCodexLoopLimit(limit)
	if len(matches) > limit {
		matches = matches[:limit]
	}
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		out = append(out, m.path)
	}
	return out, nil
}

// codexLoopSessionFile is one .jsonl session file discovered under the Codex
// sessions root, carrying the mtime the newest-first sorts order on.
type codexLoopSessionFile struct {
	path  string
	mtime time.Time
}

// collectCodexLoopSessionFiles walks root for .jsonl session files that match,
// tolerating walk-step errors and recording each survivor's mtime. Recency
// filtering and ordering stay with the callers — they differ per verb.
func collectCodexLoopSessionFiles(root string, match func(name string) bool) ([]codexLoopSessionFile, error) {
	var files []codexLoopSessionFile
	err := walkfiles.Files(root, func(p string, d os.DirEntry) error {
		if !strings.HasSuffix(d.Name(), ".jsonl") || !match(d.Name()) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		files = append(files, codexLoopSessionFile{path: p, mtime: info.ModTime()})
		return nil
	})
	return files, err
}

func normalizedCodexLoopLimit(limit int) int {
	if limit <= 0 {
		return 20
	}
	return limit
}

func resolveCodexLoopSessionPath(codexHome, sessionID, path string) (string, error) {
	if p := strings.TrimSpace(path); p != "" {
		return filepath.Clean(expandCodexLoopTilde(p)), nil
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		sessionID = strings.TrimSpace(os.Getenv("CODEX_THREAD_ID"))
	}
	if sessionID == "" {
		return "", errors.New("need --session ID, --path FILE, or CODEX_THREAD_ID")
	}
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return "", err
	}
	root := filepath.Join(home, "sessions")
	matches, err := collectCodexLoopSessionFiles(root, func(name string) bool {
		return strings.Contains(name, sessionID)
	})
	if err != nil {
		return "", fmt.Errorf("walk %s: %w", root, err)
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("session %s not found under %s", sessionID, root)
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].mtime.After(matches[j].mtime) })
	return matches[0].path, nil
}

func diagnoseCurrentCodexLoop(codexHome string) (codexLoopDiagnosis, bool, error) {
	if strings.TrimSpace(os.Getenv("CODEX_THREAD_ID")) == "" {
		return codexLoopDiagnosis{}, false, nil
	}
	resolved, err := resolveCodexLoopSessionPath(codexHome, "", "")
	if err != nil && strings.TrimSpace(codexHome) != "" {
		resolved, err = resolveCodexLoopSessionPath("", "", "")
	}
	if err != nil {
		return codexLoopDiagnosis{}, true, err
	}
	fh, err := os.Open(resolved)
	if err != nil {
		return codexLoopDiagnosis{}, true, fmt.Errorf("open %s: %w", resolved, err)
	}
	defer fh.Close()
	d, err := diagnoseCodexLoop(fh, resolved)
	if err != nil {
		return codexLoopDiagnosis{}, true, err
	}
	// Current-thread consumers include the pre-spawn launcher and dispatch gates.
	// They must bind the provider metadata to the durable launch witness just like
	// the explicit and recent-session diagnostic paths do.
	bindCodexGuardWitness(&d, codexHome, d.SessionID)
	return d, true, nil
}

func resolvedCodexLoopHome(codexHome string) (string, error) {
	home := strings.TrimSpace(codexHome)
	if home == "" {
		// TrimSpace mirrors resolveCodexHome (memq_codex.go): a whitespace- or
		// CR-polluted CODEX_HOME (setx / CRLF .env / trailing space) must still
		// fall through to ~/.codex, not become a bogus relative sessions root
		// that fails the live continuation hook open.
		home = strings.TrimSpace(os.Getenv("CODEX_HOME"))
	}
	if home == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home: %w", err)
		}
		home = filepath.Join(userHome, ".codex")
	}
	return filepath.Clean(expandCodexLoopTilde(home)), nil
}

func expandCodexLoopTilde(path string) string {
	if path == "~" || strings.HasPrefix(path, "~"+string(os.PathSeparator)) || strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			rest := strings.TrimPrefix(path, "~")
			rest = strings.TrimPrefix(rest, "/")
			rest = strings.TrimPrefix(rest, string(os.PathSeparator))
			return filepath.Join(home, filepath.FromSlash(rest))
		}
	}
	return path
}

func probeCodexLoopProvider(r io.Reader, path string) (codexLoopDiagnosis, error) {
	d := codexLoopDiagnosis{Schema: codexLoopSchema, Path: path, Verdict: "OK"}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec struct {
			Timestamp string          `json:"timestamp"`
			Type      string          `json:"type"`
			Payload   json.RawMessage `json:"payload"`
		}
		if json.Unmarshal(line, &rec) != nil || rec.Type != "session_meta" {
			continue
		}
		applyCodexLoopSessionMeta(&d, rec.Timestamp, rec.Payload)
		return d, nil
	}
	if err := sc.Err(); err != nil {
		return d, err
	}
	return d, nil
}
