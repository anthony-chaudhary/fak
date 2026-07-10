package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

const codexLoopSchema = "fak.sessions.codex_loop.v1"
const codexLoopRecentSchema = "fak.sessions.codex_loop_recent.v1"
const codexLoopHookOverrideEnv = "FAK_ALLOW_DIRECT_CODEX_CONTINUE"
const codexLoopHookDefaultBudget = 500 * time.Millisecond

// These two seams are variables so the hook's hard failure modes can be injected
// without making the general transcript diagnoser artificial. They are only mutated
// by the serial CodexLoopHook witness tests.
var codexLoopHookBudget = codexLoopHookDefaultBudget
var codexLoopHookDiagnose = diagnoseCodexLoop

type codexLoopDiagnosis struct {
	Schema            string                 `json:"schema"`
	Path              string                 `json:"path"`
	SessionID         string                 `json:"session_id,omitempty"`
	ParentSessionID   string                 `json:"parent_session_id,omitempty"`
	Originator        string                 `json:"originator,omitempty"`
	CLI               string                 `json:"cli_version,omitempty"`
	ModelProvider     string                 `json:"model_provider,omitempty"`
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
	SessionID     string `json:"session_id"`
	HookEventName string `json:"hook_event_name"`
	TurnID        string `json:"turn_id"`
}

type codexLoopHookBlock struct {
	Decision string `json:"decision"`
	Reason   string `json:"reason"`
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
	fs := flag.NewFlagSet("sessions codex-loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionID := fs.String("session", "", "Codex session id to find under --codex-home/sessions (default: $CODEX_THREAD_ID when set)")
	path := fs.String("path", "", "explicit Codex session JSONL path")
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	recent := fs.Bool("recent", false, "scan recent Codex session JSONL files instead of one path")
	sinceHours := fs.Float64("since-hours", 24, "with --recent, only scan files modified within N hours (0 = all)")
	limit := fs.Int("limit", 20, "with --recent, cap sessions scanned after newest-first sorting")
	asJSON := fs.Bool("json", false, "emit a machine-readable diagnosis")
	failOn := fs.String("fail-on", "none", "exit 1 when verdict/posture reaches this threshold: none|loop|action|unguarded (action also fails LOOP)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak sessions codex-loop [--session ID | --path FILE | --recent] [--codex-home DIR] [--json] [--fail-on none|loop|action|unguarded]")
		fs.PrintDefaults()
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
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
		gateCode, ok := codexLoopFailOnRecentExitCode(r, *failOn)
		if !ok {
			fmt.Fprintf(stderr, "fak sessions codex-loop: invalid --fail-on %q (want none, loop, action, or unguarded)\n", *failOn)
			return 2
		}
		if *asJSON {
			if code := encodeJSONOrFail(stdout, stderr, r, "fak sessions codex-loop"); code != 0 {
				return code
			}
			if gateCode != 0 {
				fmt.Fprintf(stderr, "fak sessions codex-loop: gate REFUSE fail-on=%s verdict=%s reason=%s\n", codexLoopFailOnName(*failOn), r.Verdict, codexLoopRecentGateReason(r, *failOn))
			}
			return gateCode
		}
		fmt.Fprint(stdout, renderCodexLoopRecentReport(r))
		if gateCode != 0 {
			fmt.Fprintf(stderr, "fak sessions codex-loop: gate REFUSE fail-on=%s verdict=%s reason=%s\n", codexLoopFailOnName(*failOn), r.Verdict, codexLoopRecentGateReason(r, *failOn))
		}
		return gateCode
	}
	if strings.TrimSpace(*path) != "" && strings.TrimSpace(*sessionID) != "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop: use only one of --path or --session")
		return 2
	}
	resolved, err := resolveCodexLoopSessionPath(*codexHome, *sessionID, *path)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
		return 2
	}
	fh, err := os.Open(resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: open %s: %v\n", resolved, err)
		return 1
	}
	defer fh.Close()
	d, err := diagnoseCodexLoop(fh, resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
		return 1
	}
	gateCode, ok := codexLoopFailOnDiagnosisExitCode(d, *failOn)
	if !ok {
		fmt.Fprintf(stderr, "fak sessions codex-loop: invalid --fail-on %q (want none, loop, action, or unguarded)\n", *failOn)
		return 2
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, d, "fak sessions codex-loop"); code != 0 {
			return code
		}
		if gateCode != 0 {
			fmt.Fprintf(stderr, "fak sessions codex-loop: gate REFUSE fail-on=%s verdict=%s reason=%s\n", codexLoopFailOnName(*failOn), d.Verdict, codexLoopDiagnosisGateReason(d, *failOn))
		}
		return gateCode
	}
	fmt.Fprint(stdout, renderCodexLoopDiagnosis(d))
	if gateCode != 0 {
		fmt.Fprintf(stderr, "fak sessions codex-loop: gate REFUSE fail-on=%s verdict=%s reason=%s\n", codexLoopFailOnName(*failOn), d.Verdict, codexLoopDiagnosisGateReason(d, *failOn))
	}
	return gateCode
}

// sessionsCodexLoopHook is the turn-boundary enforcement seam for #3023. Codex's
// UserPromptSubmit hook fires before a prompt becomes the next model/tool turn and
// carries the active session_id. Reuse the existing transcript diagnosis rather than
// maintaining a second provider classifier: a direct provider is blocked with the same
// typed reason the advisory `--fail-on unguarded` gate already reports.
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
	if os.Getenv(guardActiveEnv) == "1" {
		return 0
	}
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
		var block codexLoopHookBlock
		if err := json.Unmarshal(result.stdout, &block); err != nil || block.Decision != "block" {
			return 0
		}
		if _, err := stdout.Write(result.stdout); err != nil {
			return 1
		}
		return 0
	case <-timer.C:
		return 0
	}
}

func sessionsCodexLoopHookUnbounded(stdout, stderr io.Writer, stdin io.Reader, argv []string, diagnose func(io.Reader, string) (codexLoopDiagnosis, error)) int {
	fs := flag.NewFlagSet("sessions codex-loop-hook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	allowDirect := fs.Bool("allow-direct", false, "explicitly allow this intentional direct-provider continuation")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak sessions codex-loop-hook [--codex-home DIR] [--allow-direct]")
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *allowDirect || codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookOverrideEnv)) {
		return 0
	}

	var in codexLoopHookInput
	if err := json.NewDecoder(io.LimitReader(stdin, 1<<20)).Decode(&in); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: unreadable hook payload (allowing turn): %v\n", err)
		return 0
	}
	if strings.TrimSpace(in.SessionID) == "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop-hook: hook payload has no session_id (allowing turn)")
		return 0
	}

	resolved, err := resolveCodexLoopSessionPath(*codexHome, in.SessionID, "")
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: %v (allowing turn)\n", err)
		return 0
	}
	fh, err := os.Open(resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: open %s: %v (allowing turn)\n", resolved, err)
		return 0
	}
	d, diagnoseErr := diagnose(fh, resolved)
	closeErr := fh.Close()
	if diagnoseErr != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: diagnose: %v (allowing turn)\n", diagnoseErr)
		return 0
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: close %s: %v\n", resolved, closeErr)
	}
	if !codexLoopDiagnosisUnguarded(d) {
		return 0
	}

	reason := codexLoopDiagnosisGateReason(d, "unguarded") +
		": this active Codex session uses model_provider=" + strings.TrimSpace(d.ModelProvider) +
		", so fak cannot enforce the guard before the next turn. Relaunch with `fak codex`" +
		" or `fak guard -- codex`. For an intentional direct session, set " +
		codexLoopHookOverrideEnv + "=1 and resubmit."
	if err := json.NewEncoder(stdout).Encode(codexLoopHookBlock{Decision: "block", Reason: reason}); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: encode block: %v\n", err)
		return 1
	}
	return 0
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
	provider := strings.TrimSpace(d.ModelProvider)
	return provider != "" && !strings.EqualFold(provider, "fak")
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
		fh, err := os.Open(path)
		if err != nil {
			return r, fmt.Errorf("open %s: %w", path, err)
		}
		d, derr := diagnoseCodexLoop(fh, path)
		cerr := fh.Close()
		if derr != nil {
			return r, fmt.Errorf("diagnose %s: %w", path, derr)
		}
		if cerr != nil {
			return r, fmt.Errorf("close %s: %w", path, cerr)
		}
		r.Diagnoses = append(r.Diagnoses, d)
		r.Scanned++
		r.ToolCalls += d.ToolCalls
		r.ToolOutputs += d.ToolOutputs
		r.LastTokenTotalSum += d.LastTokenTotal
		if provider := strings.TrimSpace(d.ModelProvider); provider != "" {
			r.ProviderCounts[provider]++
			if !strings.EqualFold(provider, "fak") {
				r.UnguardedCount++
			}
		}
		switch d.Verdict {
		case "LOOP":
			r.LoopCount++
			switch {
			case strings.EqualFold(strings.TrimSpace(d.ModelProvider), "fak"):
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

func discoverRecentCodexLoopSessionPaths(home string, sinceHours float64, limit int) ([]string, error) {
	root := filepath.Join(home, "sessions")
	type candidate struct {
		path  string
		mtime time.Time
	}
	var cutoff time.Time
	if sinceHours > 0 {
		cutoff = time.Now().Add(-time.Duration(sinceHours * float64(time.Hour)))
	}
	var matches []candidate
	walkErr := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		if !cutoff.IsZero() && info.ModTime().Before(cutoff) {
			return nil
		}
		matches = append(matches, candidate{path: p, mtime: info.ModTime()})
		return nil
	})
	if errors.Is(walkErr, os.ErrNotExist) {
		return nil, nil
	}
	if walkErr != nil {
		return nil, fmt.Errorf("walk %s: %w", root, walkErr)
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
	type candidate struct {
		path  string
		mtime time.Time
	}
	var matches []candidate
	err = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		if !strings.Contains(d.Name(), sessionID) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		matches = append(matches, candidate{path: p, mtime: info.ModTime()})
		return nil
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
	return d, true, nil
}

func resolvedCodexLoopHome(codexHome string) (string, error) {
	home := strings.TrimSpace(codexHome)
	if home == "" {
		home = os.Getenv("CODEX_HOME")
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

func diagnoseCodexLoop(r io.Reader, path string) (codexLoopDiagnosis, error) {
	d := codexLoopDiagnosis{
		Schema:  codexLoopSchema,
		Path:    path,
		Verdict: "OK",
	}
	calls := map[string]codexPendingToolCall{}
	outcomes := map[codexOutcomeKey]*codexOutcomeAccum{}
	livelocks := map[string]*codexLivelockNotice{}
	var prevOutcome *codexOutcomeKey
	var pendingTokenOutcome *codexOutcomeKey

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Timestamp string          `json:"timestamp"`
			Type      string          `json:"type"`
			Payload   json.RawMessage `json:"payload"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		d.LastEventAt = rec.Timestamp
		switch rec.Type {
		case "session_meta":
			applyCodexLoopSessionMeta(&d, rec.Timestamp, rec.Payload)
		case "response_item":
			var item struct {
				Type      string          `json:"type"`
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
				Input     json.RawMessage `json:"input"`
				CallID    string          `json:"call_id"`
				Output    json.RawMessage `json:"output"`
			}
			if json.Unmarshal(rec.Payload, &item) != nil {
				continue
			}
			switch item.Type {
			case "function_call", "custom_tool_call":
				d.ToolCalls++
				args := item.Arguments
				if len(args) == 0 {
					args = item.Input
				}
				calls[item.CallID] = codexPendingToolCall{
					Tool:       strings.TrimSpace(item.Name),
					ArgsDigest: guardrsi.ArgsDigest(codexLoopRawText(args)),
					Timestamp:  rec.Timestamp,
				}
				pendingTokenOutcome = nil
			case "function_call_output", "custom_tool_call_output":
				d.ToolOutputs++
				call := calls[item.CallID]
				if call.Tool == "" {
					call.Tool = "unknown"
				}
				output := codexLoopRawText(item.Output)
				key := codexOutcomeKey{Tool: call.Tool, OutputDigest: digestCodexLoopText(normalizeCodexLoopText(output))}
				acc := outcomes[key]
				if acc == nil {
					acc = &codexOutcomeAccum{out: codexRepeatedOutcome{
						Tool:           key.Tool,
						OutputDigest:   key.OutputDigest,
						OutputExcerpt:  codexLoopExcerpt(output),
						FirstTimestamp: rec.Timestamp,
					}, argsDigests: map[string]bool{}}
					outcomes[key] = acc
				}
				acc.out.Count++
				acc.out.LastTimestamp = rec.Timestamp
				if call.ArgsDigest != "" {
					acc.argsDigests[call.ArgsDigest] = true
					if acc.out.FirstArgsDigest == "" {
						acc.out.FirstArgsDigest = call.ArgsDigest
					}
				}
				if prevOutcome != nil && *prevOutcome == key {
					acc.currentRun++
				} else {
					acc.currentRun = 1
				}
				if acc.currentRun > acc.out.LongestRun {
					acc.out.LongestRun = acc.currentRun
				}
				copyKey := key
				prevOutcome = &copyKey
				pendingTokenOutcome = &copyKey
				delete(calls, item.CallID)
			}
		case "event_msg":
			var ev struct {
				Type    string          `json:"type"`
				Message string          `json:"message"`
				Info    json.RawMessage `json:"info"`
				Goal    *struct {
					Objective       string `json:"objective"`
					Status          string `json:"status"`
					TokensUsed      int64  `json:"tokensUsed"`
					TimeUsedSeconds int64  `json:"timeUsedSeconds"`
				} `json:"goal"`
				Reason     string `json:"reason"`
				DurationMS int64  `json:"duration_ms"`
			}
			if json.Unmarshal(rec.Payload, &ev) != nil {
				continue
			}
			switch ev.Type {
			case "token_count":
				tok := parseCodexLoopTokenInfo(ev.Info)
				d.LastTokenTotal = tok.total
				d.LastTokenInput = tok.input
				d.LastTokenOutput = tok.output
				if pendingTokenOutcome != nil {
					if acc := outcomes[*pendingTokenOutcome]; acc != nil {
						acc.out.TokenTotal += tok.last
						acc.out.TokenEvents++
					}
					pendingTokenOutcome = nil
				}
			case "agent_message":
				if notice, ok := parseCodexLivelockNotice(rec.Timestamp, ev.Message); ok {
					cur := livelocks[notice.RepeatedCall+"\x00"+notice.Approach]
					if cur == nil {
						livelocks[notice.RepeatedCall+"\x00"+notice.Approach] = &notice
					} else {
						cur.Count++
						cur.LastTimestamp = notice.LastTimestamp
						if notice.MinRepeat < cur.MinRepeat {
							cur.MinRepeat = notice.MinRepeat
						}
						if notice.MaxRepeat > cur.MaxRepeat {
							cur.MaxRepeat = notice.MaxRepeat
						}
					}
				}
			case "thread_goal_updated":
				if ev.Goal != nil {
					d.FinalStatus = ev.Goal.Status
					d.FinalTokensUsed = ev.Goal.TokensUsed
					d.FinalTimeSeconds = ev.Goal.TimeUsedSeconds
				}
			case "turn_aborted":
				d.TurnAborted = true
				d.AbortReason = ev.Reason
				d.AbortDurationMS = ev.DurationMS
			}
		}
	}
	if err := sc.Err(); err != nil {
		return d, err
	}

	appendCodexLoopRepeatedOutcomes(&d, outcomes)
	appendCodexLoopLivelockNotices(&d, livelocks)
	classifyCodexLoopDiagnosis(&d)
	return d, nil
}

func applyCodexLoopSessionMeta(d *codexLoopDiagnosis, ts string, payload json.RawMessage) {
	// A subagent rollout starts with its own metadata, then carries the parent
	// session metadata in the inherited context. Only the first record identifies
	// this file; allowing later records to overwrite it makes every child look like
	// the same parent session and corrupts the recent-session report.
	if d.SessionID != "" {
		return
	}
	var meta struct {
		SessionID     string `json:"session_id"`
		ID            string `json:"id"`
		Timestamp     string `json:"timestamp"`
		Originator    string `json:"originator"`
		CLIVersion    string `json:"cli_version"`
		ModelProvider string `json:"model_provider"`
		Git           struct {
			CommitHash string `json:"commit_hash"`
			Branch     string `json:"branch"`
		} `json:"git"`
	}
	if json.Unmarshal(payload, &meta) == nil {
		d.SessionID = firstNonEmpty(meta.ID, meta.SessionID)
		if meta.ID != "" && meta.SessionID != "" && meta.ID != meta.SessionID {
			d.ParentSessionID = meta.SessionID
		}
		d.StartedAt = firstNonEmpty(meta.Timestamp, ts)
		d.Originator = meta.Originator
		d.CLI = meta.CLIVersion
		d.ModelProvider = meta.ModelProvider
		d.GitCommit = meta.Git.CommitHash
		d.GitBranch = meta.Git.Branch
	}
}

func appendCodexLoopRepeatedOutcomes(d *codexLoopDiagnosis, outcomes map[codexOutcomeKey]*codexOutcomeAccum) {
	for _, acc := range outcomes {
		if acc.out.Count < 3 && acc.out.LongestRun < 3 {
			continue
		}
		digests := make([]string, 0, len(acc.argsDigests))
		for digest := range acc.argsDigests {
			digests = append(digests, digest)
		}
		sort.Strings(digests)
		acc.out.ArgsDigestCount = len(digests)
		if acc.out.FirstArgsDigest == "" && len(digests) > 0 {
			acc.out.FirstArgsDigest = digests[0]
		}
		for _, digest := range digests {
			if digest == acc.out.FirstArgsDigest {
				continue
			}
			acc.out.OtherArgsDigests = append(acc.out.OtherArgsDigests, digest)
		}
		if len(acc.out.OtherArgsDigests) > 4 {
			acc.out.OtherArgsDigests = acc.out.OtherArgsDigests[:4]
		}
		d.RepeatedOutcomes = append(d.RepeatedOutcomes, acc.out)
	}
	sort.Slice(d.RepeatedOutcomes, func(i, j int) bool {
		if d.RepeatedOutcomes[i].TokenTotal != d.RepeatedOutcomes[j].TokenTotal {
			return d.RepeatedOutcomes[i].TokenTotal > d.RepeatedOutcomes[j].TokenTotal
		}
		if d.RepeatedOutcomes[i].LongestRun != d.RepeatedOutcomes[j].LongestRun {
			return d.RepeatedOutcomes[i].LongestRun > d.RepeatedOutcomes[j].LongestRun
		}
		if d.RepeatedOutcomes[i].Count != d.RepeatedOutcomes[j].Count {
			return d.RepeatedOutcomes[i].Count > d.RepeatedOutcomes[j].Count
		}
		return d.RepeatedOutcomes[i].Tool < d.RepeatedOutcomes[j].Tool
	})
	if len(d.RepeatedOutcomes) > 5 {
		d.RepeatedOutcomes = d.RepeatedOutcomes[:5]
	}
}

func appendCodexLoopLivelockNotices(d *codexLoopDiagnosis, livelocks map[string]*codexLivelockNotice) {
	for _, n := range livelocks {
		d.LivelockNotices = append(d.LivelockNotices, *n)
	}
	sort.Slice(d.LivelockNotices, func(i, j int) bool {
		if d.LivelockNotices[i].MaxRepeat != d.LivelockNotices[j].MaxRepeat {
			return d.LivelockNotices[i].MaxRepeat > d.LivelockNotices[j].MaxRepeat
		}
		return d.LivelockNotices[i].RepeatedCall < d.LivelockNotices[j].RepeatedCall
	})
}

type codexLoopTokenInfo struct {
	total  int64
	last   int64
	input  int64
	output int64
}

func parseCodexLoopTokenInfo(raw json.RawMessage) codexLoopTokenInfo {
	var info struct {
		Total struct {
			TotalTokens  int64 `json:"total_tokens"`
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"total_token_usage"`
		Last struct {
			TotalTokens  int64 `json:"total_tokens"`
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"last_token_usage"`
	}
	_ = json.Unmarshal(raw, &info)
	return codexLoopTokenInfo{
		total:  info.Total.TotalTokens,
		last:   info.Last.TotalTokens,
		input:  info.Last.InputTokens,
		output: info.Last.OutputTokens,
	}
}

var codexLivelockRE = regexp.MustCompile(`LIVELOCK_DETECTED repeat=([0-9]+) repeated_call=([^ ]+) approach=([^ .]+)`)

func parseCodexLivelockNotice(ts, msg string) (codexLivelockNotice, bool) {
	m := codexLivelockRE.FindStringSubmatch(msg)
	if len(m) != 4 {
		return codexLivelockNotice{}, false
	}
	repeat, _ := strconv.Atoi(m[1])
	return codexLivelockNotice{
		RepeatedCall:   m[2],
		Approach:       m[3],
		Count:          1,
		MinRepeat:      repeat,
		MaxRepeat:      repeat,
		FirstTimestamp: ts,
		LastTimestamp:  ts,
	}, true
}

func classifyCodexLoopDiagnosis(d *codexLoopDiagnosis) {
	if len(d.RepeatedOutcomes) == 0 {
		if len(d.LivelockNotices) > 0 {
			d.Verdict = "ACTION"
			d.Reason = "livelock_advisory_seen"
			d.NextAction = "inspect the repeated_call digest and decide whether the admitted-call advisory should become a hard fuse for this tool class"
			d.ObservabilityGaps = append(d.ObservabilityGaps, "Codex session logs carry livelock advisory text, but not a compact repeated-outcome/token-burn summary")
			return
		}
		d.Verdict = "OK"
		applyCodexLoopUnguardedGuidance(d)
		return
	}
	top := d.RepeatedOutcomes[0]
	d.Verdict = "LOOP"
	d.Reason = "repeated_tool_output"
	d.NextAction = "stop re-calling the same tool after an invariant failure; continue from the existing state or add a hard fuse for repeated admitted failures"
	if strings.EqualFold(top.Tool, "create_goal") && strings.Contains(strings.ToLower(top.OutputExcerpt), "unfinished goal") {
		d.NextAction = "for create_goal, read/continue the existing goal instead of creating a new one; hard-fuse repeated unfinished-goal failures after the first repeat"
	}
	if applyCodexLoopUnguardedGuidance(d) {
		return
	}
	d.ObservabilityGaps = append(d.ObservabilityGaps,
		"the live gateway emitted an advisory livelock note but still admitted the next identical host-tool call",
		"the Codex session final status records tokens/time but not the top repeated tool outcome that consumed them",
	)
}

func applyCodexLoopUnguardedGuidance(d *codexLoopDiagnosis) bool {
	if !codexLoopDiagnosisUnguarded(*d) {
		return false
	}
	d.NextAction = "launch future Codex sessions through `fak codex` or `fak guard -- codex`; direct model_provider=" + d.ModelProvider + " sessions cannot use the gateway's repeated-result fuse"
	if len(d.RepeatedOutcomes) > 0 {
		d.ObservabilityGaps = append(d.ObservabilityGaps,
			"this Codex session bypassed fak guard (model_provider="+d.ModelProvider+"), so the gateway could not hard-fuse the repeated tool outcome",
			"the Codex session final status records tokens/time but not the top repeated tool outcome that consumed them",
		)
		return true
	}
	d.ObservabilityGaps = append(d.ObservabilityGaps,
		"this Codex session bypassed fak guard (model_provider="+d.ModelProvider+"), so the gateway cannot hard-fuse repeated tool outcomes inside this process",
	)
	return true
}

func renderCodexLoopDiagnosis(d codexLoopDiagnosis) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak sessions codex-loop: %s\n", d.Path)
	if d.SessionID != "" {
		fmt.Fprintf(&b, "  session        : %s", d.SessionID)
		if d.ParentSessionID != "" {
			fmt.Fprintf(&b, " parent=%s", d.ParentSessionID)
		}
		if d.ModelProvider != "" {
			fmt.Fprintf(&b, " provider=%s", d.ModelProvider)
		}
		if d.Originator != "" {
			fmt.Fprintf(&b, " originator=%s", d.Originator)
		}
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "  verdict        : %s", d.Verdict)
	if d.Reason != "" {
		fmt.Fprintf(&b, " (%s)", d.Reason)
	}
	b.WriteByte('\n')
	if d.FinalStatus != "" || d.FinalTokensUsed > 0 || d.FinalTimeSeconds > 0 {
		fmt.Fprintf(&b, "  final          : status=%s tokens=%d seconds=%d\n", firstNonEmpty(d.FinalStatus, "unknown"), d.FinalTokensUsed, d.FinalTimeSeconds)
	}
	if d.TurnAborted {
		fmt.Fprintf(&b, "  abort          : reason=%s duration_ms=%d\n", firstNonEmpty(d.AbortReason, "unknown"), d.AbortDurationMS)
	}
	if d.LastTokenTotal > 0 {
		fmt.Fprintf(&b, "  last tokens    : total=%d last_input=%d last_output=%d\n", d.LastTokenTotal, d.LastTokenInput, d.LastTokenOutput)
	}
	fmt.Fprintf(&b, "  tool traffic   : calls=%d outputs=%d\n", d.ToolCalls, d.ToolOutputs)
	if len(d.RepeatedOutcomes) > 0 {
		fmt.Fprintf(&b, "  repeated tool outcomes:\n")
		for _, r := range d.RepeatedOutcomes {
			fmt.Fprintf(&b, "    %s output_digest=%s count=%d longest_run=%d tokens=%d",
				r.Tool, r.OutputDigest, r.Count, r.LongestRun, r.TokenTotal)
			if r.ArgsDigestCount > 0 {
				fmt.Fprintf(&b, " args_digests=%d", r.ArgsDigestCount)
			}
			b.WriteByte('\n')
			if r.OutputExcerpt != "" {
				fmt.Fprintf(&b, "      output: %s\n", r.OutputExcerpt)
			}
		}
	}
	if len(d.LivelockNotices) > 0 {
		fmt.Fprintf(&b, "  livelock notices:\n")
		for _, n := range d.LivelockNotices {
			fmt.Fprintf(&b, "    %s repeat=%d..%d count=%d approach=%s\n",
				n.RepeatedCall, n.MinRepeat, n.MaxRepeat, n.Count, n.Approach)
		}
	}
	if len(d.ObservabilityGaps) > 0 {
		fmt.Fprintf(&b, "  observability gaps:\n")
		for _, gap := range d.ObservabilityGaps {
			fmt.Fprintf(&b, "    - %s\n", gap)
		}
	}
	if d.NextAction != "" {
		fmt.Fprintf(&b, "  next action    : %s\n", d.NextAction)
	}
	return b.String()
}

func renderCodexLoopRecentReport(r codexLoopRecentReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak sessions codex-loop --recent: %s\n", r.CodexHome)
	fmt.Fprintf(&b, "  verdict        : %s", r.Verdict)
	if r.Reason != "" {
		fmt.Fprintf(&b, " (%s)", r.Reason)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  scope          : scanned=%d limit=%d since_hours=%s\n", r.Scanned, r.Limit, trimFloat(r.SinceHours))
	fmt.Fprintf(&b, "  session counts : LOOP=%d ACTION=%d OK=%d\n", r.LoopCount, r.ActionCount, r.OKCount)
	if len(r.ProviderCounts) > 0 {
		fmt.Fprintf(&b, "  providers      : %s", formatCodexProviderCounts(r.ProviderCounts))
		if r.UnguardedCount > 0 {
			fmt.Fprintf(&b, " unguarded=%d", r.UnguardedCount)
		}
		b.WriteByte('\n')
	}
	if r.LoopCount > 0 {
		fmt.Fprintf(&b, "  loop routes    : guarded=%d direct=%d unknown=%d\n", r.GuardedLoopCount, r.UnguardedLoopCount, r.UnknownLoopCount)
	}
	fmt.Fprintf(&b, "  tool traffic   : calls=%d outputs=%d\n", r.ToolCalls, r.ToolOutputs)
	if r.LastTokenTotalSum > 0 {
		fmt.Fprintf(&b, "  token usage    : cumulative-sum=%d (latest counter per rollout)\n", r.LastTokenTotalSum)
	}
	if r.NextAction != "" {
		fmt.Fprintf(&b, "  next action    : %s\n", r.NextAction)
	}
	if len(r.TopRepeated) > 0 {
		fmt.Fprintf(&b, "  top repeated outcomes:\n")
		for _, out := range r.TopRepeated {
			fmt.Fprintf(&b, "    %s output_digest=%s count=%d longest_run=%d tokens=%d",
				out.Tool, out.OutputDigest, out.Count, out.LongestRun, out.TokenTotal)
			if out.ArgsDigestCount > 0 {
				fmt.Fprintf(&b, " args_digests=%d", out.ArgsDigestCount)
			}
			b.WriteByte('\n')
			if out.OutputExcerpt != "" {
				fmt.Fprintf(&b, "      output: %s\n", out.OutputExcerpt)
			}
		}
	}
	if len(r.Diagnoses) > 0 {
		fmt.Fprintf(&b, "  sessions:\n")
		for _, d := range r.Diagnoses {
			label := firstNonEmpty(d.SessionID, filepath.Base(d.Path))
			fmt.Fprintf(&b, "    %s verdict=%s", label, d.Verdict)
			if d.ParentSessionID != "" {
				fmt.Fprintf(&b, " parent=%s", d.ParentSessionID)
			}
			if d.Reason != "" {
				fmt.Fprintf(&b, " reason=%s", d.Reason)
			}
			if d.LastTokenTotal > 0 {
				fmt.Fprintf(&b, " last_tokens=%d", d.LastTokenTotal)
			}
			if len(d.RepeatedOutcomes) > 0 {
				top := d.RepeatedOutcomes[0]
				fmt.Fprintf(&b, " top=%s:%d", top.Tool, top.Count)
			}
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func codexLoopRawText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		parts := make([]string, 0, len(blocks))
		for _, block := range blocks {
			if block.Text != "" {
				parts = append(parts, block.Text)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	return string(raw)
}

func formatCodexProviderCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		if strings.TrimSpace(key) != "" {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts[key]))
	}
	return strings.Join(parts, " ")
}

func normalizeCodexLoopText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func digestCodexLoopText(s string) string {
	sum := sha256.Sum256([]byte(s))
	return "sha256:" + hex.EncodeToString(sum[:])
}

var codexLoopSecretishRE = regexp.MustCompile(`(?i)(sk-[A-Za-z0-9_-]{8,}|sk-ant-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+|bearer\s+[A-Za-z0-9._-]{12,})`)

func codexLoopExcerpt(s string) string {
	s = normalizeCodexLoopText(s)
	s = codexLoopSecretishRE.ReplaceAllString(s, "[REDACTED]")
	const max = 180
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
