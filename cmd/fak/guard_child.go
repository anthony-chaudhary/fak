package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

// guard_child.go — the resolved upstream wire/credential posture, building and
// supervising the wrapped child process (incl. the budget-driven fresh-context
// restart loop), and tearing the gateway down with a faithful exit report.
// Split out of guard.go to keep the dispatch surface readable.

// guardUpstream is the resolved upstream wire + credential posture for `fak guard`: which
// provider the gateway proxies to, its base URL, the API key (if any), and — for Claude —
// whether to hold a Pro/Max subscription OAuth token upstream (pinUpstream) and where it
// came from (oauthSource).
type guardUpstream struct {
	provider     string
	autodetected bool
	baseURL      string
	apiKey       string
	pinUpstream  bool
	oauthSource  string
	// remoteServe is set when the base URL came from --remote-serve (a remote `fak serve`
	// on a lab box), so the banner can say "remote fak serve" instead of a generic
	// provider — the operator's signal that the dev turn's inference is on the lab GPU.
	remoteServe bool
	// passthroughFallback is set when the Anthropic subscription-OAuth auto-lookup found
	// no token and guard fell back to plain passthrough. That path works ONLY if the
	// wrapped agent (Claude Code) is itself logged in; cmdGuard surfaces a one-line note
	// so a cold agent that is ALSO not logged in gets a pointer home instead of an opaque
	// upstream 401 (issue #835, failure 2).
	passthroughFallback bool
	// ambientKeyOverridden is set when guard held the Pro/Max subscription OAuth token
	// upstream even though a bare ANTHROPIC_API_KEY was present in the environment. The
	// subscription is the default now regardless of that key — a global SDK key must not
	// silently bill the API account — so cmdGuard surfaces a one-line note pointing at the
	// explicit API-billing opt-in (--api-key-env ANTHROPIC_API_KEY) for discoverability.
	ambientKeyOverridden bool
}

// resolveGuardUpstream picks the upstream wire and credential posture: an explicit
// --provider wins, else the wire is inferred from the wrapped agent's name (anthropic as
// the fallback); the base URL defaults to the provider's public API. Subscription is the
// DEFAULT for Claude — when the upstream is Anthropic and no API key was EXPLICITLY named
// (--api-key-env), it sources the Pro/Max OAuth token and pins it upstream regardless of a
// bare ambient ANTHROPIC_API_KEY; --anthropic-oauth forces that and fails loud if no token
// is found. It exits(2) on an unresolvable base URL or OAuth misuse.
func resolveGuardUpstream(providerFlag, agentName, baseURLFlag, remoteServeBase, apiKeyEnv string, forceOAuth bool, oauthTokenEnv string) guardUpstream {
	// --remote-serve pins the OpenAI-compatible wire and the box's base URL: a remote
	// `fak serve` speaks the OpenAI routes the gateway proxies, and the caller has already
	// validated that it does not conflict with --provider/--base-url.
	//
	// The base MUST carry the /v1 suffix the OpenAI wire appends "/chat/completions" to:
	// the proxy planner POSTs adapter.Endpoint(BaseURL, model) = <base>/chat/completions,
	// while `fak serve` registers its route at /v1/chat/completions. normalizeRemoteServe
	// returns a bare http://HOST:PORT (so the /healthz preflight probes the ROOT health
	// route, which is NOT under /v1), so we add /v1 HERE — symmetric with guardEnvValue,
	// which adds /v1 only to the CHILD's OPENAI_BASE_URL. Without this the upstream proxy
	// hop 404s on every real turn (the /healthz preflight passes regardless, so it would
	// surface only mid-session). Idempotent: an operator base already ending in /v1 is
	// left as-is.
	remote := strings.TrimSpace(remoteServeBase) != ""
	if remote {
		providerFlag = "openai"
		baseURLFlag = guardOpenAIV1Base(strings.TrimSpace(remoteServeBase))
	}
	up, autodetected := resolveGuardProvider(providerFlag, agentName)
	resolvedBase := strings.TrimSpace(baseURLFlag)
	if resolvedBase == "" {
		resolvedBase = guardDefaultBaseURL(up)
	}
	if resolvedBase == "" {
		fmt.Fprintf(os.Stderr, "fak guard: provider %q has no public default base URL — pass --base-url\n", up)
		os.Exit(2)
	}
	apiKey := ""
	if apiKeyEnv != "" {
		apiKey = os.Getenv(apiKeyEnv)
	}

	// Subscription is the DEFAULT for Claude: whenever the upstream is Anthropic and no
	// API key was EXPLICITLY configured (--api-key-env), fak sources the Claude Pro/Max
	// OAuth token and sends it upstream as Authorization: Bearer + the oauth beta (the
	// scheme api.anthropic.com accepts an sk-ant-oat token under), holding the token
	// itself and ignoring the client's credential. A bare ANTHROPIC_API_KEY in the
	// environment NO LONGER flips this — a global SDK key must not silently bill your API
	// account when you hold a subscription. To opt INTO API billing, name the key
	// explicitly: --api-key-env ANTHROPIC_API_KEY. --anthropic-oauth forces the
	// subscription path and fails loud if no token is found.
	pinUpstream := false
	oauthSource := ""
	passthroughFallback := false
	ambientKeyOverridden := false
	if forceOAuth && up != "anthropic" {
		fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth applies only to --provider anthropic (got %q)\n", up)
		os.Exit(2)
	}
	autoOAuth := up == "anthropic" && apiKey == ""
	if forceOAuth || autoOAuth {
		tok, src, terr := resolveAnthropicOAuthToken(oauthTokenEnv)
		switch {
		case terr == nil:
			apiKey = tok
			pinUpstream = true
			oauthSource = src
			// Held the subscription token despite a bare ANTHROPIC_API_KEY in the
			// environment: flag it so cmdGuard can make the override discoverable (the
			// user may have expected that key to bill their API account).
			ambientKeyOverridden = autoOAuth && os.Getenv("ANTHROPIC_API_KEY") != ""
		case forceOAuth:
			// Explicitly requested but nothing to use — fail loud.
			fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth: %v\n", terr)
			os.Exit(2)
		case guardSubscriptionLoginPresent(oauthTokenEnv):
			// A subscription login EXISTS on disk but its token was unreadable this instant —
			// Claude Code rewrites .credentials.json ~hourly and the OAuth access token is
			// short-lived, so a boot read can catch the file mid-rotation (or holding a
			// just-expired token, which resolveAnthropicOAuthToken correctly drops rather than
			// send). Demoting to passthrough HERE would strip the placeholder ANTHROPIC_API_KEY
			// that keeps the wrapped agent from falling into its OWN /login — the 'stuck on
			// login sometimes' hang. So PIN ON INTENT with an empty boot apiKey: pinUpstream
			// stays true and the per-request APIKeyFunc (guard.go) re-reads the freshly-rotated
			// token on the first turn. effectiveAPIKey already tolerates an empty boot key
			// (func result wins; the 401 path self-heals once), so the first turn waits for the
			// rotation instead of dropping the agent into a login prompt.
			pinUpstream = true
			oauthSource = "subscription login (token rotating; resolved per request)"
		default:
			// Auto attempt found no token AND no subscription login is present at all: fall
			// back to plain passthrough — the wrapped agent's own credential (a subscription
			// login OR ANTHROPIC_API_KEY) flows upstream, so a pure API-billing user is
			// unaffected.
			passthroughFallback = true
		}
	}
	return guardUpstream{
		provider: up, autodetected: autodetected, baseURL: resolvedBase,
		apiKey: apiKey, pinUpstream: pinUpstream, oauthSource: oauthSource,
		passthroughFallback: passthroughFallback, ambientKeyOverridden: ambientKeyOverridden,
		remoteServe: remote,
	}
}

// buildGuardChild constructs the wrapped-agent command with ONLY the gateway URL injected
// into its environment (never the parent shell). In pinned subscription mode it also hands
// the client a placeholder ANTHROPIC_API_KEY (when it has none) so it talks x-api-key to the
// gateway, which ignores the placeholder and authenticates upstream with the held token.
func buildGuardChild(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) *exec.Cmd {
	// Landlock hook-floor (opt-in, Linux): rewrite the agent argv so the child is launched
	// through the fak re-exec trampoline, which applies the read-only-.git/hooks ruleset to
	// itself before exec'ing the agent. Off by default, no-op on non-Linux or when the hook
	// dirs cannot be resolved — the original command is used unchanged.
	command = maybeLandlockCommand(command)
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = os.Environ()
	for _, kv := range injected {
		child.Env = append(child.Env, kv[0]+"="+kv[1])
	}
	for _, kv := range extraEnv {
		if strings.TrimSpace(kv[0]) != "" {
			child.Env = append(child.Env, kv[0]+"="+kv[1])
		}
	}
	// Subscription mode: hand the client a PLACEHOLDER api key (only if it has none) so
	// it talks to the gateway in x-api-key mode; the gateway IGNORES the placeholder
	// (pinUpstream) and authenticates upstream with the real held OAuth token. Without it
	// the client may forward its own subscription bearer — also ignored in pinned mode —
	// so either way the held token is what reaches Anthropic.
	if pinUpstream && os.Getenv("ANTHROPIC_API_KEY") == "" {
		child.Env = append(child.Env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder")
	}
	return child
}

// maybeLandlockCommand rewrites the agent argv to run through the fak Landlock trampoline
// when the hook-floor is opted in (guard.OptedIn) AND the host is Linux. It resolves the
// repo's git dir, work-tree root, and hooks dir with git's OWN resolution — never by string-
// concatenating "<root>/.git/hooks", which would break linked worktrees and submodules where
// .git is a file. On any miss — not opted in, not Linux, fak's own path unresolvable, no git
// dir, no hook dir to protect — it returns command unchanged (the floor degrades to today's
// behavior, never blocking the spawn). The trampoline itself fails open at runtime on a
// kernel without Landlock.
func maybeLandlockCommand(command []string) []string {
	if runtime.GOOS != "linux" || !guard.OptedIn(os.Getenv) {
		return command
	}
	fakBin, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — cannot resolve fak binary (%v); spawning agent unrestricted\n", err)
		return command
	}
	gitOut := func(args ...string) string {
		out, err := exec.Command("git", args...).Output()
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(out))
	}
	gitDir := gitOut("rev-parse", "--absolute-git-dir")
	if gitDir == "" {
		fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not applied — not in a git repo; spawning agent unrestricted")
		return command
	}
	repoRoot := gitOut("rev-parse", "--show-toplevel")
	hooksPath := gitOut("rev-parse", "--git-path", "hooks")
	bare := gitOut("rev-parse", "--is-bare-repository") == "true"

	spec := guard.ResolveSpec(repoRoot, gitDir, hooksPath, bare)
	if len(spec.ReadOnlyDirs) == 0 {
		fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not applied — no hook dir resolved; spawning agent unrestricted")
		return command
	}
	return guard.TrampolineArgv(fakBin, spec, command)
}

type guardBudgetRestartEvent struct {
	Schema      string          `json:"schema"`
	FromTraceID string          `json:"from_trace_id"`
	ToTraceID   string          `json:"to_trace_id"`
	Reason      string          `json:"reason,omitempty"`
	SeedFile    string          `json:"seed_file,omitempty"`
	Seed        []agent.Message `json:"seed_messages,omitempty"`
	SeedText    string          `json:"seed_text,omitempty"`
	Note        string          `json:"note"`
}

type guardBudgetRestarter struct {
	enabled            bool
	freshContextTokens int
	limit              int
	seedDir            string
	stderr             io.Writer
	events             chan guardBudgetRestartEvent
}

func newGuardBudgetRestarter(enabled bool, freshContextTokens, limit int, seedDir string, stderr io.Writer) *guardBudgetRestarter {
	return &guardBudgetRestarter{
		enabled:            enabled,
		freshContextTokens: freshContextTokens,
		limit:              limit,
		seedDir:            strings.TrimSpace(seedDir),
		stderr:             stderr,
		events:             make(chan guardBudgetRestartEvent, 1),
	}
}

func (r *guardBudgetRestarter) Enabled() bool { return r != nil && r.enabled }

func (r *guardBudgetRestarter) OnBudgetExhausted(ctx context.Context, st gateway.SessionState, messages []agent.Message) {
	if !r.Enabled() || strings.TrimSpace(st.TraceID) == "" || strings.TrimSpace(st.ContinuationID) == "" {
		return
	}
	reset := resetServedSessionOnBudget(r.freshContextTokens)
	if reset == nil {
		return
	}
	nextTrace, seed, ok := reset(ctx, st.TraceID, messages)
	if !ok || strings.TrimSpace(nextTrace) == "" {
		return
	}
	ev := guardBudgetRestartEvent{
		Schema:      "fak.guard.budget_restart.v1",
		FromTraceID: st.TraceID,
		ToTraceID:   nextTrace,
		Reason:      st.Reason,
		Seed:        seed,
		SeedText:    guardSeedText(seed),
		Note:        "context budget exhausted; fak guard is relaunching the child under the continuation trace",
	}
	if path, err := writeGuardRestartSeedFile(r.seedDir, ev); err == nil {
		ev.SeedFile = path
	} else if r.stderr != nil {
		fmt.Fprintf(r.stderr, "fak guard: budget restart seed write failed: %v\n", err)
	}
	select {
	case r.events <- ev:
	default:
		if r.stderr != nil {
			fmt.Fprintf(r.stderr, "fak guard: budget restart event for %s dropped; restart already pending\n", st.TraceID)
		}
	}
}

func guardSeedText(seed []agent.Message) string {
	var parts []string
	for _, m := range seed {
		if c := strings.TrimSpace(m.Content); c != "" {
			parts = append(parts, c)
		}
	}
	return strings.Join(parts, "\n\n")
}

func writeGuardRestartSeedFile(dir string, ev guardBudgetRestartEvent) (string, error) {
	if strings.TrimSpace(dir) == "" {
		var err error
		dir, err = os.MkdirTemp("", "fak-guard-reset-*")
		if err != nil {
			return "", err
		}
	} else if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	name := "reset-" + guardSafeFilePart(ev.FromTraceID) + "-to-" + guardSafeFilePart(ev.ToTraceID) + ".json"
	path := filepath.Join(dir, name)
	ev.SeedFile = path
	raw, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func guardSafeFilePart(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "trace"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		return "trace"
	}
	if len(out) > 80 {
		out = out[:80]
	}
	return out
}

func guardRestartEnv(ev guardBudgetRestartEvent) [][2]string {
	env := [][2]string{
		{"FAK_RESET_FROM_TRACE", ev.FromTraceID},
		{"FAK_RESET_TRACE_ID", ev.ToTraceID},
		{"FAK_SESSION_ID", ev.ToTraceID},
		{"FAK_RESET_REASON", ev.Reason},
	}
	if ev.SeedFile != "" {
		env = append(env, [2]string{"FAK_RESET_SEED_FILE", ev.SeedFile})
	}
	return env
}

// runGuardChildAndReport runs the wrapped agent to completion, tears the gateway down,
// prints the session's adjudication + journal summary (unless quiet), flushes the durable
// trail, and exits with the child's own code — surfacing a gateway-mid-session failure as
// a non-silent error so a clean child exit never hides a downed adjudication boundary.
func runGuardChildAndReport(child *exec.Cmd, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {
	runErr := child.Run()
	finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
}

func runGuardChildSupervisedAndReport(command []string, injected [][2]string, pinUpstream bool, restarter *guardBudgetRestarter, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {
	var extraEnv [][2]string
	restarts := 0
	for {
		child := buildGuardChild(command, injected, pinUpstream, extraEnv...)
		wait := make(chan error, 1)
		if err := child.Start(); err != nil {
			finishGuardChildAndReport(err, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
			return
		}
		go func() { wait <- child.Wait() }()
		select {
		case runErr := <-wait:
			finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
			return
		case ev := <-restarter.events:
			if restarter.limit > 0 && restarts >= restarter.limit {
				if restarter.stderr != nil {
					fmt.Fprintf(restarter.stderr, "fak guard: restart limit %d reached; leaving child on drained session %s\n", restarter.limit, ev.FromTraceID)
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
				return
			}
			restarts++
			if restarter.stderr != nil {
				fmt.Fprintf(restarter.stderr, "fak guard: context budget exhausted for %s; restarting child as %s\n", ev.FromTraceID, ev.ToTraceID)
				if ev.SeedFile != "" {
					fmt.Fprintf(restarter.stderr, "fak guard: carryover seed written to %s\n", ev.SeedFile)
				}
			}
			srv.SetDefaultTraceID(ev.ToTraceID)
			extraEnv = guardRestartEnv(ev)
			// Let the triggering response finish flushing to the wrapped client before
			// stopping the process that initiated it.
			time.Sleep(750 * time.Millisecond)
			stopGuardChild(child, wait, 2*time.Second)
		}
	}
}

func stopGuardChild(child *exec.Cmd, wait <-chan error, grace time.Duration) {
	if child == nil || child.Process == nil {
		return
	}
	_ = child.Process.Signal(os.Interrupt)
	select {
	case <-wait:
		return
	case <-time.After(grace):
		_ = child.Process.Kill()
		<-wait
	}
}

// formatGuardResumeGuidance is the concise, actionable note printed when the wrapped agent
// exits abnormally (a non-zero code — a crash, an OOM, or a terminal upstream error). The
// guard process holds no agent conversation state itself — the wrapped tool owns that — so
// "resume" means re-running the same `fak guard -- <command>` with the agent's own
// resume/continue flag. The last line encodes a hard-won recovery: a guarded resume that
// dies IMMEDIATELY with "upstream model error" (a malformed-request rejection that can
// follow a mid-conversation quarantine) usually clears if that one resume is retried WITHOUT
// fak guard, then re-wrapped. Returned as a string (not printed) so it is unit-testable.
func formatGuardResumeGuidance(agentName string, code int) string {
	return fmt.Sprintf(
		"\nfak guard: %s exited abnormally (code %d).\n"+
			"  resume: re-run the same `fak guard -- %s …` — launch the agent with its own resume/continue flag (e.g. `claude --continue`) so it picks the conversation back up.\n"+
			"  this session's decision journal is replayable with `fak audit verify` (path in the audit summary above).\n"+
			"  if a guarded resume dies IMMEDIATELY with \"upstream model error\", retry that one resume WITHOUT fak guard to recover, then re-wrap.\n",
		agentName, code, agentName)
}

// formatVCacheSnapshotPointer is the exit pointer that closes the loop between the LIVE
// guard cache summary and the OFFLINE `fak vcache` family. After a session persists its
// OBSERVED provider-cache window (vcachesnapshot.Write), this line tells the operator the
// window is on disk AND names the one command that re-derives THIS session's REALIZED cache
// multiplier from it: `fak vcache score` reads the well-known snapshot path with no flags
// (it folds the same turns through the same vcacheobserve engine the summary line priced),
// and `fak vcache observe` renders the per-sub-concept panels. Without this, the snapshot is
// written silently and the related vcache tools look like they only have a synthetic forecast
// to chew on — the operator never learns the real session data is right there to replay.
//
// Empty when no turns were recorded (a run that never saw a cached turn writes an empty
// snapshot, which the score correctly treats as "no observed window" and falls open to the
// forecast), so a no-cache session stays quiet rather than printing a vacuous 0-turn pointer.
func formatVCacheSnapshotPointer(turns int, path string) string {
	if turns <= 0 {
		return ""
	}
	return fmt.Sprintf(
		"fak guard: cache window — recorded %d turn(s) to %s; replay THIS session's realized cache multiplier offline with `fak vcache score` (no flags reads this snapshot), or `fak vcache observe` for the per-sub-concept panels.\n",
		turns, path)
}

func finishGuardChildAndReport(runErr error, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {

	// Tear the gateway down and report what the kernel decided this session.
	cancel()
	serr := <-serveErr
	if !quiet {
		fmt.Fprintln(os.Stderr)
		sum := srv.AdjudicationSummary()
		fmt.Fprint(os.Stderr, formatAuditSummary(sum))
		fmt.Fprint(os.Stderr, formatAmplification(srv.KernelCounters(), sum))
		fmt.Fprint(os.Stderr, formatJournalSummary(auditJournal, auditSeq0))
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075).
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.Append("guard", agentName, cachevalueledger.DefaultLedgerRel, stats)
	}
	// Persist this session's OBSERVED provider-cache window so a later `fak vcache score`
	// (a separate process) reports the REALIZED multiplier from real traffic instead of the
	// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session,
	// and an empty window leaves the snapshot empty so the score falls open to the forecast.
	// On a clean write, point the operator at the offline vcache tools that now hold this
	// session's data — otherwise the snapshot is silent and the related vcache items look
	// like they only have a synthetic forecast to score.
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		snapPath := vcachesnapshot.DefaultPath()
		if err := vcachesnapshot.Write(snapPath, turns); err == nil && !quiet {
			fmt.Fprint(os.Stderr, formatVCacheSnapshotPointer(len(turns), snapPath))
		}
	}
	// Flush + fsync the durable trail before exit so a row returned to the agent is
	// never lost to a buffered write (Close is safe on a nil/in-memory journal).
	if auditJournal != nil {
		_ = auditJournal.Close()
	}
	// Faithfully surface the child's exit code first (so `fak guard -- claude -p …`
	// scripts see what the agent returned).
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			// An abnormal (non-zero) exit is a crash / OOM / terminal upstream error — print
			// the actionable resume note so the operator isn't left with a bare exit code.
			// Suppressed under --quiet (scripted `-p` runs) and skipped on a clean 0 exit.
			if code := ee.ExitCode(); code != 0 && !quiet {
				fmt.Fprint(os.Stderr, formatGuardResumeGuidance(agentName, code))
			}
			os.Exit(ee.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "fak guard: could not run %q: %v\n", agentName, runErr)
		os.Exit(1)
	}
	// The child succeeded — but if the gateway itself failed mid-session (Serve returned
	// something other than a clean shutdown), the adjudication boundary was down for part
	// of the run, so do not report a silent success. A clean teardown returns nil.
	if serr != nil && !errors.Is(serr, http.ErrServerClosed) && !errors.Is(serr, context.Canceled) {
		fmt.Fprintf(os.Stderr, "fak guard: gateway error during the session: %v\n", serr)
		os.Exit(1)
	}
}
