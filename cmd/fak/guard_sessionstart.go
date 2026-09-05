package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessionsteer"
)

// guard_sessionstart.go — the discoverability affordance for fak's MCP verbs (#3092).
//
// Claude Code DEFERS MCP tools: fak's mcp__fak__* verbs are surfaced by name only and must
// be paged in with a ToolSearch round-trip before they can be called. In an unattended
// /goal run nothing points the agent at them at task start, so it never searches, never
// pulls them in, and solves the task as a generic Bash/Edit coder — fak present but inert
// (the pathology behind session 2586c14b: 339 turns, 0 mcp__fak__* calls).
//
// This is a provider-native SessionStart hook. Claude Code and Codex inject its stdout into
// the FIRST turn's context as additionalContext (a one-time cost, NOT a per-prompt-prefix
// tax — so it does not fight the --expose token-thrift lever), naming the 2-3 entry verbs
// the agent should reach for first. It is the always-loaded affordance that survives the
// deferred-tool wall.
//
// Opt-out per harness via FAK_GUARD_AFFORDANCE_MODE=off (default on). Fail-open: any bad
// args or write error is a silent exit 0 — a discoverability hint must never wedge a start.

const (
	guardSessionStartEnvMode = "FAK_GUARD_AFFORDANCE_MODE"
	guardSessionStartModeOff = "off"
	guardSessionStartModeOn  = "on"
)

// guardSessionStartHint is the one-line affordance injected into the first turn. It names the
// entry verbs (in their mcp__fak__ wire form, the names the agent actually calls) and the
// two situations where fak most earns its keep: discovering the task-scoped toolbelt, and
// gating a write.
const guardSessionStartHint = "Reach for the fak substrate verbs (MCP server `fak`) to discover the task-scoped toolbelt and access kernel services. Standard client tools are already wire-gated by the kernel proxy; `mcp__fak__fak_adjudicate` / `mcp__fak__fak_admit` are only for unmanaged clients requiring manual out-of-band admission. Call `mcp__fak__fak_capabilities` to find the right fak surface for this task; `mcp__fak__fak_memory_run` for durable memory; `mcp__fak__fak_tools_search` to page in the rest. Invoke these deferred tools explicitly to page them in."

func cmdGuardSessionStart(argv []string) {
	os.Exit(runGuardSessionStartHook(os.Stdout, os.Stderr, os.Stdin, argv))
}

// runGuardSessionStart is the retained stdin-free entry: a nil payload carries no hook
// source, so it composes to the base affordance exactly as before (no look-ahead pickup).
func runGuardSessionStart(stdout, stderr io.Writer, argv []string) int {
	return runGuardSessionStartHook(stdout, stderr, nil, argv)
}

// runGuardSessionStartHook is the SessionStart-hook actuator. On the "on" mode (default) it
// prints the additionalContext JSON envelope to stdout and returns 0; "off" returns 0 with
// no output. It fails OPEN — any bad args return 0 with no injection.
//
// A stdin payload (the real hook) whose SessionStart source is "compact" ALSO injects a
// fresh same-base-SHA look-ahead lesson beside the reframed affordance (#5207). A nil stdin
// (the runGuardSessionStart entry) skips the pickup, keeping that path byte-identical.
func runGuardSessionStartHook(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	// Repair only torn ephemeral session refs before any resume/discovery path can
	// invoke Git with --all. The authoritative session registry remains untouched.
	// Fail-open here: SessionStart affordances must never wedge a fresh process.
	if reaped, err := leaseref.NewInDir("").ReapMalformedSessionRefs(context.Background()); err == nil && len(reaped) > 0 {
		fmt.Fprintf(stderr, "fak: repaired %d malformed session lease ref(s) before resume\n", len(reaped))
	}
	fs := flag.NewFlagSet("guard-sessionstart", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardSessionStartEnvMode), "off|on")
	// --managed marks a session admitted onto the long-horizon posture (a headless/fleet worker,
	// per the sessionsteer admission at install time). When set, the injected context ALSO carries
	// the persistence + managed-context RULE (spine #3512) — the soft, always-on half of the
	// long-horizon default. Attended human-driven sessions get the base affordance only.
	managedFlag := fs.Bool("managed", false, "inject the long-horizon persistence + managed-context rule")
	providerFlag := fs.String("provider", "claude", "provider emitting the SessionStart event")
	stateFlag := fs.String("state", "", "launch-scoped provider session binding")
	// --trace carries the guard trace id threaded in at install (guardSessionStartArgs). Paired
	// with the child's CLAUDE_CODE_SESSION_ID (the transcript UUID) it is the A1 identity join
	// (#4112) the resume watchdog reads to resolve a crashed UUID back to its gateway trace.
	traceFlag := fs.String("trace", "", "guard trace id to join to the transcript uuid")
	if err := fs.Parse(argv); err != nil {
		// Fail open: a discoverability hint must never wedge a session start.
		return 0
	}
	payload := readHookStdin(stdin)
	hookStart := parseGuardProviderSessionStart(payload)
	effectiveTrace := strings.TrimSpace(*traceFlag)
	provider := strings.ToLower(strings.TrimSpace(*providerFlag))
	if provider == "" {
		provider = "claude"
	}
	sessionID := hookStart.SessionID
	if sessionID == "" && provider == "claude" {
		sessionID = strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))
	}
	boundarySource := hookStart.Source
	bindCodexSession := false
	if provider == "codex" && strings.TrimSpace(*stateFlag) != "" {
		decision, err := classifyGuardCodexSessionStart(*stateFlag, hookStart.Source, sessionID)
		if err != nil {
			fmt.Fprintf(stderr, "fak: Codex session binding was not read: %v\n", err)
		} else {
			boundarySource = decision.Source
			bindCodexSession = decision.Bind
			if decision.Trace != "" {
				effectiveTrace = decision.Trace
			}
		}
	}
	boundaryRecorded := false
	if boundarySource == "clear" && sessionID != "" {
		result, err := notifyGuardProviderSessionStart(
			os.Getenv(guardLifecycleSocketEnv), os.Getenv(guardLifecycleTokenEnv),
			provider, boundarySource, sessionID, 500*time.Millisecond,
		)
		if err != nil {
			fmt.Fprintf(stderr, "fak: provider session boundary was not recorded: %v\n", err)
		} else {
			if result.Applied {
				recordGuardProviderSessionClose(result.PreviousTrace, boundarySource)
			}
			effectiveTrace = result.NewTrace
			boundaryRecorded = true
		}
	}
	if provider == "codex" && bindCodexSession && (boundarySource != "clear" || boundaryRecorded) {
		if err := writeGuardCodexSessionBinding(*stateFlag, sessionID, effectiveTrace); err != nil {
			fmt.Fprintf(stderr, "fak: Codex session binding was not recorded: %v\n", err)
		} else if err := writeCodexGuardWitness("", sessionID); err != nil {
			// The launch-scoped SessionStart hook is installed by the host-side guard
			// before Codex starts. Persist the same durable witness consumed by
			// sessions codex-loop here, before the first model/tool turn; do not wait
			// for the optional UserPromptSubmit hook.
			fmt.Fprintf(stderr, "fak: Codex guard witness was not recorded: %v\n", err)
		}
	}
	// Record the uuid<->trace join first (best-effort, fail-open), so it is written on EVERY
	// SessionStart source — independent of the affordance mode below. The affordance "off" knob
	// governs the injected hint, not the durable identity store the watchdog depends on.
	driverPID := recordGuardSessionStartIdentityFor(effectiveTrace, sessionID, provider, hookStart.Source)
	// …then register the session in the crash-survivable journal (C3, #3787), on the same terms:
	// every SessionStart source, ahead of the affordance knob, best-effort. It reuses the driver
	// pid the join above already witnessed rather than paying a second process census.
	recordGuardSessionStartJournalFor(effectiveTrace, sessionID, provider, driverPID)
	if normalizeGuardSessionStartMode(*modeFlag) == guardSessionStartModeOff {
		return 0
	}
	// Compose the injected context: the MCP-affordance hint always, plus the long-horizon rule
	// when this session was admitted MANAGED. SessionStartRule returns "" for a non-managed
	// directive, so an attended session composes to the base hint unchanged.
	additionalContext := guardSessionStartHint
	if *managedFlag {
		directive := sessionsteer.Steer(sessionsteer.SteerInput{Headless: true, DurableStore: true})
		if rule := sessionsteer.SessionStartRule(directive); rule != "" {
			additionalContext = guardSessionStartHint + "\n\n" + rule
		}
		if strings.TrimSpace(os.Getenv("FAK_TOOL_WIDTH_HINT")) != "off" {
			additionalContext += "\n\n" + sessionsteer.IndependentToolHint(true)
		}
	}
	// Emit-time reframe (#3566): route the composed additionalContext through the deterministic,
	// token-superset-safe positive-voice pass so every string fak injects at SessionStart leads
	// with the affordance. sessionsteer stays stdlib-only (tier 1) — the reframe lives here, at the
	// emit boundary, not inside the pure decision core. Idempotent, so a source string already in
	// positive voice is returned unchanged.
	// The pass is DEFAULT-ON and now runs behind the #3568 ablation lever, so #3546's control
	// arm is one env toggle (FAK_ABLATE=negframe_reframe) rather than hand-swapped strings.
	// The same call returns this turn's telemetry, recorded best-effort to the negframe
	// journal the exit summary folds into its one-line arm/residual/fallback report.
	reframed, negframeRow := guardNegframeReframe(negframe.Fak(additionalContext))
	additionalContext = reframed
	// #5207 look-ahead pickup: on a real hook payload whose SessionStart source is "compact",
	// inject the fresh same-base-SHA lesson VERBATIM after the reframed affordance — a
	// witnessed lesson is a fact to carry, not a string to positive-voice. A nil stdin (the
	// retained stdin-free entry) never triggers this, so that path stays unchanged.
	if len(payload) > 0 {
		if lesson, ok := lookaheadLessonForCompactPayload(payload); ok {
			additionalContext = additionalContext + "\n\n" + lesson
		}
	}
	// Begin (not append): SessionStart is the session boundary, so the per-turn stream the exit
	// summary folds starts fresh here rather than accumulating this workspace's whole history.
	guardNegframeBegin(guardNegframeJournalRel, negframeRow)
	// Claude Code injects a SessionStart hook's hookSpecificOutput.additionalContext into the
	// first turn's context. Emit the envelope; a marshal failure is a silent no-op (fail open).
	envelope := map[string]any{
		"hookSpecificOutput": map[string]any{
			"hookEventName":     "SessionStart",
			"additionalContext": additionalContext,
		},
	}
	data, err := json.Marshal(envelope)
	if err != nil {
		return 0
	}
	fmt.Fprintln(stdout, string(data))
	return 0
}

type guardProviderSessionStart struct {
	Source    string
	SessionID string
}

func parseGuardProviderSessionStart(payload []byte) guardProviderSessionStart {
	if len(payload) == 0 {
		return guardProviderSessionStart{}
	}
	var in struct {
		Source         string `json:"source"`
		SessionID      string `json:"session_id"`
		ThreadID       string `json:"thread_id"`
		ConversationID string `json:"conversation_id"`
	}
	if json.Unmarshal(payload, &in) != nil {
		return guardProviderSessionStart{}
	}
	id := strings.TrimSpace(in.SessionID)
	if id == "" {
		id = strings.TrimSpace(in.ThreadID)
	}
	if id == "" {
		id = strings.TrimSpace(in.ConversationID)
	}
	return guardProviderSessionStart{Source: strings.ToLower(strings.TrimSpace(in.Source)), SessionID: id}
}

func recordGuardSessionStartIdentityFor(traceID, sessionID, provider, source string) int {
	uuid := strings.TrimSpace(sessionID)
	traceID = strings.TrimSpace(traceID)
	if uuid == "" || traceID == "" {
		return 0 // a half row is not a join; FoldIdentity would skip it anyway
	}
	path := resume.IdentityLedgerPath(resolveSweepRegDir(""))
	row := resume.IdentityRow{
		TS:       time.Now().UTC().Format(time.RFC3339),
		UUID:     uuid,
		Trace:    traceID,
		Via:      "guard-sessionstart",
		Provider: strings.ToLower(strings.TrimSpace(provider)),
		Source:   strings.ToLower(strings.TrimSpace(source)),
	}
	// The join row goes down FIRST, carrying no pid, because it is the fact the resume
	// watchdog depends on (#4112) and the driver witness below reads the host process table —
	// on a loaded host that census is a second of wall clock. Folding the witness into this
	// row would put that second in front of the write, so a hook killed mid-census would lose
	// a join it used to record. Ordering keeps the older guarantee strictly ahead of the newer
	// one.
	_ = resume.AppendIdentityRow(filepath.Dir(path), row)

	// The driver process this hook is running under, when it can be WITNESSED (0 when it
	// cannot). This is the only moment on the host where the transcript UUID is known and the
	// process owning it is provably alive, so it is the only place a first-generation
	// `claude -p …` worker can be bound to a process at all (#5542). It rides a SECOND append
	// rather than the row above: the store is append-only and both folds are last-write-wins,
	// so a later row carrying the same (uuid, trace) re-states the join unchanged while adding
	// the pid FoldIdentityDriverPIDs is looking for. An unwitnessed start appends nothing, so
	// a host that can never witness a driver pays no extra row at all.
	pid := witnessGuardSessionStartDriverPID()
	if pid > 0 {
		row.TS = time.Now().UTC().Format(time.RFC3339)
		row.PID = pid
		_ = resume.AppendIdentityRow(filepath.Dir(path), row)
	}
	return pid
}

// --- driver-pid witness (#5542) -------------------------------------------------------- //
//
// `fak resume stopped` will only move a mid-tool row off DEFER when it can decide whether the
// driver that owned the transcript is still running, and the only handle it has on that is a
// pid something durably RECORDED for the session. Before this, the sole producer was the
// resume watchdog's launch ledger — which records a pid only for sessions it itself resumed —
// so every first-generation worker resolved to "no recorded driver pid" and deferred forever.
//
// The rule that keeps this honest is the same one the consumer lives by: a recorded pid is a
// CLAIM about which process owns the transcript, and a WRONG pid is strictly worse than none.
// A pid belonging to a short-lived wrapper below the driver would manufacture a false `gone`
// the moment that wrapper exits, and a resume fired on that verdict would land on a transcript
// its original driver is still writing. So this records nothing it did not witness: it walks
// the hook's own ancestor chain in ONE process-table snapshot and takes the nearest ancestor
// that IS the agent driver (guardSessionStartProcIsDriver) — primarily by process IMAGE,
// judged by the same audited image-vs-product predicate the dispatch preflight uses
// (dispatchProcessImageMatchesProduct). A bare command-line substring would not do —
// `fak guard -- claude …`, the wrapper one level ABOVE the driver on this host, contains the
// word "claude" while being a different program.
//
// The one install shape the image rule alone cannot name is a driver launched through a Node
// wrapper: it presents the image `node`, so #5542 shipped without witnessing it and that whole
// install deferred permanently (#5557). guardSessionStartNodeWrapsDriver admits exactly that
// shape and nothing more — a `node` IMAGE whose argv executes the driver's own ENTRYPOINT
// SCRIPT — which is why it cannot re-admit the wrapper the image rule exists to exclude: the
// `fak guard --` wrapper's image is `fak`, so it never reaches the argv test at all.
//
// Everything that is not that witness records 0, which the store's doc contract defines as NOT
// RECORDED: an unreadable or empty census, a chain that leaves the snapshot, a ppid cycle, a
// driver the platform names something this file has no marker for, or a walk that runs past
// its hop bound. Each of those leaves the consumer exactly where it is today — deferring —
// which is the fail-safe, not a regression.

// guardSessionStartProcRelations enumerates the host's processes through the same audited
// cross-platform census the rest of the fleet reads (`fak ps`, the resume watchdog, the
// stopped-triage liveness probe). Injectable so a test can stage an ancestor chain without
// spawning one, matching the seam shape in resume_stopped_liveness.go.
var guardSessionStartProcRelations = procguard.CollectRelations

// guardSessionStartParentPID reports the pid that spawned this hook — the entry point of the
// ancestor walk. Injectable for the same reason.
var guardSessionStartParentPID = os.Getppid

const (
	// guardSessionStartAncestorHops bounds the ancestor walk. The driver is the hook's direct
	// parent on a host that spawns hooks without a shell, and one or two hops up on one that
	// does; a bound keeps a recycled ppid on a long chain from walking somewhere unrelated.
	guardSessionStartAncestorHops = 8
	// guardSessionStartWitnessBudget caps how long the census may take before the hook gives up
	// and records no pid. The POSIX collector shells out to `ps` under a 30s timeout of its own,
	// and a SessionStart hook must never wedge a start — so a slow host loses the witness (and
	// defers, as today) rather than stalling the session.
	guardSessionStartWitnessBudget = 3 * time.Second
)

// witnessGuardSessionStartDriverPID returns the pid of the driver process this hook is running
// under, or 0 when none could be witnessed within the budget. Never errors and never blocks
// past the budget: the identity row is written either way, carrying a pid only when one was
// actually observed.
func witnessGuardSessionStartDriverPID() int {
	done := make(chan int, 1) // buffered: a late census must not block its own goroutine
	go func() {
		procs, errStr := guardSessionStartProcRelations()
		if errStr != "" || len(procs) == 0 {
			done <- 0 // no census is no witness (the #5385 empty-table shape)
			return
		}
		done <- guardSessionStartDriverPIDIn(guardSessionStartParentPID(), procs)
	}()
	select {
	case pid := <-done:
		return pid
	case <-time.After(guardSessionStartWitnessBudget):
		return 0
	}
}

// guardSessionStartDriverPIDIn is the pure half: walk up from startPID through the snapshot's
// ppid links and return the first ancestor whose process image is the agent driver. Returns 0
// when the chain leaves the snapshot, revisits a pid (a recycled ppid can close a cycle), runs
// out of parents, or exceeds the hop bound — every one of which is "not witnessed", not "gone".
func guardSessionStartDriverPIDIn(startPID int, procs []procguard.Proc) int {
	byPID := make(map[int]procguard.Proc, len(procs))
	for _, p := range procs {
		if p.PID > 0 {
			byPID[p.PID] = p
		}
	}
	seen := make(map[int]bool, guardSessionStartAncestorHops)
	pid := startPID
	for hop := 0; hop < guardSessionStartAncestorHops; hop++ {
		if pid <= 0 || seen[pid] {
			return 0
		}
		seen[pid] = true
		p, ok := byPID[pid]
		if !ok {
			return 0 // the chain leaves the table: nothing further can be witnessed
		}
		if guardSessionStartProcIsDriver(p) {
			return p.PID
		}
		if p.PPID == nil {
			return 0 // the platform did not surrender a parent for this row
		}
		pid = *p.PPID
	}
	return 0
}

// guardSessionStartProcIsDriver decides whether ONE snapshot row is the agent driver this hook
// is running under. Two arms, in strictly widening order of what they are allowed to read:
//
//  1. the process IMAGE is the driver (dispatchProcessImageMatchesProduct) — the #5542 rule,
//     unchanged, and the only arm that ever fires for a natively-installed `claude`;
//  2. the process image is `node` and its argv executes the driver's own entrypoint script —
//     the #5557 arm, which is the ONLY place this file reads a command line.
//
// Anything else is not the driver, and the walk keeps climbing.
func guardSessionStartProcIsDriver(p procguard.Proc) bool {
	return dispatchProcessImageMatchesProduct(p.Name, "claude") ||
		guardSessionStartNodeWrapsDriver(p.Name, p.Cmdline)
}

// guardSessionStartDriverEntrypointBases / …Dirs spell the driver's ENTRYPOINT SCRIPT as a
// (directory, file) pair: a `cli.js` sitting DIRECTLY inside a `claude` or `claude-code`
// directory. Both shapes the CLI is installed in present exactly that — the npm package
// (`…/node_modules/@anthropic-ai/claude-code/cli.js`) and the local install
// (`…/.claude/local/…/claude-code/cli.js`, and the `…/claude/cli.js` form the resume-liveness
// fixture in this package already models). The pair is the point: matching the file name alone
// would admit any project's `cli.js`, and matching the directory alone would admit every
// process whose argv merely mentions a path under `~/.claude`.
var (
	guardSessionStartDriverEntrypointBases = []string{"cli.js", "cli.mjs", "cli.cjs"}
	guardSessionStartDriverEntrypointDirs  = []string{"claude", "claude-code"}
)

// guardSessionStartNodeWrapsDriver reports whether a process whose IMAGE is `node` is in fact
// the agent driver (#5557). This is the one command-line read in the whole witness, and the
// two gates that keep it from re-admitting the wrapper the image rule exists to exclude are:
//
//   - the image stem must be `node`. `fak guard -- claude …` runs as the image `fak`, so it
//     never reaches the argv test — the exact negative #5542 pinned, still failing to match
//     even though its command line names the driver twice over;
//   - the argv test demands an executed SCRIPT PATH, not the word "claude". A command line
//     that merely NAMES the driver (a wrapper's tail, a `--settings` path under `~/.claude`,
//     an MCP server started from a claude config dir) carries no `<claude|claude-code>/cli.js`
//     token, so it does not match either.
//
// Token-wise rather than substring-wise for the same reason: a path is compared as a (dir,
// base) pair split on the last separator, so a directory that merely ENDS in "claude" (say
// `notclaude/cli.js`) is a different token and does not match. A node process that is
// something else entirely fails both gates and the walk simply keeps climbing — or runs out of
// hops and records nothing, which is the same honest "not witnessed" it recorded before.
func guardSessionStartNodeWrapsDriver(name, cmdline string) bool {
	if dispatchProcessNameStem(name) != "node" {
		return false
	}
	// Backslashes fold to "/" so a Windows argv splits on the same separator a POSIX one does;
	// lowercasing matches the stem predicate's own case rule.
	for _, tok := range strings.Fields(strings.ToLower(strings.ReplaceAll(cmdline, `\`, "/"))) {
		tok = strings.Trim(tok, `"'`)
		slash := strings.LastIndex(tok, "/")
		if slash < 0 {
			continue // a bare `cli.js` names no package, so it identifies nothing
		}
		if !slices.Contains(guardSessionStartDriverEntrypointBases, tok[slash+1:]) {
			continue
		}
		dir := tok[:slash]
		if i := strings.LastIndex(dir, "/"); i >= 0 {
			dir = dir[i+1:]
		}
		if slices.Contains(guardSessionStartDriverEntrypointDirs, dir) {
			return true
		}
	}
	return false
}

// normalizeGuardSessionStartMode maps the env/flag knob to on|off. Default (empty) is ON —
// the affordance is the fix, so it is on by default; a harness that wants the leanest
// surface opts out with off.
func normalizeGuardSessionStartMode(mode string) string {
	if strings.EqualFold(strings.TrimSpace(mode), guardSessionStartModeOff) {
		return guardSessionStartModeOff
	}
	return guardSessionStartModeOn
}

type guardSessionStartInstall struct {
	Applied      bool
	Mode         string
	Managed      bool
	Provider     string
	SettingsPath string
	StatePath    string
	Reason       string
}

// installGuardSessionStartHook installs a provider-native SessionStart affordance hook.
// Claude merges it into the shared --settings file; Codex receives a trusted per-launch
// config layer. Off mode or an unsupported child is a no-op. Mirrors installGuardStopHook.
func installGuardSessionStartHookForProfile(command []string, profile harnessprofile.HarnessProfile, mode string, managed bool, existingSettingsPath, traceID string) ([]string, guardSessionStartInstall, error) {
	normalized := normalizeGuardSessionStartMode(mode)
	install := guardSessionStartInstall{Mode: normalized}
	if normalized == guardSessionStartModeOff {
		install.Reason = "disabled"
		return command, install, nil
	}
	if len(command) == 0 || (!profile.HasRepoint(harnessprofile.RepointSettingsFile) && !profile.HasRepoint(harnessprofile.RepointCLIConfig)) {
		install.Reason = "non-claude-child"
		return command, install, nil
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		dir, err = guardSessionTempDir("sessionstart")
		if err != nil {
			return command, guardSessionStartInstall{}, err
		}
	}
	return installGuardSessionStartHookAtForProfile(command, profile, mode, managed, fakBin, dir, existingSettingsPath, traceID)
}

func installGuardSessionStartHookAt(command []string, mode string, managed bool, fakBin, dir, existingSettingsPath, traceID string) ([]string, guardSessionStartInstall, error) {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return installGuardSessionStartHookAtForProfile(command, profile, mode, managed, fakBin, dir, existingSettingsPath, traceID)
}

func installGuardSessionStartHookAtForProfile(command []string, profile harnessprofile.HarnessProfile, mode string, managed bool, fakBin, dir, existingSettingsPath, traceID string) ([]string, guardSessionStartInstall, error) {
	normalized := normalizeGuardSessionStartMode(mode)
	install := guardSessionStartInstall{Mode: normalized}
	if normalized == guardSessionStartModeOff {
		install.Reason = "disabled"
		return command, install, nil
	}
	if len(command) > 0 && profile.HasRepoint(harnessprofile.RepointCLIConfig) {
		return installGuardCodexSessionStartHookAt(command, managed, fakBin, dir, traceID)
	}
	if !profile.HasRepoint(harnessprofile.RepointSettingsFile) {
		install.Reason = "non-claude-child"
		return command, install, nil
	}
	var settingsPath string
	if strings.TrimSpace(existingSettingsPath) != "" {
		if err := mergeGuardSessionStartIntoSettings(existingSettingsPath, fakBin, managed, traceID); err != nil {
			return command, install, err
		}
		settingsPath = existingSettingsPath
	} else {
		if strings.TrimSpace(dir) == "" {
			return command, install, fmt.Errorf("empty SessionStart hook settings directory")
		}
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return command, install, err
		}
		settingsPath = filepath.Join(dir, "claude-sessionstart-settings.json")
		if err := writeGuardSessionStartSettings(settingsPath, fakBin, managed, traceID); err != nil {
			return command, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	install.Applied = true
	install.Managed = managed
	install.Provider = "claude"
	install.SettingsPath = settingsPath
	return command, install, nil
}

// guardSessionStartManaged decides, by default, whether a wrapped child is admitted onto the
// long-horizon MANAGED posture: a headless/fleet worker (a `-p` child, not an attended TUI) is,
// where keep-going-past-a-long-window matters most and no human is present to drive it. This is
// the default-on switch for the persistence + managed-context rule (#3512); it leans on the SAME
// headless signal the task-handoff gate uses, so the two long-horizon gates admit in lockstep.
func guardSessionStartManaged(command []string) bool {
	return !guardChildInteractive(command)
}

// guardSessionStartArgs is the hook's argv. A MANAGED (headless) session carries --managed so
// the injected context includes the long-horizon persistence + managed-context rule (#3512). A
// non-empty traceID is threaded as --trace so the running hook holds BOTH ids — the guard trace
// and (from the child env) the transcript UUID — and can record the A1 identity join (#4112).
func guardSessionStartArgs(managed bool, traceID string, providers ...string) []string {
	args := []string{"guard-sessionstart"}
	provider := "claude"
	if len(providers) > 0 && strings.TrimSpace(providers[0]) != "" {
		provider = strings.ToLower(strings.TrimSpace(providers[0]))
	}
	args = append(args, "--provider", provider)
	if managed {
		args = append(args, "--managed")
	}
	if t := strings.TrimSpace(traceID); t != "" {
		args = append(args, "--trace", t)
	}
	return args
}

// guardSessionStartMatchers builds the SessionStart hook settings entry. The affordance is
// wanted on a fresh start AND on clear/compact/resume (a compacted context may have dropped
// the original hint), so the matcher is left empty to fire on every SessionStart source.
func guardSessionStartMatchers(fakBin string, managed bool, traceID string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    guardSessionStartArgs(managed, traceID, "claude"),
		}},
	}}
}

func writeGuardSessionStartSettings(path, fakBin string, managed bool, traceID string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"SessionStart": guardSessionStartMatchers(fakBin, managed, traceID),
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeGuardSettingsFileAtomic(path, data)
}

// mergeGuardSessionStartIntoSettings adds (or replaces) the SessionStart hook in an existing
// guard settings file, preserving every other key (PreCompact/Stop/toolproc hooks).
func mergeGuardSessionStartIntoSettings(path, fakBin string, managed bool, traceID string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	settings.Hooks["SessionStart"] = guardSessionStartMatchers(fakBin, managed, traceID)
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeGuardSettingsFileAtomic(path, data)
}
