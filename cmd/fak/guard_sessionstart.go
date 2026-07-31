package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
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
// This is a Claude Code SessionStart hook. Its stdout is injected into the FIRST turn's
// context as additionalContext (a one-time cost, NOT a per-prompt-prefix tax — so it does
// not fight the --expose token-thrift lever), naming the 2-3 entry verbs the agent should
// reach for first. It is the always-loaded affordance that survives the deferred-tool wall.
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
// two situations where fak most earns its keep: pulling ranked work, and gating a write.
const guardSessionStartHint = "Reach for the fak substrate verbs (MCP server `fak`) to select ranked repo work and gate tool calls through the kernel. Call `mcp__fak__fak_index_work` to pull ranked open work for this repo; `mcp__fak__fak_admit` / `mcp__fak__fak_adjudicate` to gate/execute a tool call through the kernel; `mcp__fak__fak_memory_run` for durable memory; `mcp__fak__fak_tools_search` to page in the rest. Invoke these deferred tools explicitly to page them in."

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
	fs := flag.NewFlagSet("guard-sessionstart", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardSessionStartEnvMode), "off|on")
	// --managed marks a session admitted onto the long-horizon posture (a headless/fleet worker,
	// per the sessionsteer admission at install time). When set, the injected context ALSO carries
	// the persistence + managed-context RULE (spine #3512) — the soft, always-on half of the
	// long-horizon default. Attended human-driven sessions get the base affordance only.
	managedFlag := fs.Bool("managed", false, "inject the long-horizon persistence + managed-context rule")
	// --trace carries the guard trace id threaded in at install (guardSessionStartArgs). Paired
	// with the child's CLAUDE_CODE_SESSION_ID (the transcript UUID) it is the A1 identity join
	// (#4112) the resume watchdog reads to resolve a crashed UUID back to its gateway trace.
	traceFlag := fs.String("trace", "", "guard trace id to join to the transcript uuid")
	if err := fs.Parse(argv); err != nil {
		// Fail open: a discoverability hint must never wedge a session start.
		return 0
	}
	// Record the uuid<->trace join first (best-effort, fail-open), so it is written on EVERY
	// SessionStart source — independent of the affordance mode below. The affordance "off" knob
	// governs the injected hint, not the durable identity store the watchdog depends on.
	recordGuardSessionStartIdentity(*traceFlag)
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
	if stdin != nil {
		if lesson, ok := lookaheadLessonForCompact(stdin); ok {
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

// recordGuardSessionStartIdentity best-effort appends one uuid<->trace join row to the durable
// resume_identity.jsonl store (the A1 fold's input, #4112) under the resolved fleet regDir, so
// the resume watchdog can later resolve a crashed transcript UUID to its gateway trace long
// after the TTL-GC'd descriptor registry has forgotten the pairing. Fail-open by the hook's
// contract: a missing id (a resumed child has CLAUDE_CODE_SESSION_ID stripped, so the UUID is
// blank) or any write error is a silent no-op — the identity join must never wedge a start.
func recordGuardSessionStartIdentity(traceID string) {
	uuid := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))
	traceID = strings.TrimSpace(traceID)
	if uuid == "" || traceID == "" {
		return // a half row is not a join; FoldIdentity would skip it anyway
	}
	row := resume.IdentityRow{
		TS:    time.Now().UTC().Format(time.RFC3339),
		UUID:  uuid,
		Trace: traceID,
		Via:   "guard-sessionstart",
	}
	_ = appendJSONL(resume.IdentityLedgerPath(resolveSweepRegDir("")), row)
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
	SettingsPath string
	Reason       string
}

// installGuardSessionStartHook installs the Claude Code SessionStart affordance hook,
// MERGING it into the shared --settings file the other guard hooks already wrote
// (existingSettingsPath) so a single --settings carries them all. Off mode or a non-claude
// child is a no-op. Mirrors installGuardStopHook.
func installGuardSessionStartHook(command []string, mode string, managed bool, existingSettingsPath, traceID string) ([]string, guardSessionStartInstall, error) {
	normalized := normalizeGuardSessionStartMode(mode)
	install := guardSessionStartInstall{Mode: normalized}
	if normalized == guardSessionStartModeOff {
		install.Reason = "disabled"
		return command, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
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
	return installGuardSessionStartHookAt(command, mode, managed, fakBin, dir, existingSettingsPath, traceID)
}

func installGuardSessionStartHookAt(command []string, mode string, managed bool, fakBin, dir, existingSettingsPath, traceID string) ([]string, guardSessionStartInstall, error) {
	normalized := normalizeGuardSessionStartMode(mode)
	install := guardSessionStartInstall{Mode: normalized}
	if normalized == guardSessionStartModeOff {
		install.Reason = "disabled"
		return command, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
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
func guardSessionStartArgs(managed bool, traceID string) []string {
	args := []string{"guard-sessionstart"}
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
			Args:    guardSessionStartArgs(managed, traceID),
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
	settings, err := readGuardHookSettings(path)
	if err != nil {
		return err
	}
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	settings.Hooks["SessionStart"] = guardSessionStartMatchers(fakBin, managed, traceID)
	return writeGuardHookSettings(path, settings)
}
