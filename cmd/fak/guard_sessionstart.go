package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
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
const guardSessionStartHint = "fak substrate available (MCP server `fak`): before working as a generic coder, reach for the fak verbs. Call `mcp__fak__fak_index_work` to pull ranked open work for this repo; `mcp__fak__fak_admit` / `mcp__fak__fak_adjudicate` to gate/execute a tool call through the kernel; `mcp__fak__fak_memory_run` for durable memory; `mcp__fak__fak_tools_search` to page in the rest. These are deferred tools — you must invoke them explicitly, they will not auto-load."

func cmdGuardSessionStart(argv []string) {
	os.Exit(runGuardSessionStart(os.Stdout, os.Stderr, argv))
}

// runGuardSessionStart is the SessionStart-hook actuator. On the "on" mode (default) it
// prints the additionalContext JSON envelope to stdout and returns 0; "off" returns 0 with
// no output. It fails OPEN — any bad args return 0 with no injection.
func runGuardSessionStart(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard-sessionstart", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardSessionStartEnvMode), "off|on")
	if err := fs.Parse(argv); err != nil {
		// Fail open: a discoverability hint must never wedge a session start.
		return 0
	}
	if normalizeGuardSessionStartMode(*modeFlag) == guardSessionStartModeOff {
		return 0
	}
	// Claude Code injects a SessionStart hook's hookSpecificOutput.additionalContext into the
	// first turn's context. Emit the envelope; a marshal failure is a silent no-op (fail open).
	envelope := map[string]any{
		"hookSpecificOutput": map[string]any{
			"hookEventName":     "SessionStart",
			"additionalContext": guardSessionStartHint,
		},
	}
	data, err := json.Marshal(envelope)
	if err != nil {
		return 0
	}
	fmt.Fprintln(stdout, string(data))
	return 0
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
	SettingsPath string
	Reason       string
}

// installGuardSessionStartHook installs the Claude Code SessionStart affordance hook,
// MERGING it into the shared --settings file the other guard hooks already wrote
// (existingSettingsPath) so a single --settings carries them all. Off mode or a non-claude
// child is a no-op. Mirrors installGuardStopHook.
func installGuardSessionStartHook(command []string, mode, existingSettingsPath string) ([]string, guardSessionStartInstall, error) {
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
		dir, err = os.MkdirTemp("", "fak-guard-sessionstart-*")
		if err != nil {
			return command, guardSessionStartInstall{}, err
		}
	}
	return installGuardSessionStartHookAt(command, mode, fakBin, dir, existingSettingsPath)
}

func installGuardSessionStartHookAt(command []string, mode, fakBin, dir, existingSettingsPath string) ([]string, guardSessionStartInstall, error) {
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
		if err := mergeGuardSessionStartIntoSettings(existingSettingsPath, fakBin); err != nil {
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
		if err := writeGuardSessionStartSettings(settingsPath, fakBin); err != nil {
			return command, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	install.Applied = true
	install.SettingsPath = settingsPath
	return command, install, nil
}

// guardSessionStartMatchers builds the SessionStart hook settings entry. The affordance is
// wanted on a fresh start AND on clear/compact/resume (a compacted context may have dropped
// the original hint), so the matcher is left empty to fire on every SessionStart source.
func guardSessionStartMatchers(fakBin string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-sessionstart"},
		}},
	}}
}

func writeGuardSessionStartSettings(path, fakBin string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"SessionStart": guardSessionStartMatchers(fakBin),
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

// mergeGuardSessionStartIntoSettings adds (or replaces) the SessionStart hook in an existing
// guard settings file, preserving every other key (PreCompact/Stop/toolproc hooks).
func mergeGuardSessionStartIntoSettings(path, fakBin string) error {
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
	settings.Hooks["SessionStart"] = guardSessionStartMatchers(fakBin)
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}
