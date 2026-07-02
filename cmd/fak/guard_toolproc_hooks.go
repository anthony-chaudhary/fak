package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Seam-4 auto-install (the tool process table, docs/notes/CONCEPT-TOOL-PROCESS-
// TABLE-2026-07-02.md): a default `fak guard -- claude` session gets the
// tool-process observation hooks with no manual setup. Three hook events map
// onto the toolproc journal vocabulary:
//
//	PreToolUse  -> `fak toolproc hook pre`   (spawn: the call is now a process)
//	PostToolUse -> `fak toolproc hook post`  (exit: the call completed)
//	SessionEnd  -> `fak toolproc hook stop`  (session_end: the orphan boundary)
//
// SessionEnd — NOT Stop — carries the session boundary: Stop fires at every
// TURN end, and a background tool legitimately outlives its turn; mapping Stop
// to session_end would false-orphan every long-runner each turn. On a harness
// without SessionEnd the orphan axis simply stays dark (fail-open observation).
//
// The hook adapter itself is fail-open by design (`fak toolproc hook` always
// exits 0), so this install can never wedge the harness: worst case the
// journal is missing rows and `fak toolproc ps` reads an honest subset.
const (
	guardToolprocModeObserve = "observe"

	guardToolprocEnvMode = "FAK_GUARD_TOOLPROC_HOOKS"

	// guardToolprocJournalRel is the workspace-relative journal the hooks
	// append to — the same default `fak toolproc hook` uses, made absolute at
	// install so the hook is cwd-independent. One shared journal per
	// workspace: events carry the harness session id, so concurrent guarded
	// sessions interleave without colliding and one `fak toolproc ps` reads
	// the whole host's tool-process table.
	guardToolprocJournalRel = ".fak/toolproc/journal.jsonl"
)

type guardToolprocInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	JournalPath  string
	Reason       string
}

// installGuardToolprocHooks wires the three observation hooks into the child's
// --settings file, merging into the file the PreCompact/Stop installers
// already wrote (existingSettingsPath non-empty) so a single --settings
// carries every guard hook; otherwise it writes its own file and injects
// --settings. Off mode or a non-claude child is a no-op.
func installGuardToolprocHooks(command []string, mode, existingSettingsPath string) ([]string, [][2]string, guardToolprocInstall, error) {
	normalized, err := normalizeGuardToolprocMode(mode)
	if err != nil {
		return command, nil, guardToolprocInstall{}, err
	}
	install := guardToolprocInstall{Mode: normalized}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		dir, err = os.MkdirTemp("", "fak-guard-toolproc-*")
		if err != nil {
			return command, nil, guardToolprocInstall{}, err
		}
	}
	journal, err := filepath.Abs(guardToolprocJournalRel)
	if err != nil {
		return command, nil, install, err
	}
	return installGuardToolprocHooksAt(command, mode, existingSettingsPath, fakBin, dir, journal)
}

func installGuardToolprocHooksAt(command []string, mode, existingSettingsPath, fakBin, dir, journalPath string) ([]string, [][2]string, guardToolprocInstall, error) {
	normalized, err := normalizeGuardToolprocMode(mode)
	if err != nil {
		return command, nil, guardToolprocInstall{}, err
	}
	install := guardToolprocInstall{Mode: normalized, JournalPath: journalPath}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	if strings.TrimSpace(journalPath) == "" {
		return command, nil, install, errors.New("empty toolproc hook journal path")
	}

	var settingsPath string
	if strings.TrimSpace(existingSettingsPath) != "" {
		// Merge into the file the PreCompact/Stop installers wrote + injected.
		if err := mergeGuardToolprocIntoSettings(existingSettingsPath, fakBin, journalPath); err != nil {
			return command, nil, install, err
		}
		settingsPath = existingSettingsPath
		// command already carries --settings; do NOT inject it again.
	} else {
		if strings.TrimSpace(dir) == "" {
			return command, nil, install, errors.New("empty toolproc hook settings directory")
		}
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return command, nil, install, err
		}
		settingsPath = filepath.Join(dir, "claude-toolproc-settings.json")
		settings := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{}}
		guardToolprocSetHooks(&settings, fakBin, journalPath)
		if err := writeGuardHookSettings(settingsPath, settings); err != nil {
			return command, nil, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	install.Applied = true
	install.SettingsPath = settingsPath
	env := [][2]string{{guardToolprocEnvMode, normalized}}
	return command, env, install, nil
}

// guardToolprocSetHooks writes the three event entries onto settings,
// replacing any prior toolproc entries (idempotent re-install) and leaving
// every other event's hooks untouched.
func guardToolprocSetHooks(settings *guardPreCompactClaudeSettings, fakBin, journalPath string) {
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	for event, kind := range map[string]string{
		"PreToolUse":  "pre",
		"PostToolUse": "post",
		"SessionEnd":  "stop",
	} {
		settings.Hooks[event] = []guardPreCompactClaudeMatcher{{
			Hooks: []guardPreCompactClaudeCommand{{
				Type:    "command",
				Command: guardPreCompactHookCommand(fakBin),
				Args:    []string{"toolproc", "hook", kind, "--journal", journalPath},
			}},
		}}
	}
}

// mergeGuardToolprocIntoSettings adds (or replaces) the three toolproc hook
// events in an existing guard settings file, preserving every other key (the
// PreCompact and Stop hooks), so a single --settings carries all of them.
func mergeGuardToolprocIntoSettings(path, fakBin, journalPath string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	guardToolprocSetHooks(&settings, fakBin, journalPath)
	return writeGuardHookSettings(path, settings)
}

func writeGuardHookSettings(path string, settings guardPreCompactClaudeSettings) error {
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func normalizeGuardToolprocMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardToolprocModeObserve:
		return guardToolprocModeObserve, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	default:
		return "", fmt.Errorf("invalid --toolproc-hooks mode %q (want off or observe)", mode)
	}
}
