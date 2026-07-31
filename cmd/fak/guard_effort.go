package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// guard_effort.go — the reasoning posture a guarded session gets BY DEFAULT, decided by
// ATTENDANCE rather than by one global settings.json value.
//
// The two postures are not interchangeable, and which one is right depends on who is
// waiting for the answer:
//
//   - ULTRACODE (xhigh per-message reasoning PLUS dynamic multi-agent workflow
//     orchestration) is the right default for a USER-INITIATED session. A human is at the
//     keyboard, the question is usually the hard one they could not answer themselves, and
//     the orchestration tax is paid in wall-clock the human already accepted by asking.
//   - XHIGH (the strongest per-message reasoning short of ultracode) is the right default
//     for an AGENT session — a headless `-p` fleet worker. Nobody is watching, the work is
//     usually a bounded leaf off a dispatch queue, and ultracode's fan-out buys rigor the
//     worker's own verify step already provides while multiplying its wall-clock and spend.
//
// Why this cannot be a settings.json value: `~/.claude/settings.json` is ONE file shared by
// every session on the host, so `"ultracode": true` there leaks the orchestration tax into
// every fleet worker, and `false` denies it to the human. Guard is the only seam that knows
// which kind of session it is about to launch (guardChildInteractive — the same `-p` signal
// the task-handoff and operator-directed gates already key on), so guard is where the
// posture is resolved.
//
// HOW each posture is emitted differs, and the difference is load-bearing:
//
//   - xhigh rides the child's own `--effort xhigh` flag. It is a plain repeatable-safe flag.
//   - ultracode has NO CLI flag; it is only a settings key, so it must be MERGED into the
//     one --settings file the guard hook installers already wrote. It must NOT be handed as
//     a second `--settings '{"ultracode":true}'`: Claude Code's --settings is LAST-WINS, not
//     merged (verified against the CLI: with two --settings, keys from the first are dropped
//     entirely), so a second occurrence would silently discard the guard's ENTIRE hook stack
//     — the deny-all auto-continue Stop hook, the task-handoff gate, the toolproc journal,
//     the SessionStart affordance, and the PreCompact coherence gate. Merging into the same
//     file keeps exactly one --settings on the argv and both halves alive.
const (
	// guardEffortModeAuto resolves by attendance: interactive ⇒ ultracode, headless ⇒ xhigh.
	guardEffortModeAuto = "auto"
	// guardEffortModeUltracode forces ultracode regardless of attendance.
	guardEffortModeUltracode = "ultracode"
	// guardEffortModeXHigh forces xhigh regardless of attendance.
	guardEffortModeXHigh = "xhigh"

	// guardEffortEnvMode lets an orchestrator pin the posture without editing an argv.
	// The flag wins when the operator set it explicitly.
	guardEffortEnvMode = "FAK_GUARD_EFFORT"

	// guardEffortLevelXHigh is the value handed to the child's --effort flag. It is one of
	// the levels the CLI actually admits (low|medium|high|xhigh|max) — "ultracode" is NOT
	// an --effort level, which is why it travels as a settings key instead.
	guardEffortLevelXHigh = "xhigh"
)

// guardEffortInstall records what the posture resolver did, for the launch readout and tests.
type guardEffortInstall struct {
	Applied      bool
	Mode         string // the normalized mode that was requested (auto|ultracode|xhigh|off)
	Posture      string // the RESOLVED posture actually emitted (ultracode|xhigh); "" when none
	SettingsPath string // the --settings file ultracode was merged into; "" for the xhigh path
	Reason       string // why nothing was applied (disabled|non-claude-child|child-pinned)
}

// normalizeGuardEffortMode folds the posture knob to its canonical token, failing LOUD on an
// unrecognized value rather than silently defaulting — the same discipline
// normalizeGuardToolprocMode and normalizeManagedCacheMode use. An empty knob is `auto`.
func normalizeGuardEffortMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardEffortModeAuto:
		return guardEffortModeAuto, nil
	case guardEffortModeUltracode:
		return guardEffortModeUltracode, nil
	case guardEffortModeXHigh:
		return guardEffortModeXHigh, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	default:
		return "", fmt.Errorf("invalid --effort mode %q (want auto, ultracode, xhigh, or off)", mode)
	}
}

// guardEffortEffectiveMode resolves the posture knob from the flag and the environment. An
// EXPLICITLY-set flag always wins; otherwise FAK_GUARD_EFFORT speaks; otherwise the flag's
// own default stands. Mirrors guardTaskHandoffEffectiveMode's flag-beats-env precedence.
func guardEffortEffectiveMode(flagMode string, flagSet bool, getenv func(string) string) string {
	if flagSet {
		return flagMode
	}
	if getenv != nil {
		if env := strings.TrimSpace(getenv(guardEffortEnvMode)); env != "" {
			return env
		}
	}
	return flagMode
}

// guardEffortPosture folds the mode and the attendance signal into the single posture to
// emit. This is the attendance table:
//
//	mode       child         posture    why
//	---------  ------------  ---------  --------------------------------------------------
//	ultracode  (any)         ultracode  operator forced it; an explicit mode always wins
//	xhigh      (any)         xhigh      operator forced it; an explicit mode always wins
//	off        (any)         (none)     no posture emitted; the seat's own default stands
//	auto       interactive   ultracode  a human is waiting: buy the orchestration rigor
//	auto       headless -p   xhigh      an unattended worker: strong reasoning, no fan-out tax
//
// An unrecognized mode returns no posture; callers normalize first and surface the error.
func guardEffortPosture(mode string, interactive bool) string {
	switch mode {
	case guardEffortModeUltracode:
		return guardEffortModeUltracode
	case guardEffortModeXHigh:
		return guardEffortModeXHigh
	case guardEffortModeAuto:
		if interactive {
			return guardEffortModeUltracode
		}
		return guardEffortModeXHigh
	default: // off, or anything unnormalized
		return ""
	}
}

// guardEffortChildPinned reports whether the child argv ALREADY carries its own reasoning
// posture, in which case guard emits nothing — an explicit per-launch choice outranks a
// default. Two forms count: the child's own `--effort <level>` flag, and an inline
// `--settings` JSON naming ultracode (what a pre-guard launcher or an operator hands
// directly). Only the guard-flag segment is excluded: this scans the CHILD argv, which is
// already the post-`--` tail by the time an installer sees it.
func guardEffortChildPinned(command []string) bool {
	for _, a := range command {
		name, val, hasEq := strings.Cut(a, "=")
		switch name {
		case "--effort":
			return true
		case "--settings":
			if hasEq && guardEffortSettingsNamesUltracode(val) {
				return true
			}
		default:
			// `--settings <json>`: the value is the NEXT token, handled below.
		}
	}
	for i := 0; i+1 < len(command); i++ {
		if command[i] == "--settings" && guardEffortSettingsNamesUltracode(command[i+1]) {
			return true
		}
	}
	return false
}

// guardEffortSettingsNamesUltracode reports whether a --settings value is INLINE JSON that
// names the ultracode key. A file path is not inline JSON and returns false (guard wrote
// those itself; treating one as a pin would make the installer self-suppress on re-entry).
func guardEffortSettingsNamesUltracode(v string) bool {
	s := strings.TrimSpace(v)
	if !strings.HasPrefix(s, "{") {
		return false
	}
	var probe map[string]any
	if err := json.Unmarshal([]byte(s), &probe); err != nil {
		return false
	}
	_, ok := probe["ultracode"]
	return ok
}

// installGuardEffortPosture emits the resolved default reasoning posture onto the child.
//
// xhigh appends `--effort xhigh` to the child argv. ultracode MERGES `"ultracode": true`
// into existingSettingsPath — the file the PreCompact/Stop/toolproc/SessionStart installers
// already wrote and already put on the argv — so the session carries exactly ONE --settings
// and the hook stack survives (see the last-wins note at the top of this file). With no such
// file (every hook installer off, or a non-guard-hooked child) it writes its own settings
// file and injects the single --settings itself.
//
// No-op, with a Reason, for: mode=off, a non-Claude child (--effort/--settings are
// Claude-specific, exactly as the hook installers gate), and a child that already pinned its
// own posture.
func installGuardEffortPosture(command []string, mode, existingSettingsPath string) ([]string, guardEffortInstall, error) {
	normalized, err := normalizeGuardEffortMode(mode)
	if err != nil {
		return command, guardEffortInstall{}, err
	}
	install := guardEffortInstall{Mode: normalized}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, install, nil
	}
	if guardEffortChildPinned(command) {
		install.Reason = "child-pinned"
		return command, install, nil
	}
	posture := guardEffortPosture(normalized, guardChildInteractive(command))
	if posture == "" {
		install.Reason = "disabled"
		return command, install, nil
	}
	install.Posture = posture
	if posture == guardEffortModeXHigh {
		install.Applied = true
		return append(append([]string{}, command...), "--effort", guardEffortLevelXHigh), install, nil
	}

	// ultracode: settings-key only, so merge rather than add a second --settings.
	if path := strings.TrimSpace(existingSettingsPath); path != "" {
		if err := mergeGuardUltracodeIntoSettings(path); err != nil {
			return command, install, err
		}
		install.Applied = true
		install.SettingsPath = path
		return command, install, nil // command already carries --settings; do NOT inject it again.
	}
	dir, err := guardSessionTempDir("effort")
	if err != nil {
		return command, install, err
	}
	return installGuardEffortPostureAt(command, normalized, dir)
}

// installGuardEffortPostureAt is the no-lookup half of installGuardEffortPosture for the
// case where guard must write its OWN settings file (no hook installer produced one). dir is
// injected so a test never touches the real session temp directory.
func installGuardEffortPostureAt(command []string, mode, dir string) ([]string, guardEffortInstall, error) {
	normalized, err := normalizeGuardEffortMode(mode)
	if err != nil {
		return command, guardEffortInstall{}, err
	}
	install := guardEffortInstall{Mode: normalized}
	posture := guardEffortPosture(normalized, guardChildInteractive(command))
	if posture != guardEffortModeUltracode {
		return command, install, errors.New("guard effort: own-settings path is for the ultracode posture only")
	}
	if strings.TrimSpace(dir) == "" {
		return command, install, errors.New("empty effort posture settings directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, install, err
	}
	path := filepath.Join(dir, "claude-effort-settings.json")
	on := true
	if err := writeGuardHookSettings(path, guardPreCompactClaudeSettings{Ultracode: &on}); err != nil {
		return command, install, err
	}
	install.Applied = true
	install.Posture = posture
	install.SettingsPath = path
	return appendClaudeSettingsArg(command, path), install, nil
}

// mergeGuardUltracodeIntoSettings sets `"ultracode": true` on an existing guard settings
// file, preserving every other key (the hooks the installers wrote), so one --settings
// carries both the hook stack and the posture.
func mergeGuardUltracodeIntoSettings(path string) error {
	settings, err := readGuardHookSettings(path)
	if err != nil {
		return err
	}
	on := true
	settings.Ultracode = &on
	return writeGuardHookSettings(path, settings)
}

// guardEffortWord renders the resolved posture for the guard launch readout, so an operator
// can see WHICH default fired and WHY without re-deriving the attendance table.
func guardEffortWord(install guardEffortInstall, interactive bool) string {
	if !install.Applied {
		reason := install.Reason
		if reason == "" {
			reason = "not applied"
		}
		return fmt.Sprintf("off (--effort=%s; %s)", install.Mode, reason)
	}
	attendance := "headless -p child"
	if interactive {
		attendance = "attended interactive child"
	}
	if install.Posture == guardEffortModeUltracode {
		return fmt.Sprintf("ultracode (--effort=%s; %s — xhigh reasoning + workflow orchestration, merged into %s)",
			install.Mode, attendance, filepath.Base(install.SettingsPath))
	}
	return fmt.Sprintf("xhigh (--effort=%s; %s — strong reasoning, no orchestration tax)", install.Mode, attendance)
}
