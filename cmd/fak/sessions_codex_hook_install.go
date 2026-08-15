package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	codexRawRecoveryEnv     = "FAK_CODEX_RAW_RECOVERY"
	codexRawRecoveryValue   = "break-glass"
	codexRawRecoveryWarning = "fak codex guard: BREAK-GLASS raw Codex recovery active; fak capability floor and guard audit are not running"

	// Recovery is evaluated by the shell before invoking fak, so it remains usable
	// when fak is missing, stale, or broken. The exact value avoids accidental bypass.
	codexContinuationHookCommand        = "if [ \"${" + codexRawRecoveryEnv + ":-}\" = \"" + codexRawRecoveryValue + "\" ]; then echo '" + codexRawRecoveryWarning + "' >&2; else fak sessions codex-loop-hook 2>/dev/null || true; fi"
	codexContinuationHookCommandWindows = "if /i \"%" + codexRawRecoveryEnv + "%\"==\"" + codexRawRecoveryValue + "\" (echo " + codexRawRecoveryWarning + " 1>&2) else (fak sessions codex-loop-hook 2>nul || exit /b 0)"
)

func sessionsCodexHookInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessions codex-hook-install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	dryRun := fs.Bool("dry-run", false, "print the projected manifest without writing it")
	fs.Usage = func() { fmt.Fprintln(stderr, "usage: fak sessions codex-hook-install [--codex-home DIR] [--dry-run]") }
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	home, err := resolvedCodexLoopHome(*codexHome)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-hook-install: %v\n", err)
		return 1
	}
	path := filepath.Join(home, "hooks.json")
	manifest := map[string]any{}
	if raw, readErr := os.ReadFile(path); readErr == nil {
		if len(bytes.TrimSpace(raw)) > 0 {
			if err := json.Unmarshal(raw, &manifest); err != nil {
				fmt.Fprintf(stderr, "fak sessions codex-hook-install: decode %s: %v\n", path, err)
				return 1
			}
		}
	} else if !os.IsNotExist(readErr) {
		fmt.Fprintf(stderr, "fak sessions codex-hook-install: read %s: %v\n", path, readErr)
		return 1
	}
	installCodexContinuationHook(manifest)
	projected, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-hook-install: encode manifest: %v\n", err)
		return 1
	}
	projected = append(projected, '\n')
	if *dryRun {
		_, _ = stdout.Write(projected)
		return 0
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-hook-install: create %s: %v\n", home, err)
		return 1
	}
	if current, err := os.ReadFile(path); err == nil && bytes.Equal(current, projected) {
		fmt.Fprintf(stdout, "codex continuation hook already installed: %s\n", path)
		return 0
	}
	if err := os.WriteFile(path, projected, 0o600); err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-hook-install: write %s: %v\n", path, err)
		return 1
	}
	fmt.Fprintf(stdout, "installed codex continuation hook: %s\n", path)
	return 0
}

func installCodexContinuationHook(manifest map[string]any) {
	hooks, _ := manifest["hooks"].(map[string]any)
	if hooks == nil {
		hooks = map[string]any{}
		manifest["hooks"] = hooks
	}
	entries, _ := hooks["UserPromptSubmit"].([]any)
	filtered := make([]any, 0, len(entries)+1)
	for _, entry := range entries {
		if !containsCodexContinuationHook(entry) {
			filtered = append(filtered, entry)
		}
	}
	filtered = append(filtered, map[string]any{"hooks": []any{map[string]any{
		"type": "command", "command": codexContinuationHookCommand,
		"commandWindows": codexContinuationHookCommandWindows, "timeout": float64(30),
	}}})
	hooks["UserPromptSubmit"] = filtered
}

func containsCodexContinuationHook(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if (key == "command" || key == "commandWindows") && strings.Contains(fmt.Sprint(child), "fak sessions codex-loop-hook") {
				return true
			}
			if containsCodexContinuationHook(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if containsCodexContinuationHook(child) {
				return true
			}
		}
	}
	return false
}
