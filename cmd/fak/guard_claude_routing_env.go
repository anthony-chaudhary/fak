package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Claude Code applies the `env` block of its settings files on top of the process
// environment it was started with. A base URL pinned in the operator's user or project
// settings therefore beats the gateway URL guard put in the child's environment, and the
// child talks straight past its own guard: nothing it proposes is adjudicated and the
// launch-time lane never reaches a served tool call. A --settings file outranks those
// layers, so the routing variable is restated there.
const (
	guardClaudeSettingsEnvKey      = "env"
	guardClaudeRoutingSettingsFile = "claude-routing-settings.json"
)

// guardClaudeSettingsRoutingEnvNames is the closed set of injected variables restated in
// the settings file. Only the endpoint is pinned: model and credential names stay the
// operator's to choose.
var guardClaudeSettingsRoutingEnvNames = map[string]bool{
	"ANTHROPIC_BASE_URL": true,
}

func guardClaudeSettingsRoutingEnv(injected [][2]string) map[string]string {
	out := map[string]string{}
	for _, kv := range injected {
		name := strings.TrimSpace(kv[0])
		if guardClaudeSettingsRoutingEnvNames[name] && strings.TrimSpace(kv[1]) != "" {
			out[name] = kv[1]
		}
	}
	return out
}

// installGuardClaudeRoutingEnv restates guard's injected endpoint in the Claude child's
// --settings file, merging into the file the hook installers already wrote when there is
// one. It must run after them: it is the last writer of the `env` key.
func installGuardClaudeRoutingEnv(command []string, injected [][2]string, existingSettingsPath string) ([]string, string, error) {
	if !guardPreCompactIsClaudeCommand(command) || len(guardClaudeSettingsRoutingEnv(injected)) == 0 {
		return command, "", nil
	}
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		var err error
		dir, err = guardSessionTempDir("routingenv")
		if err != nil {
			return command, "", err
		}
	}
	return installGuardClaudeRoutingEnvAt(command, injected, existingSettingsPath, dir)
}

func installGuardClaudeRoutingEnvAt(command []string, injected [][2]string, existingSettingsPath, dir string) ([]string, string, error) {
	routing := guardClaudeSettingsRoutingEnv(injected)
	if !guardPreCompactIsClaudeCommand(command) || len(routing) == 0 {
		return command, "", nil
	}
	settingsPath := strings.TrimSpace(existingSettingsPath)
	if settingsPath != "" {
		if err := mergeGuardClaudeRoutingEnvIntoSettings(settingsPath, routing); err != nil {
			return command, "", err
		}
		return command, settingsPath, nil
	}
	if strings.TrimSpace(dir) == "" {
		return command, "", errors.New("empty Claude routing settings directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, "", err
	}
	settingsPath = filepath.Join(dir, guardClaudeRoutingSettingsFile)
	empty := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{}}
	if err := writeGuardHookSettings(settingsPath, empty); err != nil {
		return command, "", err
	}
	// Fold any --settings the caller put on the child argv first, so a caller-supplied
	// env block is merged under guard's endpoint instead of replacing it.
	command = appendClaudeSettingsArg(command, settingsPath)
	if err := mergeGuardClaudeRoutingEnvIntoSettings(settingsPath, routing); err != nil {
		return command, "", err
	}
	return command, settingsPath, nil
}

// mergeGuardClaudeRoutingEnvIntoSettings sets the routing variables inside the settings
// file's `env` object, keeping every other variable and every other top-level key.
func mergeGuardClaudeRoutingEnvIntoSettings(path string, routing map[string]string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	env := map[string]json.RawMessage{}
	if existing, ok := settings.Extra[guardClaudeSettingsEnvKey]; ok && string(existing) != "null" {
		if err := json.Unmarshal(existing, &env); err != nil {
			return fmt.Errorf("parse %q in hook settings %s: %w", guardClaudeSettingsEnvKey, path, err)
		}
	}
	for name, value := range routing {
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		env[name] = encoded
	}
	encodedEnv, err := json.Marshal(env)
	if err != nil {
		return err
	}
	if settings.Extra == nil {
		settings.Extra = map[string]json.RawMessage{}
	}
	settings.Extra[guardClaudeSettingsEnvKey] = encodedEnv
	return writeGuardHookSettings(path, settings)
}
