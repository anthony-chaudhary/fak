package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unicode/utf16"
)

const guardCodexSessionStartTimeoutSeconds = 5

var guardCodexProviderThreadID = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

type guardCodexSessionStartDecision struct {
	Boundary bool
	Bind     bool
	Source   string
	Trace    string
}

type guardCodexSessionStartBinding struct {
	ProviderSessionID string `json:"provider_session_id"`
	TraceID           string `json:"trace_id"`
}

// installGuardCodexSessionStartHookAt adds one trusted SessionFlags-layer handler. Codex
// discovers hooks per layer, so this launch-scoped array is additive to managed, user,
// project, and plugin handlers rather than replacing their arrays.
func installGuardCodexSessionStartHookAt(command []string, managed bool, fakBin, dir, traceID string) ([]string, guardSessionStartInstall, error) {
	install := guardSessionStartInstall{Mode: guardSessionStartModeOn, Managed: managed, Provider: "codex"}
	if err := guardCodexSessionStartConfigCollision(command); err != nil {
		return command, install, err
	}
	if strings.TrimSpace(dir) == "" {
		return command, install, fmt.Errorf("empty Codex SessionStart state directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, install, err
	}
	statePath := filepath.Join(dir, "codex-sessionstart-state")
	hookCommand := guardCodexSessionStartCommand(fakBin, managed, traceID, statePath)
	hookOverride := "hooks.SessionStart=[{hooks=[{type=\"command\",command=" + strconv.Quote(hookCommand) + ",timeout=" + strconv.Itoa(guardCodexSessionStartTimeoutSeconds) + "}]}]"
	stateOverride := "hooks.state={" + strconv.Quote(guardCodexSessionStartHookKey()) + "={trusted_hash=" + strconv.Quote(guardCodexSessionStartTrustedHash(hookCommand)) + "}}"
	args := []string{"-c", hookOverride, "-c", stateOverride}
	out := make([]string, 0, len(command)+len(args))
	out = append(out, command[0])
	out = append(out, args...)
	out = append(out, command[1:]...)
	install.Applied = true
	install.StatePath = statePath
	return out, install, nil
}

func guardCodexSessionStartCommand(fakBin string, managed bool, traceID, statePath string) string {
	argv := append([]string{guardPreCompactHookCommand(fakBin)}, guardSessionStartArgs(managed, traceID, "codex")...)
	argv = append(argv, "--state", statePath)
	if runtime.GOOS == "windows" {
		return guardCodexPowerShellCommand(argv)
	}
	quoted := make([]string, len(argv))
	for i, arg := range argv {
		quoted[i] = guardCodexShellQuote(arg)
	}
	return strings.Join(quoted, " ")
}

func guardCodexShellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

// Codex may run command hooks through cmd.exe or PowerShell on Windows. An encoded
// PowerShell command is valid from either outer shell and keeps paths and arguments opaque.
func guardCodexPowerShellCommand(argv []string) string {
	quoted := make([]string, len(argv))
	for i, arg := range argv {
		quoted[i] = "'" + strings.ReplaceAll(arg, "'", "''") + "'"
	}
	script := "& " + strings.Join(quoted, " ") + "; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE"
	units := utf16.Encode([]rune(script))
	raw := make([]byte, len(units)*2)
	for i, unit := range units {
		binary.LittleEndian.PutUint16(raw[i*2:], unit)
	}
	return "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + base64.StdEncoding.EncodeToString(raw)
}

// The hook trust key is Codex's synthetic SessionFlags source plus event/group/handler indexes.
func guardCodexSessionStartHookKey() string {
	if runtime.GOOS == "windows" {
		return `C:\<session-flags>\config.toml:session_start:0:0`
	}
	return `/<session-flags>/config.toml:session_start:0:0`
}

// guardCodexSessionStartTrustedHash mirrors Codex's normalized hook identity fingerprint.
// Trust is granted only to fak's exact injected handler; it does not bypass trust for sibling
// user, project, or plugin hooks.
func guardCodexSessionStartTrustedHash(command string) string {
	identity := map[string]any{
		"event_name": "session_start",
		"hooks": []any{map[string]any{
			"async":   false,
			"command": command,
			"timeout": guardCodexSessionStartTimeoutSeconds,
			"type":    "command",
		}},
	}
	data, _ := json.Marshal(identity)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum)
}

func guardCodexSessionStartConfigCollision(command []string) error {
	for i := 1; i < len(command); i++ {
		value := ""
		switch {
		case (command[i] == "-c" || command[i] == "--config") && i+1 < len(command):
			i++
			value = command[i]
		case strings.HasPrefix(command[i], "--config="):
			value = strings.TrimPrefix(command[i], "--config=")
		}
		key, _, ok := strings.Cut(value, "=")
		key = strings.TrimSpace(key)
		if ok && (key == "hooks" || key == "hooks.SessionStart" || key == "hooks.state") {
			return fmt.Errorf("Codex SessionStart adapter conflicts with per-launch config override %q", value)
		}
	}
	return nil
}

// classifyGuardCodexSessionStart distinguishes the first ordinary startup from /new. Codex
// emits source=startup for both, so the launch-scoped binding supplies the missing ordering.
func classifyGuardCodexSessionStart(statePath, source, sessionID string) (guardCodexSessionStartDecision, error) {
	source = strings.ToLower(strings.TrimSpace(source))
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return guardCodexSessionStartDecision{}, nil
	}
	raw, err := os.ReadFile(statePath)
	if err != nil && !os.IsNotExist(err) {
		return guardCodexSessionStartDecision{}, err
	}
	var bound guardCodexSessionStartBinding
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &bound); err != nil {
			return guardCodexSessionStartDecision{}, fmt.Errorf("decode Codex SessionStart binding: %w", err)
		}
	}
	if strings.TrimSpace(bound.ProviderSessionID) == sessionID {
		return guardCodexSessionStartDecision{Trace: strings.TrimSpace(bound.TraceID)}, nil
	}
	if source == "clear" {
		return guardCodexSessionStartDecision{Boundary: true, Bind: true, Source: "clear"}, nil
	}
	if source == "startup" && strings.TrimSpace(bound.ProviderSessionID) != "" {
		return guardCodexSessionStartDecision{Boundary: true, Bind: true, Source: "clear"}, nil
	}
	return guardCodexSessionStartDecision{Bind: true}, nil
}

func writeGuardCodexSessionBinding(statePath, sessionID, traceID string) error {
	statePath = strings.TrimSpace(statePath)
	sessionID = strings.TrimSpace(sessionID)
	if statePath == "" || sessionID == "" {
		return nil
	}
	data, err := json.Marshal(guardCodexSessionStartBinding{
		ProviderSessionID: sessionID,
		TraceID:           strings.TrimSpace(traceID),
	})
	if err != nil {
		return err
	}
	return writeGuardSettingsFileAtomic(statePath, append(data, '\n'))
}

// guardCodexResourceResumeCommand rebuilds the one safe interactive Codex continuation
// after guard-owned resource containment. The provider thread comes only from the exact,
// launch-scoped SessionStart binding: the guard trace is correlation metadata and is never a
// substitute thread ID. Only root-level config overrides survive; the original interactive
// prompt and any prior subcommand are deliberately not replayed.
func guardCodexResourceResumeCommand(command []string, statePath, guardTraceID string) ([]string, error) {
	if len(command) == 0 || strings.TrimSpace(command[0]) == "" {
		return nil, fmt.Errorf("Codex resource reattach has no executable command")
	}
	binding, err := readGuardCodexSessionBinding(statePath, guardTraceID)
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, len(command)+2)
	out = append(out, command[0])
	for i := 1; i < len(command); i++ {
		switch {
		case command[i] == "--":
			i = len(command)
		case command[i] == "-c" || command[i] == "--config":
			if i+1 >= len(command) {
				return nil, fmt.Errorf("Codex resource reattach found %s without a value", command[i])
			}
			out = append(out, command[i], command[i+1])
			i++
		case strings.HasPrefix(command[i], "--config="):
			out = append(out, command[i])
		}
	}
	out = append(out, "resume", binding.ProviderSessionID)
	return out, nil
}

func readGuardCodexSessionBinding(statePath, guardTraceID string) (guardCodexSessionStartBinding, error) {
	statePath = strings.TrimSpace(statePath)
	if statePath == "" {
		return guardCodexSessionStartBinding{}, fmt.Errorf("Codex SessionStart binding path is unavailable")
	}
	raw, err := os.ReadFile(statePath)
	if err != nil {
		return guardCodexSessionStartBinding{}, fmt.Errorf("read Codex SessionStart binding: %w", err)
	}
	var binding guardCodexSessionStartBinding
	if err := json.Unmarshal(raw, &binding); err != nil {
		return guardCodexSessionStartBinding{}, fmt.Errorf("decode Codex SessionStart binding: %w", err)
	}
	binding.ProviderSessionID = strings.TrimSpace(binding.ProviderSessionID)
	binding.TraceID = strings.TrimSpace(binding.TraceID)
	if binding.ProviderSessionID == "" {
		return guardCodexSessionStartBinding{}, fmt.Errorf("Codex SessionStart binding has no provider thread ID")
	}
	if !guardCodexProviderThreadID.MatchString(binding.ProviderSessionID) {
		return guardCodexSessionStartBinding{}, fmt.Errorf("Codex SessionStart binding has malformed provider thread ID")
	}
	guardTraceID = strings.TrimSpace(guardTraceID)
	if guardTraceID == "" || binding.TraceID != guardTraceID {
		return guardCodexSessionStartBinding{}, fmt.Errorf("Codex SessionStart binding does not belong to the current guard trace")
	}
	return binding, nil
}
