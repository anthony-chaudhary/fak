package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
)

const geminiSystemSettingsEnv = "GEMINI_CLI_SYSTEM_SETTINGS_PATH"

type geminiHookCommand struct {
	Name    string `json:"name,omitempty"`
	Type    string `json:"type"`
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"`
}

type geminiHookGroup struct {
	Matcher string              `json:"matcher,omitempty"`
	Hooks   []geminiHookCommand `json:"hooks"`
}

// geminiClearHookGroup adapts Gemini's native clear lifecycle event to
// the provider-neutral guard-sessionstart actuator. Gemini's /new command is an
// alias of /clear and emits the same source=clear event with a fresh session id.
func geminiClearHookGroup(fakBin string, managed bool) geminiHookGroup {
	argv := append([]string{fakBin}, guardSessionStartArgs(managed, "", "gemini")...)
	return geminiHookGroup{
		Matcher: "clear",
		Hooks: []geminiHookCommand{{
			Name:    "fak-provider-clear",
			Type:    "command",
			Command: geminiCommandLine(argv, runtime.GOOS),
			Timeout: 5000,
		}},
	}
}

func geminiCommandLine(argv []string, goos string) string {
	if goos == "windows" {
		return "& " + windowsCommandLine(argv)
	}
	quoted := make([]string, 0, len(argv))
	for _, arg := range argv {
		quoted = append(quoted, "'"+strings.ReplaceAll(arg, "'", `'"'"'`)+"'")
	}
	return strings.Join(quoted, " ")
}

func geminiSettingsSource(getenv func(string) string, goos string) string {
	if path := strings.TrimSpace(getenv(geminiSystemSettingsEnv)); path != "" {
		return path
	}
	switch goos {
	case "windows":
		root := strings.TrimRight(strings.TrimSpace(getenv("ProgramData")), `\/`)
		if root == "" {
			root = `C:\ProgramData`
		}
		return root + `\gemini-cli\settings.json`
	case "darwin":
		return "/Library/Application Support/GeminiCli/settings.json"
	default:
		return "/etc/gemini-cli/settings.json"
	}
}

// writeGeminiSettingsOverlay writes a launch-only overlay containing fak's hook
// groups while preserving any existing system settings. The source is read-only.
func writeGeminiSettingsOverlay(path, sourcePath string, additions map[string][]geminiHookGroup) error {
	root := map[string]any{}
	if strings.TrimSpace(sourcePath) != "" {
		raw, err := os.ReadFile(sourcePath)
		switch {
		case err == nil:
			if err := json.Unmarshal(stripGeminiJSONComments(raw), &root); err != nil {
				return fmt.Errorf("parse Gemini system settings %s: %w", sourcePath, err)
			}
			if root == nil {
				return fmt.Errorf("parse Gemini system settings %s: root is not an object", sourcePath)
			}
		case !os.IsNotExist(err):
			return fmt.Errorf("read Gemini system settings %s: %w", sourcePath, err)
		}
	}
	events, ok := root["hooks"].(map[string]any)
	if root["hooks"] != nil && !ok {
		return fmt.Errorf("parse Gemini system settings %s: hooks is not an object", sourcePath)
	}
	if events == nil {
		events = map[string]any{}
		root["hooks"] = events
	}
	for event, groups := range additions {
		existing, ok := events[event].([]any)
		if events[event] != nil && !ok {
			return fmt.Errorf("parse Gemini system settings %s: hooks.%s is not an array", sourcePath, event)
		}
		for _, group := range groups {
			encoded, err := json.Marshal(group)
			if err != nil {
				return err
			}
			var value any
			if err := json.Unmarshal(encoded, &value); err != nil {
				return err
			}
			existing = append(existing, value)
		}
		events[event] = existing
	}
	data, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	return writeGuardSettingsFileAtomic(path, append(data, '\n'))
}

func stripGeminiJSONComments(raw []byte) []byte {
	out := append([]byte(nil), raw...)
	inString, escaped, lineComment, blockComment := false, false, false, false
	for i := 0; i < len(out); i++ {
		if lineComment {
			if out[i] == '\n' || out[i] == '\r' {
				lineComment = false
			} else {
				out[i] = ' '
			}
			continue
		}
		if blockComment {
			if out[i] == '*' && i+1 < len(out) && out[i+1] == '/' {
				out[i], out[i+1] = ' ', ' '
				i++
				blockComment = false
			} else if out[i] != '\n' && out[i] != '\r' {
				out[i] = ' '
			}
			continue
		}
		if inString {
			if escaped {
				escaped = false
			} else if out[i] == '\\' {
				escaped = true
			} else if out[i] == '"' {
				inString = false
			}
			continue
		}
		if out[i] == '"' {
			inString = true
			continue
		}
		if out[i] == '/' && i+1 < len(out) {
			switch out[i+1] {
			case '/':
				out[i], out[i+1] = ' ', ' '
				i++
				lineComment = true
			case '*':
				out[i], out[i+1] = ' ', ' '
				i++
				blockComment = true
			}
		}
	}
	return out
}
