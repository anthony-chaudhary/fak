package toolcatalog

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// CommandResult is the stable host-side process witness. Stdout is decoded JSON;
// stderr and exit status remain audit data and never masquerade as a tool result.
type CommandResult struct {
	JSON     json.RawMessage `json:"json,omitempty"`
	Stderr   string          `json:"stderr,omitempty"`
	ExitCode int             `json:"exit_code"`
}

// RunCommand executes one explicit command adapter without a shell. Field values
// become individual argv entries, stdin bytes, or exact environment entries.
func RunCommand(ctx context.Context, registration Registration, input json.RawMessage) (CommandResult, error) {
	return RunCommandWithEnv(ctx, registration, input, nil)
}

// RunCommandWithEnv accepts an explicit base environment. A nil base inherits
// the host environment; tests and sandboxed callers can pass an empty slice.
func RunCommandWithEnv(ctx context.Context, registration Registration, input json.RawMessage, baseEnv []string) (CommandResult, error) {
	if err := ValidateRegistration(registration); err != nil {
		return CommandResult{}, err
	}
	adapter := registration.Program.Executor.Adapter
	if adapter == nil {
		return CommandResult{}, fmt.Errorf("TOOL_COMMAND_ADAPTER_UNDECLARED: %s", registration.Program.Name)
	}
	var values map[string]json.RawMessage
	decoder := json.NewDecoder(bytes.NewReader(input))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&values); err != nil {
		return CommandResult{}, fmt.Errorf("TOOL_COMMAND_INPUT: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return CommandResult{}, fmt.Errorf("TOOL_COMMAND_INPUT: trailing value")
	}
	value := func(field string) (string, error) {
		raw, ok := values[field]
		if !ok {
			return "", fmt.Errorf("TOOL_COMMAND_FIELD_MISSING: %s", field)
		}
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return "", fmt.Errorf("TOOL_COMMAND_FIELD_TYPE: %s must be string", field)
		}
		if strings.IndexByte(text, 0) >= 0 {
			return "", fmt.Errorf("TOOL_COMMAND_FIELD_NUL: %s", field)
		}
		return text, nil
	}

	argv := append([]string(nil), registration.Program.Executor.Argv...)
	for _, binding := range adapter.Argv {
		if binding.Literal != "" {
			argv = append(argv, binding.Literal)
			continue
		}
		text, err := value(binding.Field)
		if err != nil {
			if binding.OmitEmpty {
				continue
			}
			return CommandResult{}, err
		}
		if text == "" && binding.OmitEmpty {
			continue
		}
		if binding.Flag != "" {
			argv = append(argv, binding.Flag)
		}
		argv = append(argv, text)
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = registration.Program.Executor.Dir
	if adapter.Stdin != nil {
		text, err := value(adapter.Stdin.Field)
		if err != nil {
			return CommandResult{}, err
		}
		cmd.Stdin = strings.NewReader(text)
	}
	if baseEnv == nil {
		cmd.Env = os.Environ()
	} else {
		cmd.Env = append([]string(nil), baseEnv...)
	}
	envNames := make([]string, 0, len(adapter.Env))
	for name := range adapter.Env {
		envNames = append(envNames, name)
	}
	sort.Strings(envNames)
	for _, name := range envNames {
		text, err := value(adapter.Env[name])
		if err != nil {
			return CommandResult{}, err
		}
		cmd.Env = append(cmd.Env, name+"="+text)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	result := CommandResult{Stderr: stderr.String()}
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exit.ExitCode()
			return result, fmt.Errorf("TOOL_COMMAND_EXIT: %s exited %d: %s", registration.Program.Name, result.ExitCode, strings.TrimSpace(result.Stderr))
		}
		return result, fmt.Errorf("TOOL_COMMAND_START: %w", err)
	}
	result.JSON = json.RawMessage(bytes.TrimSpace(stdout.Bytes()))
	if !json.Valid(result.JSON) {
		return result, fmt.Errorf("TOOL_COMMAND_RESULT_JSON: %s", registration.Program.Name)
	}
	return result, nil
}
