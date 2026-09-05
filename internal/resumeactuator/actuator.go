// Package resumeactuator renders harness continuation commands inside FAK's managed-agent envelope.
package resumeactuator

import (
	"errors"
	"fmt"
	"strings"
)

const (
	HarnessClaude    = "claude"
	HarnessCodex     = "codex"
	HarnessOpenCode  = "opencode"
	HarnessFak       = "fak"
	HarnessFakNative = "fak-native"
)

var (
	ErrUnknownAdapter    = errors.New("unsupported resume harness")
	ErrMissingCoordinate = errors.New("missing resume coordinate")
)

// Request contains the harness-neutral identity plus the small set of coordinates
// required by the currently supported continuation adapters.
type Request struct {
	Harness     string
	Session     string
	Rollout     string
	GoalFile    string
	ResultFile  string
	CWD         string
	Prompt      string
	ClaudeExe   string
	CodexExe    string
	OpenCodeExe string
	FakExe      string
}

// Harness normalizes the legacy empty value to Claude without guessing any other value.
func (r Request) HarnessName() (string, error) {
	h := strings.ToLower(strings.TrimSpace(r.Harness))
	if h == "" {
		return HarnessClaude, nil
	}
	switch h {
	case HarnessClaude, HarnessCodex, HarnessOpenCode:
		return h, nil
	case HarnessFak, HarnessFakNative:
		return HarnessFak, nil
	default:
		return "", fmt.Errorf("%w: %s", ErrUnknownAdapter, h)
	}
}

// ContinuationArgv renders only the harness-specific continuation command. FAK's
// management envelope is deliberately added separately by ManagedArgv.
func (r Request) ContinuationArgv(fakExe string) ([]string, error) {
	h, err := r.HarnessName()
	if err != nil {
		return nil, err
	}
	switch h {
	case HarnessClaude:
		if strings.TrimSpace(r.Session) == "" {
			return nil, fmt.Errorf("%w: claude session", ErrMissingCoordinate)
		}
		exe := strings.TrimSpace(r.ClaudeExe)
		if exe == "" {
			exe = "claude"
		}
		return []string{exe, "--resume", r.Session, "-p", r.Prompt, "--dangerously-skip-permissions"}, nil
	case HarnessCodex:
		if strings.TrimSpace(r.Session) == "" {
			return nil, fmt.Errorf("%w: codex session", ErrMissingCoordinate)
		}
		if strings.TrimSpace(r.Rollout) != "" || strings.TrimSpace(r.GoalFile) != "" || strings.TrimSpace(r.ResultFile) != "" {
			coordinates := []struct{ name, value string }{
				{"rollout", r.Rollout}, {"goal_file", r.GoalFile}, {"result_file", r.ResultFile},
			}
			for _, coordinate := range coordinates {
				if strings.TrimSpace(coordinate.value) == "" {
					return nil, fmt.Errorf("%w: codex %s", ErrMissingCoordinate, coordinate.name)
				}
			}
			if strings.TrimSpace(fakExe) == "" {
				return nil, fmt.Errorf("%w: codex fak executable", ErrMissingCoordinate)
			}
			return []string{fakExe, "codex-resume", "--json", "--rollout", r.Rollout, "--cwd", r.CWD, "--prompt-file", r.GoalFile, "--result-file", r.ResultFile, r.Session}, nil
		}
		exe := strings.TrimSpace(r.CodexExe)
		if exe == "" {
			exe = "codex"
		}
		argv := []string{exe, "exec", "resume", "--json", "--dangerously-bypass-approvals-and-sandbox", r.Session}
		if strings.TrimSpace(r.Prompt) != "" {
			argv = append(argv, r.Prompt)
		}
		return argv, nil
	case HarnessOpenCode:
		if strings.TrimSpace(r.Session) == "" {
			return nil, fmt.Errorf("%w: opencode session", ErrMissingCoordinate)
		}
		exe := strings.TrimSpace(r.OpenCodeExe)
		if exe == "" {
			exe = "opencode"
		}
		argv := []string{exe, "run", "--session", r.Session}
		if strings.TrimSpace(r.Prompt) != "" {
			argv = append(argv, r.Prompt)
		}
		return argv, nil
	case HarnessFak, HarnessFakNative:
		if strings.TrimSpace(r.Session) == "" {
			return nil, fmt.Errorf("%w: fak session", ErrMissingCoordinate)
		}
		exe := strings.TrimSpace(r.FakExe)
		if exe == "" {
			exe = strings.TrimSpace(fakExe)
		}
		if exe == "" {
			exe = "fak"
		}
		argv := []string{exe, "agent", "--native", "--resume", r.Session}
		if strings.TrimSpace(r.Prompt) != "" {
			argv = append(argv, "--task", r.Prompt)
		}
		return argv, nil
	}
	panic("unreachable")
}

// ManagedArgv makes fak manage the invariant outer process. Harness adapters only
// supply the continuation command after the separator.
func (r Request) ManagedArgv(fakExe string, postureArgs, budgetArgs []string) ([]string, error) {
	if strings.TrimSpace(fakExe) == "" {
		return nil, fmt.Errorf("%w: fak management executable", ErrMissingCoordinate)
	}
	child, err := r.ContinuationArgv(fakExe)
	if err != nil {
		return nil, err
	}
	argv := []string{fakExe, "m"}
	// The resume watchdog has already classified this exact session as dead and
	// eligible. Disable m's current-thread gate for Codex continuations: the
	// watchdog itself runs from a live operator/session context, so applying that
	// unrelated context here refuses every detached recovery as "unguarded".
	if h, _ := r.HarnessName(); h == HarnessCodex {
		argv = append(argv, "--codex-loop-gate", "off")
	}
	argv = append(argv, postureArgs...)
	argv = append(argv, budgetArgs...)
	argv = append(argv, "--")
	return append(argv, child...), nil
}
