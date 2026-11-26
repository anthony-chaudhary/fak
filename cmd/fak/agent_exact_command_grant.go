package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	agentBashDefaultOpen = "default_open"
	agentBashFocused     = "focused"
)

func (af *agentFlags) setBashPolicy(policy string) error {
	switch policy {
	case "auto", agentBashDefaultOpen, agentBashFocused:
		af.bashPolicy = policy
		return nil
	default:
		return fmt.Errorf("--bash-policy %q must be auto, default_open, or focused", policy)
	}
}

func (af *agentFlags) constrainBashPolicy(governedOnly bool) error {
	if governedOnly {
		return nil
	}
	if af.bashPolicy == agentBashDefaultOpen {
		return fmt.Errorf("--bash-policy=default_open requires a native or interactive agent run")
	}
	af.bashPolicy = agentBashFocused
	return nil
}

// exactBashCommandList preserves each operator-authored command byte for byte. The
// code-tool engine compares grants exactly; trimming here would make the CLI claim a
// different command than the one it actually admits.
type exactBashCommandList []string

func (l *exactBashCommandList) String() string {
	return strings.Join(*l, ",")
}

func (l *exactBashCommandList) Set(command string) error {
	if strings.TrimSpace(command) == "" {
		return fmt.Errorf("--allow-bash-command requires a non-empty command")
	}
	*l = append(*l, command)
	return nil
}

func validateAgentBashCommandGrants(codeTools bool, grants []string) error {
	if len(grants) > 0 && !codeTools {
		return fmt.Errorf("--allow-bash-command requires --code-tools=true")
	}
	return nil
}

func armAgentCodeTools(root string, af *agentFlags) ([]agent.ToolDef, error) {
	var extraDirs []string
	if *af.skillsDir != "" {
		extraDirs = append(extraDirs, *af.skillsDir)
	}
	return agent.ArmCodeToolsWithOptions(agent.CodeToolsOptions{
		Root:                 root,
		Focused:              af.focusedBash(),
		EnableSkills:         *af.skills,
		ExtraDirs:            extraDirs,
		ExactAllowedCommands: []string(af.allowBashCommands),
		MaxCommandTime:       *af.bashCommandTimeout,
	})
}

func (af *agentFlags) focusedBash() bool {
	switch af.bashPolicy {
	case agentBashDefaultOpen:
		return false
	case agentBashFocused:
		return true
	default:
		return len(af.allowBashCommands) > 0 || *af.policyPath != "" ||
			agent.ConfiguredPosture() == adjudicator.PostureFailClosed
	}
}
