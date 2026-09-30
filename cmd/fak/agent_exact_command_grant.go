package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

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
		Focused:              true,
		EnableSkills:         *af.skills,
		ExtraDirs:            extraDirs,
		ExactAllowedCommands: []string(af.allowBashCommands),
	})
}
