// Package devhandoff defines the compatibility boundary between runtime fak and
// the separately linked fak-dev executable.
package devhandoff

import "sort"

// Command is a repository-development command implemented by fak-dev that
// runtime fak recognizes only to provide an actionable process handoff.
type Command struct {
	Name string
}

// Commands is the moved command inventory shared by both executables. Keep this
// list limited to commands fak-dev actually dispatches; runtime help and refusal
// behavior derive from it.
var Commands = []Command{
	{Name: "amd-gpu-facts"},
	{Name: "backend"},
	{Name: "blast"},
	{Name: "boundary"},
	{Name: "buildcheck"},
	{Name: "catchup"},
	{Name: "checkpoint"},
	{Name: "ci-preflight"},
	{Name: "codex-hook-census"},
	{Name: "codex-hook-gate"},
	{Name: "codex-memory"},
	{Name: "codex-tool-errors"},
	{Name: "commit-subject-coverage"},
	{Name: "feature"},
	{Name: "fleetcap"},
	{Name: "devindex"},
	{Name: "index"},
	{Name: "issue"},
	{Name: "issue-contract-repair"},
	{Name: "orient"},
	{Name: "plan-audit"},
	{Name: "project"},
	{Name: "readme-visual-audit"},
	{Name: "refactor-verify"},
	{Name: "sessiondiag"},
	{Name: "tool-coverage-audit"},
	{Name: "whats-changed"},
	{Name: "wiki"},
	{Name: "windows-setup"},
	{Name: "workflow-audit"},
}

var commandSet = func() map[string]struct{} {
	set := make(map[string]struct{}, len(Commands))
	for _, command := range Commands {
		set[command.Name] = struct{}{}
	}
	return set
}()

// IsCommand reports whether name is implemented by the separate fak-dev binary.
func IsCommand(name string) bool {
	_, ok := commandSet[name]
	return ok
}

// Names returns the moved command names in stable lexical order.
func Names() []string {
	names := make([]string, 0, len(Commands))
	for _, command := range Commands {
		names = append(names, command.Name)
	}
	sort.Strings(names)
	return names
}
