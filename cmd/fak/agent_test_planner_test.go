package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func cmdAgentWithMock(argv []string) {
	needsFixtureTask := true
	for i, arg := range argv {
		if arg == "resume" || arg == "--resume" || strings.HasPrefix(arg, "--resume=") ||
			arg == "--workflow" || strings.HasPrefix(arg, "--workflow=") ||
			arg == "--task" || strings.HasPrefix(arg, "--task=") {
			needsFixtureTask = false
			break
		}
		if i > 0 && argv[i-1] == "--task" {
			needsFixtureTask = false
			break
		}
	}
	if needsFixtureTask {
		argv = append([]string{"--task", agent.DefaultTask}, argv...)
	}
	runAgentWithPlanner(argv, agent.NewMockPlanner("test"))
}
