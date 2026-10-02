package main

import (
	"fmt"
	"time"
)

const maxAgentBashCommandTimeout = 10 * time.Minute

func validateAgentBashCommandTimeout(codeTools bool, timeout time.Duration, explicit bool) error {
	if timeout <= 0 {
		return fmt.Errorf("--bash-command-timeout must be positive")
	}
	if timeout > maxAgentBashCommandTimeout {
		return fmt.Errorf("--bash-command-timeout must not exceed %s", maxAgentBashCommandTimeout)
	}
	if explicit && !codeTools {
		return fmt.Errorf("--bash-command-timeout requires --code-tools=true")
	}
	return nil
}
