package main

import (
	"fmt"
	"strings"
	"time"
)

// serveExactBashCommandList preserves operator-authored command bytes. Whitespace
// is significant to the code-tool engine, except that an all-blank grant is invalid.
type serveExactBashCommandList []string

func (l *serveExactBashCommandList) String() string {
	return strings.Join(*l, ",")
}

func (l *serveExactBashCommandList) Set(command string) error {
	if strings.TrimSpace(command) == "" {
		return fmt.Errorf("--native-allow-bash-command requires a non-empty command")
	}
	*l = append(*l, command)
	return nil
}

// validateServeNativeBashCommandGrants rejects inert grants before model loading
// or listener startup. The flag parser already rejects blank command values.
func validateServeNativeBashCommandGrants(native, codeTools bool, grants []string) error {
	if len(grants) == 0 {
		return nil
	}
	if !native {
		return fmt.Errorf("--native-allow-bash-command requires --native")
	}
	if !codeTools {
		return fmt.Errorf("--native-allow-bash-command requires --native-code-tools=true")
	}
	return nil
}

func validateServeNativeBashCommandTimeout(native, codeTools bool, timeout time.Duration, explicit bool) error {
	if timeout <= 0 {
		return fmt.Errorf("--native-bash-command-timeout must be positive")
	}
	if timeout > maxAgentBashCommandTimeout {
		return fmt.Errorf("--native-bash-command-timeout must not exceed %s", maxAgentBashCommandTimeout)
	}
	if !explicit {
		return nil
	}
	if !native {
		return fmt.Errorf("--native-bash-command-timeout requires --native")
	}
	if !codeTools {
		return fmt.Errorf("--native-bash-command-timeout requires --native-code-tools=true")
	}
	return nil
}
