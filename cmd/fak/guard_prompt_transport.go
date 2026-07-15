package main

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const guardWindowsPromptStdinThreshold = 7 << 10

// guardPromptStdinTransport moves a large Claude print prompt off the Windows
// command line. Claude's -p/--print mode reads the prompt from stdin when the
// flag has no value. This keeps every other argument byte-for-byte unchanged.
func guardPromptStdinTransport(command []string) ([]string, string, bool) {
	return guardPromptStdinTransportForOS(command, runtime.GOOS)
}

func guardPromptStdinTransportForOS(command []string, goos string) ([]string, string, bool) {
	if goos != "windows" || len(command) < 3 {
		return command, "", false
	}
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(command[0])), ".exe")
	if name != "claude" {
		return command, "", false
	}
	for i := 1; i+1 < len(command); i++ {
		if command[i] != "-p" && command[i] != "--print" {
			continue
		}
		prompt := command[i+1]
		if len(prompt) < guardWindowsPromptStdinThreshold {
			return command, "", false
		}
		out := make([]string, 0, len(command)-1)
		out = append(out, command[:i+1]...)
		out = append(out, command[i+2:]...)
		return out, prompt, true
	}
	return command, "", false
}

func applyGuardPromptStdinTransport(child *exec.Cmd, command []string, goos string) ([]string, bool) {
	command, prompt, moved := guardPromptStdinTransportForOS(command, goos)
	if moved {
		child.Stdin = strings.NewReader(prompt)
	}
	return command, moved
}
