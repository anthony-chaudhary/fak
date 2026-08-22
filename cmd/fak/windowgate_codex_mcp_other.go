//go:build !windows

package main

import (
	"runtime"
	"time"
)

func captureCodexMCPWindowSelfcheck(string, string, time.Duration) (codexMCPWindowSelfcheckReport, error) {
	return codexMCPWindowSelfcheckReport{
		Schema:     codexMCPWindowSelfcheckSchema,
		OK:         true,
		Applicable: false,
		Platform:   runtime.GOOS,
		Reason:     "Codex MCP console-window capture is only applicable on Windows",
	}, nil
}
