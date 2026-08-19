package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/worktype"
)

const dispatchWorktypeSidecarSuffix = ".worktype.json"

func writeDispatchWorktypeSidecar(logPath, prompt string) string {
	if strings.TrimSpace(logPath) == "" {
		return ""
	}
	stem := strings.TrimSuffix(logPath, filepath.Ext(logPath))
	sessionID := filepath.Base(stem)
	row := worktype.ClassifyDispatchPrompt(sessionID, prompt)
	b, err := json.Marshal(row)
	if err != nil {
		return ""
	}
	b = append(b, '\n')
	path := stem + dispatchWorktypeSidecarSuffix
	if err := os.WriteFile(path, b, 0o600); err != nil {
		return ""
	}
	return path
}
