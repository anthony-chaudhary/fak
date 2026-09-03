package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resumebackoff"
)

// rwClaudeExe resolves the claude binary from the fleet-wide convention: FLEET_CLAUDE_EXE,
// the FAK_CLAUDE_EXE back-compat fallback, PATH, then the conventional install path.
func rwClaudeExe(home string) string {
	if v := strings.TrimSpace(os.Getenv("FLEET_CLAUDE_EXE")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_EXE")); v != "" {
		return v
	}
	for _, name := range []string{"claude", "claude.exe"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return filepath.Join(home, ".local", "bin", "claude")
}

// rwPythonExe resolves a Python interpreter for the registry-refresh child, or "".
func rwPythonExe() string {
	for _, name := range []string{"python", "python3", "py"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

// rwAccountTag is the human account tag of a config-dir basename (".claude-gem7" → "gem7").
func rwAccountTag(account string) string {
	tag := strings.TrimPrefix(strings.TrimPrefix(account, ".claude-"), ".claude")
	if tag == "" {
		return "default"
	}
	return tag
}

func rwIsDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func rwNowISO() string { return time.Now().UTC().Format("2006-01-02T15:04:05Z") }

func rwEnvBool(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func rwEnvFloat(name string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func rwEnvInt(name string, def int) int {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func rwCrashLoopBudget() int {
	return rwEnvInt("FAK_RESUME_CRASH_LOOP_BUDGET", 3)
}

func rwBackoffHistory(path string) []resumebackoff.Event {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []resumebackoff.Event
	for _, line := range bytes.Split(b, []byte{'\n'}) {
		var row struct {
			TS        string `json:"ts"`
			Session   string `json:"session"`
			Signature string `json:"signature"`
			Phase     string `json:"phase"`
		}
		if json.Unmarshal(line, &row) != nil || row.Phase != "launched" || row.Session == "" || row.Signature == "" {
			continue
		}
		if at, err := time.Parse(time.RFC3339, row.TS); err == nil {
			out = append(out, resumebackoff.Event{Session: row.Session, Signature: row.Signature, At: at})
		}
	}
	return out
}
