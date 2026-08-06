package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

const launchDoctorSchema = "fak.launch-doctor.v1"

const launchHelpText = `Usage:
  fak launch [--direct] [claude|codex] [-- provider-args...]
  fak launch install [--provider claude|codex|all] [--default claude|codex]
  fak launch uninstall [--provider claude|codex|all]
  fak launch default claude|codex
  fak launch enable|disable|status
  fak launch doctor [--json]

Direct escapes: --direct, provider-side --fak-direct, FAK_DIRECT=1, or persisted disable.
Doctor emits no prompts or forwarded provider arguments; paths are basename-redacted.
`

type launchDoctorRow struct {
	Provider       string `json:"provider"`
	Reason         string `json:"reason"`
	Action         string `json:"action,omitempty"`
	PathWinner     string `json:"path_winner,omitempty"`
	Underlying     string `json:"underlying,omitempty"`
	BypassActive   bool   `json:"bypass_active"`
	InterceptReady bool   `json:"intercept_ready"`
}

type launchDoctorReport struct {
	Schema     string            `json:"schema"`
	ConfigPath string            `json:"config_path"`
	ShimDir    string            `json:"shim_dir"`
	Default    string            `json:"default_provider,omitempty"`
	Rows       []launchDoctorRow `json:"providers"`
}

func runLaunchDoctor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit versioned JSON")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	configPath, err := launchshim.Path()
	if err != nil {
		fmt.Fprintf(stderr, "fak launch doctor: %v\n", err)
		return 1
	}
	shimDir, err := launchBinDir()
	if err != nil {
		fmt.Fprintf(stderr, "fak launch doctor: %v\n", err)
		return 1
	}
	c, loadErr := launchshim.Load()
	report := buildLaunchDoctor(c, loadErr, configPath, shimDir, exec.LookPath, os.Stat)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak launch doctor: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "LAUNCH DOCTOR config=%s shims=%s default=%s\n", report.ConfigPath, report.ShimDir, valueOr(report.Default, "none"))
		for _, row := range report.Rows {
			fmt.Fprintf(stdout, "  %-6s %-18s winner=%-12s underlying=%-12s action=%s\n", row.Provider, row.Reason, valueOr(row.PathWinner, "none"), valueOr(row.Underlying, "none"), valueOr(row.Action, "none"))
		}
	}
	if loadErr != nil {
		return 1
	}
	for _, row := range report.Rows {
		if !row.InterceptReady && row.Reason != "DISABLED" {
			return 1
		}
	}
	return 0
}

type launchLookPath func(string) (string, error)
type launchStat func(string) (os.FileInfo, error)

func buildLaunchDoctor(c launchshim.Config, loadErr error, configPath, shimDir string, look launchLookPath, stat launchStat) launchDoctorReport {
	report := launchDoctorReport{Schema: launchDoctorSchema, ConfigPath: redactLocalPath(configPath), ShimDir: redactLocalPath(shimDir), Default: c.Default}
	providers := []string{"claude", "codex"}
	for p := range c.Providers {
		if p != "claude" && p != "codex" {
			providers = append(providers, p)
		}
	}
	sort.Strings(providers)
	for _, provider := range providers {
		row := launchDoctorRow{Provider: provider, BypassActive: launchshim.EffectiveDirect(c, false)}
		if loadErr != nil {
			row.Reason, row.Action = "CONFIG_INVALID", "fak launch install --provider "+provider
			report.Rows = append(report.Rows, row)
			continue
		}
		command := strings.TrimSpace(c.Providers[provider].Command)
		if command != "" {
			row.Underlying = redactLocalPath(command)
		}
		winner, winnerErr := look(provider)
		if winnerErr == nil {
			row.PathWinner = redactLocalPath(winner)
		}
		shim := filepath.Join(shimDir, shimName(provider))
		switch {
		case c.Disabled:
			row.Reason, row.Action = "DISABLED", "fak launch enable"
		case command == "":
			row.Reason, row.Action = "UNDERLYING_MISSING", "fak launch install --provider "+provider
		case samePath(command, shim) || sameLaunchDir(command, shimDir):
			row.Reason, row.Action = "RECURSIVE", "fak launch uninstall --provider "+provider
		case statMissing(stat, command):
			row.Reason, row.Action = "UNDERLYING_MISSING", "fak launch install --provider "+provider
		case winnerErr != nil:
			row.Reason, row.Action = "NOT_ON_PATH", pathRecovery(shimDir)
		case !samePath(winner, shim):
			row.Reason, row.Action = "SHADOWED", pathRecovery(shimDir)
		default:
			row.Reason, row.InterceptReady = "READY", true
		}
		report.Rows = append(report.Rows, row)
	}
	return report
}

func statMissing(stat launchStat, path string) bool { _, err := stat(path); return err != nil }
func samePath(a, b string) bool {
	aa, _ := filepath.Abs(a)
	bb, _ := filepath.Abs(b)
	return strings.EqualFold(filepath.Clean(aa), filepath.Clean(bb))
}
func redactLocalPath(path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	return "<local>/" + filepath.Base(filepath.Clean(path))
}
func valueOr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
func pathRecovery(dir string) string { return "prepend " + redactLocalPath(dir) + " to PATH" }
