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

type launchDoctorEntryPoint struct {
	Command  string   `json:"command"`
	Provider string   `json:"provider"`
	Role     string   `json:"role"`
	Pipeline []string `json:"pipeline"`
	Ready    bool     `json:"ready"`
	Reason   string   `json:"reason"`
	Action   string   `json:"action,omitempty"`
}

const installedLaunchQualificationSchema = "fak.installed-launch-qualification.v1"

type installedLaunchQualificationProvider struct {
	Provider string   `json:"provider"`
	Harness  string   `json:"harness"`
	Chain    []string `json:"resolved_executable_chain,omitempty"`
	Status   string   `json:"status"`
	Failure  string   `json:"failure,omitempty"`
}

type installedLaunchQualificationReceipt struct {
	Schema    string                                 `json:"schema"`
	Qualified bool                                   `json:"qualified"`
	Providers []installedLaunchQualificationProvider `json:"providers"`
}

type launchDoctorReport struct {
	Schema        string                              `json:"schema"`
	ConfigPath    string                              `json:"config_path"`
	ShimDir       string                              `json:"shim_dir"`
	Default       string                              `json:"default_provider,omitempty"`
	Binary        binaryIdentity                      `json:"binary"`
	Rows          []launchDoctorRow                   `json:"providers"`
	EntryPoints   []launchDoctorEntryPoint            `json:"entry_points"`
	Qualification installedLaunchQualificationReceipt `json:"installed_launch_qualification"`
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
		fmt.Fprintln(stdout, "ENTRY POINTS")
		for _, entry := range report.EntryPoints {
			fmt.Fprintf(stdout, "  %-12s role=%-12s ready=%-5t reason=%s\n", entry.Command, entry.Role, entry.Ready, entry.Reason)
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

func repairLaunchShims() error {
	c, err := launchshim.Load()
	if err != nil {
		return err
	}
	dir, err := launchBinDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	target, err := installStableLaunchTarget(dir, exe)
	if err != nil {
		return err
	}
	for name, provider := range c.Providers {
		if provider.InstallShim {
			if err := writeLaunchShim(dir, name, target); err != nil {
				return err
			}
		}
	}
	return nil
}

type launchLookPath func(string) (string, error)
type launchStat func(string) (os.FileInfo, error)

func buildLaunchDoctor(c launchshim.Config, loadErr error, configPath, shimDir string, look launchLookPath, stat launchStat) launchDoctorReport {
	report := launchDoctorReport{Schema: launchDoctorSchema, ConfigPath: redactLocalPath(configPath), ShimDir: redactLocalPath(shimDir), Default: c.Default, Binary: buildIdentityFromRuntime()}
	providers := []string{"claude", "codex"}
	for p := range c.Providers {
		if p != "claude" && p != "codex" {
			providers = append(providers, p)
		}
	}
	sort.Strings(providers)
	for _, provider := range providers {
		row := launchDoctorRow{Provider: provider, BypassActive: launchshim.EffectiveDirect(c, false)}
		recovery := "fak launch install --provider " + provider
		if provider != "claude" && provider != "codex" {
			recovery = "fak launch add " + provider + " --command PATH"
		}
		if loadErr != nil {
			row.Reason, row.Action = "CONFIG_INVALID", recovery
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
		case command == "" && winnerErr == nil:
			row.Reason, row.Action = "UNMANAGED", recovery
		case command == "":
			row.Reason, row.Action = "UNDERLYING_MISSING", recovery
		case samePath(command, shim) || sameLaunchDir(command, shimDir):
			row.Reason, row.Action = "RECURSIVE", "fak launch uninstall --provider "+provider
		case statMissing(stat, command):
			row.Reason, row.Action = "UNDERLYING_MISSING", recovery
		case winnerErr != nil:
			row.Reason, row.Action = "NOT_ON_PATH", pathRecovery(shimDir)
		case !samePath(winner, shim):
			row.Reason, row.Action = "SHADOWED", pathRecovery(shimDir)
		default:
			row.Reason, row.InterceptReady = "READY", true
		}
		report.Rows = append(report.Rows, row)
	}
	sort.Slice(report.Rows, func(i, j int) bool { return report.Rows[i].Provider < report.Rows[j].Provider })
	report.EntryPoints = buildLaunchDoctorEntryPoints(report.Rows)
	report.Qualification = buildInstalledLaunchQualification(report.Rows)
	return report
}

func validateInstalledLaunchQualification(receipt *installedLaunchQualificationReceipt) error {
	if receipt == nil {
		return nil
	}
	if receipt.Schema != installedLaunchQualificationSchema {
		return fmt.Errorf("installed launch qualification schema %q, want %q", receipt.Schema, installedLaunchQualificationSchema)
	}
	if !receipt.Qualified {
		return fmt.Errorf("installed launch qualification is not ready")
	}
	return nil
}
func buildInstalledLaunchQualification(rows []launchDoctorRow) installedLaunchQualificationReceipt {
	receipt := installedLaunchQualificationReceipt{Schema: installedLaunchQualificationSchema, Qualified: true}
	for _, row := range rows {
		provider := installedLaunchQualificationProvider{
			Provider: row.Provider,
			Harness:  "fak-launch/" + row.Provider,
			Status:   row.Reason,
		}
		if row.InterceptReady && row.PathWinner != "" && row.Underlying != "" {
			provider.Chain = []string{row.PathWinner, "fak guard", row.Underlying}
		} else {
			receipt.Qualified = false
			provider.Failure = row.Reason
		}
		receipt.Providers = append(receipt.Providers, provider)
	}
	return receipt
}

func buildLaunchDoctorEntryPoints(rows []launchDoctorRow) []launchDoctorEntryPoint {
	var codex launchDoctorRow
	for _, row := range rows {
		if row.Provider == "codex" {
			codex = row
			break
		}
	}
	bare := launchDoctorEntryPoint{Command: "codex", Provider: "codex", Role: "canonical", Pipeline: []string{"managed-shim", "fak launch codex", "fak guard", "recorded-provider"}, Ready: codex.InterceptReady, Reason: codex.Reason, Action: codex.Action}
	pathReason, pathAction := codex.Reason, codex.Action
	if codex.InterceptReady {
		pathReason = "PATH_RESOLUTION_AMBIGUOUS"
		pathAction = "use codex; use fak launch --direct codex for recovery; track fak m codex under #8866"
	}
	return []launchDoctorEntryPoint{
		bare,
		{Command: "fak m codex", Provider: "codex", Role: "noncanonical", Pipeline: []string{"fak manage", "fak guard", "PATH codex"}, Reason: pathReason, Action: pathAction},
		{Command: "fak codex", Provider: "codex", Role: "specialized", Pipeline: []string{"freshness-admission", "loop-gate", "fak guard", "PATH codex"}, Reason: pathReason, Action: pathAction},
	}
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
