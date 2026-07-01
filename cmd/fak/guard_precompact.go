package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

const (
	guardPreCompactModeOff     = "off"
	guardPreCompactModeShadow  = "shadow"
	guardPreCompactModeEnforce = "enforce"

	guardPreCompactEnvMode       = "FAK_GUARD_PRECOMPACT_MODE"
	guardPreCompactEnvMetricsURL = "FAK_GUARD_PRECOMPACT_METRICS_URL"
	guardPreCompactMetricName    = "fak_harness_coherence_posture"
)

type guardPreCompactInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	MetricsURL   string
	Reason       string
}

type guardPreCompactClaudeSettings struct {
	Hooks map[string][]guardPreCompactClaudeMatcher `json:"hooks"`
}

type guardPreCompactClaudeMatcher struct {
	Matcher string                         `json:"matcher,omitempty"`
	Hooks   []guardPreCompactClaudeCommand `json:"hooks"`
}

type guardPreCompactClaudeCommand struct {
	Type    string   `json:"type"`
	Command string   `json:"command"`
	Args    []string `json:"args,omitempty"`
}

func cmdGuardPreCompact(argv []string) {
	os.Exit(runGuardPreCompact(os.Stdout, os.Stderr, argv))
}

func runGuardPreCompact(stdout, stderr io.Writer, argv []string) int {
	_ = stdout
	fs := flag.NewFlagSet("guard-precompact", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardPreCompactEnvMode), "off|shadow|enforce")
	metricsURLFlag := fs.String("metrics-url", os.Getenv(guardPreCompactEnvMetricsURL), "gateway /metrics URL")
	timeout := fs.Duration("timeout", 500*time.Millisecond, "maximum time to wait for the gateway posture metric")
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintf(stderr, "fak guard PreCompact: allowing Claude auto-compaction; bad hook args: %v\n", err)
		return 0
	}
	mode, err := normalizeGuardPreCompactMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard PreCompact: allowing Claude auto-compaction; %v\n", err)
		return 0
	}
	if mode == guardPreCompactModeOff {
		return 0
	}
	metricsURL := strings.TrimSpace(*metricsURLFlag)
	if metricsURL == "" {
		metricsURL = guardPreCompactMetricsURLFromBase(os.Getenv("ANTHROPIC_BASE_URL"))
	}
	if metricsURL == "" {
		fmt.Fprintln(stderr, "fak guard PreCompact: allowing Claude auto-compaction; no metrics URL configured")
		return 0
	}
	posture, err := fetchGuardPreCompactPosture(context.Background(), metricsURL, *timeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard PreCompact: allowing Claude auto-compaction; posture unavailable: %v\n", err)
		return 0
	}
	exitCode := compactcohere.PreCompactExitCode(posture)
	if mode == guardPreCompactModeShadow {
		action := "allow"
		if exitCode == 2 {
			action = "block"
		}
		fmt.Fprintf(stderr, "fak guard PreCompact: shadow would %s Claude auto-compaction (posture=%s exit=%d)\n", action, posture, exitCode)
		return 0
	}
	return exitCode
}

func installGuardPreCompactHook(command []string, mode, gwURL string) ([]string, [][2]string, guardPreCompactInstall, error) {
	normalized, err := normalizeGuardPreCompactMode(mode)
	if err != nil {
		return command, nil, guardPreCompactInstall{}, err
	}
	install := guardPreCompactInstall{Mode: normalized}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}
	dir, err := os.MkdirTemp("", "fak-guard-precompact-*")
	if err != nil {
		return command, nil, guardPreCompactInstall{}, err
	}
	return installGuardPreCompactHookAt(command, mode, gwURL, fakBin, dir)
}

func installGuardPreCompactHookAt(command []string, mode, gwURL, fakBin, dir string) ([]string, [][2]string, guardPreCompactInstall, error) {
	normalized, err := normalizeGuardPreCompactMode(mode)
	if err != nil {
		return command, nil, guardPreCompactInstall{}, err
	}
	install := guardPreCompactInstall{Mode: normalized}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	if strings.TrimSpace(dir) == "" {
		return command, nil, install, errors.New("empty PreCompact hook settings directory")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return command, nil, install, err
	}
	settingsPath := filepath.Join(dir, "claude-precompact-settings.json")
	if err := writeGuardPreCompactSettings(settingsPath, fakBin); err != nil {
		return command, nil, install, err
	}
	metricsURL := guardPreCompactMetricsURLFromBase(gwURL)
	install.Applied = true
	install.SettingsPath = settingsPath
	install.MetricsURL = metricsURL
	env := [][2]string{
		{guardPreCompactEnvMode, normalized},
		{guardPreCompactEnvMetricsURL, metricsURL},
	}
	return appendClaudeSettingsArg(command, settingsPath), env, install, nil
}

func writeGuardPreCompactSettings(path, fakBin string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"PreCompact": {{
				Matcher: "auto",
				Hooks: []guardPreCompactClaudeCommand{{
					Type:    "command",
					Command: guardPreCompactHookCommand(fakBin),
					Args:    []string{"guard-precompact"},
				}},
			}},
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func guardPreCompactHookCommand(fakBin string) string {
	fakBin = strings.TrimSpace(fakBin)
	if fakBin == "" {
		fakBin = "fak"
	}
	return fakBin
}

func appendClaudeSettingsArg(command []string, settingsPath string) []string {
	if len(command) == 0 {
		return command
	}
	out := make([]string, 0, len(command)+2)
	out = append(out, command[0], "--settings", settingsPath)
	return append(out, command[1:]...)
}

func normalizeGuardPreCompactMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --precompact-hook mode %q (want off, shadow, or enforce)", mode)
	}
}

// guardPreCompactIsClaudeCommand reports whether the wrapped agent takes the `settings-file`
// repoint — the `--settings` PreCompact/Stop hooks and `--mcp-config` self-query registration,
// all Claude-shaped. It now delegates to the profile registry (C3, #1954): a harness gets
// settings-file iff its HarnessProfile declares RepointSettingsFile, which today is exactly the
// claude profile. So the settings/MCP installers are still inert for every non-Claude agent, but
// the SELECTION is data (profile.Repoint) rather than a hardcoded name check. Delegating to
// harnessprofile.Lookup also makes the match cross-platform (a Windows-path launcher on a Linux
// runner now matches, where filepath.Base did not) — a latent-bug fix, not a behavior the tested
// paths relied on.
func guardPreCompactIsClaudeCommand(command []string) bool {
	if len(command) == 0 {
		return false
	}
	return guardProfileHasRepoint(command[0], harnessprofile.RepointSettingsFile)
}

func guardPreCompactMetricsURLFromBase(base string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		return ""
	}
	if strings.HasSuffix(base, "/v1") {
		base = strings.TrimSuffix(base, "/v1")
	}
	return base + "/metrics"
}

func fetchGuardPreCompactPosture(ctx context.Context, metricsURL string, timeout time.Duration) (compactcohere.Posture, error) {
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("metrics returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	return parseGuardPreCompactMetricsPosture(string(body))
}

func parseGuardPreCompactMetricsPosture(metrics string) (compactcohere.Posture, error) {
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		if name != guardPreCompactMetricName && !strings.HasPrefix(name, guardPreCompactMetricName+"{") {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return "", fmt.Errorf("parse %s value %q: %w", guardPreCompactMetricName, fields[1], err)
		}
		if value > 0 {
			return compactcohere.PostureBlock, nil
		}
		return compactcohere.PostureAllow, nil
	}
	return "", fmt.Errorf("metric %s not found", guardPreCompactMetricName)
}
