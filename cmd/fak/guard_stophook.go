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
)

// guard_stophook.go — the harness half of the deny-all false-stop fix.
//
// The WIRE half is unchanged and correct: when the capability floor refuses EVERY tool call
// in a turn, the gateway must report stop_reason=end_turn (else the client hangs hunting for
// a tool_use block that was dropped — the v0.15.0 contract). But end_turn halts the agent
// loop though the model wanted to act, so the turn ends with the task unfinished — a STOP the
// agent did not choose. This actuator catches that stop and, in enforce mode, RESUMES the
// agent: it is a Claude Code `Stop` hook (installed by `fak guard` into the same --settings
// file as the PreCompact hook) that reads the gateway's fak_guard_deny_all_consecutive gauge
// and, when the most recent turn was a deny-all (and we are under the retry bound), exits 2
// with a continuation instruction on stderr — which Claude Code feeds back to the model so it
// picks an allowed alternative instead of stopping. The bound (max consecutive auto-continues)
// makes it impossible to loop forever on a model that keeps re-proposing refused calls.
//
// It mirrors guard-precompact exactly: a hidden subcommand polling /metrics, fail-open on any
// unavailability (a Stop hook that cannot reach the gateway must never wedge the agent), and a
// shadow mode that logs the would-be decision while always allowing the stop.

const (
	guardStopHookEnvMode       = "FAK_GUARD_DENYALL_MODE"
	guardStopHookEnvMetricsURL = "FAK_GUARD_DENYALL_METRICS_URL"
	guardStopHookEnvMax        = "FAK_GUARD_DENYALL_MAX"

	// guardStopHookMetricName is the gateway gauge this hook polls: the count of consecutive
	// deny-all turns ending the most recent served turn (0 on a healthy completion).
	guardStopHookMetricName = "fak_guard_deny_all_consecutive"

	// guardStopHookDefaultMax bounds how many consecutive deny-all turns the hook will
	// auto-continue past before it gives up and lets the turn end — so a model that keeps
	// re-proposing refused calls cannot loop forever.
	guardStopHookDefaultMax = 3
)

// guardStopHookContinueReason is the instruction fed back to the model (via the Stop hook's
// exit-2 stderr) when fak resumes the agent past a deny-all stop. The per-call refusal detail
// is already in the transcript (the in-band `[fak] refused …` note on the ended turn); this is
// the nudge to act on it rather than stop.
const guardStopHookContinueReason = "fak guard: your previous turn proposed only tool call(s) the capability floor refused, so it ended without action (reported upstream as end_turn). Do NOT re-propose a refused call unchanged — pick an ALLOWED alternative and continue the task. If there is genuinely no allowed way to make progress, say so explicitly and then stop."

type guardStopHookInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	MetricsURL   string
	Max          int
	Reason       string
}

func cmdGuardStopHook(argv []string) {
	os.Exit(runGuardStopHook(os.Stderr, os.Stdin, argv))
}

// runGuardStopHook is the Stop-hook actuator. It returns the process exit code: 2 to BLOCK the
// stop (Claude Code continues the agent with the stderr text as guidance), 0 to allow it. It
// fails OPEN — any bad args, unknown mode, missing metrics URL, or unreachable gateway returns
// 0 so the hook can never wedge the agent — exactly the posture guard-precompact takes.
func runGuardStopHook(stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("guard-stophook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardStopHookEnvMode), "off|shadow|enforce")
	metricsURLFlag := fs.String("metrics-url", os.Getenv(guardStopHookEnvMetricsURL), "gateway /metrics URL")
	maxFlag := fs.Int("max", guardStopHookMaxFromEnv(), "max consecutive deny-all turns to auto-continue past")
	timeout := fs.Duration("timeout", 500*time.Millisecond, "maximum time to wait for the gateway gauge")
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; bad hook args: %v\n", err)
		return 0
	}
	mode, err := normalizeGuardStopHookMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; %v\n", err)
		return 0
	}
	if mode == guardPreCompactModeOff {
		return 0
	}
	// Best-effort read of Claude's Stop-hook stdin payload for stop_hook_active — used for the
	// shadow log and as a defensive signal; the gateway consecutive bound is the authoritative
	// loop guard, so a parse miss never changes the decision.
	active := readStopHookActive(stdin)
	metricsURL := strings.TrimSpace(*metricsURLFlag)
	if metricsURL == "" {
		metricsURL = guardPreCompactMetricsURLFromBase(os.Getenv("ANTHROPIC_BASE_URL"))
	}
	if metricsURL == "" {
		fmt.Fprintln(stderr, "fak guard Stop: allowing stop; no metrics URL configured")
		return 0
	}
	consecutive, err := fetchGuardStopHookConsecutive(context.Background(), metricsURL, *timeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; deny-all gauge unavailable: %v\n", err)
		return 0
	}
	maxN := *maxFlag
	if maxN <= 0 {
		maxN = guardStopHookDefaultMax
	}
	exit, block := guardStopHookDecision(consecutive, maxN, mode)
	if mode == guardPreCompactModeShadow {
		action := "allow stop"
		if block {
			action = "auto-continue (block stop)"
		}
		fmt.Fprintf(stderr, "fak guard Stop: shadow would %s (deny_all_consecutive=%d max=%d stop_hook_active=%v)\n", action, consecutive, maxN, active)
		return 0
	}
	if exit == 2 {
		// Exit 2 blocks the stop; stderr is shown to Claude as the reason to continue.
		fmt.Fprintln(stderr, guardStopHookContinueReason)
		return 2
	}
	return 0
}

// guardStopHookDecision is the PURE decision behind the hook: given the gateway's consecutive
// deny-all count, the retry bound, and the mode, return the exit code and whether it WOULD
// block. Side-effect-free so the policy is unit-tested without an HTTP gateway. The block
// window is 1..max inclusive: 0 means the last turn was a clean (non-deny-all) completion —
// allow the stop; above max means we have already auto-continued enough — give up and let the
// turn end so a stuck model cannot loop forever.
func guardStopHookDecision(consecutive, maxN int, mode string) (exit int, block bool) {
	if mode == guardPreCompactModeOff {
		return 0, false
	}
	block = consecutive >= 1 && consecutive <= maxN
	if mode == guardPreCompactModeShadow {
		return 0, block // shadow always allows the stop (exit 0) but reports the would-be block
	}
	if block {
		return 2, true
	}
	return 0, false
}

// readStopHookActive parses the stop_hook_active flag from Claude's Stop-hook stdin JSON. A nil
// reader, an empty body, or a parse miss returns false — it is advisory only.
func readStopHookActive(stdin io.Reader) bool {
	if stdin == nil {
		return false
	}
	b, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil || len(b) == 0 {
		return false
	}
	var payload struct {
		StopHookActive bool `json:"stop_hook_active"`
	}
	if json.Unmarshal(b, &payload) != nil {
		return false
	}
	return payload.StopHookActive
}

func normalizeGuardStopHookMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeEnforce:
		// Default ENFORCE: the false stop is a real defect, so the fix is on by default. It is
		// bounded (guardStopHookDefaultMax) and fully observable (the deny-all metrics + exit
		// summary), and `--deny-all-continue off` opts out. This differs deliberately from the
		// PreCompact hook's shadow default, whose enforce path can break harness context management.
		return guardPreCompactModeEnforce, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	default:
		return "", fmt.Errorf("invalid --deny-all-continue mode %q (want off, shadow, or enforce)", mode)
	}
}

func guardStopHookMaxFromEnv() int {
	if v := strings.TrimSpace(os.Getenv(guardStopHookEnvMax)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return guardStopHookDefaultMax
}

// installGuardStopHook installs the Claude Code Stop hook for a guard session. When the
// PreCompact hook already wrote a --settings file (existingSettingsPath non-empty), the Stop
// hook is MERGED into it so a single --settings carries both (--settings is a single-value flag;
// injecting it twice clobbers rather than merges). Otherwise it writes its own settings file and
// injects --settings. Off mode or a non-claude child is a no-op (command returned unchanged).
func installGuardStopHook(command []string, mode, gwURL, existingSettingsPath string, maxN int) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	install := guardStopHookInstall{Mode: normalized, Max: maxN}
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
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		dir, err = os.MkdirTemp("", "fak-guard-stophook-*")
		if err != nil {
			return command, nil, guardStopHookInstall{}, err
		}
	}
	return installGuardStopHookAt(command, mode, gwURL, fakBin, dir, existingSettingsPath, maxN)
}

func installGuardStopHookAt(command []string, mode, gwURL, fakBin, dir, existingSettingsPath string, maxN int) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	install := guardStopHookInstall{Mode: normalized, Max: maxN}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	if maxN <= 0 {
		maxN = guardStopHookDefaultMax
	}
	install.Max = maxN

	var settingsPath string
	if strings.TrimSpace(existingSettingsPath) != "" {
		// Merge the Stop hook INTO the file the PreCompact hook already wrote + injected.
		if err := mergeGuardStopHookIntoSettings(existingSettingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		settingsPath = existingSettingsPath
		// command already carries --settings; do NOT inject it again.
	} else {
		if strings.TrimSpace(dir) == "" {
			return command, nil, install, errors.New("empty Stop hook settings directory")
		}
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return command, nil, install, err
		}
		settingsPath = filepath.Join(dir, "claude-stophook-settings.json")
		if err := writeGuardStopHookSettings(settingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	metricsURL := guardPreCompactMetricsURLFromBase(gwURL)
	install.Applied = true
	install.SettingsPath = settingsPath
	install.MetricsURL = metricsURL
	env := [][2]string{
		{guardStopHookEnvMode, normalized},
		{guardStopHookEnvMetricsURL, metricsURL},
		{guardStopHookEnvMax, strconv.Itoa(maxN)},
	}
	return command, env, install, nil
}

// guardStopHookMatchers builds the Stop-hook settings entry. The Stop event takes NO matcher
// (matchers are for tool-scoped events), so the Matcher field is left empty (omitempty drops it).
func guardStopHookMatchers(fakBin string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-stophook"},
		}},
	}}
}

func writeGuardStopHookSettings(path, fakBin string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop": guardStopHookMatchers(fakBin),
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

// mergeGuardStopHookIntoSettings adds (or replaces) the Stop hook in an existing guard settings
// file, preserving every other key (e.g. the PreCompact hook), so a single --settings carries both.
func mergeGuardStopHookIntoSettings(path, fakBin string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	settings.Hooks["Stop"] = guardStopHookMatchers(fakBin)
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func fetchGuardStopHookConsecutive(ctx context.Context, metricsURL string, timeout time.Duration) (int, error) {
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("metrics returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, err
	}
	return parseGuardStopHookConsecutive(string(body))
}

// parseGuardStopHookConsecutive extracts the unlabeled fak_guard_deny_all_consecutive gauge
// value from a Prometheus scrape. Not-found is an error so the caller fails open rather than
// silently treating a missing gauge as 0 (which would never auto-continue).
func parseGuardStopHookConsecutive(metrics string) (int, error) {
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != guardStopHookMetricName {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0, fmt.Errorf("parse %s value %q: %w", guardStopHookMetricName, fields[1], err)
		}
		return int(value), nil
	}
	return 0, fmt.Errorf("metric %s not found", guardStopHookMetricName)
}
