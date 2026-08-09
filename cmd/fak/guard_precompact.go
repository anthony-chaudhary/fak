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
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

const (
	guardPreCompactModeOff     = "off"
	guardPreCompactModeShadow  = "shadow"
	guardPreCompactModeEnforce = "enforce"

	guardPreCompactEnvMode       = "FAK_GUARD_PRECOMPACT_MODE"
	guardPreCompactEnvMetricsURL = "FAK_GUARD_PRECOMPACT_METRICS_URL"
	guardPreCompactMetricName    = "fak_harness_coherence_posture"

	// guardPreCompactRelayMetricName is the gateway gauge carrying the relay's
	// advisory would-rotate signal (1 = the soft mark crossed and the leg will
	// rotate at its next safe point; 0 = disarmed). #1869 routes it through the
	// PreCompact shadow seam so operators read the relay's would-rotate next to the
	// compaction posture in ONE place. It rides the same /metrics scrape the posture
	// already reads and is shadow-only: it never changes the hook's exit code (the
	// relay rotates at its own safe point — this is observation, not compaction).
	guardPreCompactRelayMetricName = "fak_relay_would_rotate"
	// guardPreCompactRelayArmedReason is the closed relay reason token
	// (docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md) the would-rotate signal
	// carries when armed, mirroring internal/relay ArmFire.Reason and
	// session.ReasonRelayArmed.
	guardPreCompactRelayArmedReason = "RELAY_ARMED"

	// guardClaudeSettingsFlag is Claude Code's settings flag. It is LAST-WINS, not
	// merged: when an argv carries two `--settings` occurrences the earlier one's keys
	// are dropped wholesale. Guard injects its hook-settings file through this flag, so
	// a launcher that appends a second `--settings` later on the SAME argv used to
	// silently disarm guard's entire hook stack (#5510) — the session still looked
	// guarded because the guard process was there and the argv even mentioned
	// `--settings`. appendClaudeSettingsArg now guarantees exactly ONE occurrence.
	guardClaudeSettingsFlag = "--settings"

	// guardClaudeSettingsHooksKey is the one settings.json key guard owns and writes.
	// Every other top-level key belongs to the caller and is merged through verbatim.
	guardClaudeSettingsHooksKey = "hooks"

	// guardClaudeSettingsConflictReason / guardClaudeSettingsHooksReason are the named,
	// closed reasons appendClaudeSettingsArg prints when a later `--settings` payload
	// cannot be folded into guard's file. They exist so the residue this fix cannot
	// merge is LOUD rather than silent, which is the whole point of #5510.
	guardClaudeSettingsConflictReason = "SETTINGS_UNMERGEABLE"
	guardClaudeSettingsHooksReason    = "SETTINGS_HOOKS_DROPPED"
)

type guardPreCompactInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	MetricsURL   string
	Reason       string
}

// guardPreCompactClaudeSettings is the slice of Claude Code's settings.json that guard
// writes: the `hooks` map. Extra carries every OTHER top-level key VERBATIM across the
// read-modify-write round-trips the installers perform (mergeGuardStopHookIntoSettings,
// mergeGuardToolprocIntoSettings, mergeGuardSessionStartIntoSettings). Without that,
// anything folded into the file by appendClaudeSettingsArg — notably the reasoning-posture
// key a launcher used to hand the child on a SECOND `--settings` (#5510) — would be erased
// by the very next installer, because an unmodeled key vanishes on unmarshal→marshal.
type guardPreCompactClaudeSettings struct {
	Hooks map[string][]guardPreCompactClaudeMatcher `json:"hooks"`
	Extra map[string]json.RawMessage                `json:"-"`
}

// UnmarshalJSON decodes `hooks` into the typed map and parks every other top-level key in
// Extra untouched.
func (s *guardPreCompactClaudeSettings) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	s.Hooks = nil
	s.Extra = nil
	for key, value := range raw {
		if key == guardClaudeSettingsHooksKey {
			if err := json.Unmarshal(value, &s.Hooks); err != nil {
				return err
			}
			continue
		}
		if s.Extra == nil {
			s.Extra = map[string]json.RawMessage{}
		}
		s.Extra[key] = append(json.RawMessage(nil), value...)
	}
	return nil
}

// MarshalJSON re-emits `hooks` plus every preserved key. Marshalling through a map keeps the
// output key order deterministic (encoding/json sorts map keys), and a settings file with no
// Extra keys serializes byte-identically to the previous plain-struct encoding.
func (s guardPreCompactClaudeSettings) MarshalJSON() ([]byte, error) {
	raw := make(map[string]json.RawMessage, len(s.Extra)+1)
	for key, value := range s.Extra {
		if key == guardClaudeSettingsHooksKey {
			continue
		}
		raw[key] = value
	}
	hooks, err := json.Marshal(s.Hooks)
	if err != nil {
		return nil, err
	}
	raw[guardClaudeSettingsHooksKey] = hooks
	return json.Marshal(raw)
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
	os.Exit(runGuardPreCompact(os.Stdout, os.Stderr, os.Stdin, argv))
}

// runGuardPreCompact wraps the compaction-coherence decision with the #2539 PreCompact
// twin: when the compaction is ALLOWED (any exit-0 path — the reset is about to happen),
// append one compaction-boundary row per open objective so curve readers can see the
// context reset. Bounded + fail-open, gated on the guard-wired ledger env, and it never
// changes the decision's exit code. A blocked compaction (exit 2) marks no boundary —
// the context was not reset.
func runGuardPreCompact(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	code := runGuardPreCompactDecision(stdout, stderr, argv)
	if code != 2 {
		appendCompactionBoundaryFailOpen(stderr, trajctl.Stamp{SessionID: parseHookSessionID(readHookStdin(stdin))}, time.Now().UnixMilli())
		// #4118: an allowed compaction is a context reset — the latest moment the live
		// remaining budget is known before the window is handed off. Project it into the
		// transcript-UUID carry store so a later `claude --resume` re-seeds at the carried
		// budget, not a fresh cap. Fail-open; never changes the decision's exit code.
		writeDriveCarryFailOpen(time.Now())
	}
	return code
}

func runGuardPreCompactDecision(stdout, stderr io.Writer, argv []string) int {
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
	// An explicitly-passed --metrics-url PINS the signal source to that gateway's
	// /metrics. Without the pin the lifecycle-IPC preference (#3305) reads whatever
	// FAK_GUARD_LIFECYCLE_SOCKET happens to be in the environment — and `fak guard`
	// exports that var into everything it launches, so a nested `fak guard-precompact
	// --metrics-url http://…` (an operator debugging a specific gateway, or a test
	// standing up its own /metrics) silently reported the SUPERVISOR's posture and
	// never dialled the URL it was handed. Naming a URL on the command line is an
	// explicit choice of source, so it outranks the ambient env. The installed hook
	// passes no flags (see writeGuardPreCompactSettings) and carries the metrics URL in
	// FAK_GUARD_PRECOMPACT_METRICS_URL instead, so the guarded path still prefers IPC.
	// fs.Visit walks only the flags actually present in argv, which is exactly the
	// "named on the command line" vs "fell back to its env-derived default" distinction.
	metricsURLPinned := false
	if metricsURL != "" {
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "metrics-url" {
				metricsURLPinned = true
			}
		})
	}
	if metricsURL == "" {
		metricsURL = guardPreCompactMetricsURLFromBase(os.Getenv("ANTHROPIC_BASE_URL"))
	}
	fetch := fetchGuardPreCompactSignalsPreferred
	if metricsURLPinned {
		fetch = fetchGuardPreCompactSignalsHTTP
	}
	signals, source, err := fetch(context.Background(), metricsURL, *timeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard PreCompact: allowing Claude auto-compaction; posture unavailable: %v\n", err)
		return 0
	}
	posture := signals.posture
	exitCode := compactcohere.PreCompactExitCode(posture)
	if mode == guardPreCompactModeShadow {
		action := "allow"
		if exitCode == 2 {
			action = "block"
		}
		fmt.Fprintf(stderr, "fak guard PreCompact: shadow would %s Claude auto-compaction (posture=%s exit=%d)\n", action, posture, exitCode)
		surfaceGuardPreCompactRelayShadowSignal(stderr, signals.relayArmed, signals.relayPresent)
		fmt.Fprintf(stderr, "fak guard PreCompact: signals=%s\n", source)
		return 0
	}
	return exitCode
}

type guardPreCompactSignals struct {
	posture      compactcohere.Posture
	relayArmed   bool
	relayPresent bool
}

func fetchGuardPreCompactSignalsPreferred(ctx context.Context, metricsURL string, timeout time.Duration) (guardPreCompactSignals, string, error) {
	socketPath := strings.TrimSpace(os.Getenv(guardLifecycleSocketEnv))
	token := strings.TrimSpace(os.Getenv(guardLifecycleTokenEnv))
	if socketPath != "" || token != "" {
		if socketPath == "" || token == "" {
			return guardPreCompactSignals{}, "ipc", errors.New("lifecycle IPC environment incomplete")
		}
		snapshot, err := fetchGuardLifecycleSignals(socketPath, token, timeout)
		if err != nil {
			return guardPreCompactSignals{}, "ipc", fmt.Errorf("lifecycle IPC: %w", err)
		}
		posture := compactcohere.Posture(snapshot.HarnessPosture)
		if posture != compactcohere.PostureAllow && posture != compactcohere.PostureBlock {
			return guardPreCompactSignals{}, "ipc", fmt.Errorf("lifecycle IPC: invalid harness posture %q", snapshot.HarnessPosture)
		}
		return guardPreCompactSignals{
			posture:      posture,
			relayArmed:   snapshot.RelayWouldRotate,
			relayPresent: snapshot.RelayWouldRotateSeen,
		}, "ipc", nil
	}
	return fetchGuardPreCompactSignalsHTTP(ctx, metricsURL, timeout)
}

// fetchGuardPreCompactSignalsHTTP reads BOTH the compaction posture and the relay
// would-rotate gauge out of ONE gateway /metrics scrape, ignoring the lifecycle-IPC
// env entirely. It is the fallback leg of fetchGuardPreCompactSignalsPreferred and
// also the pinned path when the caller named a --metrics-url explicitly.
func fetchGuardPreCompactSignalsHTTP(ctx context.Context, metricsURL string, timeout time.Duration) (guardPreCompactSignals, string, error) {
	if metricsURL == "" {
		return guardPreCompactSignals{}, "http", errors.New("no metrics URL configured")
	}
	metrics, err := fetchGuardPreCompactMetrics(ctx, metricsURL, timeout)
	if err != nil {
		return guardPreCompactSignals{}, "http", err
	}
	posture, err := parseGuardPreCompactMetricsPosture(metrics)
	if err != nil {
		return guardPreCompactSignals{}, "http", err
	}
	relayArmed, relayPresent := parseGuardPreCompactRelayArmed(metrics)
	return guardPreCompactSignals{posture: posture, relayArmed: relayArmed, relayPresent: relayPresent}, "http", nil
}

func surfaceGuardPreCompactRelayShadowSignal(stderr io.Writer, armed, present bool) {
	if !present {
		return
	}
	if armed {
		fmt.Fprintf(stderr, "fak guard PreCompact: relay would-rotate signal armed (reason=%s); shadow-only, not triggering compaction\n", guardPreCompactRelayArmedReason)
		return
	}
	fmt.Fprintln(stderr, "fak guard PreCompact: relay would-rotate signal disarmed; shadow-only, not triggering compaction")
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
	dir, err := guardSessionTempDir("precompact")
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
	return writeGuardSettingsFileAtomic(path, data)
}

func guardPreCompactHookCommand(fakBin string) string {
	fakBin = strings.TrimSpace(fakBin)
	if fakBin == "" {
		fakBin = "fak"
	}
	return fakBin
}

// appendClaudeSettingsArg points the child at guard's hook-settings file and guarantees that
// file is the argv's ONLY `--settings`.
//
// Claude Code's `--settings` is last-wins, not merged, so simply inserting guard's file after
// argv[0] was not enough: any launcher that appended its own `--settings` later on the same
// argv — the interactive `--ultracode=on` shortcut, a tier/T0 fleet worker's launch profile —
// silently discarded guard's whole file, taking the Stop auto-continue hook, the PreToolUse
// commit-boundary gate, the toolproc journal hooks, the SessionStart affordance and the
// PreCompact coherence gate with it (#5510). Nothing reported it: the guard process was
// present and the argv still mentioned `--settings`.
//
// So a later occurrence is now FOLDED into guard's file rather than left to override it: its
// non-hook keys are merged into the file (the caller's keys win, since guard writes none of
// them) and the occurrence is removed from the argv. Guard's `hooks` key is never overridable
// from the child argv — a payload carrying one is dropped with a named line on stderr, as is
// a payload that cannot be read or parsed. Both residues are LOUD; neither is silent.
func appendClaudeSettingsArg(command []string, settingsPath string) []string {
	if len(command) == 0 {
		return command
	}
	rest, payloads := splitLaterClaudeSettingsArgs(command[1:])
	if len(payloads) > 0 {
		if err := foldClaudeSettingsIntoGuardFile(settingsPath, payloads); err != nil {
			fmt.Fprintf(os.Stderr, "fak guard: %s: a later %s on the child argv was not folded into guard's hook settings %s: %v\n",
				guardClaudeSettingsConflictReason, guardClaudeSettingsFlag, settingsPath, err)
		}
	}
	out := make([]string, 0, len(rest)+3)
	out = append(out, command[0], guardClaudeSettingsFlag, settingsPath)
	return append(out, rest...)
}

// splitLaterClaudeSettingsArgs strips every `--settings <value>` / `--settings=<value>` pair
// out of the child's own args, returning the remaining args and the payloads it removed (in
// argv order, so the last one still wins among themselves). A trailing bare `--settings` with
// no value yields an empty payload, which the fold reports rather than silently ignoring.
func splitLaterClaudeSettingsArgs(args []string) (rest, payloads []string) {
	rest = make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		if args[i] == guardClaudeSettingsFlag {
			value := ""
			if i+1 < len(args) {
				value = args[i+1]
				i++
			}
			payloads = append(payloads, value)
			continue
		}
		if value, ok := strings.CutPrefix(args[i], guardClaudeSettingsFlag+"="); ok {
			payloads = append(payloads, value)
			continue
		}
		rest = append(rest, args[i])
	}
	return rest, payloads
}

// foldClaudeSettingsIntoGuardFile merges the caller's `--settings` payloads into the settings
// file at path, which the calling installer has already written. Later payloads win over
// earlier ones, matching Claude's own last-wins order. The `hooks` key is guard's and is never
// taken from a payload; every unmergeable payload is reported (joined) so the caller can say so
// out loud. The file is only rewritten when something actually merged.
func foldClaudeSettingsIntoGuardFile(path string, payloads []string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse guard hook settings %s: %w", path, err)
	}
	var problems []error
	merged := false
	for _, payload := range payloads {
		keys, err := loadClaudeSettingsPayload(payload)
		if err != nil {
			problems = append(problems, err)
			continue
		}
		for key, value := range keys {
			if key == guardClaudeSettingsHooksKey {
				problems = append(problems, fmt.Errorf("%s: guard owns the %q key, so the child argv's hooks were not merged",
					guardClaudeSettingsHooksReason, guardClaudeSettingsHooksKey))
				continue
			}
			if settings.Extra == nil {
				settings.Extra = map[string]json.RawMessage{}
			}
			settings.Extra[key] = value
			merged = true
		}
	}
	if merged {
		if err := writeGuardHookSettings(path, settings); err != nil {
			problems = append(problems, err)
		}
	}
	return errors.Join(problems...)
}

// loadClaudeSettingsPayload resolves one `--settings` value to its top-level keys. Claude
// accepts either inline JSON (`{"ultracode":true}`) or a path to a settings file, so both are
// read here; a caller-supplied FILE is snapshotted at launch, which is the same instant guard's
// own file is written.
func loadClaudeSettingsPayload(payload string) (map[string]json.RawMessage, error) {
	trimmed := strings.TrimSpace(payload)
	if trimmed == "" {
		return nil, fmt.Errorf("empty %s value on the child argv", guardClaudeSettingsFlag)
	}
	data := []byte(trimmed)
	if !strings.HasPrefix(trimmed, "{") {
		fileData, err := os.ReadFile(trimmed)
		if err != nil {
			return nil, fmt.Errorf("read child settings file %q: %w", trimmed, err)
		}
		data = fileData
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return nil, fmt.Errorf("parse child %s payload %q: %w", guardClaudeSettingsFlag, trimmed, err)
	}
	if keys == nil {
		return nil, fmt.Errorf("child %s payload %q is not a JSON object", guardClaudeSettingsFlag, trimmed)
	}
	return keys, nil
}

// writeGuardSettingsFileAtomic writes data to path atomically: it writes a sibling
// temp file, fsyncs nothing but renames it over path, so a failed or partial write
// (ENOSPC / EACCES) leaves any PRE-EXISTING settings file intact rather than
// truncating it. This matters because the guard's merge writers rewrite the SAME
// --settings file the launched child already references, so a torn os.WriteFile there
// would hand Claude Code a corrupt settings.json; the rename makes the swap
// all-or-nothing. The 0o600 perm matches the prior direct os.WriteFile calls. The temp
// file is created in path's own directory so the rename stays on one filesystem (and
// is a same-directory replace on Windows).
func writeGuardSettingsFileAtomic(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once the rename below succeeds
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
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

// fetchGuardPreCompactMetrics scrapes the gateway /metrics body once so a single
// scrape feeds BOTH the compaction posture and the relay would-rotate shadow
// surface (#1869) — the caller parses each metric out of the returned text.
func fetchGuardPreCompactMetrics(ctx context.Context, metricsURL string, timeout time.Duration) (string, error) {
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
	return string(body), nil
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

// parseGuardPreCompactRelayArmed reads the relay would-rotate gauge (#1869) from
// the same /metrics text: armed reports whether the leg has armed (gauge > 0 =
// RELAY_ARMED, soft mark crossed), and present reports whether the gauge appeared
// at all. A missing gauge (older gateway, or a non-relay trace) returns
// (false, false) so the shadow seam stays silent for sessions with no relay
// driver rather than falsely logging "disarmed". Malformed values are skipped,
// matching the posture parser's fail-soft scan.
func parseGuardPreCompactRelayArmed(metrics string) (armed, present bool) {
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
		if name != guardPreCompactRelayMetricName && !strings.HasPrefix(name, guardPreCompactRelayMetricName+"{") {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		return value > 0, true
	}
	return false, false
}
