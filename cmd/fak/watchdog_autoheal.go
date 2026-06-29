package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const watchdogAutohealSchema = "fak.watchdog-autoheal.v1"

const (
	watchdogReasonAlive        = "WATCHDOG_ALREADY_ALIVE"
	watchdogReasonNotInstalled = "WATCHDOG_NOT_INSTALLED"
	watchdogReasonProbeFailed  = "WATCHDOG_PROBE_FAILED"
	watchdogReasonLeaseHeld    = "WATCHDOG_HEAL_IN_FLIGHT"
	watchdogReasonWarnOnly     = "WATCHDOG_AUTOHEAL_WARN_ONLY"
	watchdogReasonRestarted    = "WATCHDOG_RESTARTED"
	watchdogReasonFailed       = "WATCHDOG_RESTART_FAILED"
	watchdogReasonScheduled    = "WATCHDOG_RESTART_SCHEDULED"
	watchdogReasonDebounced    = "WATCHDOG_RESTART_DEBOUNCED"
	watchdogReasonExhausted    = "WATCHDOG_RESTART_EXHAUSTED"
	watchdogReasonPolicyBad    = "WATCHDOG_RESTART_POLICY_INVALID"
)

type watchdogAutohealMode int

const (
	watchdogAutohealOn watchdogAutohealMode = iota
	watchdogAutohealWarn
	watchdogAutohealOff
)

type watchdogService struct {
	ID       string
	Manager  string
	Unit     string
	UnitPath string
}

type watchdogProbe struct {
	Installed bool
	Alive     bool
	Detail    string
}

type watchdogAutohealSpec struct {
	watchdogService
	Probe   func(context.Context) (watchdogProbe, error)
	Restart func(context.Context) error
}

type watchdogAutohealOptions struct {
	Verb          string
	Mode          watchdogAutohealMode
	Specs         []watchdogAutohealSpec
	StateDir      string
	Clock         func() time.Time
	Sleep         func(time.Duration)
	LeaseTTL      time.Duration
	Debounce      time.Duration
	RestartPolicy watchdogRestartPolicy
}

type watchdogAutohealResult struct {
	Schema      string `json:"schema"`
	Verb        string `json:"verb"`
	ID          string `json:"id"`
	Manager     string `json:"manager"`
	Unit        string `json:"unit"`
	Action      string `json:"action"`
	Reason      string `json:"reason"`
	Summary     string `json:"summary,omitempty"`
	Attempt     uint64 `json:"attempt,omitempty"`
	Error       string `json:"error,omitempty"`
	RestartedAt int64  `json:"restarted_at_unix_nano,omitempty"`
}

type watchdogHealState struct {
	Schema                 string `json:"schema"`
	ID                     string `json:"id"`
	Attempts               uint64 `json:"attempts,omitempty"`
	LastFailureUnixNano    int64  `json:"last_failure_unix_nano,omitempty"`
	LastRestartUnixNano    int64  `json:"last_restart_unix_nano,omitempty"`
	LastReason             string `json:"last_reason,omitempty"`
	LastProbeAliveUnixNano int64  `json:"last_probe_alive_unix_nano,omitempty"`
}

type watchdogLease struct {
	Schema     string `json:"schema"`
	ID         string `json:"id"`
	Owner      string `json:"owner"`
	AcquiredAt int64  `json:"acquired_at_unix_nano"`
	ExpiresAt  int64  `json:"expires_at_unix_nano"`
}

type watchdogCommandRunner func(context.Context, string, ...string) (string, error)

type watchdogRestartPolicy struct {
	MaxAttempts uint64
	BaseDelay   time.Duration
	MaxDelay    time.Duration
	ResetAfter  time.Duration
}

type watchdogRestartDecision struct {
	Restart bool
	GiveUp  bool
	Reason  string
	Summary string
	Attempt uint64
	After   time.Duration
}

func watchdogAutohealOnStart(verb string) {
	mode := parseWatchdogAutohealMode(os.Getenv("FAK_WATCHDOG_AUTOHEAL"))
	if mode == watchdogAutohealOff {
		return
	}
	opts := defaultWatchdogAutohealOptions(verb, mode)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		results := runWatchdogAutoheal(ctx, opts)
		logWatchdogAutohealResults(os.Stderr, results)
	}()
}

func parseWatchdogAutohealMode(v string) watchdogAutohealMode {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "off", "0", "false", "no":
		return watchdogAutohealOff
	case "warn", "warning", "log", "log-only":
		return watchdogAutohealWarn
	default:
		return watchdogAutohealOn
	}
}

func defaultWatchdogAutohealOptions(verb string, mode watchdogAutohealMode) watchdogAutohealOptions {
	return watchdogAutohealOptions{
		Verb:     verb,
		Mode:     mode,
		Specs:    watchdogAutohealSpecsForGOOS(runtime.GOOS, watchdogRunCommand),
		StateDir: defaultWatchdogAutohealStateDir(),
		Clock:    time.Now,
		Sleep:    time.Sleep,
		LeaseTTL: 2 * time.Minute,
		Debounce: 10 * time.Minute,
		RestartPolicy: watchdogRestartPolicy{
			MaxAttempts: 3,
			BaseDelay:   250 * time.Millisecond,
			MaxDelay:    2 * time.Second,
		},
	}
}

func defaultWatchdogAutohealStateDir() string {
	if d := strings.TrimSpace(os.Getenv("FAK_WATCHDOG_AUTOHEAL_DIR")); d != "" {
		return d
	}
	if d, err := os.UserConfigDir(); err == nil && strings.TrimSpace(d) != "" {
		return filepath.Join(d, "fak", "watchdog-autoheal")
	}
	return filepath.Join(os.TempDir(), "fak-watchdog-autoheal")
}

func watchdogAutohealServicesForGOOS(goos string) []watchdogService {
	switch goos {
	case "windows":
		return []watchdogService{
			{ID: "fleet-resume-watchdog", Manager: "taskscheduler", Unit: "FleetResumeWatchdog"},
			{ID: "fleet-supervisor-watchdog", Manager: "taskscheduler", Unit: "FleetSupervisorWatchdog"},
			{ID: "fleet-dos-dispatch-watchdog", Manager: "taskscheduler", Unit: "FleetDOSDispatchWatchdog"},
		}
	case "darwin":
		home, _ := os.UserHomeDir()
		launchAgents := filepath.Join(home, "Library", "LaunchAgents")
		return []watchdogService{
			{ID: "fleet-dos-dispatch-watchdog", Manager: "launchd", Unit: "com.fleet.dispatch-supervisor", UnitPath: filepath.Join(launchAgents, "com.fleet.dispatch-supervisor.plist")},
			{ID: "fak-dogfood-fleet", Manager: "launchd", Unit: "com.fak.dogfood-fleet", UnitPath: filepath.Join(launchAgents, "com.fak.dogfood-fleet.plist")},
		}
	case "linux":
		return []watchdogService{
			{ID: "fleet-resume-watchdog", Manager: "systemd", Unit: "fleet-resume-watchdog.timer"},
			{ID: "fleet-supervisor-watchdog", Manager: "systemd", Unit: "fleet-supervisor-watchdog.timer"},
			{ID: "fleet-dos-dispatch-watchdog", Manager: "systemd", Unit: "fleet-dos-dispatch-watchdog.timer"},
		}
	default:
		return nil
	}
}

func watchdogAutohealSpecsForGOOS(goos string, run watchdogCommandRunner) []watchdogAutohealSpec {
	services := watchdogAutohealServicesForGOOS(goos)
	specs := make([]watchdogAutohealSpec, 0, len(services))
	for _, svc := range services {
		svc := svc
		spec := watchdogAutohealSpec{watchdogService: svc}
		switch svc.Manager {
		case "taskscheduler":
			spec.Probe = func(ctx context.Context) (watchdogProbe, error) {
				return probeScheduledTask(ctx, run, svc.Unit)
			}
			spec.Restart = func(ctx context.Context) error {
				_, err := run(ctx, "schtasks", "/Run", "/TN", svc.Unit)
				return err
			}
		case "launchd":
			spec.Probe = func(ctx context.Context) (watchdogProbe, error) {
				return probeLaunchd(ctx, run, svc.Unit, svc.UnitPath)
			}
			spec.Restart = func(ctx context.Context) error {
				return restartLaunchd(ctx, run, svc.Unit, svc.UnitPath)
			}
		case "systemd":
			spec.Probe = func(ctx context.Context) (watchdogProbe, error) {
				return probeSystemdUserUnit(ctx, run, svc.Unit)
			}
			spec.Restart = func(ctx context.Context) error {
				_, err := run(ctx, "systemctl", "--user", "restart", svc.Unit)
				return err
			}
		}
		specs = append(specs, spec)
	}
	return specs
}

func runWatchdogAutoheal(ctx context.Context, opts watchdogAutohealOptions) []watchdogAutohealResult {
	opts = normalizeWatchdogAutohealOptions(opts)
	if opts.Mode == watchdogAutohealOff || len(opts.Specs) == 0 {
		return nil
	}
	results := make([]watchdogAutohealResult, 0, len(opts.Specs))
	for _, spec := range opts.Specs {
		results = append(results, healOneWatchdog(ctx, opts, spec))
	}
	return results
}

func normalizeWatchdogAutohealOptions(opts watchdogAutohealOptions) watchdogAutohealOptions {
	if strings.TrimSpace(opts.Verb) == "" {
		opts.Verb = "fak"
	}
	if opts.Clock == nil {
		opts.Clock = time.Now
	}
	if opts.Sleep == nil {
		opts.Sleep = time.Sleep
	}
	if opts.LeaseTTL <= 0 {
		opts.LeaseTTL = 2 * time.Minute
	}
	if opts.Debounce <= 0 {
		opts.Debounce = 10 * time.Minute
	}
	if opts.RestartPolicy.MaxAttempts == 0 {
		opts.RestartPolicy = watchdogRestartPolicy{MaxAttempts: 3, BaseDelay: 250 * time.Millisecond, MaxDelay: 2 * time.Second}
	}
	return opts
}

func healOneWatchdog(ctx context.Context, opts watchdogAutohealOptions, spec watchdogAutohealSpec) watchdogAutohealResult {
	base := watchdogAutohealResult{
		Schema:  watchdogAutohealSchema,
		Verb:    opts.Verb,
		ID:      spec.ID,
		Manager: spec.Manager,
		Unit:    spec.Unit,
	}
	if spec.Probe == nil || spec.Restart == nil {
		base.Action = "probe_failed"
		base.Reason = watchdogReasonProbeFailed
		base.Summary = "watchdog autoheal spec is incomplete"
		return base
	}

	now := opts.Clock().UTC()
	probe, err := spec.Probe(ctx)
	if err != nil {
		base.Action = "probe_failed"
		base.Reason = watchdogReasonProbeFailed
		base.Error = err.Error()
		base.Summary = probe.Detail
		return base
	}
	st, _ := readWatchdogHealState(opts.StateDir, spec.ID)
	if probe.Alive {
		st = resetWatchdogAttemptsOnAlive(st, spec.ID, opts.RestartPolicy, now)
		_ = writeWatchdogHealState(opts.StateDir, st)
		base.Action = "noop"
		base.Reason = watchdogReasonAlive
		base.Summary = probe.Detail
		return base
	}
	if !probe.Installed {
		base.Action = "noop"
		base.Reason = watchdogReasonNotInstalled
		base.Summary = probe.Detail
		return base
	}
	if opts.Mode == watchdogAutohealWarn {
		base.Action = "warn"
		base.Reason = watchdogReasonWarnOnly
		base.Summary = "watchdog is installed but not active; warn-only mode suppressed restart"
		return base
	}

	if lastRestart := unixNanoTime(st.LastRestartUnixNano); !lastRestart.IsZero() && now.Sub(lastRestart) < opts.Debounce {
		base.Action = "debounced"
		base.Reason = watchdogReasonDebounced
		base.Summary = fmt.Sprintf("last restart was %s ago, inside debounce window %s", now.Sub(lastRestart).Round(time.Millisecond), opts.Debounce)
		base.RestartedAt = st.LastRestartUnixNano
		return base
	}

	release, ok, detail, err := acquireWatchdogHealLease(opts.StateDir, spec.ID, opts.LeaseTTL, now)
	if err != nil {
		base.Action = "lease_failed"
		base.Reason = watchdogReasonLeaseHeld
		base.Error = err.Error()
		return base
	}
	if !ok {
		base.Action = "in_flight"
		base.Reason = watchdogReasonLeaseHeld
		base.Summary = detail
		return base
	}
	defer release()

	st, _ = readWatchdogHealState(opts.StateDir, spec.ID)
	if lastRestart := unixNanoTime(st.LastRestartUnixNano); !lastRestart.IsZero() && now.Sub(lastRestart) < opts.Debounce {
		base.Action = "debounced"
		base.Reason = watchdogReasonDebounced
		base.Summary = fmt.Sprintf("last restart was %s ago, inside debounce window %s", now.Sub(lastRestart).Round(time.Millisecond), opts.Debounce)
		base.RestartedAt = st.LastRestartUnixNano
		return base
	}

	return restartWatchdogWithPolicy(ctx, opts, spec, st, base)
}

func restartWatchdogWithPolicy(ctx context.Context, opts watchdogAutohealOptions, spec watchdogAutohealSpec, st watchdogHealState, base watchdogAutohealResult) watchdogAutohealResult {
	attempts := st.Attempts
	lastFailure := unixNanoTime(st.LastFailureUnixNano)
	now := opts.Clock().UTC()
	if lastFailure.IsZero() {
		lastFailure = now.Add(-opts.RestartPolicy.BaseDelay)
	}

	for {
		decision := opts.RestartPolicy.Decide(attempts, lastFailure, now)
		if decision.GiveUp {
			st.ID = spec.ID
			st.Schema = watchdogAutohealSchema
			st.Attempts = attempts
			st.LastReason = decision.Reason
			_ = writeWatchdogHealState(opts.StateDir, st)
			base.Action = "give_up"
			base.Reason = decision.Reason
			base.Summary = decision.Summary
			base.Attempt = decision.Attempt
			return base
		}
		if decision.After > 0 {
			opts.Sleep(decision.After)
			now = opts.Clock().UTC()
		}
		if err := spec.Restart(ctx); err == nil {
			st = watchdogHealState{
				Schema:              watchdogAutohealSchema,
				ID:                  spec.ID,
				LastRestartUnixNano: now.UnixNano(),
				LastReason:          watchdogReasonRestarted,
			}
			_ = writeWatchdogHealState(opts.StateDir, st)
			base.Action = "restarted"
			base.Reason = watchdogReasonRestarted
			base.Summary = fmt.Sprintf("watchdog restart attempt %d/%d succeeded", decision.Attempt, opts.RestartPolicy.MaxAttempts)
			base.Attempt = decision.Attempt
			base.RestartedAt = st.LastRestartUnixNano
			return base
		} else {
			attempts = decision.Attempt
			lastFailure = now
			st.Schema = watchdogAutohealSchema
			st.ID = spec.ID
			st.Attempts = attempts
			st.LastFailureUnixNano = now.UnixNano()
			st.LastReason = watchdogReasonFailed
			_ = writeWatchdogHealState(opts.StateDir, st)
			base.Error = err.Error()
		}
	}
}

func resetWatchdogAttemptsOnAlive(st watchdogHealState, id string, policy watchdogRestartPolicy, now time.Time) watchdogHealState {
	st.Schema = watchdogAutohealSchema
	st.ID = id
	st.LastProbeAliveUnixNano = now.UnixNano()
	if st.Attempts > 0 {
		st.Attempts = policy.AttemptsAfterSuccess(st.Attempts, unixNanoTime(st.LastRestartUnixNano), now)
		if st.Attempts == 0 {
			st.LastFailureUnixNano = 0
			st.LastReason = watchdogReasonAlive
		}
	}
	return st
}

func (p watchdogRestartPolicy) Decide(attempts uint64, lastFailure, now time.Time) watchdogRestartDecision {
	if err := p.Validate(); err != nil {
		return watchdogRestartDecision{GiveUp: true, Reason: watchdogReasonPolicyBad, Attempt: attempts, Summary: "invalid restart policy: " + err.Error()}
	}
	if attempts >= p.MaxAttempts {
		return watchdogRestartDecision{
			GiveUp:  true,
			Reason:  watchdogReasonExhausted,
			Attempt: attempts,
			Summary: fmt.Sprintf("restart attempts exhausted (%d/%d)", attempts, p.MaxAttempts),
		}
	}
	lastFailure = lastFailure.UTC()
	now = now.UTC()
	if lastFailure.IsZero() {
		lastFailure = now
	}
	delay := p.backoffDelay(attempts)
	restartAt := lastFailure.Add(delay)
	after := restartAt.Sub(now)
	if after < 0 {
		after = 0
	}
	reason := watchdogReasonScheduled
	if now.After(lastFailure) && now.Before(restartAt) {
		reason = watchdogReasonDebounced
	}
	attempt := attempts + 1
	return watchdogRestartDecision{
		Restart: true,
		Reason:  reason,
		Attempt: attempt,
		After:   after,
		Summary: fmt.Sprintf("restart attempt %d/%d scheduled after %s", attempt, p.MaxAttempts, after),
	}
}

func (p watchdogRestartPolicy) Validate() error {
	if p.MaxAttempts == 0 {
		return fmt.Errorf("max_attempts must be > 0")
	}
	if p.BaseDelay <= 0 {
		return fmt.Errorf("base_delay must be > 0")
	}
	if p.MaxDelay <= 0 {
		return fmt.Errorf("max_delay must be > 0")
	}
	if p.MaxDelay < p.BaseDelay {
		return fmt.Errorf("max_delay %s is below base_delay %s", p.MaxDelay, p.BaseDelay)
	}
	if p.ResetAfter < 0 {
		return fmt.Errorf("reset_after must be >= 0")
	}
	return nil
}

func (p watchdogRestartPolicy) AttemptsAfterSuccess(attempts uint64, started, ended time.Time) uint64 {
	if attempts == 0 {
		return 0
	}
	if ended.Before(started) {
		return attempts
	}
	if p.ResetAfter <= 0 || ended.Sub(started) >= p.ResetAfter {
		return 0
	}
	return attempts
}

func (p watchdogRestartPolicy) backoffDelay(attempts uint64) time.Duration {
	delay := p.BaseDelay
	for i := uint64(0); i < attempts; i++ {
		if delay >= p.MaxDelay {
			return p.MaxDelay
		}
		if delay > p.MaxDelay/2 {
			return p.MaxDelay
		}
		delay *= 2
	}
	if delay > p.MaxDelay {
		return p.MaxDelay
	}
	return delay
}

func readWatchdogHealState(dir, id string) (watchdogHealState, error) {
	b, err := os.ReadFile(watchdogHealStatePath(dir, id))
	if errors.Is(err, os.ErrNotExist) {
		return watchdogHealState{Schema: watchdogAutohealSchema, ID: id}, nil
	}
	if err != nil {
		return watchdogHealState{}, err
	}
	var st watchdogHealState
	if err := json.Unmarshal(b, &st); err != nil {
		return watchdogHealState{Schema: watchdogAutohealSchema, ID: id}, nil
	}
	if st.ID == "" {
		st.ID = id
	}
	if st.Schema == "" {
		st.Schema = watchdogAutohealSchema
	}
	return st, nil
}

func writeWatchdogHealState(dir string, st watchdogHealState) error {
	if strings.TrimSpace(dir) == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(st)
	if err != nil {
		return err
	}
	return os.WriteFile(watchdogHealStatePath(dir, st.ID), append(b, '\n'), 0o644)
}

func acquireWatchdogHealLease(dir, id string, ttl time.Duration, now time.Time) (func(), bool, string, error) {
	if strings.TrimSpace(dir) == "" {
		return func() {}, true, "", nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, false, "", err
	}
	path := watchdogHealLeasePath(dir, id)
	payload := watchdogLease{
		Schema:     watchdogAutohealSchema,
		ID:         id,
		Owner:      fmt.Sprintf("%d", os.Getpid()),
		AcquiredAt: now.UnixNano(),
		ExpiresAt:  now.Add(ttl).UnixNano(),
	}
	data, _ := json.Marshal(payload)
	for i := 0; i < 2; i++ {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			if _, werr := f.Write(append(data, '\n')); werr != nil {
				_ = f.Close()
				_ = os.Remove(path)
				return nil, false, "", werr
			}
			_ = f.Close()
			return func() { _ = os.Remove(path) }, true, "", nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, false, "", err
		}
		held, detail := liveWatchdogLease(path, now)
		if held {
			return nil, false, detail, nil
		}
		_ = os.Remove(path)
	}
	return nil, false, "lost lease create race", nil
}

func liveWatchdogLease(path string, now time.Time) (bool, string) {
	b, err := os.ReadFile(path)
	if err != nil {
		return true, "lease exists but could not be read"
	}
	var l watchdogLease
	if err := json.Unmarshal(b, &l); err != nil {
		return false, "stale/corrupt lease"
	}
	if l.ExpiresAt <= 0 || now.UnixNano() >= l.ExpiresAt {
		return false, "expired lease"
	}
	remaining := time.Duration(l.ExpiresAt - now.UnixNano()).Round(time.Millisecond)
	return true, "another fak start is healing this watchdog; lease expires in " + remaining.String()
}

func watchdogHealStatePath(dir, id string) string {
	return filepath.Join(dir, watchdogSafeFilePart(id)+".json")
}

func watchdogHealLeasePath(dir, id string) string {
	return filepath.Join(dir, watchdogSafeFilePart(id)+".lock")
}

func watchdogSafeFilePart(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "watchdog"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "watchdog"
	}
	return b.String()
}

func unixNanoTime(ns int64) time.Time {
	if ns <= 0 {
		return time.Time{}
	}
	return time.Unix(0, ns).UTC()
}

func logWatchdogAutohealResults(w io.Writer, results []watchdogAutohealResult) {
	for _, r := range results {
		if !watchdogAutohealShouldLog(r) {
			continue
		}
		b, err := json.Marshal(r)
		if err != nil {
			continue
		}
		fmt.Fprintf(w, "fak watchdog-autoheal: %s\n", b)
	}
}

func watchdogAutohealShouldLog(r watchdogAutohealResult) bool {
	switch r.Action {
	case "restarted", "warn", "give_up", "probe_failed", "lease_failed":
		return true
	default:
		return false
	}
}

func watchdogRunCommand(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func probeScheduledTask(ctx context.Context, run watchdogCommandRunner, task string) (watchdogProbe, error) {
	script := fmt.Sprintf("$t=Get-ScheduledTask -TaskName %s -ErrorAction SilentlyContinue; if ($null -eq $t) { exit 3 }; $t.State", psSingleQuoted(task))
	out, err := run(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	state := strings.TrimSpace(out)
	if err != nil && state == "" {
		return watchdogProbe{Installed: false, Alive: false, Detail: "scheduled task not installed"}, nil
	}
	alive := strings.EqualFold(state, "Ready") || strings.EqualFold(state, "Running")
	return watchdogProbe{Installed: true, Alive: alive, Detail: "scheduled task state=" + state}, nil
}

func probeLaunchd(ctx context.Context, run watchdogCommandRunner, label, plist string) (watchdogProbe, error) {
	if _, err := run(ctx, "launchctl", "list", label); err == nil {
		return watchdogProbe{Installed: true, Alive: true, Detail: "launchd job loaded"}, nil
	}
	if strings.TrimSpace(plist) != "" {
		if _, err := os.Stat(plist); err == nil {
			return watchdogProbe{Installed: true, Alive: false, Detail: "launchd plist exists but job is not loaded"}, nil
		}
	}
	return watchdogProbe{Installed: false, Alive: false, Detail: "launchd job not installed"}, nil
}

func restartLaunchd(ctx context.Context, run watchdogCommandRunner, label, plist string) error {
	target := fmt.Sprintf("gui/%d/%s", os.Getuid(), label)
	if _, err := run(ctx, "launchctl", "list", label); err == nil {
		_, err = run(ctx, "launchctl", "kickstart", "-k", target)
		return err
	}
	if strings.TrimSpace(plist) == "" {
		return fmt.Errorf("launchd plist path is empty for %s", label)
	}
	if _, err := os.Stat(plist); err != nil {
		return err
	}
	_, err := run(ctx, "launchctl", "bootstrap", fmt.Sprintf("gui/%d", os.Getuid()), plist)
	if err != nil {
		return err
	}
	_, err = run(ctx, "launchctl", "kickstart", "-k", target)
	return err
}

func probeSystemdUserUnit(ctx context.Context, run watchdogCommandRunner, unit string) (watchdogProbe, error) {
	enabledOut, enabledErr := run(ctx, "systemctl", "--user", "is-enabled", unit)
	activeOut, activeErr := run(ctx, "systemctl", "--user", "is-active", unit)
	enabled := strings.TrimSpace(enabledOut)
	active := strings.TrimSpace(activeOut)
	installed := enabledErr == nil || enabled == "enabled" || enabled == "static" || enabled == "linked" || enabled == "disabled"
	if !installed {
		return watchdogProbe{Installed: false, Alive: false, Detail: "systemd user unit not installed"}, nil
	}
	return watchdogProbe{Installed: true, Alive: activeErr == nil && active == "active", Detail: "systemd unit active=" + active + " enabled=" + enabled}, nil
}

func psSingleQuoted(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
