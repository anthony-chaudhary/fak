package fleetpane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

const (
	Schema          = "fleet-control-pane/1"
	ConfigSchema    = "fleet-control-pane.config/1"
	LoopListSchema  = "fleet-control-pane.loop-list/1"
	LoopCheckSchema = "fleet-control-pane.loop-check/1"
	LoopAuditSchema = "fleet-control-pane.loop-audit/1"
	FleetSchema     = "fleet-control-pane.fleet/1"

	LoopAuditSelfName = "gardening-loops-audit"
)

var (
	healthyLoopValues = map[string]bool{
		"OK": true, "HEALTHY": true, "RUNNING": true, "ALIVE": true,
		"READY": true, "PASS": true, "GREEN": true,
	}
	actionLoopValues = map[string]bool{
		"ACTION": true, "DEAD": true, "STALLED": true, "UNHEALTHY": true,
		"FAIL": true, "FAILED": true, "ERROR": true, "RED": true, "MISSING": true,
	}
	loopBrokenStates = map[string]bool{"UNAVAILABLE": true, "TIMEOUT": true}
)

type Runner interface {
	Run(ctx context.Context, req RunRequest) RunResult
	LookPath(file string) (string, error)
}

type RunRequest struct {
	Args    []string
	Cwd     string
	Env     map[string]string
	Timeout time.Duration
}

type RunResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
	TimedOut bool
	Err      error
}

type OSRunner struct{}

func (OSRunner) LookPath(file string) (string, error) {
	if file == "" {
		return "", exec.ErrNotFound
	}
	if _, err := os.Stat(file); err == nil {
		return file, nil
	}
	return exec.LookPath(file)
}

func (OSRunner) Run(ctx context.Context, req RunRequest) RunResult {
	if len(req.Args) == 0 {
		return RunResult{ExitCode: -1, Err: errors.New("empty command")}
	}
	if req.Timeout <= 0 {
		req.Timeout = 30 * time.Second
	}
	runCtx, cancel := context.WithTimeout(ctx, req.Timeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, req.Args[0], req.Args[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	// OSRunner.Run executes an arbitrary caller-supplied argv (a loop status_cmd, the
	// supervisor/monitor status command, a build/`go test`) that can fork its own
	// descendant tree. Bare CommandContext ctx-cancel is single-PID (windowgate only
	// hides the window; it is not a tree kill), so a Timeout/cancel of a hung command
	// orphans that subtree (#3108, sibling of #2989/#3106). Tree-kill on cancel and
	// bound the reap, mirroring internal/nightrun/run.go and the other generic-runner
	// sites (witness/edittx/swebench).
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
	if req.Cwd != "" {
		cmd.Dir = req.Cwd
	}
	if len(req.Env) > 0 {
		env := os.Environ()
		for key, val := range req.Env {
			env = append(env, key+"="+val)
		}
		cmd.Env = env
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	out := RunResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		ExitCode: 0,
		Err:      err,
		TimedOut: runCtx.Err() == context.DeadlineExceeded,
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		out.ExitCode = exitErr.ExitCode()
	} else if err != nil {
		out.ExitCode = -1
	}
	return out
}

type Config struct {
	Root             string
	Python           string
	UserHome         string
	JobDir           string
	ClaudeExe        string
	RegistryDir      string
	MachineDir       string
	MachineID        string
	WatchdogLogDir   string
	GitRemote        string
	Target           int
	SessionWindowH   float64
	RefreshTimeoutS  int
	MachineStaleMin  float64
	Loops            map[string]LoopSpec
	SupervisorStatus []string
	MonitorStatus    []string
	ControlTick      map[string]any
	Watchdogs        map[string]map[string]any
	Paths            map[string]string
	LocalExists      bool
}

type LoopSpec struct {
	Enabled         bool
	StatusCmd       []string
	RecoverCmd      []string
	AutoRecover     bool
	Action          string
	TimeoutS        int
	RecoverTimeoutS int
	Cwd             string
	OKReturnCodes   []int
}

type SourceEntry struct {
	Source        string `json:"source"`
	Path          string `json:"path"`
	DisplayPath   string `json:"display_path"`
	Enabled       bool   `json:"enabled"`
	HasStatusCmd  bool   `json:"has_status_cmd"`
	HasRecoverCmd bool   `json:"has_recover_cmd"`
	AutoRecover   bool   `json:"auto_recover"`
}

type Readiness struct {
	OK      bool     `json:"ok"`
	Detail  string   `json:"detail"`
	Command []string `json:"command,omitempty"`
}

type LoopListItem struct {
	Name            string        `json:"name"`
	Enabled         bool          `json:"enabled"`
	Ready           bool          `json:"ready"`
	Source          string        `json:"source"`
	Sources         []SourceEntry `json:"sources"`
	Overridden      bool          `json:"overridden"`
	StatusCmd       []string      `json:"status_cmd"`
	StatusReady     Readiness     `json:"status_ready"`
	RecoverCmd      []string      `json:"recover_cmd"`
	RecoverReady    Readiness     `json:"recover_ready"`
	AutoRecover     bool          `json:"auto_recover"`
	Action          string        `json:"action"`
	Cwd             string        `json:"cwd"`
	TimeoutS        int           `json:"timeout_s,omitempty"`
	RecoverTimeoutS int           `json:"recover_timeout_s,omitempty"`
}

type LoopListDoc struct {
	Schema       string            `json:"schema"`
	GeneratedUTC string            `json:"generated_utc"`
	OK           bool              `json:"ok"`
	Count        int               `json:"count"`
	Enabled      int               `json:"enabled"`
	Disabled     int               `json:"disabled"`
	Blocked      int               `json:"blocked"`
	Paths        map[string]string `json:"paths"`
	Loops        []LoopListItem    `json:"loops"`
	Commands     []string          `json:"commands,omitempty"`
}

type LoopCheck struct {
	Name          string         `json:"name"`
	Enabled       bool           `json:"enabled"`
	Cmd           []string       `json:"cmd,omitempty"`
	AutoRecover   bool           `json:"auto_recover,omitempty"`
	HasRecoverCmd bool           `json:"has_recover_cmd,omitempty"`
	Action        string         `json:"action,omitempty"`
	State         string         `json:"state"`
	Reason        string         `json:"reason,omitempty"`
	ReturnCode    *int           `json:"returncode,omitempty"`
	Payload       map[string]any `json:"payload,omitempty"`
	Detail        string         `json:"detail,omitempty"`
	Stdout        string         `json:"stdout,omitempty"`
	Stderr        string         `json:"stderr,omitempty"`
}

type LoopCheckDoc struct {
	Schema       string    `json:"schema"`
	GeneratedUTC string    `json:"generated_utc"`
	OK           bool      `json:"ok"`
	Verdict      string    `json:"verdict"`
	NeedsAction  bool      `json:"needs_action"`
	Loop         string    `json:"loop"`
	Recover      bool      `json:"recover"`
	Apply        bool      `json:"apply"`
	Reason       string    `json:"reason,omitempty"`
	Available    []string  `json:"available,omitempty"`
	Check        LoopCheck `json:"check"`
	Recovery     *Recovery `json:"recovery"`
}

type Recovery struct {
	Name      string   `json:"name"`
	DryRun    bool     `json:"dry_run"`
	Command   []string `json:"command"`
	LoopState string   `json:"loop_state"`
	OK        bool     `json:"ok"`
	Skipped   bool     `json:"skipped"`
	Reason    string   `json:"reason,omitempty"`
}

type LoopAuditRow struct {
	Name       string `json:"name"`
	Bucket     string `json:"bucket"`
	State      string `json:"state"`
	Detail     string `json:"detail"`
	ActionHint string `json:"action_hint"`
	ReturnCode *int   `json:"returncode,omitempty"`
}

type LoopAuditDoc struct {
	Schema       string         `json:"schema"`
	GeneratedUTC string         `json:"generated_utc"`
	OK           bool           `json:"ok"`
	Counts       map[string]int `json:"counts"`
	Loops        []LoopAuditRow `json:"loops"`
	Requested    []string       `json:"requested"`
	Missing      []string       `json:"missing"`
}

type StatusDoc struct {
	Schema       string         `json:"schema"`
	GeneratedUTC string         `json:"generated_utc"`
	AppVersion   string         `json:"app_version"`
	Machine      map[string]any `json:"machine"`
	Config       map[string]any `json:"config"`
	Refresh      any            `json:"refresh"`
	Git          map[string]any `json:"git"`
	Registry     map[string]any `json:"registry"`
	ControlTick  map[string]any `json:"control_tick"`
	Watchdogs    map[string]any `json:"watchdogs"`
	Loops        map[string]any `json:"loops"`
	Supervisor   map[string]any `json:"supervisor"`
	WorkerHealth WorkerHealth   `json:"worker_health"`
	SetupPlan    map[string]any `json:"setup_plan"`
	Actions      []string       `json:"actions"`
	Verdict      string         `json:"verdict"`
}

type FleetDoc struct {
	Schema           string           `json:"schema"`
	GeneratedUTC     string           `json:"generated_utc"`
	MachineDir       string           `json:"machine_dir"`
	StaleMin         float64          `json:"stale_min"`
	IncludeLiveLocal bool             `json:"include_live_local"`
	RefreshLiveLocal bool             `json:"refresh_live_local"`
	CurrentVersion   string           `json:"current_version"`
	Written          any              `json:"written"`
	Verdict          string           `json:"verdict"`
	States           map[string]int   `json:"states"`
	Versions         map[string]int   `json:"versions"`
	Totals           map[string]int   `json:"totals"`
	Machines         []map[string]any `json:"machines"`
}

type Options struct {
	Runner Runner
	Now    func() time.Time
}

func (o Options) runner() Runner {
	if o.Runner != nil {
		return o.Runner
	}
	return OSRunner{}
}

func (o Options) now() time.Time {
	if o.Now != nil {
		return o.Now().UTC()
	}
	return time.Now().UTC()
}

func GeneratedUTC(opts Options) string {
	return opts.now().Format(time.RFC3339)
}

func LoopList(cfg Config, opts Options) LoopListDoc {
	runner := opts.runner()
	sources := SourceEntries(cfg.Root)
	names := loopNames(cfg.Loops)
	loops := make([]LoopListItem, 0, len(names))
	enabledCount := 0
	blocked := 0
	for _, name := range names {
		spec := cfg.Loops[name]
		entries := sources[name]
		source := "effective"
		if len(entries) > 0 {
			source = entries[len(entries)-1].Source
		}
		statusCmd := ExpandCmd(spec.StatusCmd, cfg)
		recoverCmd := ExpandCmd(spec.RecoverCmd, cfg)
		statusReady := CommandReadiness(statusCmd, cfg.Root, runner)
		recoverReady := Readiness{OK: !spec.AutoRecover, Detail: "recover command is not configured"}
		if len(recoverCmd) > 0 {
			recoverReady = CommandReadiness(recoverCmd, cfg.Root, runner)
		}
		ready := !spec.Enabled || (statusReady.OK && (!spec.AutoRecover || recoverReady.OK))
		if spec.Enabled {
			enabledCount++
			if !ready {
				blocked++
			}
		}
		loops = append(loops, LoopListItem{
			Name:            name,
			Enabled:         spec.Enabled,
			Ready:           ready,
			Source:          source,
			Sources:         entries,
			Overridden:      len(entries) > 1,
			StatusCmd:       statusCmd,
			StatusReady:     statusReady,
			RecoverCmd:      recoverCmd,
			RecoverReady:    recoverReady,
			AutoRecover:     spec.AutoRecover,
			Action:          spec.Action,
			Cwd:             spec.Cwd,
			TimeoutS:        spec.TimeoutS,
			RecoverTimeoutS: spec.RecoverTimeoutS,
		})
	}
	doc := LoopListDoc{
		Schema:       LoopListSchema,
		GeneratedUTC: GeneratedUTC(opts),
		OK:           blocked == 0,
		Count:        len(loops),
		Enabled:      enabledCount,
		Disabled:     len(loops) - enabledCount,
		Blocked:      blocked,
		Paths:        cfg.Paths,
		Loops:        loops,
	}
	if enabledCount == 0 {
		doc.Commands = []string{loopScaffoldTemplateCommand(cfg), loopAddTemplateCommand(cfg)}
	}
	return doc
}

func CommandReadiness(cmd []string, root string, runner Runner) Readiness {
	if len(cmd) == 0 {
		return Readiness{OK: false, Detail: "command is not configured"}
	}
	var missing []string
	if !commandExists(cmd[0], runner) {
		missing = append(missing, "command not found: "+cmd[0])
	}
	for _, part := range cmd[1:] {
		if commandInputMissing(part, root) {
			missing = append(missing, "missing input: "+part)
		}
	}
	detail := displayCommand(cmd)
	if len(missing) > 0 {
		if detail != "" {
			detail += "; "
		}
		detail += strings.Join(missing, "; ")
	}
	return Readiness{OK: len(missing) == 0, Detail: detail, Command: cmd}
}

func commandExists(command string, runner Runner) bool {
	if command == "" {
		return false
	}
	if _, err := os.Stat(command); err == nil {
		return true
	}
	if runner == nil {
		runner = OSRunner{}
	}
	_, err := runner.LookPath(command)
	return err == nil
}

func commandInputMissing(part, root string) bool {
	if part == "" || strings.HasPrefix(part, "-") {
		return false
	}
	ext := strings.ToLower(filepath.Ext(part))
	if ext != ".py" && ext != ".ps1" && ext != ".sh" && ext != ".cmd" && ext != ".bat" {
		return false
	}
	path := resolvePathString(part, root)
	_, err := os.Stat(path)
	return err != nil
}

func CollectLoop(ctx context.Context, name string, spec LoopSpec, cfg Config, opts Options) LoopCheck {
	if !spec.Enabled {
		return LoopCheck{Name: name, Enabled: false, State: "SKIPPED", Reason: "disabled"}
	}
	runner := opts.runner()
	cmd := ExpandCmd(spec.StatusCmd, cfg)
	base := LoopCheck{
		Name:          name,
		Enabled:       true,
		Cmd:           cmd,
		AutoRecover:   spec.AutoRecover,
		HasRecoverCmd: len(spec.RecoverCmd) > 0,
		Action:        spec.Action,
	}
	if len(cmd) == 0 {
		base.State = "UNAVAILABLE"
		base.Reason = "no status_cmd configured"
		return base
	}
	if !commandExists(cmd[0], runner) {
		base.State = "UNAVAILABLE"
		base.Reason = "command not found: " + cmd[0]
		return base
	}
	timeout := time.Duration(spec.TimeoutS) * time.Second
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	res := runner.Run(ctx, RunRequest{
		Args:    cmd,
		Cwd:     loopCwd(spec, cfg),
		Env:     envForTools(cfg),
		Timeout: timeout,
	})
	if res.TimedOut {
		base.State = "TIMEOUT"
		base.Reason = fmt.Sprintf("status command timed out after %ds", int(timeout/time.Second))
		return base
	}
	if res.Err != nil && res.ExitCode < 0 {
		base.State = "UNAVAILABLE"
		base.Reason = res.Err.Error()
		return base
	}
	state, payload, detail := ClassifyLoopStatus(res, spec)
	code := res.ExitCode
	base.State = state
	base.ReturnCode = &code
	base.Payload = payload
	base.Detail = detail
	base.Stdout = tail(strings.TrimSpace(res.Stdout), 1000)
	base.Stderr = tail(strings.TrimSpace(res.Stderr), 1000)
	return base
}

func ClassifyLoopStatus(res RunResult, spec LoopSpec) (string, map[string]any, string) {
	payload := ReadJSONFromText(res.Stdout)
	if !expectedReturnCodes(spec)[res.ExitCode] {
		detail := strings.TrimSpace(res.Stderr)
		if detail == "" {
			detail = strings.TrimSpace(res.Stdout)
		}
		if detail == "" {
			detail = fmt.Sprintf("returncode %d", res.ExitCode)
		}
		return "ACTION", payload, tail(detail, 1000)
	}
	if len(payload) > 0 {
		if ok, exists := payload["ok"].(bool); exists {
			return okState(ok), payload, stringValueFirst(payload, "reason", "detail")
		}
		for _, key := range []string{"verdict", "status", "state", "health"} {
			value, exists := payload[key]
			if !exists || value == nil {
				continue
			}
			raw := fmt.Sprint(value)
			folded := strings.ToUpper(raw)
			if healthyLoopValues[folded] {
				return "OK", payload, key + "=" + raw
			}
			if actionLoopValues[folded] {
				return "ACTION", payload, key + "=" + raw
			}
			return "UNKNOWN", payload, key + "=" + raw
		}
		return "UNKNOWN", payload, "JSON did not include ok/verdict/status/state/health"
	}
	if strings.TrimSpace(res.Stdout) != "" {
		return "UNKNOWN", payload, tail(strings.TrimSpace(res.Stdout), 1000)
	}
	return "OK", payload, ""
}

func LoopCheckPlan(ctx context.Context, cfg Config, name string, recover bool, apply bool, opts Options) LoopCheckDoc {
	loopName := strings.TrimSpace(name)
	available := loopNames(cfg.Loops)
	spec, ok := cfg.Loops[loopName]
	if loopName == "" || !ok {
		reason := "unknown loop"
		if loopName == "" {
			reason = "loop name is required"
		}
		return LoopCheckDoc{
			Schema:       LoopCheckSchema,
			GeneratedUTC: GeneratedUTC(opts),
			OK:           false,
			Verdict:      "UNKNOWN",
			NeedsAction:  true,
			Loop:         loopName,
			Recover:      recover,
			Apply:        apply,
			Reason:       reason,
			Available:    available,
			Check:        LoopCheck{},
			Recovery:     nil,
		}
	}
	check := CollectLoop(ctx, loopName, spec, cfg, opts)
	needsAction := LoopCheckNeedsAction(check)
	recovery := (*Recovery)(nil)
	if recover || apply {
		recovery = RecoveryAction(loopName, spec, check, cfg, !apply, true)
	}
	fatal := check.State == "UNAVAILABLE" || check.State == "TIMEOUT"
	okDoc := !fatal && (recovery == nil || recovery.OK)
	verdict := "OK"
	if needsAction {
		verdict = "ACTION"
	}
	return LoopCheckDoc{
		Schema:       LoopCheckSchema,
		GeneratedUTC: GeneratedUTC(opts),
		OK:           okDoc,
		Verdict:      verdict,
		NeedsAction:  needsAction,
		Loop:         loopName,
		Recover:      recover,
		Apply:        apply,
		Check:        check,
		Recovery:     recovery,
	}
}

func RecoveryAction(name string, spec LoopSpec, check LoopCheck, cfg Config, dryRun bool, includeNotNeeded bool) *Recovery {
	recoverCmd := ExpandCmd(spec.RecoverCmd, cfg)
	action := &Recovery{Name: "loop:" + name, DryRun: dryRun, Command: recoverCmd, LoopState: check.State}
	if check.State == "OK" || check.State == "SKIPPED" {
		if includeNotNeeded {
			action.OK = true
			action.Skipped = true
			action.Reason = "loop state " + check.State + "; recovery not needed"
			return action
		}
		return nil
	}
	if len(recoverCmd) == 0 {
		action.OK = false
		action.Skipped = true
		action.Reason = "recover_cmd not configured"
		return action
	}
	action.OK = false
	action.Skipped = true
	action.Reason = "native fleetpane does not apply recovery commands yet"
	return action
}

func LoopAudit(ctx context.Context, cfg Config, names []string, opts Options) LoopAuditDoc {
	requestedSet := map[string]bool{}
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name != "" {
			requestedSet[name] = true
		}
	}
	var requested []string
	for name := range requestedSet {
		requested = append(requested, name)
	}
	sort.Strings(requested)

	var selected []string
	for _, name := range loopNames(cfg.Loops) {
		spec := cfg.Loops[name]
		if !spec.Enabled || name == LoopAuditSelfName {
			continue
		}
		if len(requestedSet) > 0 && !requestedSet[name] {
			continue
		}
		selected = append(selected, name)
	}
	counts := map[string]int{"healthy": 0, "action": 0, "broken": 0}
	rows := make([]LoopAuditRow, 0, len(selected))
	for _, name := range selected {
		spec := cfg.Loops[name]
		check := CollectLoop(ctx, name, spec, cfg, opts)
		bucket := "healthy"
		if loopBrokenStates[check.State] {
			bucket = "broken"
		} else if LoopCheckNeedsAction(check) {
			bucket = "action"
		}
		counts[bucket]++
		rows = append(rows, LoopAuditRow{
			Name:       name,
			Bucket:     bucket,
			State:      check.State,
			Detail:     LoopAuditDetail(check),
			ActionHint: spec.Action,
			ReturnCode: check.ReturnCode,
		})
	}
	counts["total"] = len(rows)
	found := map[string]bool{}
	for _, name := range selected {
		found[name] = true
	}
	var missing []string
	for _, name := range requested {
		if !found[name] {
			missing = append(missing, name)
		}
	}
	return LoopAuditDoc{
		Schema:       LoopAuditSchema,
		GeneratedUTC: GeneratedUTC(opts),
		OK:           counts["broken"] == 0,
		Counts:       counts,
		Loops:        rows,
		Requested:    requested,
		Missing:      missing,
	}
}

func LoopAuditDetail(check LoopCheck) string {
	if len(check.Payload) > 0 {
		for _, key := range []string{"reason", "summary", "verdict", "why", "headline", "message"} {
			if value, ok := check.Payload[key].(string); ok && strings.TrimSpace(value) != "" {
				return tail(strings.TrimSpace(value), 200)
			}
		}
	}
	raw := check.Detail
	if raw == "" {
		raw = check.Reason
	}
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return tail(line, 200)
		}
	}
	return ""
}

func CollectStatus(ctx context.Context, cfg Config, opts Options) StatusDoc {
	registry := readJSONMap(filepath.Join(cfg.RegistryDir, "sessions.json"))
	loopsDoc := collectLoops(ctx, cfg, opts, map[string]bool{})
	status := StatusDoc{
		Schema:       Schema,
		GeneratedUTC: GeneratedUTC(opts),
		AppVersion:   AppVersion(cfg.Root),
		Machine: map[string]any{
			"id":       MachineID(cfg),
			"host":     hostName(),
			"platform": runtime.GOOS + "/" + runtime.GOARCH,
			"go":       runtime.Version(),
		},
		Config: map[string]any{
			"root":                cfg.Root,
			"local_config":        cfg.Paths["local"],
			"local_config_exists": cfg.LocalExists,
			"registry_dir":        cfg.RegistryDir,
			"machine_dir":         cfg.MachineDir,
			"user_home":           cfg.UserHome,
			"job_dir":             cfg.JobDir,
			"target":              cfg.Target,
			"session_window_h":    cfg.SessionWindowH,
		},
		Refresh:      nil,
		Git:          collectGit(ctx, cfg, opts),
		Registry:     SummarizeRegistry(registry, opts),
		ControlTick:  collectControlTick(cfg),
		Watchdogs:    collectWatchdogs(cfg),
		Loops:        loopsDoc,
		Supervisor:   collectSupervisor(ctx, cfg, opts),
		WorkerHealth: collectWorkerHealth(ctx, cfg, opts),
		SetupPlan:    map[string]any{"ok": true, "steps": []any{}, "note": "native fleetpane read-only subset"},
	}
	status.Actions = recommendedActions(status)
	status.Verdict = "OK"
	if len(status.Actions) > 0 {
		status.Verdict = "ACTION"
	}
	return status
}

func collectLoops(ctx context.Context, cfg Config, opts Options, skip map[string]bool) map[string]any {
	names := loopNames(cfg.Loops)
	checks := make([]LoopCheck, 0, len(names))
	enabled := 0
	for _, name := range names {
		spec := cfg.Loops[name]
		if !spec.Enabled || skip[name] {
			continue
		}
		enabled++
		checks = append(checks, CollectLoop(ctx, name, spec, cfg, opts))
	}
	states := map[string]int{}
	for _, check := range checks {
		states[check.State]++
	}
	out := map[string]any{
		"count":      len(checks),
		"configured": len(names),
		"enabled":    enabled,
		"disabled":   len(names) - enabled,
		"states":     states,
		"checks":     checks,
	}
	if enabled == 0 {
		out["commands"] = []string{loopScaffoldTemplateCommand(cfg), loopAddTemplateCommand(cfg)}
	}
	return out
}

func collectGit(ctx context.Context, cfg Config, opts Options) map[string]any {
	if _, err := os.Stat(filepath.Join(cfg.Root, ".git")); err != nil {
		return map[string]any{"available": false, "reason": "not a git repo"}
	}
	runner := opts.runner()
	if !commandExists("git", runner) {
		return map[string]any{"available": false, "reason": "git not on PATH"}
	}
	runGit := func(args ...string) RunResult {
		return runner.Run(ctx, RunRequest{
			Args:    append([]string{"git"}, args...),
			Cwd:     cfg.Root,
			Timeout: 10 * time.Second,
		})
	}
	branch := strings.TrimSpace(runGit("rev-parse", "--abbrev-ref", "HEAD").Stdout)
	head := strings.TrimSpace(runGit("rev-parse", "--short=12", "HEAD").Stdout)
	status := runGit("status", "--porcelain=v1").Stdout
	counts := map[string]int{"modified": 0, "deleted": 0, "untracked": 0, "unmerged": 0}
	var entries []string
	for _, line := range strings.Split(status, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		entries = append(entries, line)
		if strings.HasPrefix(line, "??") {
			counts["untracked"]++
			continue
		}
		if len(line) >= 2 && (line[0] == 'U' || line[1] == 'U' || line[:2] == "AA" || line[:2] == "DD") {
			counts["unmerged"]++
			continue
		}
		if strings.Contains(line[:mathx.MinInt(2, len(line))], "D") {
			counts["deleted"]++
		} else {
			counts["modified"]++
		}
	}
	return map[string]any{
		"available":         true,
		"branch":            branch,
		"head":              head,
		"counts":            counts,
		"dirty_total":       len(entries),
		"dirty_sample":      firstStrings(entries, 25),
		"dirty_omitted":     max(0, len(entries)-25),
		"merge_in_progress": fileExists(filepath.Join(cfg.Root, ".git", "MERGE_HEAD")),
		"safe_ff":           map[string]any{"ok": nil, "state": "unavailable", "reason": "native fleetpane read-only subset does not assess safe-ff"},
		"worktrees":         map[string]any{"available": false, "reason": "native fleetpane read-only subset does not inspect worktrees"},
	}
}

func SummarizeRegistry(registry map[string]any, opts Options) map[string]any {
	if len(registry) == 0 {
		return map[string]any{"exists": false, "sessions": 0, "actions": map[string]int{}, "categories": map[string]int{}}
	}
	sessions := sliceAny(registry["sessions"])
	accounts := sliceAny(registry["accounts"])
	actions := map[string]int{}
	categories := map[string]int{}
	dispositions := map[string]int{}
	for _, item := range sessions {
		session, ok := item.(map[string]any)
		if !ok {
			continue
		}
		actions[stringValue(session["action"])]++
		categories[stringValue(session["category"])]++
		dispositions[stringValue(session["disp"])]++
	}
	available := 0
	var blocked []map[string]any
	for _, item := range accounts {
		account, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if boolValue(account["available"]) {
			available++
		} else {
			blocked = append(blocked, blockedAccountSnapshot(account))
		}
	}
	availableTags := []string{}
	for _, item := range accounts {
		account, ok := item.(map[string]any)
		if ok && boolValue(account["available"]) {
			tag := stringValue(account["tag"])
			if tag == "" {
				tag = stringValue(account["account"])
			}
			availableTags = append(availableTags, tag)
		}
	}
	blockedKinds := map[string]int{}
	authBlockedAccounts := 0
	usageBlockedAccounts := 0
	for _, account := range blocked {
		kind := stringValueDefault(account["block_kind"], "unknown")
		blockedKinds[kind]++
		if accountBlockIsAuthLike(account) {
			authBlockedAccounts++
		}
		if accountBlockIsUsageLike(account) {
			usageBlockedAccounts++
		}
	}
	return map[string]any{
		"exists":               true,
		"schema":               registry["schema"],
		"generated_utc":        registry["generated_utc"],
		"age_min":              parseISOAgeMin(stringValue(registry["generated_utc"]), opts),
		"sessions":             len(sessions),
		"categories":           categories,
		"actions":              actions,
		"dispositions":         dispositions,
		"auto_resume":          actions["AUTO_RESUME"],
		"surface":              actions["SURFACE"],
		"auth_blocked":         dispositions["INFRA_AUTH"],
		"auth_blocked_actions": actions["BLOCKED_AUTH"],
		"supervised":           actions["SUPERVISED"],
		"accounts": map[string]any{
			"total":               len(accounts),
			"available":           available,
			"blocked_count":       len(blocked),
			"blocked_kinds":       blockedKinds,
			"auth_blocked_count":  authBlockedAccounts,
			"usage_blocked_count": usageBlockedAccounts,
			"available_tags":      availableTags,
			"blocked":             blocked,
		},
	}
}

func collectControlTick(cfg Config) map[string]any {
	taskName := stringValueDefault(cfg.ControlTick["task_name"], "FleetControlPaneTick")
	register := stringValue(cfg.ControlTick["register_script"])
	if runtime.GOOS != "windows" {
		if posix := stringValue(cfg.ControlTick["posix_register_script"]); posix != "" {
			register = posix
		}
	}
	registerPath := resolvePathString(register, cfg.Root)
	runner := filepath.Join(cfg.RegistryDir, "control_pane_tick")
	if runtime.GOOS == "windows" {
		runner += ".cmd"
	} else {
		runner += ".sh"
	}
	return map[string]any{
		"task_name":              taskName,
		"register_script":        displayPath(registerPath, cfg.Root),
		"register_script_exists": fileExists(registerPath),
		"runner":                 displayPath(runner, cfg.Root),
		"runner_exists":          fileExists(runner),
		"runner_required":        false,
		"interval_min":           cfg.ControlTick["interval_min"],
		"task": map[string]any{
			"supported": false,
			"installed": false,
			"reason":    "native fleetpane read-only subset does not inspect scheduler tasks",
		},
	}
}

func collectWatchdogs(cfg Config) map[string]any {
	out := map[string]any{}
	names := make([]string, 0, len(cfg.Watchdogs))
	for name := range cfg.Watchdogs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		spec := cfg.Watchdogs[name]
		script := resolvePathString(stringValue(spec["script"]), cfg.Root)
		register := resolvePathString(stringValue(spec["register_script"]), cfg.Root)
		out[name] = map[string]any{
			"task_name":              stringValueDefault(spec["task_name"], name),
			"script":                 displayPath(script, cfg.Root),
			"script_exists":          fileExists(script),
			"register_script":        displayPath(register, cfg.Root),
			"register_script_exists": fileExists(register),
			"interval_min":           spec["interval_min"],
			"task": map[string]any{
				"supported": false,
				"installed": false,
				"reason":    "native fleetpane read-only subset does not inspect scheduler tasks",
			},
		}
	}
	return out
}

func collectSupervisor(ctx context.Context, cfg Config, opts Options) map[string]any {
	rawCmd := cfg.SupervisorStatus
	cmd := ExpandCmd(rawCmd, cfg)
	if len(cmd) == 0 {
		return map[string]any{"available": false, "reason": "no supervisor_status_cmd configured"}
	}
	runner := opts.runner()
	if !commandExists(cmd[0], runner) {
		return map[string]any{"available": false, "reason": "command not found: " + cmd[0], "cmd": cmd}
	}
	res := runner.Run(ctx, RunRequest{Args: cmd, Cwd: cfg.Root, Env: envForTools(cfg), Timeout: 20 * time.Second})
	if res.TimedOut {
		return map[string]any{"available": false, "reason": "supervisor status timed out", "cmd": cmd}
	}
	payload := ReadJSONFromText(res.Stdout)
	if len(payload) > 0 {
		return map[string]any{"available": true, "cmd": cmd, "returncode": res.ExitCode, "payload": payload}
	}
	if res.ExitCode != 0 {
		reason := strings.TrimSpace(res.Stderr)
		if reason == "" {
			reason = strings.TrimSpace(res.Stdout)
		}
		return map[string]any{"available": false, "returncode": res.ExitCode, "reason": tail(reason, 1000), "cmd": cmd}
	}
	return map[string]any{"available": false, "reason": "supervisor status was not JSON", "cmd": cmd}
}

// collectWorkerHealth runs the configured `fak fleet monitor --json` command and
// folds its class histogram into the pane's one health summary (#2035). It is the
// monitor analog of collectSupervisor: a failed/unconfigured/timed-out monitor
// yields an honest unavailable WorkerHealth (reason set, counts zero-filled) rather
// than a misleading all-zero histogram.
func collectWorkerHealth(ctx context.Context, cfg Config, opts Options) WorkerHealth {
	cmd := ExpandCmd(cfg.MonitorStatus, cfg)
	if len(cmd) == 0 {
		return WorkerHealth{Counts: emptyHealthCounts(), Reason: "no monitor_status_cmd configured"}
	}
	runner := opts.runner()
	if !commandExists(cmd[0], runner) {
		return WorkerHealth{Counts: emptyHealthCounts(), Reason: "command not found: " + cmd[0]}
	}
	res := runner.Run(ctx, RunRequest{Args: cmd, Cwd: cfg.Root, Env: envForTools(cfg), Timeout: 20 * time.Second})
	if res.TimedOut {
		return WorkerHealth{Counts: emptyHealthCounts(), Reason: "monitor status timed out"}
	}
	if strings.TrimSpace(res.Stdout) == "" {
		reason := strings.TrimSpace(res.Stderr)
		if reason == "" {
			reason = fmt.Sprintf("monitor exited %d with no output", res.ExitCode)
		}
		return WorkerHealth{Counts: emptyHealthCounts(), Reason: tail(reason, 1000)}
	}
	health := WorkerHealthFromMonitorJSON([]byte(res.Stdout))
	health.Source = cmd
	return health
}

func recommendedActions(status StatusDoc) []string {
	var actions []string
	if exists, _ := status.Config["local_config_exists"].(bool); !exists {
		actions = append(actions, "Run `python tools/fleet_control_pane.py init` once on this machine.")
	}
	if exists, _ := status.Registry["exists"].(bool); !exists {
		actions = append(actions, "Run `python tools/fleet_control_pane.py status --refresh --write` to create the session registry.")
	}
	if loops, ok := status.Loops["checks"].([]LoopCheck); ok {
		for _, check := range loops {
			if LoopCheckNeedsAction(check) {
				action := check.Action
				if action == "" {
					action = check.Detail
				}
				if action == "" {
					action = check.Reason
				}
				if action == "" {
					action = "inspect loop status"
				}
				actions = append(actions, fmt.Sprintf("Loop %s is %s: %s.", check.Name, check.State, action))
			}
		}
	}
	if available, _ := status.Supervisor["available"].(bool); !available {
		actions = append(actions, "Supervisor status unavailable: "+stringValueDefault(status.Supervisor["reason"], "unavailable")+".")
	}
	return actions
}

func FleetView(ctx context.Context, cfg Config, includeLiveLocal bool, refreshLiveLocal bool, opts Options) FleetDoc {
	machineDir := cfg.MachineDir
	staleMin := cfg.MachineStaleMin
	var machines []map[string]any
	_ = filepath.WalkDir(machineDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d == nil || d.IsDir() || filepath.Ext(path) != ".json" {
			return nil
		}
		machines = append(machines, SummarizeMachineSnapshot(path, readJSONMap(path), staleMin, opts))
		return nil
	})
	if includeLiveLocal {
		localPath := filepath.Join(machineDir, MachineID(cfg)+".json")
		localStatus := statusDocToMap(CollectStatus(ctx, cfg, opts))
		localMachine := SummarizeMachineSnapshot(localPath, localStatus, staleMin, opts)
		localMachine["live_local"] = true
		localMachine["refresh_local"] = refreshLiveLocal
		localID := stringValue(localMachine["id"])
		filtered := machines[:0]
		for _, machine := range machines {
			if stringValue(machine["id"]) != localID {
				filtered = append(filtered, machine)
			}
		}
		machines = append(filtered, localMachine)
	}
	sort.Slice(machines, func(i, j int) bool {
		return stringValue(machines[i]["id"]) < stringValue(machines[j]["id"])
	})
	currentVersion := AppVersion(cfg.Root)
	versionMismatches := 0
	for _, machine := range machines {
		if drift := machineVersionDrift(machine, currentVersion); drift != nil {
			machine["version_drift"] = drift
			versionMismatches++
		}
	}
	states := map[string]int{}
	versions := map[string]int{}
	totals := map[string]int{}
	for _, machine := range machines {
		state := stringValueDefault(machine["state"], "UNKNOWN")
		states[state]++
		versions[stringValueDefault(machine["app_version"], "unknown")]++
		totals["sessions"] += intValueDefault(machine["sessions"], 0)
		totals["actions"] += intValueDefault(machine["actions_count"], 0)
		totals["auth_blocked"] += intValueDefault(machine["auth_blocked"], 0)
		totals["auto_resume"] += intValueDefault(machine["auto_resume"], 0)
		totals["seat_capacity"] += intValueDefault(machine["seat_capacity"], 0)
		totals["throttled_seats"] += intValueDefault(machine["throttled_seats"], 0)
		totals["healthy_seats"] += intValueDefault(machine["healthy_seats"], 0)
		totals["surface"] += intValueDefault(machine["surface"], 0)
		if accounts := mapValue(machine["accounts"]); len(accounts) > 0 {
			totals["blocked_accounts"] += intValueDefault(accounts["blocked_count"], len(sliceAny(accounts["blocked"])))
			totals["auth_blocked_accounts"] += intValueDefault(accounts["auth_blocked_count"], 0)
		}
		if loops := mapValue(machine["loops"]); len(loops) > 0 {
			totals["loop_actions"] += intValueDefault(loops["action"], 0)
		}
		if git := mapValue(machine["git"]); len(git) > 0 {
			totals["dirty_paths"] += intValueDefault(git["dirty_total"], 0)
		}
	}
	totals["version_mismatches"] = versionMismatches
	verdict := "OK"
	if len(machines) == 0 {
		verdict = "EMPTY"
	} else if versionMismatches > 0 {
		verdict = "ACTION"
	} else {
		for _, machine := range machines {
			if map[string]bool{"ACTION": true, "STALE": true, "UNKNOWN": true, "INVALID": true}[stringValue(machine["state"])] {
				verdict = "ACTION"
				break
			}
		}
	}
	return FleetDoc{
		Schema:           FleetSchema,
		GeneratedUTC:     GeneratedUTC(opts),
		MachineDir:       machineDir,
		StaleMin:         staleMin,
		IncludeLiveLocal: includeLiveLocal,
		RefreshLiveLocal: refreshLiveLocal,
		CurrentVersion:   currentVersion,
		Written:          nil,
		Verdict:          verdict,
		States:           states,
		Versions:         versions,
		Totals:           totals,
		Machines:         machines,
	}
}

func SummarizeMachineSnapshot(path string, status map[string]any, staleMin float64, opts Options) map[string]any {
	if len(status) == 0 {
		return map[string]any{"id": strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)), "source": path, "state": "INVALID", "reason": "missing or malformed JSON"}
	}
	machine := mapValue(status["machine"])
	reg := mapValue(status["registry"])
	accts := mapValue(reg["accounts"])
	loops := mapValue(status["loops"])
	gitStatus := mapValue(status["git"])
	safeFF := mapValue(gitStatus["safe_ff"])
	age := parseISOAgeMin(stringValue(status["generated_utc"]), opts)
	verdict := stringValueDefault(status["verdict"], "UNKNOWN")
	state := verdict
	if age == nil {
		state = "UNKNOWN"
	} else if *age > staleMin {
		state = "STALE"
	}
	loopChecks := make([]map[string]any, 0)
	for _, item := range sliceAny(loops["checks"]) {
		switch check := item.(type) {
		case LoopCheck:
			loopChecks = append(loopChecks, loopCheckSnapshot(check))
		case map[string]any:
			loopChecks = append(loopChecks, loopCheckSnapshotMap(check))
		}
	}
	loopActions := 0
	for _, check := range loopChecks {
		state := stringValueDefault(check["state"], "UNKNOWN")
		if state != "OK" && state != "SKIPPED" {
			loopActions++
		}
	}
	return map[string]any{
		"id":              stringValueDefault(machine["id"], strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))),
		"host":            stringValueDefault(machine["host"], strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))),
		"app_version":     status["app_version"],
		"source":          path,
		"generated_utc":   status["generated_utc"],
		"age_min":         age,
		"state":           state,
		"verdict":         verdict,
		"actions_count":   len(sliceAny(status["actions"])),
		"actions":         firstAny(sliceAny(status["actions"]), 12),
		"sessions":        intValueDefault(reg["sessions"], 0),
		"categories":      mapValue(reg["categories"]),
		"session_actions": mapValue(reg["actions"]),
		"auth_blocked":    intValueDefault(reg["auth_blocked"], 0),
		"auto_resume":     intValueDefault(reg["auto_resume"], 0),
		"surface":         intValueDefault(reg["surface"], 0),
		"accounts": map[string]any{
			"available":           intValueDefault(accts["available"], 0),
			"total":               intValueDefault(accts["total"], 0),
			"available_tags":      sliceAny(accts["available_tags"]),
			"blocked":             sliceAny(accts["blocked"]),
			"blocked_count":       intValueDefault(accts["blocked_count"], len(sliceAny(accts["blocked"]))),
			"auth_blocked_count":  intValueDefault(accts["auth_blocked_count"], 0),
			"usage_blocked_count": intValueDefault(accts["usage_blocked_count"], 0),
			"blocked_kinds":       mapValue(accts["blocked_kinds"]),
		},
		"supervisor": map[string]any{
			"available": mapValue(status["supervisor"])["available"],
			"verdict":   mapValue(mapValue(status["supervisor"])["payload"])["verdict"],
			"action":    "",
		},
		"control_tick": map[string]any{},
		"watchdogs":    map[string]any{},
		"setup_plan":   map[string]any{},
		"loops": map[string]any{
			"count":      intValueDefault(loops["count"], 0),
			"configured": intValueDefault(loops["configured"], intValueDefault(loops["count"], 0)),
			"enabled":    intValueDefault(loops["enabled"], intValueDefault(loops["count"], 0)),
			"disabled":   intValueDefault(loops["disabled"], 0),
			"states":     mapValue(loops["states"]),
			"action":     loopActions,
			"checks":     loopChecks,
			"commands":   firstAny(sliceAny(loops["commands"]), 3),
		},
		"git": map[string]any{
			"branch":            gitStatus["branch"],
			"head":              gitStatus["head"],
			"dirty_total":       intValueDefault(gitStatus["dirty_total"], 0),
			"merge_in_progress": boolValue(gitStatus["merge_in_progress"]),
			"safe_ff_state":     safeFF["state"],
			"safe_ff_ok":        safeFF["ok"],
			"safe_ff_reason":    safeFF["reason"],
		},
	}
}

func WriteJSON(w interface{ Write([]byte) (int, error) }, doc any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	return enc.Encode(doc)
}

func ReadJSONFromText(raw string) map[string]any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(raw), &doc); err == nil {
		return doc
	}
	for idx, ch := range raw {
		if ch != '{' {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(raw[idx:]))
		dec.UseNumber()
		doc = map[string]any{}
		if err := dec.Decode(&doc); err == nil {
			return doc
		}
	}
	return map[string]any{}
}
