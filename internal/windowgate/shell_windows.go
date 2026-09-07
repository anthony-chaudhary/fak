//go:build windows

package windowgate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// PowerShell command complexity categories.
const (
	CategoryLightweight = "lightweight"
	CategoryMedium      = "medium"
	CategoryHeavy       = "heavy"
	CategoryDefault     = "default"
)

// Adaptive base deadlines by complexity category.
const (
	DeadlineLightweight = 15 * time.Second
	DeadlineMedium      = 45 * time.Second
	DeadlineHeavy       = 120 * time.Second
	DeadlineDefault     = 30 * time.Second
)

// Default streaming extension parameters.
const (
	DefaultExtensionSlice   = 10 * time.Second
	DefaultMaxDeadline      = 5 * time.Minute
	DefaultWarningThreshold = 0.70
)

var (
	reHeavyCategory       = regexp.MustCompile(`(?i)\b(dism(\.exe)?|Install-Module|Import-Module|Update-Module|Enable-WindowsOptionalFeature|Disable-WindowsOptionalFeature|Install-Package|Install-WindowsFeature|choco|winget)\b`)
	reMediumCategory      = regexp.MustCompile(`(?i)\b(Get-CimInstance|Get-WmiObject|Get-Service|Restart-Service|Start-Service|Stop-Service|Get-Process|Invoke-RestMethod|Invoke-WebRequest|Test-NetConnection)\b`)
	reLightweightCategory = regexp.MustCompile(`(?i)\b(Get-ChildItem|Get-Date|Get-Item|Get-ItemProperty|Get-ItemPropertyValue|Test-Path|Get-Location|Get-Content|Get-Command|dir|ls|echo)\b`)
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPage            uint64
	AvailPage            uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")

func detectHighLoadWindows() bool {
	var m memoryStatusEx
	m.Length = uint32(unsafe.Sizeof(m))
	r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m)))
	if r != 0 && m.MemoryLoad >= 85 {
		return true
	}
	return false
}

// ClassifyScriptCategory inspects a PowerShell script to determine its complexity category.
func ClassifyScriptCategory(script string) string {
	s := strings.TrimSpace(script)
	if reHeavyCategory.MatchString(s) {
		return CategoryHeavy
	}
	if reMediumCategory.MatchString(s) {
		return CategoryMedium
	}
	if reLightweightCategory.MatchString(s) {
		return CategoryLightweight
	}
	return CategoryDefault
}

// ResolveBaseDeadline computes the initial execution deadline for a given script and category.
func ResolveBaseDeadline(script string, category string) time.Duration {
	cat := strings.ToLower(strings.TrimSpace(category))
	switch cat {
	case CategoryLightweight, "light":
		return DeadlineLightweight
	case CategoryMedium, "cim", "wmi":
		return DeadlineMedium
	case CategoryHeavy, "dism", "module":
		return DeadlineHeavy
	case CategoryDefault:
		return DeadlineDefault
	case "":
		auto := ClassifyScriptCategory(script)
		switch auto {
		case CategoryHeavy:
			return DeadlineHeavy
		case CategoryMedium:
			return DeadlineMedium
		case CategoryLightweight:
			return DeadlineLightweight
		default:
			return DeadlineDefault
		}
	default:
		return DeadlineDefault
	}
}

// BuildPowerShellArgs returns standard non-interactive execution arguments.
func BuildPowerShellArgs(script string) []string {
	return []string{
		"-NoLogo",
		"-NoProfile",
		"-NonInteractive",
		"-ExecutionPolicy",
		"Bypass",
		"-Command",
		script,
	}
}

// PowerShellOptions provides configuration values for running a PowerShell script.
type PowerShellOptions struct {
	Category        string
	Timeout         time.Duration
	MaxDeadline     time.Duration
	ExtensionSlice  time.Duration
	PreferredEngine string
	WorkingDir      string
	Env             []string
	Logger          func(format string, args ...any)
}

// PowerShellConfig holds resolved configuration and injection seams.
type PowerShellConfig struct {
	Category         string
	Timeout          time.Duration
	MaxDeadline      time.Duration
	ExtensionSlice   time.Duration
	WarningThreshold float64
	PreferredEngine  string
	WorkingDir       string
	Env              []string
	Logger           func(format string, args ...any)

	// Test seams
	LookPath       func(string) (string, error)
	DetectHighLoad func() bool
	CommandContext func(ctx context.Context, name string, args ...string) *exec.Cmd
}

// PowerShellResult contains execution outputs, timings, and metadata.
type PowerShellResult struct {
	Stdout     string        `json:"stdout"`
	Stderr     string        `json:"stderr"`
	ExitCode   int           `json:"exit_code"`
	Duration   time.Duration `json:"duration"`
	Engine     string        `json:"engine"`
	TimedOut   bool          `json:"timed_out,omitempty"`
	Extensions int           `json:"extensions,omitempty"`
	Advisories []string      `json:"advisories,omitempty"`
}

// PowerShellOption mutates PowerShellConfig.
type PowerShellOption func(*PowerShellConfig)

// WithCategory sets the command complexity category.
func WithCategory(category string) PowerShellOption {
	return func(c *PowerShellConfig) { c.Category = category }
}

// WithTimeout sets an explicit base timeout, overriding automatic category resolution.
func WithTimeout(timeout time.Duration) PowerShellOption {
	return func(c *PowerShellConfig) { c.Timeout = timeout }
}

// WithMaxDeadline sets the hard cap for adaptive deadline streaming extensions.
func WithMaxDeadline(maxDeadline time.Duration) PowerShellOption {
	return func(c *PowerShellConfig) { c.MaxDeadline = maxDeadline }
}

// WithExtensionSlice sets the incremental deadline extension added when output streams.
func WithExtensionSlice(slice time.Duration) PowerShellOption {
	return func(c *PowerShellConfig) { c.ExtensionSlice = slice }
}

// WithWarningThreshold sets the fraction of active deadline at which an advisory warning is emitted.
func WithWarningThreshold(threshold float64) PowerShellOption {
	return func(c *PowerShellConfig) { c.WarningThreshold = threshold }
}

// WithPreferredEngine sets the preferred PowerShell executable (e.g. "pwsh" or "powershell.exe").
func WithPreferredEngine(engine string) PowerShellOption {
	return func(c *PowerShellConfig) { c.PreferredEngine = engine }
}

// WithWorkingDir sets the working directory for PowerShell execution.
func WithWorkingDir(dir string) PowerShellOption {
	return func(c *PowerShellConfig) { c.WorkingDir = dir }
}

// WithEnv sets the environment variables for PowerShell execution.
func WithEnv(env []string) PowerShellOption {
	return func(c *PowerShellConfig) { c.Env = env }
}

// WithLogger sets a custom advisory logger function.
func WithLogger(logger func(format string, args ...any)) PowerShellOption {
	return func(c *PowerShellConfig) { c.Logger = logger }
}

// WithOptions applies a PowerShellOptions struct.
func WithOptions(opts PowerShellOptions) PowerShellOption {
	return func(c *PowerShellConfig) {
		if opts.Category != "" {
			c.Category = opts.Category
		}
		if opts.Timeout > 0 {
			c.Timeout = opts.Timeout
		}
		if opts.MaxDeadline > 0 {
			c.MaxDeadline = opts.MaxDeadline
		}
		if opts.ExtensionSlice > 0 {
			c.ExtensionSlice = opts.ExtensionSlice
		}
		if opts.PreferredEngine != "" {
			c.PreferredEngine = opts.PreferredEngine
		}
		if opts.WorkingDir != "" {
			c.WorkingDir = opts.WorkingDir
		}
		if len(opts.Env) > 0 {
			c.Env = opts.Env
		}
		if opts.Logger != nil {
			c.Logger = opts.Logger
		}
	}
}

// WithDetectHighLoad injects a load detector for testing.
func WithDetectHighLoad(fn func() bool) PowerShellOption {
	return func(c *PowerShellConfig) { c.DetectHighLoad = fn }
}

// WithLookPath injects an executable lookup seam for testing.
func WithLookPath(fn func(string) (string, error)) PowerShellOption {
	return func(c *PowerShellConfig) { c.LookPath = fn }
}

// WithCommandContext injects a command constructor for testing.
func WithCommandContext(fn func(context.Context, string, ...string) *exec.Cmd) PowerShellOption {
	return func(c *PowerShellConfig) { c.CommandContext = fn }
}

func killProcessTree(pid int, job *JobObject) {
	if job != nil {
		_ = job.Close()
	}
	if pid > 0 {
		tk := exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(pid))
		ConfigureBackgroundCommand(tk)
		_ = tk.Run()
	}
}

func getPID(cmd *exec.Cmd) int {
	if cmd != nil && cmd.Process != nil {
		return cmd.Process.Pid
	}
	return 0
}

func resetTimer(t *time.Timer, d time.Duration) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	t.Reset(d)
}

func drainChannel(ch chan struct{}) {
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}

type enginePlan struct {
	primary  string
	fallback string
}

func planEngines(cfg *PowerShellConfig) enginePlan {
	if cfg.PreferredEngine != "" {
		pref := cfg.PreferredEngine
		if strings.EqualFold(pref, "pwsh") || strings.EqualFold(pref, "pwsh.exe") {
			return enginePlan{primary: pref, fallback: "powershell.exe"}
		}
		if strings.EqualFold(pref, "powershell") || strings.EqualFold(pref, "powershell.exe") {
			return enginePlan{primary: pref, fallback: "pwsh"}
		}
		return enginePlan{primary: pref, fallback: ""}
	}
	return enginePlan{primary: "pwsh", fallback: "powershell.exe"}
}

func spawnCommand(ctx context.Context, engine string, script string, cfg *PowerShellConfig) (*exec.Cmd, *JobObject, io.ReadCloser, io.ReadCloser, error) {
	cmd := cfg.CommandContext(ctx, engine, BuildPowerShellArgs(script)...)
	ConfigureBackgroundCommand(cmd)
	if cfg.WorkingDir != "" {
		cmd.Dir = cfg.WorkingDir
	}
	if len(cfg.Env) > 0 {
		cmd.Env = cfg.Env
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		_ = stdoutPipe.Close()
		return nil, nil, nil, nil, err
	}

	job, err := StartInNewJob(cmd)
	if err != nil {
		cmd = cfg.CommandContext(ctx, engine, BuildPowerShellArgs(script)...)
		ConfigureBackgroundCommand(cmd)
		if cfg.WorkingDir != "" {
			cmd.Dir = cfg.WorkingDir
		}
		if len(cfg.Env) > 0 {
			cmd.Env = cfg.Env
		}
		outP, errOut := cmd.StdoutPipe()
		errP, errErr := cmd.StderrPipe()
		if errOut != nil || errErr != nil {
			if outP != nil {
				_ = outP.Close()
			}
			return nil, nil, nil, nil, err
		}
		stdoutPipe = outP
		stderrPipe = errP
		if plainErr := cmd.Start(); plainErr != nil {
			_ = stdoutPipe.Close()
			_ = stderrPipe.Close()
			return nil, nil, nil, nil, plainErr
		}
		job = nil
	}

	return cmd, job, stdoutPipe, stderrPipe, nil
}

// RunPowerShell runs a PowerShell script with dynamic adaptive deadlines,
// console window suppression, dual-engine fallback, streaming extensions, and process containment.
func RunPowerShell(ctx context.Context, script string, opts ...PowerShellOption) (PowerShellResult, error) {
	if err := ctx.Err(); err != nil {
		return PowerShellResult{ExitCode: -1}, err
	}

	cfg := &PowerShellConfig{
		MaxDeadline:      DefaultMaxDeadline,
		ExtensionSlice:   DefaultExtensionSlice,
		WarningThreshold: DefaultWarningThreshold,
		Env:              os.Environ(),
		LookPath:         exec.LookPath,
		DetectHighLoad:   detectHighLoadWindows,
		CommandContext:   exec.CommandContext,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}

	var advisories []string
	var advMu sync.Mutex
	logAdvisory := func(msg string) {
		advMu.Lock()
		advisories = append(advisories, msg)
		advMu.Unlock()
		if cfg.Logger != nil {
			cfg.Logger("%s", msg)
		} else {
			log.Printf("%s", msg)
		}
	}

	baseDeadline := cfg.Timeout
	if baseDeadline <= 0 {
		baseDeadline = ResolveBaseDeadline(script, cfg.Category)
	}
	if cfg.DetectHighLoad != nil && cfg.DetectHighLoad() {
		scaled := time.Duration(float64(baseDeadline) * 1.5)
		msg := fmt.Sprintf("[windowgate:powershell] ADVISORY: high system load detected; scaling base deadline to %v", scaled)
		logAdvisory(msg)
		baseDeadline = scaled
	}

	activeDeadline := baseDeadline
	maxDeadline := cfg.MaxDeadline
	if maxDeadline <= 0 {
		maxDeadline = DefaultMaxDeadline
	}
	if maxDeadline < activeDeadline {
		maxDeadline = activeDeadline
	}

	extensionSlice := cfg.ExtensionSlice
	if extensionSlice <= 0 {
		extensionSlice = DefaultExtensionSlice
	}

	warningThreshold := cfg.WarningThreshold
	if warningThreshold <= 0 || warningThreshold >= 1.0 {
		warningThreshold = DefaultWarningThreshold
	}

	plan := planEngines(cfg)
	selectedEngine := plan.primary

	// Probe primary engine
	if plan.fallback != "" {
		if _, err := cfg.LookPath(plan.primary); err != nil {
			msg := fmt.Sprintf("[windowgate:powershell] ADVISORY: %s not found in PATH; falling back to %s", plan.primary, plan.fallback)
			logAdvisory(msg)
			selectedEngine = plan.fallback
			plan.fallback = ""
		}
	}

	cmd, job, stdoutPipe, stderrPipe, startErr := spawnCommand(ctx, selectedEngine, script, cfg)
	if startErr != nil && plan.fallback != "" && ctx.Err() == nil {
		msg := fmt.Sprintf("[windowgate:powershell] ADVISORY: failed to start %s (%v); falling back to %s", selectedEngine, startErr, plan.fallback)
		logAdvisory(msg)
		selectedEngine = plan.fallback
		plan.fallback = ""
		cmd, job, stdoutPipe, stderrPipe, startErr = spawnCommand(ctx, selectedEngine, script, cfg)
	}
	if startErr != nil {
		if ctx.Err() != nil {
			return PowerShellResult{
				ExitCode:   -1,
				Engine:     selectedEngine,
				Advisories: advisories,
			}, ctx.Err()
		}
		return PowerShellResult{
			ExitCode:   -1,
			Engine:     selectedEngine,
			Advisories: advisories,
		}, fmt.Errorf("windowgate: start %s: %w", selectedEngine, startErr)
	}

	var (
		stdoutBuf bytes.Buffer
		stderrBuf bytes.Buffer
		bufMu     sync.Mutex
		dataCh    = make(chan struct{}, 64)
		streamWg  sync.WaitGroup
	)

	streamPipe := func(r io.Reader, buf *bytes.Buffer) {
		defer streamWg.Done()
		chunk := make([]byte, 4096)
		for {
			n, err := r.Read(chunk)
			if n > 0 {
				bufMu.Lock()
				buf.Write(chunk[:n])
				bufMu.Unlock()

				select {
				case dataCh <- struct{}{}:
				default:
				}
			}
			if err != nil {
				break
			}
		}
	}

	streamWg.Add(2)
	go streamPipe(stdoutPipe, &stdoutBuf)
	go streamPipe(stderrPipe, &stderrBuf)

	doneCh := make(chan error, 1)
	go func() {
		doneCh <- cmd.Wait()
	}()

	startTime := time.Now()
	lastExtension := startTime
	var extensionsCount int
	var warned bool
	var timedOut bool
	var waitErr error

	deadlineTimer := time.NewTimer(activeDeadline)
	defer deadlineTimer.Stop()

	warnDuration := time.Duration(float64(activeDeadline) * warningThreshold)
	warnTimer := time.NewTimer(warnDuration)
	defer warnTimer.Stop()

loop:
	for {
		select {
		case <-ctx.Done():
			killProcessTree(getPID(cmd), job)
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
			}
			<-doneCh
			streamWg.Wait()
			waitErr = ctx.Err()
			break loop

		case <-deadlineTimer.C:
			timedOut = true
			killProcessTree(getPID(cmd), job)
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
			}
			<-doneCh
			streamWg.Wait()
			waitErr = context.DeadlineExceeded
			break loop

		case <-warnTimer.C:
			if !warned {
				warned = true
				msg := fmt.Sprintf("[windowgate:powershell] ADVISORY: command reached 70%% of deadline (%v); continuing execution", activeDeadline)
				logAdvisory(msg)
			}

		case <-dataCh:
			drainChannel(dataCh)
			now := time.Now()
			if activeDeadline < maxDeadline {
				remaining := startTime.Add(activeDeadline).Sub(now)
				if remaining > 0 && (now.Sub(lastExtension) >= extensionSlice/2 || remaining < extensionSlice) {
					newDeadline := activeDeadline + extensionSlice
					if newDeadline > maxDeadline {
						newDeadline = maxDeadline
					}
					if newDeadline > activeDeadline {
						activeDeadline = newDeadline
						lastExtension = now
						extensionsCount++
						warned = false

						newRem := startTime.Add(activeDeadline).Sub(now)
						if newRem > 0 {
							resetTimer(deadlineTimer, newRem)
						}
						newWarnAt := startTime.Add(time.Duration(float64(activeDeadline) * warningThreshold))
						if newWarnAt.After(now) {
							resetTimer(warnTimer, newWarnAt.Sub(now))
						}
					}
				}
			}

		case waitErr = <-doneCh:
			streamWg.Wait()
			if job != nil {
				_ = job.Close()
			}
			break loop
		}
	}

	bufMu.Lock()
	stdoutStr := stdoutBuf.String()
	stderrStr := stderrBuf.String()
	bufMu.Unlock()

	exitCode := 0
	if timedOut {
		exitCode = -1
	} else if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}
	}

	advMu.Lock()
	finalAdvisories := append([]string(nil), advisories...)
	advMu.Unlock()

	result := PowerShellResult{
		Stdout:     stdoutStr,
		Stderr:     stderrStr,
		ExitCode:   exitCode,
		Duration:   time.Since(startTime),
		Engine:     selectedEngine,
		TimedOut:   timedOut,
		Extensions: extensionsCount,
		Advisories: finalAdvisories,
	}

	if timedOut {
		return result, context.DeadlineExceeded
	}
	if ctx.Err() != nil {
		return result, ctx.Err()
	}
	return result, waitErr
}
