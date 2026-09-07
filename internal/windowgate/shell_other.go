//go:build !windows

package windowgate

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
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

// RunPowerShell provides a portable mock/fallback implementation off Windows.
// If pwsh is present in PATH, it runs pwsh; otherwise returns a simulated result.
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
		DetectHighLoad:   func() bool { return false },
		CommandContext:   exec.CommandContext,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}

	pwshPath, err := cfg.LookPath("pwsh")
	if err != nil {
		advisory := "[windowgate:powershell] ADVISORY: pwsh not found off Windows; returning simulated result"
		if cfg.Logger != nil {
			cfg.Logger("%s", advisory)
		}
		return PowerShellResult{
			Stdout:     "simulated non-windows output\n",
			Stderr:     "",
			ExitCode:   0,
			Duration:   10 * time.Millisecond,
			Engine:     "stub",
			TimedOut:   false,
			Extensions: 0,
			Advisories: []string{advisory},
		}, nil
	}

	baseDeadline := cfg.Timeout
	if baseDeadline <= 0 {
		baseDeadline = ResolveBaseDeadline(script, cfg.Category)
	}

	runCtx, cancel := context.WithTimeout(ctx, baseDeadline)
	defer cancel()

	cmd := cfg.CommandContext(runCtx, pwshPath, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script)
	if cfg.WorkingDir != "" {
		cmd.Dir = cfg.WorkingDir
	}
	if len(cfg.Env) > 0 {
		cmd.Env = cfg.Env
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	runErr := cmd.Run()
	duration := time.Since(start)

	exitCode := 0
	timedOut := false
	if runErr != nil {
		if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
			timedOut = true
			exitCode = -1
		} else {
			var exitErr *exec.ExitError
			if errors.As(runErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = -1
			}
		}
	}

	res := PowerShellResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		ExitCode: exitCode,
		Duration: duration,
		Engine:   "pwsh",
		TimedOut: timedOut,
	}
	if timedOut {
		return res, context.DeadlineExceeded
	}
	if errors.Is(runCtx.Err(), context.Canceled) {
		return res, context.Canceled
	}
	return res, runErr
}
