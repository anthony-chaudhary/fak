package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

type guardManagedCommandInputs struct {
	OutputProfile           *string
	WorkProfile             *string
	CodexLoopGate           *string
	CodexHome               *string
	CodexLoopGateSinceHours *float64
	CodexLoopGateLimit      *int
	SessionPressureGate     *string
	Provider                *string
	APIKeyEnv               *string
	Model                   *string
	Banner                  *string
	DebugStats              *bool
	Quiet                   *bool
	SetFlags                map[string]bool
}

type guardManagedCommandPreparation struct {
	Command                []string
	ResponseProfileCapture *guardProfileCapture
	LaunchPlan             guardLaunchPlan
	AgentName              string
	DebugStatsStderr       bool
	BannerMode             string
}

// prepareGuardManagedCommand resolves the wrapped command and the launch gates that
// must run before any gateway listener binds. It exits with the same CLI status as the
// caller when a profile, pressure gate, or startup presentation is invalid.
func prepareGuardManagedCommand(fs *flag.FlagSet, launchPlan guardLaunchPlan, in guardManagedCommandInputs) guardManagedCommandPreparation {
	outputProfile := in.OutputProfile
	workProfile := in.WorkProfile
	codexLoopGate := in.CodexLoopGate
	codexHome := in.CodexHome
	codexLoopGateSinceHours := in.CodexLoopGateSinceHours
	codexLoopGateLimit := in.CodexLoopGateLimit
	sessionPressureGate := in.SessionPressureGate
	provider := in.Provider
	apiKeyEnv := in.APIKeyEnv
	model := in.Model
	bannerFlag := in.Banner
	debugStats := in.DebugStats
	quiet := in.Quiet
	guardSetFlags := in.SetFlags

	command := launchPlan.executableCommand() // everything after the flags (and after `--`) is the wrapped agent.
	profilesExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "output-profile" || f.Name == "work-profile" {
			profilesExplicit = true
		}
	})
	command, responseProfileCapture, profileErr := injectGuardProfiles(command, *outputProfile, *workProfile, profilesExplicit)
	if profileErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", profileErr)
		os.Exit(2)
	}
	if len(command) == 0 {
		fs.Usage()
		os.Exit(2)
	}
	launchPlan = launchPlan.withExecutableCommand(command)
	if code := runGuardStoragePressureGate(os.Stderr, defaultGuardStoragePressureDeps()); code != 0 {
		os.Exit(code)
	}
	agentName := launchPlan.agentName()
	if cfg, ok := guardCodexLoopGateConfigForProfile(launchPlan.harnessProfile(), launchPlan.executableCommand(), *codexLoopGate, *codexHome, *codexLoopGateSinceHours, *codexLoopGateLimit, *quiet); ok {
		if code := runCodexLoopGate(os.Stderr, cfg); code != 0 {
			os.Exit(code)
		}
	}
	sessionPressure, specErr := parseGuardSessionPressureSpec(*sessionPressureGate)
	if specErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: --session-pressure-gate %q: %v\n", *sessionPressureGate, specErr)
		os.Exit(2)
	}
	if code := runGuardSessionPressureGate(os.Stderr, guardSessionPressureGateConfig{
		Threshold:     sessionPressure.Threshold,
		SinceDays:     sessionPressure.SinceDays,
		Max:           sessionPressure.Max,
		Quiet:         *quiet,
		ReportPath:    sessionPressure.ReportPath,
		LaunchModel:   *model,
		Justification: sessionPressure.Justification,
	}); code != 0 {
		os.Exit(code)
	}

	// Cooldown-aware seat selection: a bare `fak guard -- claude` (no --rotate) resolves its
	// account purely from the environment and, unlike `fak accounts launch --rotate`, never
	// consults the fleet-shared cooldown store — so it would launch against an account the
	// launcher just watched bounce off its own weekly/usage cap, burning a turn on a walled
	// seat. Only meaningful on the subscription-OAuth Anthropic path: an explicit --api-key-env
	// is API billing (one key, no rotation) and a non-Claude child has no Claude seat to rotate.
	// When the currently-resolved seat is actively cooled and a live alternate exists, redirect
	// $CLAUDE_CONFIG_DIR to it BEFORE resolveGuardUpstream and the child spawn, so every
	// downstream consumer (fak's own OAuth read, the failover seed, the cap-recovery transcript
	// path, and the child's inherited env — all of which re-read the env var) follows to the live
	// seat. Fail-open: any doubt leaves the resolved dir untouched (see guardRotateOffCooldown).
	if provResolved, _ := launchPlan.resolveProvider(*provider); provResolved == "anthropic" && strings.TrimSpace(*apiKeyEnv) == "" {
		guardHomeDir, _ := os.UserHomeDir()
		if newDir, rotated := guardRotateOffCooldown(guardHomeDir, guardDefaultAccountsRegistryPath(guardHomeDir), time.Now(), guardRotateWarnWriter(os.Stderr, *quiet)); rotated {
			_ = os.Setenv("CLAUDE_CONFIG_DIR", newDir)
		}
	}

	// Decide whether the per-turn `fak-turn …` economy line streams to the SHARED terminal
	// stderr. On an attended interactive launch the wrapped agent (Claude Code) paints a
	// full-screen alternate-screen TUI over THIS terminal, so a per-turn stderr write lands
	// on top of it and corrupts the session view; there the economy belongs in the `fak info`
	// split pane (the dedicated fak section) + the exit summary, not the agent pane. An
	// explicit --debug-stats still streams here; headless/piped runs keep it (no TUI to
	// corrupt). See guardDebugStatsToSharedStderr.
	debugStatsStderr := guardDebugStatsToSharedStderr(
		*debugStats, *quiet, guardSetFlags["debug-stats"],
		cmdGuardStdinInteractive(), launchPlan.interactive())

	// Startup-banner verbosity: resolve --banner now, fail-loud on a bad value before
	// any gateway binds. AUTO/empty selects the private delayed-progress-only mode for
	// both interactive and noninteractive launches. See guard_banner.go.
	bannerMode, bannerErr := guardBannerModeDecision(*bannerFlag, *quiet, cmdGuardStdinInteractive(), launchPlan.interactive())
	if bannerErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", bannerErr)
		os.Exit(2)
	}

	return guardManagedCommandPreparation{
		Command:                command,
		ResponseProfileCapture: responseProfileCapture,
		LaunchPlan:             launchPlan,
		AgentName:              agentName,
		DebugStatsStderr:       debugStatsStderr,
		BannerMode:             bannerMode,
	}
}
