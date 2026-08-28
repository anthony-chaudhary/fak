package main

import (
	"fmt"
	"io"
	"strings"
)

// guard_banner.go decides how much of the guard's startup report reaches the terminal.
//
// A default `fak guard -- claude` used to print the FULL startup report — banner block,
// hook installers, MCP/capability notes, auth posture, ~20 dense lines — straight to the
// shared stderr, seconds before the wrapped agent's full-screen TUI painted over it. On
// an attended launch that is a wall of text nobody can act on in the moment; the detail
// is real, it is just in the wrong place at the wrong time. So cmdGuard now renders the
// full report to a buffer regardless, hands it to the in-process gateway
// (Server.SetStartupReport), and prints per --banner:
//
//	auto (default)  delayed startup progress only, for interactive and noninteractive
//	                launches. Healthy startup emits no report, identity, or settle lines.
//	full            always the full report (today's pre-flag behavior, forced).
//	compact         always the compact banner.
//	animate         always request the attended animation (with its existing compact
//	                fallback when animation is unavailable).
//	off             no banner at all (narrower than --quiet, which also silences the
//	                exit summary and per-turn notes).
//
// The full text stays one command away for the session's whole life:
// `fak info --startup` reads it back off the gateway's /debug/vars.
const (
	guardBannerAuto    = "auto"
	guardBannerFull    = "full"
	guardBannerCompact = "compact"
	guardBannerOff     = "off"
	// guardBannerProgress is the internal healthy-default mode. It deliberately is not an
	// accepted --banner value: auto/empty resolve here so startup keeps the delayed progress
	// surface without emitting any startup-report or launch-animation bytes.
	guardBannerProgress = "progress"
	// guardBannerAnimate is the explicit attended-animation mode: instead of the compact banner's
	// three static lines flashing before the agent's TUI paints over them, play a short in-place
	// icon animation that lands on one iconic identity line (see guard_launch_anim.go). It
	// degrades to the compact banner off a color TTY or under FAK_GUARD_LAUNCH_ANIM=off, so the
	// byte-clean / no-motion paths are preserved; the render site (guard.go) owns that fallback.
	guardBannerAnimate = "animate"
)

// guardBannerModeDecision resolves the --banner flag to a concrete mode. Precedence,
// highest first: --quiet silences everything (its existing contract — the banner is part
// of what it already suppressed); an explicit full/compact/animate/off is a knowing choice
// and wins; AUTO/empty resolves to the private progress-only mode for every launch shape.
// An unknown value is a loud usage error, never a silent fallback.
func guardBannerModeDecision(banner string, quiet, stdinInteractive, childInteractive bool) (string, error) {
	mode := strings.ToLower(strings.TrimSpace(banner))
	if mode == "" {
		mode = guardBannerAuto
	}
	switch mode {
	case guardBannerAuto, guardBannerFull, guardBannerCompact, guardBannerOff, guardBannerAnimate:
	default:
		return "", fmt.Errorf("--banner must be auto, full, compact, animate, or off; got %q", banner)
	}
	if quiet {
		return guardBannerOff, nil
	}
	if mode != guardBannerAuto {
		return mode, nil
	}
	return guardBannerProgress, nil
}

// guardDumpStartupOnLaunchFail decides whether a launch failure still needs the full report
// spilled. Only explicit full already printed it; every other mode, including progress, spills.
func guardDumpStartupOnLaunchFail(bannerMode string) bool {
	return bannerMode != guardBannerFull
}

// printGuardCompactBanner is the attended-launch banner: three lines instead of the
// ~20-line full report. It keeps the identity line (version + short build id — the "+"
// dirty marker is the staleness tell, same as the fak info pane header), the gateway URL
// (the one value every other surface hangs off), and a COPY-PASTEABLE command that
// prints the full report on demand. Prior-run refusals stay in that report: attended
// launch has a hard three-line budget so stale history cannot scroll the child UI.
func printGuardCompactBanner(w io.Writer, version, shortBuild, gwURL string, command []string, refusalCarryForward []guardRefusalCarry) {
	identity := version
	if strings.TrimSpace(shortBuild) != "" {
		identity += " (" + shortBuild + ")"
	} else {
		// No short build id means the running binary carries no VCS stamp — it cannot attest
		// its commit, so it is indistinguishable from a stale one (#3306). The compact banner
		// is fixed at three lines, so surface the defect in the identity itself rather than a
		// new row: "(no stamp)" tells an attended operator to `fak self-update --force`.
		identity += " (no stamp)"
	}
	fmt.Fprintf(w, "fak guard %s — kernel-adjudicated: %s\n", identity, strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway %s — every tool call crosses the capability floor; audit journal + /metrics live there\n", gwURL)
	fmt.Fprintf(w, "  full startup report: fak info --startup --gateway-url %s   (or relaunch with --banner=full)\n", gwURL)
}
