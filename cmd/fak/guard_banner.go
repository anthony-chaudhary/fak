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
//	auto (default)  compact for an ATTENDED INTERACTIVE launch; the FULL report for
//	                headless / piped / scripted launches, byte-for-byte — a captured
//	                fleet log wants the detail and has no TUI to corrupt.
//	full            always the full report (today's pre-flag behavior, forced).
//	compact         always the compact banner.
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
)

// guardBannerModeDecision resolves the --banner flag to a concrete mode. Precedence,
// highest first: --quiet silences everything (its existing contract — the banner is part
// of what it already suppressed); an explicit full/compact/off is a knowing choice and
// wins; AUTO keys on the same attended-vs-headless split guardDebugStatsToSharedStderr
// and --split auto use — an interactive stdin AND an interactive child mean a human is
// about to hand the terminal to a full-screen agent UI, so the wall of text buys nothing
// there. An unknown value is a loud usage error, never a silent fallback.
func guardBannerModeDecision(banner string, quiet, stdinInteractive, childInteractive bool) (string, error) {
	mode := strings.ToLower(strings.TrimSpace(banner))
	if mode == "" {
		mode = guardBannerAuto
	}
	switch mode {
	case guardBannerAuto, guardBannerFull, guardBannerCompact, guardBannerOff:
	default:
		return "", fmt.Errorf("--banner must be auto, full, compact, or off; got %q", banner)
	}
	if quiet {
		return guardBannerOff, nil
	}
	if mode != guardBannerAuto {
		return mode, nil
	}
	if stdinInteractive && childInteractive {
		return guardBannerCompact, nil
	}
	return guardBannerFull, nil
}

// printGuardCompactBanner is the attended-launch banner: three lines instead of the
// ~20-line full report. It keeps the identity line (version + short build id — the "+"
// dirty marker is the staleness tell, same as the fak info pane header), the gateway URL
// (the one value every other surface hangs off), and a COPY-PASTEABLE command that
// prints the full report on demand. The prior-run refusal carry-forward still prints in
// full — it is the one actionable block an operator must see BEFORE re-attempting work,
// so compacting must never hide it.
func printGuardCompactBanner(w io.Writer, version, shortBuild, gwURL string, command []string, refusalCarryForward []guardRefusalCarry) {
	identity := version
	if strings.TrimSpace(shortBuild) != "" {
		identity += " (" + shortBuild + ")"
	}
	fmt.Fprintf(w, "fak guard %s — kernel-adjudicated: %s\n", identity, strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway %s — every tool call crosses the capability floor; audit journal + /metrics live there\n", gwURL)
	fmt.Fprintf(w, "  full startup report: fak info --startup --gateway-url %s   (or relaunch with --banner=full)\n", gwURL)
	fmt.Fprint(w, formatGuardRefusalCarryForward(refusalCarryForward))
}
