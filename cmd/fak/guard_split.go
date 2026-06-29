package main

import (
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// `fak guard --split`: the default-launch UI/UX upgrade. A bare `fak guard -- claude` hands
// the whole terminal to Claude Code (alternate screen + full repaint), so fak — the kernel
// adjudicating every tool call and the cache economy it is winning — goes invisible for the
// rest of the session. --split fixes that by opening a 20% pane running `fak info` BESIDE the
// 80% agent pane, so the live cache/token economy and the floor's safety counters stay on
// screen the whole session.
//
// The constraint that shapes the design: `fak guard` holds the gateway IN THIS PROCESS (an
// OS-picked loopback port, torn down when the agent exits). So the split must be INLINE — open
// the overlay pane in the CURRENT terminal window and run the agent inline in this pane —
// never a fresh window, which would orphan the gateway this process owns. The overlay polls
// this guard's own loopback gateway (auth-exempt on loopback), so the bearer is never placed
// on a pane command line.
//
// Like the rest of cmd/fak this is a PURE plan builder (buildGuardSplitPlan, zero side
// effects) plus a thin runner (openGuardInfoPane) that execs the resolved multiplexer argv.

// guardSplitGOOS / guardSplitLookPath are the indirection seams for multiplexer detection so a
// test can pin the platform and the available multiplexers without touching the real host
// (the os/exec-style seam used throughout cmd/fak).
var (
	guardSplitGOOS     = runtime.GOOS
	guardSplitLookPath = exec.LookPath
)

// guardSplitPlan is the resolved inline-split plan. Spawn is the multiplexer argv that opens
// the 20% overlay pane in the CURRENT window; Overlay is the fak-info command that pane runs
// (recorded for the dry-run/json surfaces). Host names the resolved multiplexer ("tmux" |
// "wt" | "none"). There is no Agent/Claude field: the agent always launches inline in the
// current pane AFTER this plan opens the overlay, so guard keeps its in-process gateway.
type guardSplitPlan struct {
	Host     string   `json:"host"`
	Where    string   `json:"where"`
	Geometry string   `json:"geometry"`
	Spawn    []string `json:"spawn,omitempty"`
	Overlay  []string `json:"overlay"`
	Fallback string   `json:"fallback,omitempty"`
}

// buildGuardSplitPlan resolves the multiplexer and assembles the argv that opens the 20%
// overlay pane in the current window. It is pure: goos/getenv/lookPath are injected, and
// overlayArgs are the child argv AFTER the fak executable (selfExe is prepended here).
// Detection order: inside tmux ($TMUX) -> inside Windows Terminal ($WT_SESSION + `wt`) ->
// none. Both supported hosts split the CURRENT window so the overlay lands beside the inline
// agent pane.
func buildGuardSplitPlan(goos string, getenv func(string) string, lookPath func(string) (string, error), selfExe, where string, overlayArgs []string) (guardSplitPlan, error) {
	where = strings.TrimSpace(strings.ToLower(where))
	if where == "" {
		where = "bottom"
	}
	if where != "bottom" && where != "right" {
		return guardSplitPlan{}, fmt.Errorf("--split-where must be %q or %q, got %q", "bottom", "right", where)
	}
	overlayCmd := append([]string{selfExe}, overlayArgs...)
	plan := guardSplitPlan{
		Where:    where,
		Geometry: guardSplitGeometryLabel(where),
		Overlay:  overlayCmd,
	}

	// 1. Inside tmux: split the current window for the 20% overlay pane. -v stacks panes
	// (bottom strip), -h splits side-by-side (right column). The agent then launches inline
	// in the current (80%) pane.
	if strings.TrimSpace(getenv("TMUX")) != "" {
		orient := "-v"
		if where == "right" {
			orient = "-h"
		}
		plan.Host = "tmux"
		plan.Spawn = append([]string{"tmux", "split-window", orient, "-l", "20%", "--"}, overlayCmd...)
		return plan, nil
	}

	// 2. Inside Windows Terminal: $WT_SESSION is set for any shell running in a WT pane, which
	// is the reliable "we are inside WT" signal (the analogue of $TMUX). `wt -w 0` targets the
	// CURRENT window, so split-pane adds the overlay beside this session rather than opening a
	// new window (which would orphan guard's in-process gateway). -H makes a bottom strip, -V a
	// right column; -s 0.2 sizes the new (overlay) pane to 20%.
	if goos == "windows" && strings.TrimSpace(getenv("WT_SESSION")) != "" {
		if _, err := lookPath("wt"); err == nil {
			orient := "-H"
			if where == "right" {
				orient = "-V"
			}
			plan.Host = "wt"
			spawn := []string{"wt", "-w", "0", "split-pane", orient, "-s", "0.2"}
			plan.Spawn = append(spawn, overlayCmd...)
			return plan, nil
		}
	}

	// 3. No usable multiplexer context: render an actionable fallback (open a pane yourself and
	// run the overlay in it).
	plan.Host = "none"
	plan.Fallback = guardSplitFallbackRecipe(overlayCmd)
	return plan, nil
}

func guardSplitGeometryLabel(where string) string {
	if where == "right" {
		return "agent 80% (left) / fak info 20% (right column)"
	}
	return "agent 80% (top) / fak info 20% (bottom strip)"
}

// guardSplitFallbackRecipe is printed when no multiplexer context is found: how to open a
// second pane and the exact overlay command to run in it.
func guardSplitFallbackRecipe(overlayCmd []string) string {
	var b strings.Builder
	b.WriteString("fak guard --split: no terminal multiplexer context found (looked for $TMUX and, on Windows, $WT_SESSION + `wt`).\n")
	b.WriteString("open a second pane yourself (tmux split, or Windows Terminal Alt+Shift+- / Alt+Shift+plus) and run the 20% fak-info overlay there:\n")
	fmt.Fprintf(&b, "  %s\n", strings.Join(overlayCmd, " "))
	return b.String()
}

// renderGuardSplitPlan renders the plan for the optional --split-dry-run surface: the
// geometry, the resolved multiplexer, the spawn argv, and the overlay command. It never
// prints a bearer (the overlay carries only the loopback gateway URL, never a token).
func renderGuardSplitPlan(p guardSplitPlan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard --split · %s\n", p.Geometry)
	fmt.Fprintf(&b, "host: %s\n", p.Host)
	if p.Host == "none" {
		b.WriteString(p.Fallback)
		return b.String()
	}
	fmt.Fprintf(&b, "spawn: %s\n", strings.Join(p.Spawn, " "))
	b.WriteString("  agent pane (80%): current pane (inline launch after the overlay pane opens)\n")
	fmt.Fprintf(&b, "  fak info pane (20%%): %s\n", strings.Join(p.Overlay, " "))
	return b.String()
}

// guardSplitEnabled resolves the --split tri-state (auto|on|off) into a decision. AUTO (the
// default) enables the inline fak-info pane ONLY for an attended interactive launch inside a
// known terminal multiplexer (tmux or Windows Terminal), and never recursively (the spawned
// pane and the agent inherit FAK_GUARD_SPLIT=1). on forces it (the runner prints the fallback
// recipe when no multiplexer is found); off disables it. Every non-interactive / headless /
// CI / plain-terminal launch falls through to a no-op, so the default is invisible and
// harmless exactly where a split cannot help — there is zero behavior change for those paths.
func guardSplitEnabled(mode string, getenv func(string) string, stdinInteractive, childInteractive bool) (bool, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "auto":
		if strings.TrimSpace(getenv("FAK_GUARD_SPLIT")) != "" {
			return false, nil // already inside a fak split — never nest.
		}
		if !stdinInteractive || !childInteractive {
			return false, nil // headless / piped / -p run: nothing to sit beside.
		}
		inMux := strings.TrimSpace(getenv("TMUX")) != "" || strings.TrimSpace(getenv("WT_SESSION")) != ""
		return inMux, nil
	case "on", "true", "1", "yes":
		return true, nil
	case "off", "false", "0", "no":
		return false, nil
	default:
		return false, fmt.Errorf("--split must be auto|on|off, got %q", mode)
	}
}

// guardChildInteractive reports whether the wrapped agent argv looks like an attended
// interactive session (vs a headless one-shot). A headless `-p` / `--print` run paints no
// alternate screen and exits, so a split pane beside it is pointless — those default to false.
func guardChildInteractive(command []string) bool {
	for _, a := range command {
		if a == "-p" || a == "--print" || strings.HasPrefix(a, "--print=") {
			return false
		}
	}
	return true
}

// openGuardInfoPane builds the inline-split plan and opens the 20% fak-info pane beside the
// current one, pointed at this guard's own loopback gateway. It is best-effort: a missing
// multiplexer prints the fallback recipe and a failed spawn prints a note — neither is fatal,
// because the agent still launches inline in this pane (just without the overlay). execCommand
// is the os/exec seam shared with claude_mac_fak.go so a test can capture the spawn.
func openGuardInfoPane(stderr io.Writer, getenv func(string) string, where, gwURL string, interval time.Duration) {
	selfExe := tuiExecutable()
	plan, err := buildGuardSplitPlan(guardSplitGOOS, getenv, guardSplitLookPath, selfExe, where, guardInfoPaneOverlayArgs(gwURL, interval))
	if err != nil {
		fmt.Fprintf(stderr, "fak guard: --split: %v\n", err)
		return
	}
	if plan.Host == "none" {
		fmt.Fprint(stderr, plan.Fallback)
		return
	}
	cmd := execCommand(plan.Spawn[0], plan.Spawn[1:]...)
	cmd.Stderr = stderr // wt/tmux are thin clients to the running multiplexer; stdout stays clean.
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(stderr, "fak guard: --split: opening the fak-info pane failed: %v (continuing without it)\n", err)
		return
	}
	fmt.Fprintf(stderr, "fak guard: --split · opened a 20%% fak-info pane (%s); launching the agent in this pane ...\n", plan.Geometry)
}

// guardInfoPaneOverlayArgs is the single source of truth for the `fak info` child argv the
// split pane runs, so the dry-run preview and the live spawn can never drift. Kept beside
// openGuardInfoPane (which calls the same shape inline) so the two stay in lockstep.
func guardInfoPaneOverlayArgs(gwURL string, interval time.Duration) []string {
	return []string{"info", "--gateway-url", gwURL, "--interval", interval.String()}
}

// renderGuardInfoPaneDryRun resolves the split plan and renders it WITHOUT spawning anything —
// the --split-dry-run surface. An operator can preview the resolved multiplexer, the 80/20
// geometry, and the exact `fak info` pane command before handing the terminal to the agent,
// instead of having to launch the split (which takes over the screen) to find out what it does.
// A bad --split-where is the only error; it returns the message and a non-zero code.
func renderGuardInfoPaneDryRun(getenv func(string) string, where, gwURL string, interval time.Duration) (string, int) {
	selfExe := tuiExecutable()
	plan, err := buildGuardSplitPlan(guardSplitGOOS, getenv, guardSplitLookPath, selfExe, where, guardInfoPaneOverlayArgs(gwURL, interval))
	if err != nil {
		return fmt.Sprintf("fak guard: --split: %v\n", err), 2
	}
	return renderGuardSplitPlan(plan), 0
}
