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
// OS-picked loopback port, torn down when the agent exits). So the AGENT must launch INLINE in
// the current pane — never a fresh window, which would orphan the gateway this process owns.
// The overlay pane prefers the current window too (tmux / Windows Terminal / iTerm2 split it);
// Apple Terminal is the one host with NO split panes, so there the overlay opens as a
// companion Terminal WINDOW — still only a poller of this guard's loopback gateway, so nothing
// is orphaned. The overlay polls this guard's own loopback gateway (auth-exempt on loopback),
// so the bearer is never placed on a pane command line.
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

// guardSplitPlan is the resolved inline-split plan. Spawn is the host argv that opens
// the overlay surface; Overlay is the fak-info command that surface runs (recorded for
// the dry-run/json surfaces). Host names the resolved split host ("tmux" | "wt" |
// "iterm2" | "terminal-app" | "none"). There is no Agent/Claude field: the agent always
// launches inline in the current pane AFTER this plan opens the overlay, so guard keeps
// its in-process gateway.
type guardSplitPlan struct {
	Host     string   `json:"host"`
	Where    string   `json:"where"`
	Geometry string   `json:"geometry"`
	Spawn    []string `json:"spawn,omitempty"`
	Overlay  []string `json:"overlay"`
	Fallback string   `json:"fallback,omitempty"`
}

// buildGuardSplitPlan resolves the split host and assembles the argv that opens the
// overlay surface. It is pure: goos/getenv/lookPath are injected, and overlayArgs are
// the child argv AFTER the fak executable (selfExe is prepended here). Detection order:
// inside tmux ($TMUX) -> inside Windows Terminal ($WT_SESSION + `wt`) -> a macOS
// terminal app scriptable via osascript (iTerm2 splits the current window; Apple
// Terminal has no split panes, so it gets a companion window) -> none.
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
	// (bottom strip), -h splits side-by-side (right column). -d creates the overlay pane
	// WITHOUT making it the active one, so keyboard focus stays on the agent pane the
	// operator actually types into — the fak-info pane is read-only observability that
	// should never steal the cursor. The agent then launches inline in the current (80%)
	// pane, which is still the focused one.
	if strings.TrimSpace(getenv("TMUX")) != "" {
		orient := "-v"
		if where == "right" {
			orient = "-h"
		}
		plan.Host = "tmux"
		plan.Spawn = append([]string{"tmux", "split-window", orient, "-d", "-l", "20%", "--"}, overlayCmd...)
		return plan, nil
	}

	// 2. Inside Windows Terminal: $WT_SESSION is set for any shell running in a WT pane, which
	// is the reliable "we are inside WT" signal (the analogue of $TMUX). `wt -w 0` targets the
	// CURRENT window, so split-pane adds the overlay beside this session rather than opening a
	// new window (which would orphan guard's in-process gateway). -H makes a bottom strip, -V a
	// right column; -s 0.2 sizes the new (overlay) pane to 20%.
	//
	// split-pane focuses the NEW (overlay) pane, which would leave the cursor on the read-only
	// fak-info pane instead of the agent. wt has no "don't focus" flag, but it runs a chain of
	// `;`-separated subcommands against the same -w 0 window, so we append `move-focus` back to
	// the agent pane: it sits ABOVE a bottom strip (-H) so focus returns UP, and to the LEFT of
	// a right column (-V) so focus returns LEFT. The agent then launches inline in that focused
	// pane.
	if goos == "windows" && strings.TrimSpace(getenv("WT_SESSION")) != "" {
		if _, err := lookPath("wt"); err == nil {
			orient, focusBack := "-H", "up"
			if where == "right" {
				orient, focusBack = "-V", "left"
			}
			plan.Host = "wt"
			spawn := []string{"wt", "-w", "0", "split-pane", orient, "-s", "0.2"}
			spawn = append(spawn, overlayCmd...)
			spawn = append(spawn, ";", "move-focus", focusBack)
			plan.Spawn = spawn
			return plan, nil
		}
	}

	// 3. macOS terminal apps, scripted via osascript (always present on macOS). A stock Mac
	// ships neither tmux nor Windows Terminal, so before this rung an attended `fak guard
	// -- claude` in Terminal.app/iTerm2 silently skipped the split and fak stayed invisible
	// for the whole session — the one attended platform with NO fak surface at all.
	if app := macSplitTerminalApp(goos, getenv); app != "" {
		if _, err := lookPath("osascript"); err == nil {
			if app == "iterm2" {
				// iTerm2 is a true inline split of the CURRENT window. Its AppleScript verbs
				// name the DIVIDER: "split horizontally" draws a horizontal divider (new pane
				// BELOW — the bottom strip), "split vertically" a vertical one (new pane to the
				// RIGHT — the right column). The API has no pane-size parameter, so the split
				// is even, not 80/20 — the geometry label says so rather than claiming 20%.
				// The new session is created without being selected, so keyboard focus stays
				// in the current (agent) pane.
				verb, geometry := "split horizontally", "agent (top) / fak info (bottom pane — iTerm2 even split)"
				if where == "right" {
					verb, geometry = "split vertically", "agent (left) / fak info (right pane — iTerm2 even split)"
				}
				plan.Host = "iterm2"
				plan.Geometry = geometry
				plan.Spawn = []string{"osascript", "-e", fmt.Sprintf(
					`tell application "iTerm2" to tell current session of current window to %s with same profile command "%s"`,
					verb, appleScriptQuote(shellJoin(overlayCmd)))}
				return plan, nil
			}
			// Apple Terminal has no split panes, so the overlay opens as a companion WINDOW
			// (`do script` runs the overlay in a fresh window; --split-where has nothing to
			// place). That never orphans the gateway — the agent still launches inline in
			// THIS window; the companion only polls the loopback gateway URL. `do script`
			// focuses the new window, so the script re-fronts the agent window (index 1)
			// to keep the cursor where the operator types. The typed command line carries
			// its own close, so the window does not outlive the overlay process (#5482 —
			// guardSplitAppleTerminalCommand).
			plan.Host = "terminal-app"
			plan.Geometry = "agent (this window) / fak info (companion Terminal window)"
			plan.Spawn = []string{"osascript",
				"-e", `tell application "Terminal"`,
				"-e", `set agentWindow to front window`,
				"-e", `do script "` + appleScriptQuote(guardSplitAppleTerminalCommand(overlayCmd, guardSplitAppleCloseOnExit(getenv))) + `"`,
				"-e", `set index of agentWindow to 1`,
				"-e", `end tell`,
			}
			return plan, nil
		}
	}

	// 4. No usable multiplexer context: render an actionable fallback (open a pane yourself and
	// run the overlay in it).
	plan.Host = "none"
	plan.Fallback = guardSplitFallbackRecipe(overlayCmd)
	return plan, nil
}

// macSplitTerminalApp resolves which scriptable macOS terminal app this session is
// attended in: "iterm2" (true inline split panes), "terminal-app" (Apple Terminal —
// no split panes, companion window instead), or "" (neither, or not macOS). iTerm2
// is recognized by either of its two markers ($ITERM_SESSION_ID survives shells that
// rewrite $TERM_PROGRAM); Apple Terminal only by $TERM_PROGRAM. Every other
// TERM_PROGRAM (vscode, ...) stays "" — an unknown host must keep today's silent
// no-op, not gain a surprise osascript spawn.
func macSplitTerminalApp(goos string, getenv func(string) string) string {
	if goos != "darwin" {
		return ""
	}
	if strings.TrimSpace(getenv("ITERM_SESSION_ID")) != "" || strings.TrimSpace(getenv("TERM_PROGRAM")) == "iTerm.app" {
		return "iterm2"
	}
	if strings.TrimSpace(getenv("TERM_PROGRAM")) == "Apple_Terminal" {
		return "terminal-app"
	}
	return ""
}

// guardSplitAppleCloseTail is the tail appended to the companion window's command line so the
// WINDOW closes when the overlay ends (#5482).
//
// What it replaces was a bare `; exit` whose comment claimed the window would then close itself
// "with the default close-on-clean-exit profile". There is no such default: Apple Terminal's
// stock Basic profile ships `When the shell exits: Don't close the window`. So the overlay
// process exited on time (the #2340 --max-idle backstop), the shell exited, and the window
// stayed parked on `[Process completed]` — one dead window per guarded session, accumulating
// monotonically (an attended Terminal reached ~100 of them across two days, ~84% of its
// windows). #2340 made the PROCESS exit; on this host the window is not the process, so process
// exit is necessary but not sufficient and the close has to be explicit.
//
// Four properties the shape is chosen for:
//
//   - It cannot mis-target. The spawn script above does resolve `front window` — but only to
//     re-front the AGENT window in the same breath as `do script`. The CLOSE runs minutes to
//     hours later, when the front window is whatever the operator has since focused; closing
//     that would destroy a live terminal, which is far worse than leaking a dead one. So the
//     close selects by the overlay's OWN tty, read inside the overlay's own shell. `every window
//     whose ...` is a filter, so no match (the operator already closed it) is a silent no-op
//     rather than an error.
//   - It preserves a crash. The close is gated on the overlay exiting 0, so a real failure keeps
//     its window — and the only copy of the diagnostic — on screen. Both routine ends exit 0
//     (info.go: the "gateway closed — guarded session ended" close and the --max-idle backstop),
//     as does an in-band Ctrl-C/q quit, which is an operator saying "done with this overlay". A
//     flag error, a failed fetch, or an uncaught signal is non-zero, and stays parked.
//   - It is inert to quoting. The tty goes to osascript as a run-handler ARGUMENT, never
//     interpolated into AppleScript source, so nothing a device path (or an overlay argument
//     further up the line) could contain can open a second AppleScript statement.
//   - It degrades to the old behavior. The overlay command still LEADS the line, so a shell that
//     cannot parse the tail still runs the dashboard; and `exit` still runs on both paths, so a
//     close that fails leaves exactly the pre-#5482 dead window, never a live shell nobody reaps.
//
// The nohup + `&` + `exit` ordering is deliberate on both counts: zsh (the macOS login shell)
// HUPs background jobs as it exits, so the closer must be immune to that; and the shell wants to
// be GONE when Terminal handles the close, because the profile default ("Ask before closing: if
// there are processes other than the login shell and ...") can otherwise raise a confirmation
// dialog for a window that still has a process on its tty.
//
// NOT VERIFIED END TO END: this was written without a Mac. guard_split_close_test.go witnesses
// the generated command line — that Terminal then closes that window, silently, is unwitnessed
// and wants a Mac operator (#5482).
const guardSplitAppleCloseTail = `__fak_rc=$?; if [ "$__fak_rc" = 0 ]; then nohup /usr/bin/osascript -e 'on run {overlayTTY}' -e 'tell application "Terminal" to close (every window whose tty of tab 1 is overlayTTY)' -e 'end run' "$(tty)" >/dev/null 2>&1 & fi; exit $__fak_rc`

// guardSplitAppleTerminalCommand builds the single shell line Apple Terminal's `do script` types
// into the companion window: the overlay command, then either the self-closing tail (#5482, the
// default) or the bare `; exit` it replaced.
func guardSplitAppleTerminalCommand(overlayCmd []string, closeOnExit bool) string {
	if !closeOnExit {
		return shellJoin(overlayCmd) + "; exit"
	}
	return shellJoin(overlayCmd) + "; " + guardSplitAppleCloseTail
}

// guardSplitAppleCloseOnExit resolves the #5482 escape hatch. Closing is the DEFAULT — the whole
// point is that auto-close stops depending on a profile setting the operator never chose —
// and FAK_SPLIT_CLOSE_ON_EXIT=0|off|false|no restores the lingering `[Process completed]` window
// for an operator who wants to read one post mortem. It reads the same injected getenv the rest
// of the plan builder does, so the knob stays inside the pure plan and is covered by its tests.
func guardSplitAppleCloseOnExit(getenv func(string) string) bool {
	switch strings.TrimSpace(strings.ToLower(getenv("FAK_SPLIT_CLOSE_ON_EXIT"))) {
	case "0", "off", "false", "no":
		return false
	}
	return true
}

// appleScriptQuote escapes s for embedding inside a double-quoted AppleScript string
// literal. AppleScript strings have exactly two escapes: backslash and double-quote.
func appleScriptQuote(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return strings.ReplaceAll(s, `"`, `\"`)
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
	b.WriteString("fak guard --split: no splittable terminal context found (looked for $TMUX; on Windows, $WT_SESSION + `wt`; on macOS, $TERM_PROGRAM naming iTerm2 or Apple Terminal).\n")
	b.WriteString("open a second pane/window yourself (tmux split, Windows Terminal Alt+Shift+- / Alt+Shift+plus, or a second terminal window) and run the fak-info overlay there:\n")
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
	// The pane labels stay honest per host: only tmux/wt can actually promise 80/20
	// (iTerm2's AppleScript split is even; Apple Terminal gets a companion window).
	agentLabel, infoLabel := "agent pane (80%): current pane", "fak info pane (20%)"
	switch p.Host {
	case "iterm2":
		agentLabel, infoLabel = "agent pane: current pane", "fak info pane (even split)"
	case "terminal-app":
		agentLabel, infoLabel = "agent: this window", "fak info (companion window)"
	}
	fmt.Fprintf(&b, "  %s (inline launch after the overlay opens)\n", agentLabel)
	fmt.Fprintf(&b, "  %s: %s\n", infoLabel, strings.Join(p.Overlay, " "))
	return b.String()
}

// guardSplitEnabled resolves the --split tri-state (auto|on|off) into a decision. AUTO (the
// default) enables the inline fak-info pane ONLY for an attended interactive launch inside a
// known splittable terminal context (tmux, Windows Terminal, or — via the guardSplitGOOS
// seam — a scriptable macOS terminal app: iTerm2 / Apple Terminal), and never recursively
// (the spawned pane and the agent inherit FAK_GUARD_SPLIT=1). on forces it (the runner
// prints the fallback recipe when no host is found); off disables it. Every
// non-interactive / headless / CI / plain-terminal launch falls through to a no-op, so the
// default is invisible and harmless exactly where a split cannot help — there is zero
// behavior change for those paths.
func guardSplitEnabled(mode string, getenv func(string) string, stdinInteractive, childInteractive bool) (bool, error) {
	normalized := strings.TrimSpace(strings.ToLower(mode))
	// The nesting guard applies to EVERY enabling mode, not just auto: the spawned overlay
	// pane and the inline agent both inherit FAK_GUARD_SPLIT=1 (set in cmdGuard), so a second
	// `fak guard` launched inside this session — including an explicit --split=on a wrapper or
	// the agent itself runs — must never re-split an already-split pane. Previously only the
	// auto branch consulted FAK_GUARD_SPLIT, so --split=on (and its true/1/yes aliases) would
	// recursively split. off short-circuits before this so a deliberate opt-OUT is honored
	// even inside a split.
	switch normalized {
	case "off", "false", "0", "no":
		return false, nil
	}
	if strings.TrimSpace(getenv("FAK_GUARD_SPLIT")) != "" {
		return false, nil // already inside a fak split — never nest, regardless of mode.
	}
	switch normalized {
	case "", "auto":
		if !stdinInteractive || !childInteractive {
			return false, nil // headless / piped / -p run: nothing to sit beside.
		}
		inMux := strings.TrimSpace(getenv("TMUX")) != "" || strings.TrimSpace(getenv("WT_SESSION")) != "" ||
			macSplitTerminalApp(guardSplitGOOS, getenv) != ""
		return inMux, nil
	case "on", "true", "1", "yes":
		return true, nil
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
	// Open the overlay pane WITHOUT blocking the agent launch. On Windows `wt.exe` is an AppX
	// execution alias whose cold start alone costs ~200ms, and `tmux split-window` round-trips to
	// the running server too — cmd.Run() would pay that on the CRITICAL PATH, between the gateway
	// going healthy and Claude starting, so the operator waits ~200ms longer to reach a working
	// agent. The pane is pure observability that sits BESIDE the agent; it does not need to be up
	// first. So fire-and-reap it in the background and return immediately, letting the caller
	// launch the agent now; the pane appears a beat later. A spawn that fails to even start is
	// still reported (the multiplexer client itself is missing), but a slow client no longer
	// gates the launch.
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(stderr, "fak guard: --split: opening the fak-info pane failed: %v (continuing without it)\n", err)
		return
	}
	go func() { _ = cmd.Wait() }() // reap the multiplexer client so it never lingers as a zombie.
	// A successful split is visible by construction. Do not narrate it into the agent
	// pane; failures below remain actionable and therefore visible.
}

// guardInfoPaneMaxIdle is the #2340 backstop applied to the AUTO-spawned split pane: if its
// gateway never answers within this long, the pane self-exits instead of polling a dead URL
// forever and leaking an OpenConsole pane + its threads into the WindowsTerminal host (measured
// live: one host at 1576 threads / 31k handles because these panes never exited). It is safe to
// set aggressively because the backstop ONLY fires for a pane that was NEVER healthy — a gateway
// that came up and later died still gets the friendlier "session ended" close, and a healthy
// dashboard pane runs indefinitely regardless of this value. Manual `fak info` keeps default 0.
const guardInfoPaneMaxIdle = 5 * time.Minute

// guardInfoPaneOverlayArgs is the single source of truth for the `fak info` child argv the
// split pane runs, so the dry-run preview and the live spawn can never drift. Kept beside
// openGuardInfoPane (which calls the same shape inline) so the two stay in lockstep. The
// auto-spawned pane always carries the #2340 --max-idle backstop (see guardInfoPaneMaxIdle).
func guardInfoPaneOverlayArgs(gwURL string, interval time.Duration) []string {
	return []string{"info", "--gateway-url", gwURL, "--interval", interval.String(), "--max-idle", guardInfoPaneMaxIdle.String()}
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
