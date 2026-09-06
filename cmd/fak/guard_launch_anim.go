package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"golang.org/x/term"
)

// guard_launch_anim.go — the ATTENDED launch animation.
//
// A default `fak guard -- claude` used to print its startup report (once ~20 dense lines,
// then a compact 3-line banner) to the shared stderr in the instant before the wrapped
// agent's full-screen TUI painted over it. Even compacted, an attended operator reads that
// as a "wall of text that flashes" — text appears, then is gone the moment the agent draws.
// The detail is real; it just wants to LIVE in `fak info --startup` (where it already does),
// not flash-and-vanish.
//
// Explicit --banner=animate plays a short, in-place, self-clearing ICON animation — a
// single line that redraws over itself (carriage return, no scroll)
// showing fak's capability floor rising under the agent, then landing on ONE iconic identity
// line. It is a loading motif in the spirit of the host agent's own spinner: motion, not a
// paragraph. The full report stays one command away (`fak info --startup`) and is spilled in
// full ONLY when the launch actually fails (guard_child.go's spawn path), where the wall of
// text is exactly what an operator needs and no child TUI exists to corrupt.
//
// Test surface: everything asserted on is a PURE, byte-clean string builder
// (guardLaunchAnimFrames / guardLaunchSettleLines) plus a pure gate (guardLaunchAnimEnabled).
// The cursor control, color, and the real per-frame sleep live in the impure player
// (playGuardLaunchAnimation), which takes an injected sleep so a test drives it at zero delay.
// Color and cursor escapes are layered only at write time and only for a real TTY — the same
// discipline info_color.go and the guard exit summary already follow.

const (
	// guardLaunchAnimEnv is the per-harness opt-out: FAK_GUARD_LAUNCH_ANIM=off falls back to
	// the static compact banner (for an operator who wants the plain three lines, no motion).
	guardLaunchAnimEnv    = "FAK_GUARD_LAUNCH_ANIM"
	guardLaunchAnimEnvOff = "off"

	// guardLaunchFrameDelay paces the in-place redraw. guardLaunchTrackCells frames at this
	// delay is ~0.5s of motion — a satisfying "launch" beat, below the threshold where an
	// attended operator would feel the guard is slow to hand over.
	guardLaunchFrameDelay   = 55 * time.Millisecond
	guardLaunchTrackCells   = 8
	guardLaunchDefaultWidth = 80

	// guardLaunchClearLine is the carriage-return + clear-to-end-of-line the player writes
	// before every frame so the animation redraws IN PLACE on one row instead of scrolling a
	// fresh line each tick (the scroll would be the very "wall of text" this replaces).
	// Standard VT, already relied on across the guard's TTY output (the exit-summary reset,
	// the info-pane in-place redraw).
	guardLaunchClearLine = "\r\x1b[2K"

	// The floor glyphs and the settled mark. Filled cells are fak's capability floor coming
	// up under the agent; the diamond is the "floor is up, agent seated on it" resting state.
	guardLaunchFilledCell = '▰'
	guardLaunchEmptyCell  = '▱'
	guardLaunchSettleMark = '◆'
)

// guardLaunchSpinner is the single-cell braille working cycle — the classic, universally
// legible "in progress" glyph, one display cell per frame so the width cap is exact.
var guardLaunchSpinner = []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

// guardLaunchPhases narrate the three acts of a fak launch as the floor rises: the kernel
// floor arming, the gateway opening, the agent linking through it. The final frames rest on
// the last phase.
var guardLaunchPhases = []string{"arming capability floor", "opening gateway", "linking kernel"}

// guardLaunchAnimEnabled decides whether the attended launch may play the icon animation this
// run. It requires the resolved animate banner mode AND a real stderr TTY (a piped/redirected
// stderr must stay byte-clean — the carriage-return redraw would corrupt a captured log) AND
// color allowed (NO_COLOR is the community opt-out the rest of the guard honors; the animation
// IS color+cursor motion, so NO_COLOR degrades to the plain compact banner) AND no explicit
// env opt-out. Pure over its inputs so the gate is unit-tested without a terminal.
func guardLaunchAnimEnabled(bannerMode string, stderrTTY, noColor bool, env string) bool {
	if bannerMode != guardBannerAnimate {
		return false
	}
	if !stderrTTY || noColor {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(env), guardLaunchAnimEnvOff) {
		return false
	}
	return true
}

// guardLaunchAnimFrames builds the ordered, byte-clean frames of the launch animation. Each is
// a single line already capped to width display cells so the in-place redraw can never wrap
// onto a second row (a wrap would scroll and defeat the whole one-line motif). The floor fills
// left-to-right across guardLaunchTrackCells while the spinner cycles and the phase word
// advances. Color is layered by the player, never here.
func guardLaunchAnimFrames(width int) []string {
	if width <= 0 {
		width = guardLaunchDefaultWidth
	}
	// One frame per floor cell, plus a final fully-filled frame, so the floor visibly completes.
	n := guardLaunchTrackCells + 1
	frames := make([]string, 0, n)
	ver := appversion.Current()
	for i := 0; i < n; i++ {
		fill := i
		if fill > guardLaunchTrackCells {
			fill = guardLaunchTrackCells
		}
		bar := strings.Repeat(string(guardLaunchFilledCell), fill) +
			strings.Repeat(string(guardLaunchEmptyCell), guardLaunchTrackCells-fill)
		glyph := guardLaunchSpinner[i%len(guardLaunchSpinner)]
		phase := guardLaunchPhases[i*len(guardLaunchPhases)/n]
		line := fmt.Sprintf("%c fak %s  %s  %s", glyph, ver, bar, phase)
		frames = append(frames, takeCellsTUI(line, width))
	}
	return frames
}

// guardLaunchSettleLines builds the byte-clean resting state the animation lands on once the
// floor is up: ONE iconic identity line (settle mark + version/build + the agent now on the
// floor + the gateway every other surface hangs off), then a dim pointer that BOTH names the
// full report's on-demand home AND states the one case it is spilled inline (a failed launch)
// — the two situations the goal calls out. The prior-run refusal carry-forward, if any,
// follows VERBATIM: it is the one block compacting must never hide, since an operator must act
// on it before re-attempting work. Every line is width-capped; color/dim is layered by the
// player.
func guardLaunchSettleLines(version, shortBuild, agent, gwURL string, width int, refusals []guardRefusalCarry) []string {
	if width <= 0 {
		width = guardLaunchDefaultWidth
	}
	identity := version
	if strings.TrimSpace(shortBuild) != "" {
		identity += " (" + shortBuild + ")"
	} else {
		// No short build id means the running binary carries no VCS stamp — indistinguishable
		// from a stale one (#3306). Surface it in the identity itself, same as the compact banner.
		identity += " (no stamp)"
	}
	head := fmt.Sprintf("%c fak guard %s", guardLaunchSettleMark, identity)
	if a := strings.TrimSpace(agent); a != "" {
		head += " — " + a + " on the kernel floor"
	}
	head += "  ·  gateway " + gwURL
	lines := []string{takeCellsTUI(head, width)}
	lines = append(lines, takeCellsTUI("  full startup report: fak info --startup --gateway-url "+gwURL+" (spilled here only if launch fails)", width))
	return lines
}

// playGuardLaunchAnimation plays frames in place on one line, then reveals the settle lines.
// color wraps the moving frames in the guard's cyan-bold (the exit-summary heading hue) and
// dims the settle pointer; sleep is injected so a test runs it at zero delay. w is the terminal
// (os.Stderr in production). The last frame is cleared before the settle so no animation
// residue survives into the scrollback the wrapped agent's alt-screen restores on exit.
func playGuardLaunchAnimation(w io.Writer, frames, settle []string, color bool, sleep func(time.Duration)) {
	for _, f := range frames {
		fmt.Fprint(w, guardLaunchClearLine)
		fmt.Fprint(w, guardLaunchPaint(f, tuiSGRCyanBold, color))
		sleep(guardLaunchFrameDelay)
	}
	// Land: clear the last frame, then print each settle line on its own row.
	fmt.Fprint(w, guardLaunchClearLine)
	for i, line := range settle {
		switch {
		case i == 0:
			fmt.Fprintln(w, guardLaunchPaint(line, tuiSGRCyanBold, color))
		case i == 1:
			fmt.Fprintln(w, guardLaunchPaint(line, tuiSGRDim, color))
		default:
			// Refusal carry-forward rows stay plain — an operator must read them, not skim them.
			fmt.Fprintln(w, line)
		}
	}
}

// guardLaunchPaint wraps s in an SGR pair for a color TTY, or returns it byte-clean otherwise.
// The whole-row wrap (open … reset) matches the info-pane color discipline: escapes are zero
// display cells and the reset ends every row, so a hue can never bleed past its line.
func guardLaunchPaint(s, sgr string, color bool) string {
	if !color || sgr == "" {
		return s
	}
	return sgr + s + tuiSGRReset
}

// printGuardLaunchAnimation is the production entry the guard's banner switch calls for an
// attended interactive launch: build the frames + settle lines for the current identity and
// play them to w (os.Stderr) with real per-frame sleeps. It mirrors printGuardCompactBanner's
// role so the switch can pick between the two, and always lands on a stable identity line, the
// `fak info --startup` pointer, and any refusal carry-forward.
func printGuardLaunchAnimation(w io.Writer, version, shortBuild, gwURL string, command []string, refusals []guardRefusalCarry, width int) {
	agent := ""
	if len(command) > 0 {
		agent = guardLaunchAgentName(command[0])
	}
	frames := guardLaunchAnimFrames(width)
	settle := guardLaunchSettleLines(version, shortBuild, agent, gwURL, width, refusals)
	playGuardLaunchAnimation(w, frames, settle, true, time.Sleep)
}

// guardLaunchAgentName is the short, human name of the wrapped agent for the settle line — the
// command's base with any executable extension trimmed ("C:\\...\\claude.exe" -> "claude").
func guardLaunchAgentName(cmd0 string) string {
	// Normalize backslashes before filepath.Base: it only splits the HOST separator,
	// so a Windows launch path (C:\tools\Claude.exe) fed to the Linux test runner would
	// otherwise come back whole. Same host-independence fix as wipFenceSlug.
	base := filepath.Base(strings.ReplaceAll(strings.TrimSpace(cmd0), `\`, "/"))
	for _, ext := range []string{".exe", ".cmd", ".bat"} {
		if strings.EqualFold(filepath.Ext(base), ext) {
			return strings.ToLower(strings.TrimSuffix(base, filepath.Ext(base)))
		}
	}
	return strings.ToLower(base)
}

// guardLaunchStderrWidth measures the terminal width for the animation, or 0 ("unknown", the
// builders default to guardLaunchDefaultWidth) off a TTY. Keeps the render site a one-liner.
func guardLaunchStderrWidth() int {
	if w, _, err := term.GetSize(int(os.Stderr.Fd())); err == nil && w > 0 {
		return w
	}
	return 0
}
