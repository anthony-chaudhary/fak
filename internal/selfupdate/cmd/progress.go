package selfupdatecmd

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"golang.org/x/term"
)

const (
	selfUpdateBarLength   = 20
	selfUpdateFilledGlyph = "█"
	selfUpdateEmptyGlyph  = "░"
)

var (
	selfUpdateWriterIsTerminal = func(w io.Writer) bool {
		if f, ok := w.(*os.File); ok {
			return term.IsTerminal(int(f.Fd()))
		}
		return false
	}

	selfUpdateTerminalWidth = func(w io.Writer) int {
		if f, ok := w.(*os.File); ok {
			if width, _, err := term.GetSize(int(f.Fd())); err == nil && width > 0 {
				return width
			}
		}
		return 80
	}
)

var (
	selfUpdateVerboseMu sync.Mutex
	selfUpdateVerbose   bool
)

func setSelfUpdateVerbose(v bool) {
	selfUpdateVerboseMu.Lock()
	defer selfUpdateVerboseMu.Unlock()
	selfUpdateVerbose = v
}

func isSelfUpdateVerbose() bool {
	selfUpdateVerboseMu.Lock()
	defer selfUpdateVerboseMu.Unlock()
	return selfUpdateVerbose
}

// isInteractiveProgressBar reports whether the single-line interactive progress bar should be used.
// It is active only on an interactive TTY and when verbose mode is disabled.
func isInteractiveProgressBar() bool {
	if isSelfUpdateVerbose() {
		return false
	}
	return selfUpdateWriterIsTerminal(selfUpdateProgress)
}

// formatSelfUpdateProgressBar formats the single-line progress bar text:
//
//	fak self-update · [████████░░░░░░░░░░░░]  40%  building fak-dev companion
//
// Clamped to terminal width if width > 0.
func formatSelfUpdateProgressBar(percent int, operation string, width int) string {
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}
	filled := (percent * selfUpdateBarLength) / 100
	if filled < 0 {
		filled = 0
	}
	if filled > selfUpdateBarLength {
		filled = selfUpdateBarLength
	}
	empty := selfUpdateBarLength - filled
	bar := strings.Repeat(selfUpdateFilledGlyph, filled) + strings.Repeat(selfUpdateEmptyGlyph, empty)

	raw := fmt.Sprintf("fak self-update · [%s] %3d%%  %s", bar, percent, strings.TrimSpace(operation))
	if width > 0 {
		runes := []rune(raw)
		if len(runes) > width {
			raw = string(runes[:width])
		}
	}
	return raw
}

// drawSelfUpdateProgressBar writes the progress bar on an interactive TTY with \r\x1b[2K prefix and no trailing newline.
func drawSelfUpdateProgressBar(w io.Writer, percent int, operation string) {
	width := selfUpdateTerminalWidth(w)
	bar := formatSelfUpdateProgressBar(percent, operation, width)
	fmt.Fprintf(w, "\r\x1b[2K%s", bar)
	selfUpdateProgressState.barDrawn = true
}

// clearSelfUpdateProgressBar erases the single-line progress bar if one was drawn.
func clearSelfUpdateProgressBar() {
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	clearSelfUpdateProgressBarLocked()
}

func clearSelfUpdateProgressBarLocked() {
	if selfUpdateProgressState.barDrawn || isInteractiveProgressBar() {
		fmt.Fprint(selfUpdateProgress, "\r\x1b[2K")
		selfUpdateProgressState.barDrawn = false
	}
}

// settleSelfUpdateProgressBar finalizes the single-line progress bar with a newline if drawn.
func settleSelfUpdateProgressBar() {
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	if selfUpdateProgressState.barDrawn && isInteractiveProgressBar() {
		fmt.Fprintln(selfUpdateProgress)
		selfUpdateProgressState.barDrawn = false
	}
}

// WriteSelfUpdateLog multiplexes a log line onto w while de-conflicting with any active single-line progress bar.
// If an in-place progress bar is currently active, it clears the bar (\r\x1b[2K), writes the log message
// with a trailing newline, and redraws the progress bar below.
func WriteSelfUpdateLog(w io.Writer, msg string) {
	if w == nil {
		w = selfUpdateProgress
		if w == nil {
			w = os.Stderr
		}
	}
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()

	active := selfUpdateProgressState.barDrawn
	if active {
		fmt.Fprint(w, "\r\x1b[2K")
		selfUpdateProgressState.barDrawn = false
	}
	if !strings.HasSuffix(msg, "\n") {
		msg += "\n"
	}
	fmt.Fprint(w, msg)
	if active {
		drawSelfUpdateProgressBar(w, selfUpdateProgressState.percent, selfUpdateProgressState.operation)
	}
}

// ProgressReporter provides a direct API for managing progress bar state and display.
type ProgressReporter struct {
	w          io.Writer
	verbose    bool
	isTerminal func(io.Writer) bool
	termWidth  func(io.Writer) int
	barDrawn   bool
}

// NewProgressReporter constructs a reporter targeting w.
func NewProgressReporter(w io.Writer, verbose bool) *ProgressReporter {
	return &ProgressReporter{
		w:          w,
		verbose:    verbose,
		isTerminal: selfUpdateWriterIsTerminal,
		termWidth:  selfUpdateTerminalWidth,
	}
}

// IsInteractive reports whether interactive progress bar display is active.
func (r *ProgressReporter) IsInteractive() bool {
	if r.verbose {
		return false
	}
	if r.isTerminal != nil {
		return r.isTerminal(r.w)
	}
	return false
}

// Report updates progress. In interactive TTY mode it draws the single progress bar;
// otherwise it emits standard progress lines.
func (r *ProgressReporter) Report(percent int, operation string) {
	if r.IsInteractive() {
		width := 80
		if r.termWidth != nil {
			width = r.termWidth(r.w)
		}
		bar := formatSelfUpdateProgressBar(percent, operation, width)
		fmt.Fprintf(r.w, "\r\x1b[2K%s", bar)
		r.barDrawn = true
		return
	}
	fmt.Fprintf(r.w, "self-update: progress=%d%% operation=%q\n", percent, strings.TrimSpace(operation))
}

// Clear erases the progress bar if drawn.
func (r *ProgressReporter) Clear() {
	if r.barDrawn || r.IsInteractive() {
		fmt.Fprint(r.w, "\r\x1b[2K")
		r.barDrawn = false
	}
}

// Settle finishes the progress bar cleanly by printing a trailing newline.
func (r *ProgressReporter) Settle() {
	if r.barDrawn && r.IsInteractive() {
		fmt.Fprintln(r.w)
		r.barDrawn = false
	}
}
