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
	selfUpdateProgressMu sync.Mutex
	selfUpdateVerboseMu  sync.Mutex
	selfUpdateVerbose    bool
)

func authoritativeProgressWriter() io.Writer {
	if selfUpdateProgress != nil {
		return selfUpdateProgress
	}
	return os.Stderr
}

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
	return selfUpdateWriterIsTerminal(authoritativeProgressWriter())
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
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	clearSelfUpdateProgressBarLocked()
}

func clearSelfUpdateProgressBarLocked() {
	if selfUpdateProgressState.barDrawn || isInteractiveProgressBar() {
		fmt.Fprint(authoritativeProgressWriter(), "\r\x1b[2K")
		selfUpdateProgressState.barDrawn = false
	}
}

// settleSelfUpdateProgressBar finalizes the single-line progress bar with a newline if drawn.
func settleSelfUpdateProgressBar() {
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()
	if selfUpdateProgressState.barDrawn && isInteractiveProgressBar() {
		fmt.Fprintln(authoritativeProgressWriter())
		selfUpdateProgressState.barDrawn = false
	}
}

// WriteSelfUpdateLog multiplexes a log line onto w while de-conflicting with any active single-line progress bar.
// If an in-place progress bar is currently active, it clears the bar (\r\x1b[2K) on the authoritative progress stream,
// writes the log message to w with a trailing newline, and redraws the progress bar on the authoritative progress stream.
func WriteSelfUpdateLog(w io.Writer, msg string) {
	progressStream := authoritativeProgressWriter()
	if w == nil {
		w = progressStream
	}
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()

	active := selfUpdateProgressState.barDrawn
	if active {
		fmt.Fprint(progressStream, "\r\x1b[2K")
		selfUpdateProgressState.barDrawn = false
	}
	if !strings.HasSuffix(msg, "\n") {
		msg += "\n"
	}
	fmt.Fprint(w, msg)
	if active {
		drawSelfUpdateProgressBar(progressStream, selfUpdateProgressState.percent, selfUpdateProgressState.operation)
	}
}

// ProgressReporter provides a direct API for managing progress bar state and display.
type ProgressReporter struct {
	mu         sync.Mutex
	w          io.Writer
	verbose    bool
	isTerminal func(io.Writer) bool
	termWidth  func(io.Writer) int
	barDrawn   bool
}

// NewProgressReporter constructs a reporter targeting w.
func NewProgressReporter(w io.Writer, verbose bool) *ProgressReporter {
	if w == nil {
		w = authoritativeProgressWriter()
	}
	return &ProgressReporter{
		w:          w,
		verbose:    verbose,
		isTerminal: selfUpdateWriterIsTerminal,
		termWidth:  selfUpdateTerminalWidth,
	}
}

func (r *ProgressReporter) writer() io.Writer {
	if r.w != nil {
		return r.w
	}
	return authoritativeProgressWriter()
}

func (r *ProgressReporter) isAuthoritative() bool {
	return r.writer() == authoritativeProgressWriter()
}

// IsInteractive reports whether interactive progress bar display is active.
func (r *ProgressReporter) IsInteractive() bool {
	if r.verbose {
		return false
	}
	if r.isTerminal != nil {
		return r.isTerminal(r.writer())
	}
	return false
}

// Report updates progress. In interactive TTY mode it draws the single progress bar;
// otherwise it emits standard progress lines.
func (r *ProgressReporter) Report(percent int, operation string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()

	w := r.writer()
	if r.IsInteractive() {
		width := 80
		if r.termWidth != nil {
			width = r.termWidth(w)
		}
		bar := formatSelfUpdateProgressBar(percent, operation, width)
		fmt.Fprintf(w, "\r\x1b[2K%s", bar)
		r.barDrawn = true
		if r.isAuthoritative() {
			selfUpdateProgressState.percent = percent
			selfUpdateProgressState.operation = operation
			selfUpdateProgressState.barDrawn = true
		}
		return
	}
	fmt.Fprintf(w, "self-update: progress=%d%% operation=%q\n", percent, strings.TrimSpace(operation))
	if r.isAuthoritative() {
		selfUpdateProgressState.percent = percent
		selfUpdateProgressState.operation = operation
		selfUpdateProgressState.barDrawn = false
	}
}

// Clear erases the progress bar if drawn.
func (r *ProgressReporter) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()

	w := r.writer()
	if r.barDrawn || r.IsInteractive() {
		fmt.Fprint(w, "\r\x1b[2K")
		r.barDrawn = false
		if r.isAuthoritative() {
			selfUpdateProgressState.barDrawn = false
		}
	}
}

// Settle finishes the progress bar cleanly by printing a trailing newline.
func (r *ProgressReporter) Settle() {
	r.mu.Lock()
	defer r.mu.Unlock()
	selfUpdateProgressMu.Lock()
	defer selfUpdateProgressMu.Unlock()
	selfUpdateProgressState.Lock()
	defer selfUpdateProgressState.Unlock()

	w := r.writer()
	if r.barDrawn && r.IsInteractive() {
		fmt.Fprintln(w)
		r.barDrawn = false
		if r.isAuthoritative() {
			selfUpdateProgressState.barDrawn = false
		}
	}
}
