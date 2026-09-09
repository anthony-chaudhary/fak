package demoui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// DefaultBarLength is the default character width of the progress bar graphic.
	DefaultBarLength = 16

	// DefaultCadence is the animation tick rate (~10 frames/sec).
	DefaultCadence = 100 * time.Millisecond

	// FilledGlyph is the UTF-8 block character used for completed progress.
	FilledGlyph = "█"

	// EmptyGlyph is the UTF-8 shade character used for remaining progress.
	EmptyGlyph = "░"
)

// LoadingOptions configures an animated model loading display.
type LoadingOptions struct {
	// Writer is where the animated line is rendered (defaults to os.Stderr if nil).
	Writer io.Writer

	// Label is the title of the operation (e.g. "Loading model" or "Loading model (qwen38)").
	Label string

	// Total is the expected total units (e.g. tensors or bytes).
	// When <= 0, the display runs in indeterminate mode with a bouncing graphic pulse.
	Total int64

	// Current is the initial progress (typically 0).
	Current int64

	// Unit is an optional suffix for progress counts (e.g. "tensors", "GB").
	Unit string

	// Detail is optional transient status text (e.g. "allocating KV cache").
	Detail string

	// BarLength is the width of the progress bar graphic in runes (default 16).
	BarLength int

	// EstimatedDuration is an optional estimated total duration for indeterminate loading.
	// When set and Total <= 0, an ETA is estimated relative to this duration.
	EstimatedDuration time.Duration

	// HideTimer disables the elapsed time counter when true.
	HideTimer bool

	// HideETA disables the estimated time remaining calculation when true.
	HideETA bool

	// HideGraphic disables the graphical progress bar / pulse when true.
	HideGraphic bool

	// Cadence controls the refresh interval of the animation (default 100ms).
	Cadence time.Duration

	// Quiet suppresses all terminal output when true.
	Quiet bool
}

// LoadingDisplay controls a live, animated model loading status line with a spinner,
// graphical progress bar, elapsed timer, and estimated time remaining (ETA).
type LoadingDisplay struct {
	mu                sync.Mutex
	writer            io.Writer
	label             string
	total             int64
	current           int64
	unit              string
	detail            string
	barLength         int
	estimatedDuration time.Duration
	hideTimer         bool
	hideETA           bool
	hideGraphic       bool
	cadence           time.Duration
	quiet             bool

	startTime    time.Time
	frame        int
	indetPos     int
	indetDir     int
	maxWidth     int32
	stopped      bool
	doneChan     chan struct{}
	finishedChan chan struct{}
}

// StartLoadingDisplay starts an animated model loading display in a background goroutine
// and returns the active LoadingDisplay controller. Call Stop() or Done() when finished.
func StartLoadingDisplay(opts LoadingOptions) *LoadingDisplay {
	w := opts.Writer
	if w == nil {
		w = os.Stderr
	}
	barLen := opts.BarLength
	if barLen <= 0 {
		barLen = DefaultBarLength
	}
	cadence := opts.Cadence
	if cadence <= 0 {
		cadence = DefaultCadence
	}
	label := opts.Label
	if label == "" {
		label = "Loading model"
	}

	d := &LoadingDisplay{
		writer:            w,
		label:             label,
		total:             opts.Total,
		current:           opts.Current,
		unit:              opts.Unit,
		detail:            opts.Detail,
		barLength:         barLen,
		estimatedDuration: opts.EstimatedDuration,
		hideTimer:         opts.HideTimer,
		hideETA:           opts.HideETA,
		hideGraphic:       opts.HideGraphic,
		cadence:           cadence,
		quiet:             opts.Quiet,
		startTime:         time.Now(),
		indetDir:          1,
		doneChan:          make(chan struct{}),
		finishedChan:      make(chan struct{}),
	}

	if d.quiet {
		close(d.finishedChan)
		return d
	}

	go d.loop()
	return d
}

// ModelLoadingSpinner starts an animated model loading display with a spinner,
// bouncing graphical progress bar, and elapsed timer on w, returning an idempotent stop func.
func ModelLoadingSpinner(w io.Writer, label string) (stop func()) {
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:            w,
		Label:             label,
		EstimatedDuration: 10 * time.Second,
	})
	return disp.Stop
}

// NewModelLoadingProgress creates and starts an animated loading display with a known total count.
func NewModelLoadingProgress(w io.Writer, label string, total int64) *LoadingDisplay {
	return StartLoadingDisplay(LoadingOptions{
		Writer:  w,
		Label:   label,
		Total:   total,
		Current: 0,
		Unit:    "tensors",
	})
}

// Update updates the current progress and total units thread-safely.
func (d *LoadingDisplay) Update(current, total int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.current = current
	d.total = total
}

// SetCurrent updates the current progress value thread-safely.
func (d *LoadingDisplay) SetCurrent(current int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.current = current
}

// SetTotal updates the total progress value thread-safely.
func (d *LoadingDisplay) SetTotal(total int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.total = total
}

// SetDetail sets or clears the transient detail text thread-safely.
func (d *LoadingDisplay) SetDetail(detail string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.detail = detail
}

// SetLabel updates the display label thread-safely.
func (d *LoadingDisplay) SetLabel(label string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.label = label
}

// Tick increments the current progress counter by delta thread-safely.
func (d *LoadingDisplay) Tick(delta int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.current += delta
}

// Elapsed returns the duration since the loading display was started.
func (d *LoadingDisplay) Elapsed() time.Duration {
	d.mu.Lock()
	defer d.mu.Unlock()
	return time.Since(d.startTime)
}

// Stop stops the display animation and cleanly clears the line from the terminal.
// It is idempotent and safe to defer.
func (d *LoadingDisplay) Stop() {
	d.mu.Lock()
	if d.stopped {
		d.mu.Unlock()
		return
	}
	d.stopped = true
	close(d.doneChan)
	w := d.writer
	quiet := d.quiet
	d.mu.Unlock()

	<-d.finishedChan

	if quiet {
		return
	}

	width := int(atomic.LoadInt32(&d.maxWidth)) + 2
	if width < 30 {
		width = 30
	}
	blank := strings.Repeat(" ", width)
	fmt.Fprintf(w, "\r%s\r", blank)
}

// Done stops the animation, clears the line, and prints a final completion line with newline.
func (d *LoadingDisplay) Done(finalMessage string) {
	d.mu.Lock()
	if d.stopped {
		d.mu.Unlock()
		return
	}
	d.stopped = true
	close(d.doneChan)
	w := d.writer
	quiet := d.quiet
	d.mu.Unlock()

	<-d.finishedChan

	if quiet {
		return
	}

	width := int(atomic.LoadInt32(&d.maxWidth)) + 2
	if width < 30 {
		width = 30
	}
	blank := strings.Repeat(" ", width)
	fmt.Fprintf(w, "\r%s\r%s\n", blank, finalMessage)
}

// Donef formats a message and calls Done.
func (d *LoadingDisplay) Donef(format string, args ...any) {
	d.Done(fmt.Sprintf(format, args...))
}

func (d *LoadingDisplay) loop() {
	defer close(d.finishedChan)

	// Render the initial frame immediately for instant visual feedback.
	d.renderFrame()

	ticker := time.NewTicker(d.cadence)
	defer ticker.Stop()

	for {
		select {
		case <-d.doneChan:
			return
		case <-ticker.C:
			d.renderFrame()
		}
	}
}

func (d *LoadingDisplay) renderFrame() {
	d.mu.Lock()
	line := d.formatLineLocked()
	d.frame++
	d.advanceIndeterminateLocked()
	w := d.writer
	d.mu.Unlock()

	runeCount := int32(len([]rune(line)))
	for {
		cur := atomic.LoadInt32(&d.maxWidth)
		if runeCount <= cur || atomic.CompareAndSwapInt32(&d.maxWidth, cur, runeCount) {
			break
		}
	}

	// Pad out to maxWidth to completely overwrite previous redraws.
	pad := ""
	curMax := atomic.LoadInt32(&d.maxWidth)
	if int(curMax) > int(runeCount) {
		pad = strings.Repeat(" ", int(curMax)-int(runeCount))
	}

	fmt.Fprintf(w, "\r%s%s", line, pad)
}

func (d *LoadingDisplay) advanceIndeterminateLocked() {
	pw := 4
	if d.barLength < 8 {
		pw = 2
	}
	maxOffset := d.barLength - pw
	if maxOffset <= 0 {
		return
	}
	if d.indetDir == 0 {
		d.indetDir = 1
	}
	next := d.indetPos + d.indetDir
	if next >= maxOffset {
		d.indetPos = maxOffset
		d.indetDir = -1
	} else if next <= 0 {
		d.indetPos = 0
		d.indetDir = 1
	} else {
		d.indetPos = next
	}
}

func (d *LoadingDisplay) formatLineLocked() string {
	elapsed := time.Since(d.startTime)
	glyph := spinFrames[d.frame%len(spinFrames)]

	return formatLoadingLine(
		glyph,
		d.label,
		d.current,
		d.total,
		d.unit,
		d.detail,
		d.barLength,
		d.indetPos,
		elapsed,
		d.estimatedDuration,
		d.hideTimer,
		d.hideETA,
		d.hideGraphic,
	)
}

// formatLoadingLine constructs the formatted loading status line.
func formatLoadingLine(
	glyph rune,
	label string,
	current, total int64,
	unit, detail string,
	barLength, indetPos int,
	elapsed time.Duration,
	estDuration time.Duration,
	hideTimer, hideETA, hideGraphic bool,
) string {
	if label == "" {
		label = "Loading model"
	}
	var parts []string
	parts = append(parts, fmt.Sprintf("%c %s…", glyph, label))

	if !hideGraphic && barLength > 0 {
		bar := renderProgressBar(current, total, barLength, indetPos)
		if bar != "" {
			parts = append(parts, bar)
		}
	}

	if !hideTimer {
		parts = append(parts, formatDuration(elapsed))
	}

	if !hideETA {
		eta := calculateETA(current, total, elapsed, estDuration)
		if eta != "" {
			parts = append(parts, fmt.Sprintf("(%s)", eta))
		}
	}

	if detail != "" {
		parts = append(parts, detail)
	} else if total > 0 {
		if unit != "" {
			parts = append(parts, fmt.Sprintf("%d/%d %s", current, total, unit))
		} else {
			parts = append(parts, fmt.Sprintf("%d/%d", current, total))
		}
	} else if current > 0 {
		if unit != "" {
			parts = append(parts, fmt.Sprintf("%d %s", current, unit))
		} else {
			parts = append(parts, fmt.Sprintf("%d", current))
		}
	}

	return strings.Join(parts, " ")
}

// renderProgressBar renders either a determinate filled progress bar or an indeterminate bouncing pulse.
func renderProgressBar(current, total int64, barLength, indetPos int) string {
	if barLength <= 0 {
		return ""
	}
	if total > 0 {
		pct := int(float64(current) * 100 / float64(total))
		if pct < 0 {
			pct = 0
		}
		if pct > 100 {
			pct = 100
		}
		filled := (pct * barLength) / 100
		if filled < 0 {
			filled = 0
		}
		if filled > barLength {
			filled = barLength
		}
		empty := barLength - filled
		bar := strings.Repeat(FilledGlyph, filled) + strings.Repeat(EmptyGlyph, empty)
		return fmt.Sprintf("[%s] %3d%%", bar, pct)
	}

	// Indeterminate bouncing graphic pulse
	pw := 4
	if barLength < 8 {
		pw = 2
	}
	maxOffset := barLength - pw
	pos := indetPos
	if pos > maxOffset {
		pos = maxOffset
	}
	if pos < 0 {
		pos = 0
	}
	before := strings.Repeat(EmptyGlyph, pos)
	pulse := strings.Repeat(FilledGlyph, pw)
	after := strings.Repeat(EmptyGlyph, barLength-pw-pos)
	return fmt.Sprintf("[%s%s%s]", before, pulse, after)
}

// calculateETA estimates time remaining based on current progress and elapsed time.
func calculateETA(current, total int64, elapsed time.Duration, estDuration time.Duration) string {
	if total > 0 {
		if current >= total {
			return "ETA 0s"
		}
		if current > 0 && elapsed > 0 {
			rate := float64(current) / elapsed.Seconds()
			if rate > 0 {
				remItems := total - current
				remSec := float64(remItems) / rate
				remDur := time.Duration(remSec * float64(time.Second))
				return "ETA ~" + formatDuration(remDur)
			}
		}
		return "ETA --"
	}

	if estDuration > 0 {
		if elapsed < estDuration {
			rem := estDuration - elapsed
			return "ETA ~" + formatDuration(rem)
		}
		return "ETA ~imminent"
	}

	return ""
}

// formatDuration formats a duration concisely for the progress timer and ETA:
// < 60s: "X.Ys"
// < 1h:  "XmYs"
// >= 1h: "XhYmZs"
func formatDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(100 * time.Millisecond)
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	totalSec := int(d.Seconds())
	if d < time.Hour {
		m := totalSec / 60
		s := totalSec % 60
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	h := totalSec / 3600
	m := (totalSec % 3600) / 60
	s := totalSec % 60
	return fmt.Sprintf("%dh%02dm%02ds", h, m, s)
}

var (
	progressTensorsRegex = regexp.MustCompile(`(\d+)/(\d+)\s+tensors`)
	progressPctRegex     = regexp.MustCompile(`(\d+(?:\.\d+)?)%`)
	progressRateRegex    = regexp.MustCompile(`([\d\.]+\s+[GMK]?B/s)`)
)

// ParseProgressLine parses progress information from standard log lines (e.g. from LoadProfiler)
// and updates the display accordingly. Returns true if progress was updated.
func (d *LoadingDisplay) ParseProgressLine(line string) bool {
	updated := false
	if m := progressTensorsRegex.FindStringSubmatch(line); len(m) == 3 {
		cur, err1 := strconv.ParseInt(m[1], 10, 64)
		tot, err2 := strconv.ParseInt(m[2], 10, 64)
		if err1 == nil && err2 == nil {
			d.Update(cur, tot)
			updated = true
		}
	} else if m := progressPctRegex.FindStringSubmatch(line); len(m) == 2 {
		pct, err := strconv.ParseFloat(m[1], 64)
		if err == nil && d.total > 0 {
			d.SetCurrent(int64(pct * float64(d.total) / 100.0))
			updated = true
		}
	}
	if m := progressRateRegex.FindStringSubmatch(line); len(m) == 2 {
		d.SetDetail(m[1])
		updated = true
	}
	return updated
}

// ProgressWriter returns an io.Writer that parses written log lines to advance the display.
func (d *LoadingDisplay) ProgressWriter() io.Writer {
	pr, pw := io.Pipe()
	go func() {
		scanner := bufio.NewScanner(pr)
		for scanner.Scan() {
			d.ParseProgressLine(scanner.Text())
		}
	}()
	return pw
}
