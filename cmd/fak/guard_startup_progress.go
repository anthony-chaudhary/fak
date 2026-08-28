package main

import (
	"fmt"
	"io"
	"sync"
	"time"
)

const guardStartupProgressDelay = 150 * time.Millisecond

// guardStartupProgress exposes the post-report gap without turning a healthy
// launch into another banner. The timer makes the current phase visible only
// after startup is observably slow; a TTY gets one rewritten line and captured
// logs get one compact row per subsequent phase. Phase names are fixed caller
// literals: command arguments, environment values, paths, and credentials never
// enter this renderer.
type guardStartupProgress struct {
	mu          sync.Mutex
	w           io.Writer
	interactive bool
	enabled     bool
	phase       string
	visible     bool
	finished    bool
	started     time.Time
	timer       *time.Timer
}

func newGuardStartupProgress(w io.Writer, enabled, interactive bool, delay time.Duration) *guardStartupProgress {
	p := &guardStartupProgress{w: w, enabled: enabled && w != nil, interactive: interactive, started: time.Now()}
	if !p.enabled {
		return p
	}
	if delay <= 0 {
		delay = guardStartupProgressDelay
	}
	p.timer = time.AfterFunc(delay, p.reveal)
	return p
}

func (p *guardStartupProgress) Phase(phase string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.enabled || p.finished || phase == "" || phase == p.phase {
		return
	}
	p.phase = phase
	if p.visible {
		p.renderLocked(false)
	}
}

func (p *guardStartupProgress) reveal() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.enabled || p.finished || p.phase == "" {
		return
	}
	p.visible = true
	p.renderLocked(false)
}

func (p *guardStartupProgress) Started() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.finished {
		return
	}
	p.finished = true
	if p.timer != nil {
		p.timer.Stop()
	}
	if !p.enabled || !p.visible {
		return
	}
	p.phase = "child registration/started"
	p.renderLocked(true)
}

func (p *guardStartupProgress) Stop() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.finished {
		return
	}
	p.finished = true
	if p.timer != nil {
		p.timer.Stop()
	}
	if p.enabled && p.visible && p.interactive {
		fmt.Fprint(p.w, "\r\x1b[2K")
	}
}

// Abort preserves a visible slow-start phase as a complete line before the
// caller prints its error. Stop cannot do this job: its success/early-return
// contract clears a transient TTY row, which previously let the following
// diagnostic begin in the middle of that row and then erased it on defer.
func (p *guardStartupProgress) Abort() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.finished {
		return
	}
	p.finished = true
	if p.timer != nil {
		p.timer.Stop()
	}
	if p.enabled && p.visible && p.interactive {
		fmt.Fprintln(p.w)
	}
}

// EndLine makes room for a non-fatal startup diagnostic, then leaves progress
// active so a later phase can reuse the next terminal line in place.
func (p *guardStartupProgress) EndLine() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.finished && p.enabled && p.visible && p.interactive {
		fmt.Fprintln(p.w)
	}
}

func (p *guardStartupProgress) renderLocked(final bool) {
	elapsed := time.Since(p.started).Round(time.Millisecond)
	if p.interactive {
		fmt.Fprintf(p.w, "\r\x1b[2Kfak guard · starting: %s elapsed=%s", p.phase, elapsed)
		if final {
			fmt.Fprintln(p.w)
		}
		return
	}
	fmt.Fprintf(p.w, "fak guard: startup phase=%s elapsed=%s\n", p.phase, elapsed)
}
