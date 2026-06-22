// Package demoui holds the small, cross-cutting helpers the on-box demos
// (cmd/demorace, cmd/ctxdemo, cmd/simpledemo, ...) share, so they all report the
// SAME thing about the machine and never freeze on a long blocking phase.
//
// Two concerns live here, both in service of one property: a watched demo should
// always be SHOWING something — what hardware it is using, and that it is still
// working right now.
//
//   - Probe() reports the real compute surface: logical cores, the matmul worker
//     width actually in use, the compute backends compiled into this build, and
//     whether any of them is a hardware ACCELERATOR. A demo renders this instead of
//     silently running on "whatever", so a viewer can see it is saturating the box —
//     and a GPU build (-tags cuda/metal/vulkan on matching hardware) lights up the
//     accelerator field automatically, with no demo-side change.
//   - Beat() / Spinner() keep the screen alive during a blocking phase (model load,
//     prefill measurement) that would otherwise emit nothing for tens of seconds.
//     Beat is for an event-stream demo (it heartbeats a callback ~1×/s while a call
//     runs in the background); Spinner is its terminal twin (an animated stderr line).
//
// The package is deliberately tiny and depends only on internal/compute and
// internal/model, so every demo can adopt it without pulling in heavy state.
package demoui

import (
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Hardware is the compute surface a demo is running on — the honest answer to "what
// is this using?". It is JSON-tagged so a web demo can hand it straight to the page.
type Hardware struct {
	LogicalCores int      `json:"logical_cores"` // runtime.NumCPU()
	GOMAXPROCS   int      `json:"gomaxprocs"`    // Go scheduler width
	Workers      int      `json:"workers"`       // matmul worker width actually in use (the real parallelism)
	Backends     []string `json:"backends"`      // compute backends compiled into THIS build (cpu-ref always present)
	Accelerator  string   `json:"accelerator"`   // "" when CPU-only; else the registered device backend name (cuda/metal/vulkan)
	AccelTier    string   `json:"accel_tier,omitempty"`
	Summary      string   `json:"summary"` // one human-readable line, ready to render
}

// Probe reads the live machine surface. It is cheap (no allocation beyond the
// backend-name slice) and safe to call per request, so a demo can re-probe after a
// -jobs/-budget cap has changed the worker width and report the post-cap number.
//
// "Accelerator" is the first registered backend that is NOT the reference floor — a
// real device (cuda/metal/vulkan) only registers when the build was tagged for it on
// matching hardware, so on a default CPU build this is "" and the summary says so
// plainly rather than implying a GPU that isn't there.
func Probe() Hardware {
	hw := Hardware{
		LogicalCores: runtime.NumCPU(),
		GOMAXPROCS:   runtime.GOMAXPROCS(0),
		Workers:      model.NumWorkers(),
		Backends:     compute.Registered(),
	}
	for _, name := range hw.Backends {
		if be, ok := compute.Lookup(name); ok && be.Class() != compute.Reference {
			hw.Accelerator = be.Name()
			hw.AccelTier = be.Tier()
			break
		}
	}
	hw.Summary = hw.describe()
	return hw
}

func (h Hardware) describe() string {
	if h.Accelerator != "" {
		dev := h.Accelerator
		if h.AccelTier != "" {
			dev += " (" + h.AccelTier + ")"
		}
		return fmt.Sprintf("%d cores · %d matmul workers · accelerator: %s", h.LogicalCores, h.Workers, dev)
	}
	return fmt.Sprintf("%d cores · %d matmul workers · pure-Go Q8 CPU (no GPU backend in this build)", h.LogicalCores, h.Workers)
}

// Beat runs work() to completion while invoking beat(elapsed) approximately every
// `every` from the CALLING goroutine. That ordering is the point: a demo whose emit
// path is single-threaded (an SSE writer is not safe for concurrent writes) can
// heartbeat a blocking call without a data race — work() runs on a background
// goroutine, and every tick callback runs here on the caller's goroutine, never
// concurrently with the caller's other emits.
//
// It returns only after work() has returned. A zero/negative `every` disables ticking
// (work still runs and is still waited on), which lets a caller pass a configurable
// cadence (or 0 to opt out) through one code path.
func Beat(every time.Duration, beat func(elapsed time.Duration), work func()) {
	done := make(chan struct{})
	go func() { defer close(done); work() }()
	if every <= 0 {
		<-done
		return
	}
	start := time.Now()
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-done:
			return
		case <-t.C:
			beat(time.Since(start))
		}
	}
}

var spinFrames = []rune{'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'}

// Spinner animates a one-line status on w (typically os.Stderr) until the returned
// stop func is called, after which the line is cleared. It is the terminal twin of
// Beat: a model load that would print nothing for 30s instead shows a live
// "⠙ Loading model… 12.3s" that advances ~8×/s. stop() is idempotent and safe to
// defer. The clear is done by overwriting with spaces + carriage return (no ANSI), so
// it behaves the same on a plain Windows console as on a VT-capable terminal.
func Spinner(w io.Writer, label string) (stop func()) {
	done := make(chan struct{})
	finished := make(chan struct{}) // closed when the animator goroutine has returned
	var stopped sync.Once
	width := int32(0) // widest line written, for a clean clear
	go func() {
		defer close(finished)
		start := time.Now()
		t := time.NewTicker(120 * time.Millisecond)
		defer t.Stop()
		i := 0
		for {
			select {
			case <-done:
				return
			case <-t.C:
				line := fmt.Sprintf("%c %s… %.1fs", spinFrames[i%len(spinFrames)], label, time.Since(start).Seconds())
				if n := int32(len([]rune(line))); n > atomic.LoadInt32(&width) {
					atomic.StoreInt32(&width, n)
				}
				fmt.Fprintf(w, "\r%s ", line)
				i++
			}
		}
	}()
	return func() {
		stopped.Do(func() {
			close(done)
			<-finished // the animator may be mid-Fprintf(w); wait for it to return before the clear write below (and the caller's read of w) so neither races the animator
			n := int(atomic.LoadInt32(&width)) + 1
			blank := make([]rune, n)
			for i := range blank {
				blank[i] = ' '
			}
			fmt.Fprintf(w, "\r%s\r", string(blank))
		})
	}
}
