//go:build windows

package main

import (
	"sync"
	"time"
)

// newInfoResizeChan (Windows): Windows raises no SIGWINCH, so the only way to notice that the
// pane changed size is to look. This is the per-tick GetSize poll the rest of the overlay has
// always documented (info_focus.go's header, info_resize_other.go) — it returned a nil channel
// before, which made every one of those comments a promise the code did not keep.
//
// Why the gap mattered rather than being a cosmetic TODO: the interactive frame is padded to
// exactly the measured pane height and redrawn in place, so a height LARGER than the real pane
// overflows it. The pane scrolls, and what stays on screen is the padded tail — blank rows. On
// Windows nothing could ever correct that: there is no SIGWINCH, and `fak guard --split` hands
// keyboard focus straight back to the agent pane (guard_split.go), so the focus-in re-measure
// never fires for the info pane either. A single bad startup measure stood for the whole session
// and the 20% strip just looked empty.
//
// The watcher is seeded with the geometry the overlay is about to paint at, so it also catches a
// startup measure that was wrong from the first frame, not only a later resize — see
// watchInfoTermSize. The loop's own per-frame re-measure is the backstop; this makes the repaint
// land in a quarter second instead of on the next poll interval.
func newInfoResizeChan(baseW, baseH int) (<-chan struct{}, func()) {
	ticker := time.NewTicker(infoResizePollInterval)
	done := make(chan struct{})
	out := watchInfoTermSize(baseW, baseH, ticker.C, infoTermSize, done)
	var once sync.Once
	return out, func() {
		once.Do(func() {
			ticker.Stop()
			close(done)
		})
	}
}
