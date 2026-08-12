package main

import (
	"os"
	"time"

	"golang.org/x/term"
)

// Pane-geometry plumbing shared by both resize builds (info_resize_windows.go polls,
// info_resize_other.go waits on SIGWINCH). Everything here is platform-neutral so the polling
// core is unit-testable on any OS, not only the one that ships it.

// infoTermSize is the seam over "how big is the pane right now". It reads the REAL os.Stdout fd,
// which is the same source the startup measure in runInfo uses, so the loop's re-measure and the
// resize watcher can never disagree about what they are looking at. It is a var so a test can pin
// a size without a TTY; the default is the live read, so no production path changes.
var infoTermSize = func() (width, height int, err error) {
	return term.GetSize(int(os.Stdout.Fd()))
}

// infoResizePollInterval is how often the Windows resize watcher samples the pane. Windows has no
// SIGWINCH, so a poll is the only way to notice a resize. It is short enough that a drag-resize
// (or a pane whose host sizes it a beat after the process starts) repaints while the operator is
// still looking at it, and cheap enough to be free: one GetConsoleScreenBufferInfo per interval,
// and a repaint is emitted ONLY when the numbers actually change.
const infoResizePollInterval = 250 * time.Millisecond

// watchInfoTermSize is the polling core: it samples size on every tick and emits ONE coalesced
// repaint signal whenever the measured pane differs from the last geometry it knows about.
//
// baseW/baseH seed that "last known" geometry with what the overlay is about to PAINT at, not
// with the watcher's own first sample. That distinction is the whole point: a pane whose real
// size was already different from the startup measure (a --split pane sampled before its host
// finished laying out the split) never changes again afterwards, so a watcher that baselined off
// its own first read would see a steady size and stay silent forever — leaving the overlay
// painting an oversized, bottom-padded frame that overflows the pane and shows only blank rows.
// Seeded, the first tick disagrees with the seed and repaints.
//
// A failed or non-positive read is skipped rather than treated as a resize to 0 (a pane can
// report nothing mid-teardown, and a 0 would be read as "size unknown" and blow the layout up).
// The send is non-blocking onto a buffered channel, so a burst of ticks during a drag-resize
// collapses to one pending repaint and the watcher never blocks on a busy loop. It returns when
// done closes or tick is closed.
func watchInfoTermSize(baseW, baseH int, tick <-chan time.Time, size func() (int, int, error), done <-chan struct{}) <-chan struct{} {
	out := make(chan struct{}, 1)
	go func() {
		w, h := baseW, baseH
		for {
			select {
			case <-done:
				return
			case _, ok := <-tick:
				if !ok {
					return
				}
				nw, nh, err := size()
				if err != nil || nw <= 0 || nh <= 0 {
					continue
				}
				if nw == w && nh == h {
					continue
				}
				w, h = nw, nh
				select {
				case out <- struct{}{}:
				default:
				}
			}
		}
	}()
	return out
}
