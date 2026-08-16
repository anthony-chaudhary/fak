//go:build !windows

package main

import (
	"os"
	"os/signal"
	"sync"
	"syscall"
)

// newInfoResizeChan wires the live `fak info` overlay's terminal-resize signal. On a real OS
// the controlling terminal raises SIGWINCH when the pane size changes, so the overlay can
// re-measure (term.GetSize) and force ONE clean repaint at the new geometry instead of drawing
// the rest of the session at the stale startup size. It returns the delivery channel and a stop
// func the loop defers to tear the notifier down.
//
// SIGWINCH is NOT always delivered to a pane whose tab is hidden (some terminals coalesce or
// drop it while backgrounded), which is exactly why the focus-in edge ALSO re-measures — the
// two signals are complementary, not redundant. The Windows build (info_resize_windows.go) has
// no SIGWINCH to wait on and polls GetSize instead.
//
// baseW/baseH are the geometry the overlay is about to paint at. The polling build seeds its
// "last known size" with them so a startup measure that was already wrong repaints on the first
// poll; the kernel hands SIGWINCH to this build on every real change, so they are unused here.
// The signature is shared so the call site stays platform-free.
func newInfoResizeChan(_, _ int) (<-chan struct{}, func()) {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGWINCH)
	out := make(chan struct{}, 1)
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-done:
				return
			case <-sig:
				// Coalesce: a burst of SIGWINCH (a drag-resize) only needs one repaint.
				select {
				case out <- struct{}{}:
				default:
				}
			}
		}
	}()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			signal.Stop(sig)
			close(done)
		})
	}
	return out, stop
}
