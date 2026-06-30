//go:build windows

package main

// newInfoResizeChan (Windows): Windows has no SIGWINCH, so there is no terminal-resize signal
// to deliver. It returns a nil channel — a receive on a nil channel blocks forever, so the
// overlay's `case <-resizeCh:` is a clean no-op — and a no-op stop func. The overlay still
// catches resizes on Windows because it re-measures (term.GetSize) on every focus-in edge and,
// for the focus-never-arrives case, the loop's per-tick remeasure keeps the geometry current.
// This mirrors the serve_signals_{windows,other}.go build-tag seam (Windows has no SIGHUP
// either) so the platform split stays uniform across the kernel.
func newInfoResizeChan() (<-chan struct{}, func()) {
	return nil, func() {}
}
