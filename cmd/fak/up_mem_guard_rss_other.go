//go:build !(darwin && cgo)

package main

// platformCurrentRSS reports no OS-level current RSS on this platform, so the
// mem guard falls back to the Go runtime's own accounting. The guard is a
// macOS-turnkey mitigation; other platforms keep the historical unbounded
// holder unless an operator sets --max-rss.
func platformCurrentRSS() uint64 { return 0 }
