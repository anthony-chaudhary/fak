//go:build !(darwin && cgo)

package harnessres

// DarwinSelfRSSReader returns nil when this build cannot read Darwin current
// self RSS through Mach. It does not substitute peak RSS or Go-runtime bytes for
// current process residency, so callers can report an unavailable reader.
func DarwinSelfRSSReader() func() (uint64, bool) { return nil }
