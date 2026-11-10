//go:build !unix

package main

import (
	"io"
	"os"
)

// mapView materializes the whole file into process memory. There is no
// address-space mapping on this platform, so every byte is already resident
// by the time a worker reads it; the prefault walk degenerates to a
// readability check and Advised stays false.
func (r *MappedRegion) mapView(f *os.File) ([]byte, error) {
	return io.ReadAll(f)
}

// advise reports false: there is no pager to advise here.
func (r *MappedRegion) advise() bool { return false }

// pageUnit is nominal on !unix; it only shapes PrefaultReport arithmetic.
func (r *MappedRegion) pageUnit() int64 { return 4096 }

// release is a no-op: the GC owns the materialized buffer.
func (r *MappedRegion) release() error { return nil }

// isUnixPlatform anchors the Advised contract check in the test.
func (r *MappedRegion) isUnixPlatform() bool { return false }
