// Package boundedlog caps append-only JSONL ledgers written by
// open-append-close callers (fak-private#3064, disk gardening). Rotation is
// one-generation: before an append, a ledger at or past its byte cap is
// renamed to "<path>.1", replacing any older ".1"; the append lands in a
// fresh active file. Disk use stays bounded by roughly twice the cap per
// ledger; no record is ever rewritten in place.
// Crash safety rests on the single os.Rename: a crash before it leaves the
// over-cap file intact; after it, a complete ".1" and no active file. Holders
// of the old file keep writing into ".1", so their lines survive; racing
// rotations can at worst drop the older generation - history the cap was
// about to retire - never corrupting a line or exceeding the bound.
package boundedlog

import (
	"errors"
	"io/fs"
	"os"
)

// RotatedSuffix is the suffix of the single retained older generation.
const RotatedSuffix = ".1"

// RotatedPath returns the retained-generation path for a ledger.
func RotatedPath(path string) string { return path + RotatedSuffix }

// RotateIfOver renames path to RotatedPath(path), replacing any previous
// rotated generation, when path exists and its size is at least maxBytes.
// A non-positive maxBytes disables the cap; missing ledgers and ones a
// concurrent rotator already moved are not errors. Non-regular files are
// never rotated - the caller's open-time checks decide what to do with them.
func RotateIfOver(path string, maxBytes int64) (bool, error) {
	if maxBytes <= 0 {
		return false, nil
	}
	info, err := os.Lstat(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	if !info.Mode().IsRegular() || info.Size() < maxBytes {
		return false, nil
	}
	if err := os.Rename(path, RotatedPath(path)); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// Segments returns the ledger's existing files, oldest first: the rotated
// generation (if present) followed by the active file (if present).
func Segments(path string) []string {
	var out []string
	for _, p := range []string{RotatedPath(path), path} {
		if _, err := os.Lstat(p); err == nil {
			out = append(out, p)
		}
	}
	return out
}
