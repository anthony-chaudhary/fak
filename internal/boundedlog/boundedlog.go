// Package boundedlog keeps an append-only log file (JSONL or plain text) under a
// byte ceiling by size-triggered rotation with exactly one retained generation.
//
// A writer calls RotateIfOver(path, cap) before it opens path for append: when the
// active file has grown past cap it is renamed to path+".1" (replacing any older
// .1), so the next append starts a fresh active file. Disk use is therefore bounded
// at roughly 2*cap plus one append. A reader calls Segments(path) and reads the
// returned files in order (oldest first) so a query spans the rotation boundary.
//
// Rotation is a best-effort hygiene step: callers ignore its error and append
// anyway, so a failed rename never costs a log row. Do NOT use this for a log that
// is hash-chained or replayed as full history (tamper evidence, balances, event
// sourcing) — dropping the old generation would break those readers.
//
// jsonlledger.AppendBounded is the line-append sibling (rotate-then-append in one
// call, surfacing rotation errors); boundedlog is the pre-open step for writers
// that own their own open (or hand the fd to a child process) and must never let
// rotation fail the append.
package boundedlog

import (
	"errors"
	"io/fs"
	"os"
)

// Common caps. DefaultMaxBytes suits operational/telemetry logs; AuditMaxBytes is
// the larger ceiling for audit-type ledgers whose window is worth keeping longer.
const (
	DefaultMaxBytes int64 = 16 << 20 // 16 MiB
	AuditMaxBytes   int64 = 32 << 20 // 32 MiB
)

// RotatedPath is the path of the single retained older generation of path.
func RotatedPath(path string) string { return path + ".1" }

// RotateIfOver renames path to RotatedPath(path) when path's size exceeds
// maxBytes, replacing any previous rotated generation. It reports whether a
// rotation happened. A missing file, a non-regular file, or maxBytes <= 0 is a
// no-op (false, nil).
func RotateIfOver(path string, maxBytes int64) (bool, error) {
	if path == "" || maxBytes <= 0 {
		return false, nil
	}
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	if !info.Mode().IsRegular() || info.Size() <= maxBytes {
		return false, nil
	}
	rotated := RotatedPath(path)
	if err := os.Rename(path, rotated); err != nil {
		// Windows refuses to rename over an existing file; drop the old
		// generation and retry once.
		if rmErr := os.Remove(rotated); rmErr != nil && !errors.Is(rmErr, fs.ErrNotExist) {
			return false, err
		}
		if err := os.Rename(path, rotated); err != nil {
			return false, err
		}
	}
	return true, nil
}

// Segments returns the existing segments of path oldest-first: the rotated
// generation (if present) followed by the active file (if present). A reader that
// reads each segment in order sees every retained row in append order.
func Segments(path string) []string {
	var out []string
	for _, p := range []string{RotatedPath(path), path} {
		if info, err := os.Stat(p); err == nil && info.Mode().IsRegular() {
			out = append(out, p)
		}
	}
	return out
}

// ReadAll returns the concatenated contents of Segments(path), oldest first. A
// segment that vanishes or fails to read between Stat and Read is skipped; a log
// with no segments yields (nil, fs.ErrNotExist).
func ReadAll(path string) ([]byte, error) {
	segs := Segments(path)
	if len(segs) == 0 {
		return nil, fs.ErrNotExist
	}
	var out []byte
	for _, p := range segs {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if len(out) > 0 && len(b) > 0 && out[len(out)-1] != '\n' {
			out = append(out, '\n')
		}
		out = append(out, b...)
	}
	return out, nil
}
