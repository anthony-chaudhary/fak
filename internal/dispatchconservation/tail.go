package dispatchconservation

import (
	"bufio"
	"io"
	"os"
	"strings"
)

const DefaultTailLines = 10000

// ReadTailBytes returns the last n bytes of a file as a string ("" when the file
// is unreadable — a missing artifact is never an error here, it is just absent
// evidence). Byte-bounded, not line-bounded, on purpose: this is the SAME window
// the witness sweep's own no-commit classifier reads
// (internal/dispatchtick.WitnessTailBytes / tools/issue_resolve_dispatch.py
// `_log_tail_text`), so a marker this package finds is a marker that classifier
// could also have found. A worker log runs to megabytes of streamed turns; the
// guard exit summary lives at the very end, so the tail is the whole signal.
//
// The cut is at a raw byte offset, so the first line may be a partial UTF-8
// sequence. Callers only ever substring-match ASCII-anchored markers well inside
// the window, so a torn leading rune is harmless (and a torn rune can only lose a
// match at the very START of the window, never invent one).
func ReadTailBytes(path string, n int) string {
	if n <= 0 {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	size := st.Size()
	if size > int64(n) {
		if _, err := f.Seek(size-int64(n), io.SeekStart); err != nil {
			return ""
		}
	}
	raw, err := io.ReadAll(io.LimitReader(f, int64(n)))
	if err != nil {
		return ""
	}
	return string(raw)
}

// ReadTailLines bounds memory and parse work to the newest max non-empty rows.
// A ring buffer preserves append order without reading the whole file into RAM.
func ReadTailLines(path string, max int) []string {
	if max <= 0 {
		max = DefaultTailLines
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	ring := make([]string, max)
	count := 0
	sc := bufio.NewScanner(f)
	buf := make([]byte, 64<<10)
	sc.Buffer(buf, 2<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		ring[count%max] = line
		count++
	}
	n := count
	if n > max {
		n = max
	}
	out := make([]string, 0, n)
	start := count - n
	for i := 0; i < n; i++ {
		out = append(out, ring[(start+i)%max])
	}
	return out
}
