package toolproc

import (
	"bufio"
	"io"
	"os"
)

// JournalTailWindowBytes bounds how much of the shared, append-only journal a
// single hook firing parses. The journal grows without bound and is re-read on
// every PreToolUse/PostToolUse firing — each a fresh `fak` process — so a full
// parse is O(journal) twice per tool call, the dominant soft-page-fault
// contributor on a busy multi-session box (#3154). A firing only needs the
// current call's own recent history: its pre-spawn (to pair the exit) and any
// background job it is polling. Both live near the tail; older, fully-terminal
// events are inert for this firing. A correlation target older than the window
// degrades to the plain single-event mapping — the same graceful fallback
// HookEvents already applies when a shape does not match — so the bound trades a
// rare, fail-open observability gap for a per-firing cost that no longer scales
// with total history.
const JournalTailWindowBytes = 4 << 20 // 4 MiB

// ParseTailFile parses at most the last JournalTailWindowBytes of the journal at
// path, aligned to a record boundary. A missing file yields no events (nil,
// nil), matching a fresh workspace — the same fail-open behavior the hook had
// when it opened the journal directly. A journal within the window is parsed
// whole, so behavior only diverges from a full parse once the file is large
// enough to matter.
func ParseTailFile(path string) ([]Event, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	return ParseTail(f, JournalTailWindowBytes)
}

// ParseTail parses events from the tail of r. When r is seekable and larger than
// maxBytes (> 0), only the last maxBytes are read and the partial record the
// seek lands mid-way through is discarded, so parsing starts on a clean line
// boundary. A non-seekable r, a non-positive maxBytes, or an r at/under maxBytes
// is parsed whole — identical to ParseEvents. Bounding the read keeps per-firing
// cost O(window) regardless of how large the shared journal grows.
func ParseTail(r io.Reader, maxBytes int64) ([]Event, error) {
	s, ok := r.(io.Seeker)
	if !ok || maxBytes <= 0 {
		return ParseEvents(r)
	}
	size, err := s.Seek(0, io.SeekEnd)
	if err != nil {
		return nil, err
	}
	if size <= maxBytes {
		// Rewind: probing the size consumed the reader to EOF.
		if _, err := s.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		return ParseEvents(r)
	}
	if _, err := s.Seek(size-maxBytes, io.SeekStart); err != nil {
		return nil, err
	}
	br := bufio.NewReader(r)
	// Drop the partial record the mid-file seek landed inside; the next record
	// starts after the first newline. No newline in the whole window (a single
	// record larger than the window) means nothing parseable — fail open.
	if _, err := br.ReadString('\n'); err != nil {
		if err == io.EOF {
			return nil, nil
		}
		return nil, err
	}
	return ParseEvents(br)
}
