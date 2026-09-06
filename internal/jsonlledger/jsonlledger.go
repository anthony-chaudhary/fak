// Package jsonlledger holds the shared JSONL-ledger row helpers the report
// packages (cadencereport, milestonereport, programreport, …) each used to
// copy-paste: Parse scans a JSONL ledger into typed rows, and LatestBefore finds
// the newest prior row. Each caller keeps its own row type and delegates here so
// the duplicated bodies live in exactly one place.
package jsonlledger

import (
	"bufio"
	"bytes"
	"encoding/json"
	"hash/fnv"
	"io"
	"os"
	"strconv"
	"strings"
)

// Parse scans content as JSONL, unmarshaling each non-blank line into a T and
// appending it when keep(row) reports true. Blank and malformed lines are
// skipped. A nil keep accepts every well-formed row. The 1 MiB line buffer
// matches the copies this consolidates, so long ledger lines still parse.
func Parse[T any](content string, keep func(T) bool) []T {
	var rows []T
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if keep != nil && !keep(row) {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// LatestBefore returns the row in prior with the greatest (date, tiebreak) sort
// key, skipping any row whose non-empty tiebreak equals the reference row's (its
// own prior generation), or (zero, false) when none remain. date and tiebreak
// extract the primary sort key and the stable-sort tiebreaker from a row. It
// consolidates the identical "find the previous ledger row" scan the report
// packages each carried.
func LatestBefore[T any](row T, prior []T, date, tiebreak func(T) string) (T, bool) {
	self := tiebreak(row)
	var best *T
	var bestDate, bestTiebreak string
	for i := range prior {
		tb := tiebreak(prior[i])
		if tb != "" && tb == self {
			continue
		}
		d := date(prior[i])
		if best == nil || d > bestDate || (d == bestDate && tb >= bestTiebreak) {
			best = &prior[i]
			bestDate = d
			bestTiebreak = tb
		}
	}
	if best == nil {
		var zero T
		return zero, false
	}
	return *best, true
}

// Checkpoint records where a prior TailFold stopped so the next call can fold
// only the bytes appended since. Hold it in memory across reads or persist it —
// every field is JSON-encodable. The zero Checkpoint requests a full fold.
type Checkpoint[S any] struct {
	Path        string `json:"path"`   // file this checkpoint describes
	Offset      int64  `json:"offset"` // bytes folded: end of the last complete line
	Size        int64  `json:"size"`   // file size observed when the checkpoint was taken
	ModTimeNano int64  `json:"mtime"`  // file mtime observed, UnixNano
	Boundary    string `json:"fp"`     // fingerprint of the bytes ending at Offset
	State       S      `json:"state"`  // accumulated fold
}

// boundaryWindow bounds how many bytes before Offset the rotation fingerprint
// covers. It keeps the identity check O(1) while still catching an in-place
// rewrite that shifts a ledger's contents — e.g. a capped ledger dropping its
// oldest row rewrites the whole file, changing the bytes ending at the old
// Offset, so the fingerprint no longer matches and TailFold re-folds in full.
const boundaryWindow = 256

// TailFold folds only the JSONL rows appended to path since ckpt was taken.
// When ckpt still describes the file — same path, not shrunk below Offset, and
// the bytes ending at Offset unchanged — it seeks to ckpt.Offset and folds only
// the newly appended complete lines into ckpt.State via step. Otherwise (a zero
// ckpt, a different path, a shorter file, or a rewritten prefix) it re-folds the
// whole file from initial. Blank and malformed lines are skipped, matching
// Parse; a trailing line with no newline is left unconsumed until the writer
// completes it. The returned checkpoint carries the advanced offset and the new
// fold — feed it back on the next call.
//
// Correctness rests on one assumption: when the file grows, the bytes before the
// prior Offset are unchanged (an append). A writer that rewrites the prefix in
// place is caught only when that rewrite alters the boundaryWindow bytes ending
// at Offset; a rewrite that preserves them exactly would be mis-resumed. Every
// current caller either appends or shifts contents (which changes those bytes),
// so the assumption holds — revisit it before pointing TailFold at a new writer.
func TailFold[T, S any](path string, ckpt Checkpoint[S], initial S, step func(S, T) S) (Checkpoint[S], error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// A missing file folds to initial; a checkpoint that named a
			// now-deleted file is discarded as a rotation.
			return Checkpoint[S]{Path: path, State: initial}, nil
		}
		return ckpt, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return ckpt, err
	}
	size := info.Size()
	mod := info.ModTime().UnixNano()

	resume := ckpt.Path == path && ckpt.Offset > 0 && size >= ckpt.Offset
	if resume {
		fp, err := boundaryAt(f, ckpt.Offset)
		if err != nil {
			return ckpt, err
		}
		if fp != ckpt.Boundary {
			resume = false // shrank, rotated, or the prefix was rewritten
		}
	}

	state := initial
	var start int64
	if resume {
		state, start = ckpt.State, ckpt.Offset
	}

	if size > start {
		buf := make([]byte, size-start)
		n, err := f.ReadAt(buf, start)
		if err != nil && err != io.EOF {
			return ckpt, err
		}
		buf = buf[:n]
		for {
			i := bytes.IndexByte(buf, '\n')
			if i < 0 {
				break // partial trailing line: leave it for the next call
			}
			line := bytes.TrimSpace(buf[:i])
			start += int64(i) + 1
			buf = buf[i+1:]
			if len(line) == 0 {
				continue
			}
			var row T
			if err := json.Unmarshal(line, &row); err != nil {
				continue
			}
			state = step(state, row)
		}
	}

	fp, err := boundaryAt(f, start)
	if err != nil {
		return ckpt, err
	}
	return Checkpoint[S]{Path: path, Offset: start, Size: size, ModTimeNano: mod, Boundary: fp, State: state}, nil
}

// boundaryAt fingerprints up to boundaryWindow bytes ending at offset — the
// signature TailFold compares to decide whether a file's prefix is unchanged.
func boundaryAt(f *os.File, offset int64) (string, error) {
	if offset <= 0 {
		return "", nil
	}
	start := offset - boundaryWindow
	if start < 0 {
		start = 0
	}
	buf := make([]byte, offset-start)
	if _, err := f.ReadAt(buf, start); err != nil && err != io.EOF {
		return "", err
	}
	h := fnv.New64a()
	_, _ = h.Write(buf)
	return strconv.FormatUint(h.Sum64(), 16), nil
}
