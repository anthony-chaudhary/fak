// Package benchckpt is the shared per-cell write-ahead checkpoint the compute-bench
// executors (modelbench, fanrun, and the fanbench/turnbench siblings that follow)
// write through so a crash at cell N does not discard the cells 1..N-1 already
// measured. See issue #2382.
//
// The gap it closes: every compute-bench executor builds its ENTIRE grid in memory
// and emits the artifact only after the last cell, so a crash at minute 40 of a
// 45-minute sweep discards every measured cell. modelbench's -preflight/-smoke ladder
// guards the multi-minute LOAD, but nothing protects the grid AFTER load, which is
// where the wall-clock cost actually accrues.
//
// The shape (a write-ahead ledger, one JSON line per cell):
//
//	line 1 : {"schema":"benchckpt/1","fingerprint":{...grid identity...}}
//	line k : {"key":"<coordinate>","cell":{...measured cell...}}
//
// Durability discipline (issue #2386): every write is an append of ONE complete line
// followed by fsync — never a whole-file open()+write. So a kill -9 can at worst tear
// the trailing line; Open tolerates that torn line by skipping any record that fails
// to parse, and every earlier cell survives. Coordinate identity is the caller's full
// cell key; a resumed run whose fingerprint (grid/model/seed) differs from the
// checkpoint refuses with ErrFingerprintMismatch rather than silently mixing
// incompatible cells.
package benchckpt

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

// Schema is the ledger schema tag (bumped on a breaking record-shape change).
const Schema = "benchckpt/1"

// ErrFingerprintMismatch is returned by Open when an existing checkpoint's grid
// identity (grid/model/seed/…) differs from the one the caller is resuming with. It is
// a typed refusal so a caller can distinguish "wrong grid, do not merge" from an I/O
// error and surface it, rather than blending incompatible cells into one artifact.
var ErrFingerprintMismatch = errors.New("benchckpt: checkpoint fingerprint mismatch")

// Fingerprint is the grid identity a checkpoint is bound to: the caller supplies the
// coordinates that must match for a resume to be valid (e.g. {"grid":"16,64,256",
// "model":"smollm2-135m","seed":7}). It is compared canonically (map keys sort in
// encoding/json), so equal fingerprints round-trip byte-identically.
type Fingerprint map[string]any

// header is the first line of every ledger.
type header struct {
	Schema      string      `json:"schema"`
	Fingerprint Fingerprint `json:"fingerprint"`
}

// record is one appended cell line.
type record struct {
	Key  string          `json:"key"`
	Cell json.RawMessage `json:"cell"`
}

// Ledger is an open write-ahead checkpoint. It is NOT safe for concurrent use; the
// bench executors run their grid serially, one cell at a time.
type Ledger struct {
	f     *os.File
	fp    Fingerprint
	order []string                   // recorded keys, in the order first seen
	cells map[string]json.RawMessage // key -> measured cell
}

// Open opens (or creates) the checkpoint at path bound to fp.
//
//   - New/empty file: the header line is written (append + fsync) and an empty ledger
//     returned.
//   - Existing non-empty file: the header fingerprint MUST equal fp or Open returns
//     ErrFingerprintMismatch. The already-recorded cells are loaded so Missing can
//     skip them; a torn trailing line (from a crash mid-append) is skipped rather than
//     failing the resume.
//
// The returned Ledger appends to the file; call Close when the grid is done.
func Open(path string, fp Fingerprint) (*Ledger, error) {
	if fp == nil {
		fp = Fingerprint{}
	}
	l := &Ledger{fp: fp, cells: map[string]json.RawMessage{}}

	info, statErr := os.Stat(path)
	existing := statErr == nil && info.Size() > 0
	if existing {
		if err := l.load(path, fp); err != nil {
			return nil, err
		}
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	l.f = f

	if !existing {
		if err := l.writeLine(header{Schema: Schema, Fingerprint: fp}); err != nil {
			f.Close()
			return nil, err
		}
	}
	return l, nil
}

// load reads an existing ledger: validates the header fingerprint and collects the
// recorded cells, tolerating a torn trailing line.
func (l *Ledger) load(path string, want Fingerprint) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	first := true
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		if first {
			first = false
			var h header
			if err := json.Unmarshal(line, &h); err != nil {
				return fmt.Errorf("benchckpt: unreadable header in %s: %w", path, err)
			}
			if !fingerprintsEqual(h.Fingerprint, want) {
				return fmt.Errorf("%w: checkpoint=%s requested=%s",
					ErrFingerprintMismatch, mustJSON(h.Fingerprint), mustJSON(want))
			}
			continue
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			// A torn final line from a crash mid-append: stop reading records here.
			// Every complete earlier record already loaded stays valid.
			break
		}
		if _, seen := l.cells[r.Key]; !seen {
			l.order = append(l.order, r.Key)
		}
		l.cells[r.Key] = r.Cell
	}
	return nil
}

// Append write-ahead-records one measured cell under key: one complete JSON line
// followed by fsync, before the caller moves to the next cell. A duplicate key
// overwrites the in-memory cell (last write wins) but still appends a line, so the
// on-disk log stays append-only.
func (l *Ledger) Append(key string, cell any) error {
	data, err := json.Marshal(cell)
	if err != nil {
		return fmt.Errorf("benchckpt: marshal cell %q: %w", key, err)
	}
	if err := l.writeLine(record{Key: key, Cell: json.RawMessage(data)}); err != nil {
		return err
	}
	if _, seen := l.cells[key]; !seen {
		l.order = append(l.order, key)
	}
	l.cells[key] = json.RawMessage(data)
	return nil
}

// writeLine marshals v, appends it as one line, and fsyncs — the write-ahead unit.
func (l *Ledger) writeLine(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := l.f.Write(data); err != nil {
		return err
	}
	return l.f.Sync()
}

// Has reports whether key already has a recorded cell.
func (l *Ledger) Has(key string) bool {
	_, ok := l.cells[key]
	return ok
}

// Missing is the resume filter: given the full set of grid coordinates want, it returns
// exactly those not yet recorded, in the input order. A fresh run returns want
// unchanged; a fully-recorded grid returns an empty slice.
func (l *Ledger) Missing(want []string) []string {
	out := make([]string, 0, len(want))
	for _, k := range want {
		if !l.Has(k) {
			out = append(out, k)
		}
	}
	return out
}

// Cell returns the recorded cell for key (decoded into v, which must be a pointer). It
// reports false if key was never recorded.
func (l *Ledger) Cell(key string, v any) (bool, error) {
	raw, ok := l.cells[key]
	if !ok {
		return false, nil
	}
	if v == nil {
		return true, nil
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return true, fmt.Errorf("benchckpt: decode cell %q: %w", key, err)
	}
	return true, nil
}

// Keys returns the recorded coordinate keys in the order first seen — the order in
// which a caller reassembles the final artifact from the checkpoint.
func (l *Ledger) Keys() []string {
	return append([]string(nil), l.order...)
}

// Len is the number of distinct recorded cells.
func (l *Ledger) Len() int { return len(l.cells) }

// Close flushes and closes the underlying file.
func (l *Ledger) Close() error {
	if l.f == nil {
		return nil
	}
	err := l.f.Close()
	l.f = nil
	return err
}

// fingerprintsEqual compares two fingerprints by canonical JSON so numeric/key-order
// differences that JSON round-trips away do not count as a mismatch.
func fingerprintsEqual(a, b Fingerprint) bool {
	return bytes.Equal(mustJSON(a), mustJSON(b))
}

func mustJSON(fp Fingerprint) []byte {
	if fp == nil {
		fp = Fingerprint{}
	}
	// A Fingerprint that round-trips through JSON (the on-disk form) so the in-memory
	// and reloaded values compare equal regardless of numeric concrete type.
	round, err := json.Marshal(fp)
	if err != nil {
		return []byte(fmt.Sprintf("%v", fp))
	}
	var norm map[string]any
	if err := json.Unmarshal(round, &norm); err != nil {
		return round
	}
	out, err := json.Marshal(norm)
	if err != nil {
		return round
	}
	return out
}
