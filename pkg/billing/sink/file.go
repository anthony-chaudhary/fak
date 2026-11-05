package sink

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// File is a JSONL Backend: it appends one record per line to a file,
// creating the file when absent, and syncs after every batch so a
// delivered record survives a crash or restart. The file is opened
// append-only, so a restarted producer keeps appending after earlier
// content instead of overwriting it.
//
// Lines are produced by encoding/json over the UsageRecord struct, so
// field order is fixed by the struct layout - deterministic across runs
// and platforms - and no raw newline can ever appear inside a line.
//
// Append is all-or-nothing with respect to file content: if any part of
// a batch fails, the partial bytes are rolled back and the file is left
// exactly as the batch found it, so a retried batch re-appends cleanly
// instead of leaving a torn line behind.
type File struct {
	path string
	f    *os.File
	w    *bufio.Writer

	mu     sync.Mutex
	closed bool
}

// NewFile opens path for append-only JSONL output, creating it with
// mode 0644 when absent.
func NewFile(path string) (*File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &File{path: path, f: f, w: bufio.NewWriter(f)}, nil
}

// Path reports the file's path.
func (file *File) Path() string { return file.path }

// Append writes one line per record, flushes the writer, and syncs the
// file. On any failure it rolls the file back to its pre-batch size and
// returns the error.
func (file *File) Append(batch []billing.UsageRecord) error {
	file.mu.Lock()
	defer file.mu.Unlock()
	if file.closed {
		return ErrClosed
	}
	start, err := file.f.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	for i := range batch {
		line, err := json.Marshal(&batch[i])
		if err != nil {
			return err
		}
		if _, err := file.w.Write(line); err != nil {
			file.rollback(start)
			return err
		}
		if err := file.w.WriteByte('\n'); err != nil {
			file.rollback(start)
			return err
		}
	}
	if err := file.w.Flush(); err != nil {
		file.rollback(start)
		return err
	}
	if err := file.f.Sync(); err != nil {
		file.rollback(start)
		return err
	}
	return nil
}

// Close closes the underlying file. It is idempotent; later calls
// return nil.
func (file *File) Close() error {
	file.mu.Lock()
	defer file.mu.Unlock()
	if file.closed {
		return nil
	}
	file.closed = true
	return file.f.Close()
}

// rollback restores the file to its pre-batch size, discarding any
// partially written batch bytes. Callers hold mu.
func (file *File) rollback(start int64) {
	file.w.Reset(file.f)
	_ = file.f.Truncate(start)
}

// ReadJSONL reads back a JSONL record file written by File (or any
// conforming emitter) and decodes it strictly: exactly one record per
// line, each line decoded through billing.DecodeUsageRecord, which
// refuses unknown fields and trailing data. An interior empty line, a
// missing final newline, or any line that fails strict decoding is an
// error. A zero-byte file yields no records and no error. Decoding does
// not check semantic rules; callers that need them call Validate on
// each record.
func ReadJSONL(path string) ([]billing.UsageRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	lines := bytes.Split(data, []byte{'\n'})
	if len(lines[len(lines)-1]) != 0 {
		return nil, fmt.Errorf("%w: final line lacks its newline (truncated write?)", ErrMalformedLine)
	}
	lines = lines[:len(lines)-1]
	recs := make([]billing.UsageRecord, 0, len(lines))
	for i, line := range lines {
		if len(line) == 0 {
			return nil, fmt.Errorf("%w: line %d is empty", ErrMalformedLine, i+1)
		}
		rec, err := billing.DecodeUsageRecord(line)
		if err != nil {
			return nil, err
		}
		recs = append(recs, rec)
	}
	return recs, nil
}
