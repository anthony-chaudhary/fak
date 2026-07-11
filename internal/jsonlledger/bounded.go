package jsonlledger

import (
	"bytes"
	"errors"
	"io"
	"os"
)

// DefaultActiveBytes is the maximum active-ledger size used by unattended
// nightrun writers. One sealed generation may coexist beside it.
const DefaultActiveBytes int64 = 8 << 20

// AppendBounded appends one complete JSONL line, rotating the prior active file
// to path+".1" before the append that would cross maxBytes. Rotation lives at
// the write site, so an unattended writer cannot bypass the disk bound.
func AppendBounded(path string, line []byte, maxBytes int64) error {
	if maxBytes <= 0 {
		maxBytes = DefaultActiveBytes
	}
	if len(line) == 0 || line[len(line)-1] != '\n' {
		line = append(append([]byte(nil), line...), '\n')
	}
	if st, err := os.Stat(path); err == nil && st.Size()+int64(len(line)) > maxBytes {
		sealed := path + ".1"
		_ = os.Remove(sealed)
		if err := os.Rename(path, sealed); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	if _, err = f.Write(line); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// ReadTail returns at most maxBytes from the end of path, dropping the first
// partial row when the window starts mid-line. Missing/unreadable files return
// nil, matching the ledgers' fail-open first-run contract.
func ReadTail(path string, maxBytes int64) []byte {
	if maxBytes <= 0 {
		maxBytes = DefaultActiveBytes
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil
	}
	start := st.Size() - maxBytes
	if start < 0 {
		start = 0
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil
	}
	b, err := io.ReadAll(io.LimitReader(f, maxBytes))
	if err != nil {
		return nil
	}
	if start > 0 {
		if i := bytes.IndexByte(b, '\n'); i >= 0 {
			b = b[i+1:]
		} else {
			return nil
		}
	}
	return b
}
