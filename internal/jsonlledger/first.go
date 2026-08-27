package jsonlledger

import (
	"bufio"
	"encoding/json"
	"io"
)

const maxRowBytes = 4 * 1024 * 1024

// First decodes the first non-blank JSONL row from r. A malformed first row is
// returned as an error rather than skipped, preserving strict schema-probe
// behavior while keeping scanner setup in one dependency-safe package.
func First[T any](r io.Reader) (row T, found bool, err error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxRowBytes)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		if err := json.Unmarshal(line, &row); err != nil {
			return row, false, err
		}
		return row, true, nil
	}
	if err := sc.Err(); err != nil {
		return row, false, err
	}
	return row, false, nil
}
