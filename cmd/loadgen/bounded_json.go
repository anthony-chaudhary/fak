// Inspired by ai-dynamo/aiperf, src/aiperf/dataset/json_stream.py:1 at
// ab610add4e8876fca223197596c728e17365f381 (Apache-2.0; idea only, no code
// copied): dataset dumps are written as bounded JSON chunks rather than one
// unbounded document.

package main

import (
	"encoding/json"
	"errors"
)

// defaultJSONChunkBytes is the upstream 1 MiB chunk bound.
const defaultJSONChunkBytes = 1 << 20

var errJSONChunkBound = errors.New("jsonchunks: non-positive chunk bound")

// BoundedJSONChunks encodes records into JSON-array chunks of at most
// maxBytes each. A single record larger than the bound gets a chunk of its
// own (it cannot be split without changing the record). maxBytes <= 0 selects
// the 1 MiB default only when exactly zero; a negative bound is refused.
func BoundedJSONChunks(records []any, maxBytes int) ([][]byte, error) {
	if maxBytes < 0 {
		return nil, errJSONChunkBound
	}
	if maxBytes == 0 {
		maxBytes = defaultJSONChunkBytes
	}
	var chunks [][]byte
	cur := []byte{'['}
	count := 0
	flush := func() {
		if count > 0 {
			chunks = append(chunks, append(cur, ']'))
		}
		cur, count = []byte{'['}, 0
	}
	for _, r := range records {
		enc, err := json.Marshal(r)
		if err != nil {
			return nil, err
		}
		sep := 0
		if count > 0 {
			sep = 1
		}
		if count > 0 && len(cur)+sep+len(enc)+1 > maxBytes {
			flush()
			sep = 0
		}
		if sep == 1 {
			cur = append(cur, ',')
		}
		cur = append(cur, enc...)
		count++
	}
	flush()
	return chunks, nil
}
