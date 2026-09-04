package harnessprotocol

import (
	"bufio"
	"encoding/json"
	"io"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// ReadJSONL decodes a stream of newline-delimited JSON envelopes from an io.Reader.
func ReadJSONL(r io.Reader) ([]harnesskit.Envelope, error) {
	var out []harnesskit.Envelope
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	for s.Scan() {
		var e harnesskit.Envelope
		if err := json.Unmarshal(s.Bytes(), &e); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, s.Err()
}

// WriteJSONL encodes a slice of envelopes as newline-delimited JSON to an io.Writer.
func WriteJSONL(w io.Writer, events []harnesskit.Envelope) error {
	enc := json.NewEncoder(w)
	for _, e := range events {
		if err := enc.Encode(e); err != nil {
			return err
		}
	}
	return nil
}
