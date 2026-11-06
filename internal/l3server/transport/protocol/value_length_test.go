package protocol

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: tiny in-memory frames; no listener or large allocation.
func TestDecodeValueResponseBoundsDeclaredLength(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name  string
		body  []byte
		value []byte
		found bool
		ok    bool
	}{
		{name: "missing_body"},
		{name: "absent", body: []byte{0}, ok: true},
		{name: "short_present", body: []byte{1}},
		{name: "empty_present", body: []byte{1, 0, 0, 0, 0}, value: []byte{}, found: true, ok: true},
		{name: "exact", body: []byte{1, 1, 0, 0, 0, 'v'}, value: []byte("v"), found: true, ok: true},
		{name: "trailing_bytes", body: []byte{1, 1, 0, 0, 0, 'v', 'x'}, value: []byte("v"), found: true, ok: true},
		{name: "truncated", body: []byte{1, 2, 0, 0, 0, 'v'}},
		// These complete five-byte bodies never need their declared allocation.
		// The old decoder panics on 32-bit targets for the last three rows.
		{name: "signed_add_boundary", body: []byte{1, 0xfa, 0xff, 0xff, 0x7f}},
		{name: "signed_add_overflow", body: []byte{1, 0xfb, 0xff, 0xff, 0x7f}},
		{name: "signed_conversion_overflow", body: []byte{1, 0, 0, 0, 0x80}},
		{name: "unsigned_end_overflow", body: []byte{1, 0xff, 0xff, 0xff, 0xff}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var frame bytes.Buffer
			if err := WriteMessage(&frame, Message{Header: Header{OpCode: RespValue}, Body: tc.body}); err != nil {
				t.Fatal(err)
			}
			msg, err := ReadMessage(&frame)
			if err != nil {
				t.Fatalf("complete small frame refused: %v", err)
			}
			defer PutBodyBuf(msg.Body, msg.BodyPoolIdx)
			value, found, err := DecodeValueResponse(msg.Body)
			if (err == nil) != tc.ok || found != tc.found || !bytes.Equal(value, tc.value) || (value == nil) != (tc.value == nil) {
				t.Fatalf("value=%v found=%t err=%v, want value=%v found=%t success=%t", value, found, err, tc.value, tc.found, tc.ok)
			}
		})
	}
}
