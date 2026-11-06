package protocol

import (
	"bytes"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: tiny in-memory frames; no listener, clock waits or shard setup.
func TestDecodeKVBodyRequiresFlaggedTTL(t *testing.T) {
	t.Parallel()
	// Independent wire fixture: key "k", value "v", TTL 750 ms, little-endian.
	body := []byte{1, 0, 'k', 1, 0, 0, 0, 'v', 0xee, 2, 0, 0, 0, 0, 0, 0}
	const prefix = 8
	for _, flags := range []uint8{FlagWithTTL, FlagWithTTL | FlagReplicated} {
		for n := 0; n <= 8; n++ {
			t.Run(fmt.Sprintf("flags=%d/ttl_bytes=%d", flags, n), func(t *testing.T) {
				var frame bytes.Buffer
				msg := Message{Header: Header{OpCode: OpSet, Flags: flags}, Body: body[:prefix+n]}
				if err := WriteMessage(&frame, msg); err != nil {
					t.Fatal(err)
				}
				decoded, err := ReadMessage(&frame)
				if err != nil {
					t.Fatalf("complete frame refused before body validation: %v", err)
				}
				defer PutBodyBuf(decoded.Body, decoded.BodyPoolIdx)
				key, value, ttl, err := DecodeKVBody(decoded.Body, decoded.Header.Flags)
				if n < 8 {
					if err == nil || key != nil || value != nil || ttl != 0 {
						t.Fatalf("incomplete TTL: key=%v value=%v ttl=%d err=%v", key, value, ttl, err)
					}
					return
				}
				if err != nil || string(key) != "k" || string(value) != "v" || ttl != 750 {
					t.Fatalf("complete TTL: key=%v value=%v ttl=%d err=%v", key, value, ttl, err)
				}
			})
		}
	}
	for _, n := range []int{prefix, len(body)} {
		key, value, ttl, err := DecodeKVBody(body[:n], FlagReplicated)
		if err != nil || string(key) != "k" || string(value) != "v" || ttl != 0 {
			t.Fatalf("unflagged body length %d: key=%v value=%v ttl=%d err=%v", n, key, value, ttl, err)
		}
	}
	for _, tc := range []struct {
		fill byte
		want int64
	}{{0, 0}, {0xff, -1}} {
		complete := append([]byte(nil), body...)
		for i := prefix; i < len(complete); i++ {
			complete[i] = tc.fill
		}
		key, value, ttl, err := DecodeKVBody(complete, FlagWithTTL)
		if err != nil || string(key) != "k" || string(value) != "v" || ttl != tc.want {
			t.Fatalf("complete TTL %d: key=%v value=%v ttl=%d err=%v", tc.want, key, value, ttl, err)
		}
	}
}
