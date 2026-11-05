package protocol

import (
	"bytes"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Estimate only: bounded in-memory bodies at the uint16 wire-length boundary.
func TestDecodeKeyLengthUsesIndexWidth(t *testing.T) {
	t.Parallel()
	const duration int64 = 0x0102030405060708
	for _, tc := range []struct {
		name     string
		encode   func([]byte) []byte
		decode   func([]byte) ([]byte, int64, error)
		duration int64
	}{
		{"key", EncodeKeyBody, func(body []byte) ([]byte, int64, error) {
			key, err := DecodeKeyBody(body)
			return key, 0, err
		}, 0},
		{"pattern", EncodeKeysBody, func(body []byte) ([]byte, int64, error) {
			key, err := DecodeKeysBody(body)
			return key, 0, err
		}, 0},
		{"lease", func(key []byte) []byte { return EncodeLeaseBody(key, duration) }, DecodeLeaseBody, duration},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, size := range []int{0, 1, (1 << 16) - 3, (1 << 16) - 2, (1 << 16) - 1} {
				t.Run(fmt.Sprintf("length_%d", size), func(t *testing.T) {
					want := bytes.Repeat([]byte{0xa5}, size)
					body := tc.encode(want)
					for _, complete := range [][]byte{body, append(append([]byte(nil), body...), 0xee, 0xff)} {
						got, ttl, err := tc.decode(complete)
						if err != nil || !bytes.Equal(got, want) || ttl != tc.duration {
							t.Fatalf("complete body: length=%d duration=%d err=%v, want length=%d duration=%d", len(got), ttl, err, size, tc.duration)
						}
					}
					got, ttl, err := tc.decode(body[:len(body)-1])
					if err == nil || got != nil || ttl != 0 {
						t.Fatalf("truncated body: length=%d duration=%d err=%v, want refusal", len(got), ttl, err)
					}
				})
			}
		})
	}
}
