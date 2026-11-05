package protocol

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: tiny complete frames; the old overflow requires a 32-bit target.
func TestDecodeKVBodyBoundsDeclaredLength(t *testing.T) {
	t.Parallel()
	for _, declared := range []uint32{0, 1, 2, 0x7ffffff8, 0x7ffffff9, 0x7fffffff, 0x80000000, 0xffffffff} {
		for _, flags := range []uint8{FlagNone, FlagReplicated} {
			t.Run(fmt.Sprintf("length_%08x/flags_%d", declared, flags), func(t *testing.T) {
				// key "k", one physical value byte. Declared zero retains trailing-byte
				// compatibility; one is exact; larger claims must be refused.
				body := []byte{1, 0, 'k', 0, 0, 0, 0, 'v'}
				binary.LittleEndian.PutUint32(body[3:7], declared)
				var frame bytes.Buffer
				if err := WriteMessage(&frame, Message{
					Header: Header{OpCode: OpSet, Flags: flags, RequestID: 7}, Body: body,
				}); err != nil {
					t.Fatal(err)
				}
				msg, err := ReadMessage(&frame)
				if err != nil {
					t.Fatalf("complete small frame refused: %v", err)
				}
				defer PutBodyBuf(msg.Body, msg.BodyPoolIdx)
				if msg.Header.OpCode != OpSet || msg.Header.Flags != flags || msg.Header.RequestID != 7 || !bytes.Equal(msg.Body, body) || frame.Len() != 0 {
					t.Fatalf("frame changed: header=%+v body=%v remaining=%d", msg.Header, msg.Body, frame.Len())
				}
				key, value, ttl, err := DecodeKVBody(msg.Body, msg.Header.Flags)
				if declared > 1 {
					if err == nil || err.Error() != "kv body value truncated" || key != nil || value != nil || ttl != 0 {
						t.Fatalf("invalid length: key=%v value=%v ttl=%d err=%v", key, value, ttl, err)
					}
					return
				}
				wantValue := []byte{}
				if declared == 1 {
					wantValue = []byte("v")
				}
				if err != nil || string(key) != "k" || !bytes.Equal(value, wantValue) || value == nil || ttl != 0 {
					t.Fatalf("valid length: key=%v value=%v ttl=%d err=%v, want value=%v", key, value, ttl, err, wantValue)
				}
			})
		}
	}
}
