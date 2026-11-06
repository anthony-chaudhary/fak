package protocol

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: tiny in-memory frames; run on a 32-bit target to witness the old panic.
func TestBatchDecodersBoundDeclaredLengths(t *testing.T) {
	t.Parallel()
	for _, declared := range []uint32{0, 1, 2, 0x7ffffff4, 0x7ffffff5, 0x7ffffff6, 0x7ffffff7, 0x7ffffffb, 0x7ffffffc, 0x80000000, 0xffffffff} {
		t.Run(fmt.Sprintf("length_%08x", declared), func(t *testing.T) {
			// Each complete frame contains one payload byte. Zero preserves
			// accepted trailing bytes; one is exact; every larger claim truncates.
			mset := []byte{1, 0, 0, 0, 1, 0, 'k', 0, 0, 0, 0, 'v'}
			binary.LittleEndian.PutUint32(mset[7:11], declared)
			multi := []byte{1, 0, 0, 0, 1, 0, 0, 0, 0, 'v'}
			binary.LittleEndian.PutUint32(multi[5:9], declared)
			result := []byte{0, 0, 0, 0, 'v'}
			binary.LittleEndian.PutUint32(result[:4], declared)

			for _, tc := range []struct {
				name   string
				opcode byte
				body   []byte
			}{
				{name: "mset", opcode: OpMSet, body: mset},
				{name: "multi_value", opcode: RespMultiValue, body: multi},
				{name: "mset_result", opcode: RespMSetResult, body: result},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var frame bytes.Buffer
					if err := WriteMessage(&frame, Message{Header: Header{OpCode: tc.opcode}, Body: tc.body}); err != nil {
						t.Fatal(err)
					}
					msg, err := ReadMessage(&frame)
					if err != nil {
						t.Fatalf("complete small frame refused: %v", err)
					}
					defer PutBodyBuf(msg.Body, msg.BodyPoolIdx)
					var got []byte
					var wantError string
					switch tc.name {
					case "mset":
						var keys, values [][]byte
						keys, values, err = DecodeMSetBody(msg.Body)
						wantError = "mset body value 0 truncated"
						if declared <= 1 {
							if err != nil || len(keys) != 1 || string(keys[0]) != "k" || len(values) != 1 {
								t.Fatalf("keys=%v values=%v err=%v", keys, values, err)
							}
							got = values[0]
						} else if keys != nil || values != nil {
							t.Fatalf("failed MSET leaked partial results: %v %v", keys, values)
						}
					case "multi_value":
						var values [][]byte
						var founds []bool
						values, founds, err = DecodeMultiValueResponse(msg.Body)
						wantError = "multi value response value 0 truncated"
						if declared <= 1 {
							if err != nil || len(values) != 1 || len(founds) != 1 || !founds[0] {
								t.Fatalf("values=%v founds=%v err=%v", values, founds, err)
							}
							got = values[0]
						} else if values != nil || founds != nil {
							t.Fatalf("failed multivalue leaked partial results: %v %v", values, founds)
						}
					case "mset_result":
						got, err = DecodeMSetResultResponse(msg.Body)
						wantError = fmt.Sprintf("mset result truncated: have 5, need %d", uint64(declared)+4)
					}
					if declared > 1 {
						if err == nil || err.Error() != wantError || got != nil {
							t.Fatalf("value=%v err=%v, want nil and %q", got, err, wantError)
						}
						return
					}
					want := []byte("v")[:int(declared)]
					if err != nil || got == nil || !bytes.Equal(got, want) {
						t.Fatalf("value=%v err=%v, want non-nil %v", got, err, want)
					}
				})
			}
		})
	}
}
