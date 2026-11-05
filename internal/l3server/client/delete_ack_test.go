package client

import (
	"bufio"
	"bytes"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: buffered frames; no listener, handshake, or daemon.
func TestDeleteRequiresOKResponse(t *testing.T) {
	t.Parallel()
	for _, op := range []struct {
		name  string
		flags uint8
		call  func(*Client, []byte) error
	}{
		{"delete", protocol.FlagNone, (*Client).Delete},
		{"replicated", protocol.FlagReplicated, (*Client).DeleteReplicated},
	} {
		for _, tc := range []struct {
			name string
			code uint8
			body []byte
			ok   bool
		}{
			{"ok", protocol.RespOK, nil, true},
			{"ok_body_ignored", protocol.RespOK, []byte("metadata"), true},
			{"error", protocol.RespError, []byte("denied"), false},
			{"not_ready", protocol.RespNotReady, protocol.EncodeNotReadyResponse(0, 1, "starting"), false},
			{"oom", protocol.RespOOM, protocol.EncodeOOMResponse(100, 8, 8, "full"), false},
			{"value", protocol.RespValue, protocol.EncodeValueResponse([]byte("v"), true), false},
			{"mset_result", protocol.RespMSetResult, protocol.EncodeMSetResultResponse([]byte{0}), false},
			{"unknown", 0xff, nil, false},
		} {
			t.Run(op.name+"/"+tc.name, func(t *testing.T) {
				var responses, requests bytes.Buffer
				if err := protocol.WriteMessage(&responses, protocol.Message{
					Header: protocol.Header{OpCode: tc.code, RequestID: 42}, Body: tc.body,
				}); err != nil {
					t.Fatal(err)
				}
				c := &Client{codec: &protocol.Codec{
					Reader: bufio.NewReader(&responses), Writer: bufio.NewWriter(&requests),
				}}
				c.reqID.Store(41)
				err := op.call(c, []byte("key"))
				if (err == nil) != tc.ok {
					t.Fatalf("response 0x%02x: err=%v, want success=%t", tc.code, err, tc.ok)
				}
				if !tc.ok {
					want := fmt.Sprintf("unexpected DELETE response opcode: 0x%02x", tc.code)
					if tc.code == protocol.RespError {
						want = "server error: denied"
					}
					if err.Error() != want {
						t.Fatalf("error = %q, want %q", err, want)
					}
				}
				request, err := protocol.ReadMessage(&requests)
				if err != nil {
					t.Fatal(err)
				}
				defer protocol.PutBodyBuf(request.Body, request.BodyPoolIdx)
				if request.Header.OpCode != protocol.OpDelete || request.Header.RequestID != 42 || request.Header.Flags != op.flags || !bytes.Equal(request.Body, protocol.EncodeKeyBody([]byte("key"))) || requests.Len() != 0 || c.reqID.Load() != 42 {
					t.Fatalf("DELETE request changed: header=%+v body=%v remaining=%d counter=%d", request.Header, request.Body, requests.Len(), c.reqID.Load())
				}
			})
		}
	}
}
