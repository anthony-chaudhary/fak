package client

import (
	"bufio"
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: synthetic buffered frames, with no listener or handshake.
func TestSetRequiresOKResponse(t *testing.T) {
	t.Parallel()
	for _, op := range []struct {
		name  string
		flags uint8
		call  func(*Client, []byte, []byte, int64) error
	}{
		{"set", protocol.FlagWithTTL, (*Client).Set},
		{"replicated", protocol.FlagWithTTL | protocol.FlagReplicated, (*Client).SetReplicated},
	} {
		for _, tc := range []struct {
			name string
			code uint8
			body []byte
			ok   bool
		}{
			{"ok", protocol.RespOK, nil, true},
			{"error", protocol.RespError, []byte("denied"), false},
			{"oom", protocol.RespOOM, protocol.EncodeOOMResponse(100, 8, 8, "full"), false},
			{"not_ready", protocol.RespNotReady, protocol.EncodeNotReadyResponse(0, 1, "starting"), false},
			{"value", protocol.RespValue, protocol.EncodeValueResponse([]byte("v"), true), false},
			{"unknown", 0xff, nil, false},
		} {
			t.Run(op.name+"/"+tc.name, func(t *testing.T) {
				var responses, requests bytes.Buffer
				if err := protocol.WriteMessage(&responses, protocol.Message{
					Header: protocol.Header{OpCode: tc.code, RequestID: 1}, Body: tc.body,
				}); err != nil {
					t.Fatal(err)
				}
				c := &Client{codec: &protocol.Codec{
					Reader: bufio.NewReader(&responses), Writer: bufio.NewWriter(&requests),
				}}
				err := op.call(c, []byte("k"), []byte("v"), 25)
				if (err == nil) != tc.ok {
					t.Fatalf("response 0x%02x: err=%v, want success=%t", tc.code, err, tc.ok)
				}
				if tc.code == protocol.RespError && err.Error() != "server error: denied" {
					t.Fatalf("legacy server error changed: %v", err)
				}
				request, err := protocol.ReadMessage(&requests)
				if err != nil {
					t.Fatal(err)
				}
				defer protocol.PutBodyBuf(request.Body, request.BodyPoolIdx)
				key, value, ttl, err := protocol.DecodeKVBody(request.Body, request.Header.Flags)
				if err != nil || request.Header.OpCode != protocol.OpSet || request.Header.RequestID != 1 || request.Header.Flags != op.flags || string(key) != "k" || string(value) != "v" || ttl != 25 || requests.Len() != 0 {
					t.Fatalf("SET request changed: header=%+v key=%v value=%v ttl=%d err=%v remaining=%d", request.Header, key, value, ttl, err, requests.Len())
				}
			})
		}
	}
}
