package client

import (
	"bufio"
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: synthetic buffered frames, with no listener or handshake.
func TestMSetRequiresAcknowledgment(t *testing.T) {
	t.Parallel()
	keys, values := [][]byte{[]byte("a"), []byte("b")}, [][]byte{[]byte("one"), []byte("two")}
	for _, tc := range []struct {
		name     string
		code     uint8
		body     []byte
		statuses []byte
		wantErr  bool
	}{
		{"ok", protocol.RespOK, nil, nil, false},
		{"ok_body_ignored", protocol.RespOK, []byte("metadata"), nil, false},
		{"partial", protocol.RespMSetResult, protocol.EncodeMSetResultResponse([]byte{0, 1}), []byte{0, 1}, false},
		{"empty_result", protocol.RespMSetResult, protocol.EncodeMSetResultResponse(nil), []byte{}, false},
		{"malformed_result", protocol.RespMSetResult, []byte{2, 0, 0, 0, 1}, nil, true},
		{"error", protocol.RespError, []byte("denied"), nil, true},
		{"oom", protocol.RespOOM, protocol.EncodeOOMResponse(100, 8, 8, "full"), nil, true},
		{"not_ready", protocol.RespNotReady, protocol.EncodeNotReadyResponse(0, 1, "starting"), nil, true},
		{"value", protocol.RespValue, protocol.EncodeValueResponse([]byte("v"), true), nil, true},
		{"unknown", 0xff, nil, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var responses, requests bytes.Buffer
			if err := protocol.WriteMessage(&responses, protocol.Message{
				Header: protocol.Header{OpCode: tc.code, RequestID: 1}, Body: tc.body,
			}); err != nil {
				t.Fatal(err)
			}
			c := &Client{codec: &protocol.Codec{
				Reader: bufio.NewReader(&responses), Writer: bufio.NewWriter(&requests),
			}}
			statuses, err := c.MSet(keys, values)
			if (err != nil) != tc.wantErr || (statuses == nil) != (tc.statuses == nil) || !bytes.Equal(statuses, tc.statuses) {
				t.Fatalf("response 0x%02x: statuses=%v err=%v, want statuses=%v error=%t", tc.code, statuses, err, tc.statuses, tc.wantErr)
			}
			if tc.code == protocol.RespError && err.Error() != "server error: denied" {
				t.Fatalf("legacy server error changed: %v", err)
			}
			request, err := protocol.ReadMessage(&requests)
			if err != nil {
				t.Fatal(err)
			}
			defer protocol.PutBodyBuf(request.Body, request.BodyPoolIdx)
			if request.Header.OpCode != protocol.OpMSet || request.Header.RequestID != 1 || request.Header.Flags != 0 || !bytes.Equal(request.Body, protocol.EncodeMSetBody(keys, values)) || requests.Len() != 0 {
				t.Fatalf("MSET request changed: header=%+v body=%v remaining=%d", request.Header, request.Body, requests.Len())
			}
		})
	}
}
