package client

import (
	"bufio"
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: synthetic buffered frames, with no listener or handshake.
func TestReadRequiresExpectedResponse(t *testing.T) {
	t.Parallel()
	for _, op := range []struct {
		name         string
		requestCode  uint8
		responseCode uint8
		wrongCode    uint8
		hitBody      []byte
		wantValue    []byte
		call         func(*Client, []byte) ([]byte, bool, error)
	}{
		{
			name: "get", requestCode: protocol.OpGet, responseCode: protocol.RespValue,
			wrongCode: protocol.RespOK, hitBody: []byte{1, 1, 0, 0, 0, 'v'}, wantValue: []byte("v"),
			call: func(c *Client, key []byte) ([]byte, bool, error) {
				value, err := c.Get(key)
				return value, value != nil, err
			},
		},
		{
			name: "exists", requestCode: protocol.OpTest, responseCode: protocol.RespOK,
			wrongCode: protocol.RespValue, hitBody: []byte{1},
			call: func(c *Client, key []byte) ([]byte, bool, error) {
				found, err := c.Exists(key)
				return nil, found, err
			},
		},
	} {
		for _, tc := range []struct {
			name  string
			code  uint8
			body  []byte
			ok    bool
			found bool
		}{
			{"hit", op.responseCode, op.hitBody, true, true},
			{"miss", op.responseCode, []byte{0}, true, false},
			{"error", protocol.RespError, []byte("denied"), false, false},
			{"not_ready_zero", protocol.RespNotReady, protocol.EncodeNotReadyResponse(0, 2, "starting"), false, false},
			{"not_ready_one", protocol.RespNotReady, protocol.EncodeNotReadyResponse(1, 2, "starting"), false, false},
			{"wrong_kind", op.wrongCode, op.hitBody, false, false},
			{"unknown", 0xff, op.hitBody, false, false},
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
				value, found, err := op.call(c, []byte("k"))
				var wantValue []byte
				if tc.found {
					wantValue = op.wantValue
				}
				if (err == nil) != tc.ok || found != tc.found || !bytes.Equal(value, wantValue) || (value == nil) != (wantValue == nil) {
					t.Fatalf("response 0x%02x: value=%v found=%t err=%v, want value=%v found=%t success=%t", tc.code, value, found, err, wantValue, tc.found, tc.ok)
				}
				if tc.code == protocol.RespError && err.Error() != "server error: denied" {
					t.Fatalf("legacy server error changed: %v", err)
				}
				request, err := protocol.ReadMessage(&requests)
				if err != nil {
					t.Fatal(err)
				}
				defer protocol.PutBodyBuf(request.Body, request.BodyPoolIdx)
				key, err := protocol.DecodeKeyBody(request.Body)
				if err != nil || request.Header.OpCode != op.requestCode || request.Header.RequestID != 1 || request.Header.Flags != protocol.FlagNone || string(key) != "k" || requests.Len() != 0 {
					t.Fatalf("read request changed: header=%+v key=%v err=%v remaining=%d", request.Header, key, err, requests.Len())
				}
			})
		}
	}
}
