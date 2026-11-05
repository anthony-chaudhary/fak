package client

import (
	"bufio"
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: buffered frames, without a listener, handshake, or daemon.
func TestMGetRequiresMultiValueResponse(t *testing.T) {
	t.Parallel()
	hit := []byte{2, 0, 0, 0, 1, 1, 0, 0, 0, 'v', 0}
	for _, tc := range []struct {
		name      string
		code      uint8
		body      []byte
		empty     bool
		values    [][]byte
		founds    []bool
		wantError string
	}{
		{name: "hit_and_miss", code: protocol.RespMultiValue, body: hit, values: [][]byte{[]byte("v"), nil}, founds: []bool{true, false}},
		{name: "all_misses", code: protocol.RespMultiValue, body: []byte{2, 0, 0, 0, 0, 0}, values: [][]byte{nil, nil}, founds: []bool{false, false}},
		{name: "empty_request", code: protocol.RespMultiValue, body: []byte{0, 0, 0, 0}, empty: true, values: [][]byte{}, founds: []bool{}},
		{name: "malformed_multivalue", code: protocol.RespMultiValue, body: []byte{1, 0, 0, 0, 1}, wantError: "multi value response value 0 length truncated"},
		{name: "server_error", code: protocol.RespError, body: []byte("denied"), wantError: "server error: denied"},
		{name: "not_ready", code: protocol.RespNotReady, body: protocol.EncodeNotReadyResponse(0, 2, "starting")},
		{name: "mset_result", code: protocol.RespMSetResult, body: protocol.EncodeMSetResultResponse([]byte{0, 0})},
		{name: "wrong_kind_hit", code: protocol.RespOK, body: hit},
		{name: "unknown_hit", code: 0xff, body: hit},
	} {
		t.Run(tc.name, func(t *testing.T) {
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
			keys := [][]byte{[]byte("a"), []byte("b")}
			wantBody := []byte{2, 0, 0, 0, 1, 0, 'a', 1, 0, 'b'}
			if tc.empty {
				keys = nil
				wantBody = []byte{0, 0, 0, 0}
			}
			values, founds, err := c.MGet(keys)
			wantError := tc.wantError
			if tc.code != protocol.RespMultiValue && tc.code != protocol.RespError {
				wantError = fmt.Sprintf("unexpected MGET response opcode: 0x%02x", tc.code)
			}
			if (err == nil) != (wantError == "") || !reflect.DeepEqual(values, tc.values) || !reflect.DeepEqual(founds, tc.founds) {
				t.Fatalf("response 0x%02x: values=%v founds=%v err=%v, want values=%v founds=%v error=%q", tc.code, values, founds, err, tc.values, tc.founds, wantError)
			}
			if err != nil && err.Error() != wantError {
				t.Fatalf("error=%q, want %q", err, wantError)
			}
			request, err := protocol.ReadMessage(&requests)
			if err != nil {
				t.Fatal(err)
			}
			defer protocol.PutBodyBuf(request.Body, request.BodyPoolIdx)
			if request.Header.OpCode != protocol.OpMGet || request.Header.RequestID != 42 || request.Header.Flags != protocol.FlagNone || !bytes.Equal(request.Body, wantBody) || requests.Len() != 0 || c.reqID.Load() != 42 {
				t.Fatalf("MGET request changed: header=%+v body=%v remaining=%d counter=%d", request.Header, request.Body, requests.Len(), c.reqID.Load())
			}
		})
	}
}
