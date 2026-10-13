package client

import (
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// fak-test:runtime fast est=1s
func TestClientHandshakeAdmission(t *testing.T) {
	for _, tc := range []struct {
		name    string
		op      uint8
		body    string
		wrongID bool
		accept  bool
	}{
		{name: "native_protocol", op: protocol.RespOK, body: `{"status":"ok","api_version":1,"protocol_version":1,"ready":false}`, accept: true},
		{name: "error_response", op: protocol.RespError, body: `{"status":"incompatible"}`},
		{name: "wrong_opcode", op: protocol.RespValue, body: `{"status":"ok","api_version":1,"protocol_version":1}`},
		{name: "wrong_request", op: protocol.RespOK, body: `{"status":"ok","api_version":1,"protocol_version":1}`, wrongID: true},
		{name: "wrong_api", op: protocol.RespOK, body: `{"status":"ok","api_version":2,"protocol_version":1}`},
		{name: "wrong_protocol", op: protocol.RespOK, body: `{"status":"ok","api_version":1,"protocol_version":2}`},
		{name: "wrong_status", op: protocol.RespOK, body: `{"status":"incompatible","api_version":1,"protocol_version":1}`},
		{name: "malformed_json", op: protocol.RespOK, body: `{`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			go func() {
				defer close(done)
				conn, err := listener.Accept()
				if err != nil {
					return
				}
				defer conn.Close()
				_ = conn.SetDeadline(time.Now().Add(time.Second))
				codec := protocol.NewCodec(conn)
				request, err := codec.ReadMessage()
				if err != nil {
					return
				}
				// Keep the valid GET control usable against the pre-handshake client.
				if request.Header.OpCode == protocol.OpHandshake {
					id := request.Header.RequestID
					if tc.wrongID {
						id++
					}
					if codec.WriteMessage(protocol.Message{Header: protocol.Header{OpCode: tc.op, RequestID: id}, Body: []byte(tc.body)}) != nil {
						return
					}
					if !tc.accept {
						return
					}
					request, err = codec.ReadMessage()
					if err != nil {
						return
					}
				}
				if tc.accept && request.Header.OpCode == protocol.OpGet {
					_ = codec.WriteMessage(protocol.Message{Header: protocol.Header{OpCode: protocol.RespValue, RequestID: request.Header.RequestID}, Body: []byte{0}})
				}
			}()
			c, err := New(listener.Addr().String())
			if tc.accept {
				if err != nil {
					t.Errorf("valid native protocol rejected: %v", err)
				} else {
					value, getErr := c.Get([]byte("missing"))
					if getErr != nil || value != nil {
						t.Errorf("native GET control: value=%v err=%v", value, getErr)
					}
				}
			} else if !errors.Is(err, ErrHandshake) || c != nil {
				t.Error("unvalidated peer admitted as healthy client")
			}
			if c != nil {
				_ = c.Close()
			}
			_ = listener.Close()
			<-done
		})
	}
}

// fak-test:runtime fast est=1s
func TestClientHandshakeDeadline(t *testing.T) {
	for _, respond := range []bool{false, true} {
		name := "stalled_peer_closed"
		if respond {
			name = "deadline_reset_after_hello"
		}
		t.Run(name, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			defer listener.Close()
			done := make(chan error, 1)
			go func() {
				conn, err := listener.Accept()
				if err != nil {
					done <- err
					return
				}
				defer conn.Close()
				_ = conn.SetDeadline(time.Now().Add(time.Second))
				codec := protocol.NewCodec(conn)
				hello, err := codec.ReadMessage()
				if err != nil {
					done <- err
					return
				}
				if !respond {
					_, err = codec.ReadMessage()
					done <- err
					return
				}
				err = codec.WriteMessage(protocol.Message{Header: protocol.Header{OpCode: protocol.RespOK, RequestID: hello.Header.RequestID}, Body: []byte(`{"status":"ok","api_version":1,"protocol_version":1}`)})
				if err != nil {
					done <- err
					return
				}
				get, err := codec.ReadMessage()
				if err != nil {
					done <- err
					return
				}
				time.Sleep(100 * time.Millisecond)
				done <- codec.WriteMessage(protocol.Message{Header: protocol.Header{OpCode: protocol.RespValue, RequestID: get.Header.RequestID}, Body: []byte{0}})
			}()
			c, err := newWithTimeout(listener.Addr().String(), 50*time.Millisecond)
			if respond {
				if err != nil {
					t.Errorf("valid hello: %v", err)
				} else {
					_, err = c.Get([]byte("missing"))
					if err != nil {
						t.Errorf("handshake deadline leaked into GET: %v", err)
					}
					_ = c.Close()
				}
			} else if c != nil || !errors.Is(err, ErrHandshake) {
				if c != nil {
					_ = c.Close()
				}
				t.Errorf("stalled admission: client=%v err=%v", c, err)
			}
			peerErr := <-done
			if !respond && !errors.Is(peerErr, io.EOF) {
				t.Errorf("failed socket not closed: %v", peerErr)
			}
			if respond && peerErr != nil {
				t.Errorf("valid peer: %v", peerErr)
			}
		})
	}
}
