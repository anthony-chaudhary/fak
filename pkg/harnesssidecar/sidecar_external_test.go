package harnesssidecar_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"reflect"
	"testing"
	"time"

	hs "github.com/anthony-chaudhary/fak/pkg/harnesssidecar"
)

func hello(name string, caps []string) hs.Handshake {
	return hs.Handshake{Protocol: hs.ProtocolVersion, Identity: hs.Identity{Name: name, Version: "1", Digest: hs.ContractDigest(caps)}, Capabilities: caps, Limits: hs.Limits{MaxFrame: 4096, MaxInflight: 2, CancelGrace: time.Second}}
}

func TestEquivalentInProcessAndSidecarTrace(t *testing.T) {
	handler := hs.HandlerFunc(func(_ context.Context, m string, p json.RawMessage, send func(json.RawMessage) error) error {
		if err := send(json.RawMessage(`{"seq":1}`)); err != nil {
			return err
		}
		return send(append(json.RawMessage(nil), p...))
	})
	direct := collectDirect(t, handler)
	clientConn, serverConn := net.Pipe()
	defer clientConn.Close()
	defer serverConn.Close()
	serverHello, clientHello := hello("go-sidecar", []string{"stream"}), hello("host", []string{"stream", "extra-host-only"})
	server := hs.NewServer(serverConn, serverConn, serverHello, clientHello, handler)
	done := make(chan error, 1)
	go func() { done <- server.Serve(context.Background()) }()
	codec := hs.NewCodec(clientConn, clientConn, 4096)
	writeWire(t, codec, map[string]any{"kind": "handshake", "handshake": clientHello})
	var ack map[string]any
	if err := codec.Read(&ack); err != nil {
		t.Fatal(err)
	}
	writeWire(t, codec, map[string]any{"kind": "request", "request": hs.Request{ID: "1", Method: "fixture", Payload: json.RawMessage(`{"seq":2}`)}})
	var trace []json.RawMessage
	for {
		var raw struct {
			Kind     string       `json:"kind"`
			Response *hs.Response `json:"response"`
		}
		if err := codec.Read(&raw); err != nil {
			t.Fatal(err)
		}
		if raw.Response.Done {
			break
		}
		trace = append(trace, raw.Response.Payload)
	}
	if !reflect.DeepEqual(direct, trace) {
		t.Fatalf("semantic trace differs direct=%s sidecar=%s", direct, trace)
	}
	clientConn.Close()
	<-done
}

func collectDirect(t *testing.T, h hs.Handler) []json.RawMessage {
	t.Helper()
	var out []json.RawMessage
	err := h.Handle(context.Background(), "fixture", json.RawMessage(`{"seq":2}`), func(p json.RawMessage) error { out = append(out, append(json.RawMessage(nil), p...)); return nil })
	if err != nil {
		t.Fatal(err)
	}
	return out
}
func writeWire(t *testing.T, c *hs.Codec, v any) {
	t.Helper()
	if err := c.Write(v); err != nil {
		t.Fatal(err)
	}
}

func TestHandshakeRejectsWideningAndDigestMismatch(t *testing.T) {
	want := hello("host", []string{"stream"})
	widened := hello("child", []string{"stream", "tools"})
	if err := hs.ValidateHandshake(want, widened); !errors.Is(err, hs.ErrWidening) {
		t.Fatalf("want widening: %v", err)
	}
	bad := hello("child", []string{"stream"})
	bad.Identity.Digest = "tampered"
	if err := hs.ValidateHandshake(want, bad); !errors.Is(err, hs.ErrProtocol) {
		t.Fatalf("want digest refusal: %v", err)
	}
}

func TestMalformedAndOversizeFramesFailClosed(t *testing.T) {
	var wire bytes.Buffer
	binary.Write(&wire, binary.BigEndian, uint32(5000))
	wire.Write(make([]byte, 10))
	codec := hs.NewCodec(&wire, io.Discard, 1024)
	var out any
	if err := codec.Read(&out); !errors.Is(err, hs.ErrProtocol) {
		t.Fatalf("want size refusal: %v", err)
	}
	wire.Reset()
	binary.Write(&wire, binary.BigEndian, uint32(3))
	wire.WriteString("bad")
	codec = hs.NewCodec(&wire, io.Discard, 1024)
	if err := codec.Read(&out); !errors.Is(err, hs.ErrProtocol) {
		t.Fatalf("want malformed refusal: %v", err)
	}
}

func TestCancellationReachesHandler(t *testing.T) {
	client, peer := net.Pipe()
	defer client.Close()
	defer peer.Close()
	entered := make(chan struct{})
	canceled := make(chan struct{})
	h := hs.HandlerFunc(func(ctx context.Context, _ string, _ json.RawMessage, _ func(json.RawMessage) error) error {
		close(entered)
		<-ctx.Done()
		close(canceled)
		return ctx.Err()
	})
	host, child := hello("host", []string{"cancel"}), hello("child", []string{"cancel"})
	s := hs.NewServer(peer, peer, child, host, h)
	go s.Serve(context.Background())
	c := hs.NewCodec(client, client, 4096)
	writeWire(t, c, map[string]any{"kind": "handshake", "handshake": host})
	var ack any
	c.Read(&ack)
	writeWire(t, c, map[string]any{"kind": "request", "request": hs.Request{ID: "cancel-me", Method: "wait"}})
	<-entered
	writeWire(t, c, map[string]any{"kind": "cancel", "cancel_id": "cancel-me"})
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("cancel did not reach handler")
	}
}

func TestSchemaAndGeneratedClientsPinProtocol(t *testing.T) {
	paths := []string{
		"schema/protocol.schema.json",
		"clients/typescript/protocol.ts",
		"clients/python/protocol.py",
	}
	for _, path := range paths {
		blob, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(blob, []byte(hs.ProtocolVersion)) {
			t.Fatalf("%s does not pin %s", path, hs.ProtocolVersion)
		}
	}
	blob, _ := os.ReadFile(paths[0])
	if !json.Valid(blob) {
		t.Fatal("protocol schema is not valid JSON")
	}
}

func TestCapabilityTokenRequiredBeforeHandler(t *testing.T) {
	client, peer := net.Pipe()
	defer client.Close()
	defer peer.Close()
	called := make(chan struct{}, 1)
	h := hs.HandlerFunc(func(context.Context, string, json.RawMessage, func(json.RawMessage) error) error {
		called <- struct{}{}
		return nil
	})
	auth := hs.AuthorizerFunc(func(_ context.Context, capability, token string) error {
		if capability == "tools" && token == "scoped-token" {
			return nil
		}
		return errors.New("denied")
	})
	host, child := hello("host", []string{"tools"}), hello("child", []string{"tools"})
	s := hs.NewAuthorizedServer(peer, peer, child, host, h, auth)
	go s.Serve(context.Background())
	c := hs.NewCodec(client, client, 4096)
	writeWire(t, c, map[string]any{"kind": "handshake", "handshake": host})
	var ack any
	if err := c.Read(&ack); err != nil {
		t.Fatal(err)
	}
	writeWire(t, c, map[string]any{"kind": "request", "request": hs.Request{ID: "denied", Method: "invoke", Capability: "tools"}})
	var denied struct {
		Kind     string       `json:"kind"`
		Response *hs.Response `json:"response"`
	}
	if err := c.Read(&denied); err != nil {
		t.Fatal(err)
	}
	if denied.Response == nil || denied.Response.Error != "capability denied" {
		t.Fatalf("bad denial: %+v", denied)
	}
	select {
	case <-called:
		t.Fatal("handler reached without token")
	default:
	}
	writeWire(t, c, map[string]any{"kind": "request", "request": hs.Request{ID: "allowed", Method: "invoke", Capability: "tools", CapabilityToken: "scoped-token"}})
	for {
		var got struct {
			Kind     string       `json:"kind"`
			Response *hs.Response `json:"response"`
		}
		if err := c.Read(&got); err != nil {
			t.Fatal(err)
		}
		if got.Response.Done {
			break
		}
	}
	select {
	case <-called:
	case <-time.After(time.Second):
		t.Fatal("authorized handler not reached")
	}
}

func TestPeerCrashFailsClosed(t *testing.T) {
	client, peer := net.Pipe()
	host, child := hello("host", []string{"stream"}), hello("child", []string{"stream"})
	s := hs.NewServer(peer, peer, child, host, hs.HandlerFunc(func(context.Context, string, json.RawMessage, func(json.RawMessage) error) error { return nil }))
	done := make(chan error, 1)
	go func() { done <- s.Serve(context.Background()) }()
	c := hs.NewCodec(client, client, 4096)
	writeWire(t, c, map[string]any{"kind": "handshake", "handshake": host})
	var ack any
	if err := c.Read(&ack); err != nil {
		t.Fatal(err)
	}
	client.Close()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("peer crash reported clean completion")
		}
	case <-time.After(time.Second):
		t.Fatal("server did not fail closed after peer crash")
	}
}
