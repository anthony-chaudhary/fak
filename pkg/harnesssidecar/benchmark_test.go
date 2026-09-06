package harnesssidecar

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"testing"
	"time"
)

var (
	benchSinkFrame  frame
	benchSinkLimits Limits
	benchSinkDigest string
)

func BenchmarkFrameEncodeDecode(b *testing.B) {
	sample := frame{
		Kind: "request",
		Request: &Request{
			ID:      "req-bench-1",
			Method:  "tool.invoke",
			Payload: json.RawMessage(`{"command":"echo","args":["hello","world"]}`),
		},
	}

	var wire bytes.Buffer
	c := NewCodec(&wire, &wire, 64*1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wire.Reset()
		c.r.Reset(&wire)
		if err := c.Write(sample); err != nil {
			b.Fatal(err)
		}
		var out frame
		if err := c.Read(&out); err != nil {
			b.Fatal(err)
		}
		benchSinkFrame = out
	}
}

func BenchmarkFrameEncode(b *testing.B) {
	sample := frame{
		Kind: "request",
		Request: &Request{
			ID:      "req-bench-1",
			Method:  "tool.invoke",
			Payload: json.RawMessage(`{"command":"echo","args":["hello","world"]}`),
		},
	}
	c := NewCodec(bytes.NewReader(nil), io.Discard, 64*1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.Write(sample); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFrameDecode(b *testing.B) {
	sample := frame{
		Kind: "request",
		Request: &Request{
			ID:      "req-bench-1",
			Method:  "tool.invoke",
			Payload: json.RawMessage(`{"command":"echo","args":["hello","world"]}`),
		},
	}
	var buf bytes.Buffer
	prep := NewCodec(bytes.NewReader(nil), &buf, 64*1024)
	if err := prep.Write(sample); err != nil {
		b.Fatal(err)
	}
	raw := buf.Bytes()
	r := bytes.NewReader(raw)
	c := NewCodec(r, io.Discard, 64*1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.Reset(raw)
		c.r.Reset(r)
		var out frame
		if err := c.Read(&out); err != nil {
			b.Fatal(err)
		}
		benchSinkFrame = out
	}
}

func BenchmarkLimitsValidation(b *testing.B) {
	cases := []Limits{
		{MaxFrame: 64 * 1024, MaxInflight: 16, CancelGrace: time.Second},
		{MaxFrame: 1024 * 1024, MaxInflight: 64, CancelGrace: 5 * time.Second},
		{MaxFrame: 4096, MaxInflight: 2, CancelGrace: 500 * time.Millisecond},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l := cases[i%len(cases)]
		if err := l.Validate(); err != nil {
			b.Fatal(err)
		}
		benchSinkLimits = l.normalized()
	}
}

func BenchmarkValidateHandshake(b *testing.B) {
	caps := []string{"stream", "tools", "cancel", "state"}
	want := Handshake{
		Protocol:     ProtocolVersion,
		Identity:     Identity{Name: "host", Version: "1.0", Digest: ContractDigest(caps)},
		Capabilities: caps,
		Limits:       Limits{MaxFrame: 64 * 1024, MaxInflight: 16, CancelGrace: time.Second},
	}
	got := Handshake{
		Protocol:     ProtocolVersion,
		Identity:     Identity{Name: "worker", Version: "1.0", Digest: ContractDigest(caps)},
		Capabilities: caps,
		Limits:       Limits{MaxFrame: 64 * 1024, MaxInflight: 16, CancelGrace: time.Second},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateHandshake(want, got); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkContractDigest(b *testing.B) {
	caps := []string{"stream", "tools", "cancel", "state", "memory", "auth"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkDigest = ContractDigest(caps)
	}
}

func TestLimitsValidate(t *testing.T) {
	valid := Limits{
		MaxFrame:    1024,
		MaxInflight: 4,
		CancelGrace: time.Second,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid limits, got: %v", err)
	}

	invalidFrames := valid
	invalidFrames.MaxFrame = 0
	if err := invalidFrames.Validate(); !errors.Is(err, ErrProtocol) {
		t.Fatalf("expected ErrProtocol for zero MaxFrame, got: %v", err)
	}

	invalidInflight := valid
	invalidInflight.MaxInflight = 0
	if err := invalidInflight.Validate(); !errors.Is(err, ErrProtocol) {
		t.Fatalf("expected ErrProtocol for zero MaxInflight, got: %v", err)
	}

	invalidGrace := valid
	invalidGrace.CancelGrace = 0
	if err := invalidGrace.Validate(); !errors.Is(err, ErrProtocol) {
		t.Fatalf("expected ErrProtocol for zero CancelGrace, got: %v", err)
	}
}

func TestBenchmarkOperationsSanity(t *testing.T) {
	sample := frame{
		Kind: "request",
		Request: &Request{
			ID:      "req-sanity",
			Method:  "tool.invoke",
			Payload: json.RawMessage(`{"command":"echo"}`),
		},
	}
	var wire bytes.Buffer
	c := NewCodec(&wire, &wire, 64*1024)
	if err := c.Write(sample); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	var out frame
	if err := c.Read(&out); err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if out.Request == nil || out.Request.ID != "req-sanity" {
		t.Fatalf("unexpected frame: %+v", out)
	}

	limits := Limits{MaxFrame: 1024, MaxInflight: 2, CancelGrace: time.Second}
	if err := limits.Validate(); err != nil {
		t.Fatalf("validate failed: %v", err)
	}
	norm := limits.normalized()
	if norm.MaxFrame != 1024 {
		t.Fatalf("unexpected normalized: %+v", norm)
	}

	caps := []string{"stream", "tools"}
	h := Handshake{
		Protocol:     ProtocolVersion,
		Identity:     Identity{Name: "worker", Version: "1.0", Digest: ContractDigest(caps)},
		Capabilities: caps,
		Limits:       limits,
	}
	if err := ValidateHandshake(h, h); err != nil {
		t.Fatalf("handshake validation failed: %v", err)
	}
}
