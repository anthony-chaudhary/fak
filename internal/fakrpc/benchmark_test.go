package fakrpc

import (
	"bytes"
	"testing"
)

// BenchmarkRPCOperation measures end-to-end client/server RPC dispatch: client request
// encoding, server-side request decoding, worker execution, response frame encoding,
// and client-side frame decoding with sha256 verification.
func BenchmarkRPCOperation(b *testing.B) {
	req := Request{
		Nonce:   "bench-nonce-001",
		Kind:    KindDevTurn,
		Model:   "glm-5.2",
		Payload: "run benchmark validation suite",
		Params:  map[string]string{"max_tokens": "512", "temperature": "0.2"},
	}
	respPayload := []byte(`{"schema":"glm-gpu-witness/1","ok":true,"tokens":512}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reqBytes, err := req.Encode()
		if err != nil {
			b.Fatalf("Encode: %v", err)
		}
		decodedReq, err := DecodeRequest(reqBytes)
		if err != nil {
			b.Fatalf("DecodeRequest: %v", err)
		}
		frameBytes := EncodeFrame(decodedReq.Nonce, 0, respPayload)
		frame, err := DecodeFrame(frameBytes)
		if err != nil {
			b.Fatalf("DecodeFrame: %v", err)
		}
		if !frame.OK() || frame.Nonce != req.Nonce {
			b.Fatalf("unexpected frame outcome: %+v", frame)
		}
	}
}

// BenchmarkRequestEncode measures serialization of a Request into single-line JSON.
func BenchmarkRequestEncode(b *testing.B) {
	req := Request{
		Nonce:   "bench-nonce-001",
		Kind:    KindDevTurn,
		Model:   "glm-5.2",
		Payload: "run benchmark validation suite",
		Params:  map[string]string{"max_tokens": "512"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := req.Encode(); err != nil {
			b.Fatalf("Encode: %v", err)
		}
	}
}

// BenchmarkRequestDecode measures parsing and validation of a spooled JSON request line.
func BenchmarkRequestDecode(b *testing.B) {
	req := Request{
		Nonce:   "bench-nonce-001",
		Kind:    KindDevTurn,
		Model:   "glm-5.2",
		Payload: "run benchmark validation suite",
		Params:  map[string]string{"max_tokens": "512"},
	}
	raw, err := req.Encode()
	if err != nil {
		b.Fatalf("setup Encode: %v", err)
	}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeRequest(raw); err != nil {
			b.Fatalf("DecodeRequest: %v", err)
		}
	}
}

// BenchmarkFrameEncode measures FAKRES framing, including sha256 checksum generation.
func BenchmarkFrameEncode(b *testing.B) {
	body := bytes.Repeat([]byte("a"), 4096)
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EncodeFrame("bench-nonce", 0, body)
	}
}

// BenchmarkDecodeFrame measures parsing, length verification, and sha256 integrity checks.
func BenchmarkDecodeFrame(b *testing.B) {
	body := bytes.Repeat([]byte("a"), 4096)
	framed := EncodeFrame("bench-nonce", 0, body)
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frame, err := DecodeFrame(framed)
		if err != nil {
			b.Fatalf("DecodeFrame: %v", err)
		}
		if !frame.OK() {
			b.Fatalf("expected frame.OK() to be true")
		}
	}
}
