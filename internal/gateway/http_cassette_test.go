package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHTTPTransactionCassetteWitness(t *testing.T) {
	// First witness requirements (#9910):
	// 1. Capture one full HTTP streaming request (method, path, headers, body).
	// 2. Stream response chunks with observed relative timings.
	// 3. Replay byte-for-byte through ResponseWriter.
	// 4. Effects explicitly disabled during replay.
	// 5. Timing explicitly classified as "observed".

	traceID := "trace-cassette-test-12345"
	reqBody := `{"model":"qwen3.8","messages":[{"role":"user","content":"Hello cassette"}],"stream":true}`
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer fake-token")

	respHeaders := make(http.Header)
	respHeaders.Set("Content-Type", "text/event-stream")
	respHeaders.Set("Cache-Control", "no-cache")

	streamChunks := []string{
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
		"data: [DONE]\n\n",
	}

	delays := []time.Duration{
		5 * time.Millisecond,
		10 * time.Millisecond,
		2 * time.Millisecond,
	}

	// 1 & 2. Record full HTTP transaction cassette
	cassette, err := RecordStreamingHTTPTransaction(
		traceID,
		httpReq,
		[]byte(reqBody),
		http.StatusOK,
		respHeaders,
		streamChunks,
		delays,
	)
	if err != nil {
		t.Fatalf("RecordStreamingHTTPTransaction failed: %v", err)
	}

	// 4 & 5. Verify cassette metadata
	if !cassette.EffectsDisabled {
		t.Fatal("expected EffectsDisabled = true")
	}
	if cassette.TimingClassification != "observed" {
		t.Fatalf("expected timing classification 'observed', got %s", cassette.TimingClassification)
	}
	if cassette.ResponseStatus != http.StatusOK {
		t.Fatalf("expected status 200, got %d", cassette.ResponseStatus)
	}
	if len(cassette.ResponseChunks) != 3 {
		t.Fatalf("expected 3 recorded chunks, got %d", len(cassette.ResponseChunks))
	}

	// Round-trip serialize / deserialize verification
	serialized, err := cassette.Serialize()
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}
	deserialized, err := DeserializeCassette(serialized)
	if err != nil {
		t.Fatalf("DeserializeCassette failed: %v", err)
	}
	if deserialized.TraceID != traceID || deserialized.TimingClassification != "observed" {
		t.Fatalf("deserialization mismatch: %+v", deserialized)
	}

	// 3. Replay byte-for-byte into ResponseWriter
	rec := httptest.NewRecorder()
	if err := ReplayHTTPCassette(rec, deserialized); err != nil {
		t.Fatalf("ReplayHTTPCassette failed: %v", err)
	}

	// Verify headers in replayed response
	if rec.Header().Get("X-Fak-Cassette-Replay") != "true" {
		t.Fatal("expected X-Fak-Cassette-Replay header")
	}
	if rec.Header().Get("X-Fak-Effects-Disabled") != "true" {
		t.Fatal("expected X-Fak-Effects-Disabled header")
	}
	if rec.Header().Get("X-Fak-Timing-Class") != "observed" {
		t.Fatalf("expected timing header 'observed', got %s", rec.Header().Get("X-Fak-Timing-Class"))
	}
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected Content-Type text/event-stream, got %s", rec.Header().Get("Content-Type"))
	}

	// Verify exact byte-for-byte replayed payload
	var wantPayload bytes.Buffer
	for _, c := range streamChunks {
		wantPayload.WriteString(c)
	}

	gotPayload := rec.Body.Bytes()
	if !bytes.Equal(gotPayload, wantPayload.Bytes()) {
		t.Fatalf("replayed payload mismatch:\ngot  =%q\nwant =%q", string(gotPayload), wantPayload.String())
	}
}
