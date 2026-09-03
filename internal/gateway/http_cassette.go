package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// HTTPRecordedChunk represents one streamed chunk captured at the wire boundary.
type HTTPRecordedChunk struct {
	Index      int           `json:"index"`
	Data       string        `json:"data"`
	RelativeMs time.Duration `json:"relative_ms"`
}

// HTTPTransactionCassette stores a complete HTTP transaction for deterministic replay.
type HTTPTransactionCassette struct {
	TraceID              string              `json:"trace_id"`
	RequestMethod        string              `json:"request_method"`
	RequestPath          string              `json:"request_path"`
	RequestHeaders       map[string][]string `json:"request_headers"`
	RequestBody          string              `json:"request_body"`
	ResponseStatus       int                 `json:"response_status"`
	ResponseHeaders      map[string][]string `json:"response_headers"`
	ResponseChunks       []HTTPRecordedChunk `json:"response_chunks"`
	TotalDurationMs      time.Duration       `json:"total_duration_ms"`
	TimingClassification string              `json:"timing_classification"` // strictly "observed"
	EffectsDisabled      bool                `json:"effects_disabled"`
}

// RecordStreamingHTTPTransaction captures an HTTP request and its streaming response chunks into a cassette.
func RecordStreamingHTTPTransaction(
	traceID string,
	req *http.Request,
	body []byte,
	status int,
	respHeaders http.Header,
	chunks []string,
	chunkDelays []time.Duration,
) (HTTPTransactionCassette, error) {
	if req == nil {
		return HTTPTransactionCassette{}, fmt.Errorf("request must not be nil")
	}

	recordedChunks := make([]HTTPRecordedChunk, len(chunks))
	var elapsed time.Duration
	for i, c := range chunks {
		var delay time.Duration
		if i < len(chunkDelays) {
			delay = chunkDelays[i]
		}
		elapsed += delay
		recordedChunks[i] = HTTPRecordedChunk{
			Index:      i,
			Data:       c,
			RelativeMs: elapsed,
		}
	}

	headersCopy := make(map[string][]string, len(req.Header))
	for k, v := range req.Header {
		headersCopy[k] = append([]string(nil), v...)
	}

	respHeadersCopy := make(map[string][]string, len(respHeaders))
	for k, v := range respHeaders {
		respHeadersCopy[k] = append([]string(nil), v...)
	}

	return HTTPTransactionCassette{
		TraceID:              traceID,
		RequestMethod:        req.Method,
		RequestPath:          req.URL.Path,
		RequestHeaders:       headersCopy,
		RequestBody:          string(body),
		ResponseStatus:       status,
		ResponseHeaders:      respHeadersCopy,
		ResponseChunks:       recordedChunks,
		TotalDurationMs:      elapsed,
		TimingClassification: "observed",
		EffectsDisabled:      true,
	}, nil
}

// ReplayHTTPCassette streams the cassette back through http.ResponseWriter byte-for-byte
// with effects disabled and timing explicitly reported as observed.
func ReplayHTTPCassette(w http.ResponseWriter, cassette HTTPTransactionCassette) error {
	if w == nil {
		return fmt.Errorf("response writer must not be nil")
	}

	// Set replayed response headers
	for k, vals := range cassette.ResponseHeaders {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.Header().Set("X-Fak-Cassette-Replay", "true")
	w.Header().Set("X-Fak-Effects-Disabled", "true")
	w.Header().Set("X-Fak-Timing-Class", cassette.TimingClassification)

	w.WriteHeader(cassette.ResponseStatus)

	flusher, canFlush := w.(http.Flusher)

	// Stream each recorded chunk byte-for-byte in exact order
	for _, chunk := range cassette.ResponseChunks {
		if _, err := ioWriteString(w, chunk.Data); err != nil {
			return fmt.Errorf("replay write chunk %d: %w", chunk.Index, err)
		}
		if canFlush {
			flusher.Flush()
		}
	}

	return nil
}

func ioWriteString(w http.ResponseWriter, s string) (int, error) {
	return w.Write([]byte(s))
}

// SerializeCassette encodes the cassette into JSONL format.
func (c HTTPTransactionCassette) Serialize() ([]byte, error) {
	return json.Marshal(c)
}

// DeserializeCassette parses a cassette from JSON bytes.
func DeserializeCassette(b []byte) (HTTPTransactionCassette, error) {
	var c HTTPTransactionCassette
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	err := dec.Decode(&c)
	return c, err
}
