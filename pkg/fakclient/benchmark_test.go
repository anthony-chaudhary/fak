package fakclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"
)

var (
	benchSinkBytes    []byte
	benchSinkResponse SyscallResponse
	benchSinkAPIError *APIError
	benchSinkChanges  ChangesResponse
	benchSinkRuntime  RuntimeWireEvent
)

func BenchmarkClientRequestEncoding(b *testing.B) {
	reqMinimal := SyscallRequest{
		Tool:      "read_file",
		Arguments: json.RawMessage(`{"path":"internal/kernel/kernel.go"}`),
	}
	reqFull := SyscallRequest{
		Tool:      "write_file",
		Arguments: json.RawMessage(`{"path":"internal/kernel/kernel.go","content":"package kernel"}`),
		ReadOnly:  false,
		Witness:   "sha256:4a8b7921c3d1f05e192a837c92b3a4f102938475",
		TraceID:   "t-9f8e7d6c-5b4a-3210-fedc-ba9876543210",
		Principal: "tenant-enterprise-cluster-alpha",
		Preferences: Preference{
			RequireWitness:     true,
			WitnessRoute:       "cas-strict",
			WaitMode:           "bounded-10s",
			TransformMode:      "canonical",
			Disclosure:         "redacted",
			Timeout:            "30s",
			ResumeNotification: "webhook",
		},
	}
	reqAdmit := AdmitRequest{
		Tool:    "fetch_url",
		Result:  json.RawMessage(`{"status":200,"body":"sanitized response content"}`),
		Witness: "sha256:b5c891e4f3a7",
		TraceID: "t-trace-admit-01",
	}

	b.Run("Minimal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			data, err := json.Marshal(reqMinimal)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkBytes = data
		}
	})

	b.Run("Full", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			data, err := json.Marshal(reqFull)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkBytes = data
		}
	})

	b.Run("Admit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			data, err := json.Marshal(reqAdmit)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkBytes = data
		}
	})
}

func BenchmarkParseVerdict(b *testing.B) {
	allowJSON := []byte(`{"verdict":{"kind":"ALLOW","by":"policy-fastpath"},"result":{"status":"OK","content":"{\"success\":true}"},"trace_id":"t-7f3a9c"}`)
	denyJSON := []byte(`{"verdict":{"kind":"DENY","reason":"SELF_MODIFY","by":"selfmod_guard","disposition":"ESCALATE","detail":{"claim":"fak/internal/kernel/kernel.go","caller":"agent-42","boundary":"l0"}},"trace_id":"t-7f3a9c"}`)
	transformJSON := []byte(`{"verdict":{"kind":"TRANSFORM","reason":"PARAM_NORMALIZED","by":"sanitizer","disposition":"RETRYABLE"},"repaired_arguments":{"path":"internal/kernel/kernel.go","mode":"ro"},"trace_id":"t-7f3a9c"}`)
	quarantineJSON := []byte(`{"verdict":{"kind":"QUARANTINE","reason":"SECRET_EXFIL","by":"egress_filter","disposition":"TERMINAL"},"result":{"status":"OK","content":"","meta":{"admit":"quarantined","page_out":"blob:9f8a"}},"trace_id":"t-7f3a9c"}`)

	b.Run("Allow", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(allowJSON)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var resp SyscallResponse
			if err := json.Unmarshal(allowJSON, &resp); err != nil {
				b.Fatal(err)
			}
			benchSinkResponse = resp
		}
	})

	b.Run("Deny", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(denyJSON)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var resp SyscallResponse
			if err := json.Unmarshal(denyJSON, &resp); err != nil {
				b.Fatal(err)
			}
			benchSinkResponse = resp
		}
	})

	b.Run("Transform", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(transformJSON)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var resp SyscallResponse
			if err := json.Unmarshal(transformJSON, &resp); err != nil {
				b.Fatal(err)
			}
			benchSinkResponse = resp
		}
	})

	b.Run("Quarantine", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(quarantineJSON)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var resp SyscallResponse
			if err := json.Unmarshal(quarantineJSON, &resp); err != nil {
				b.Fatal(err)
			}
			benchSinkResponse = resp
		}
	})
}

func BenchmarkParseAPIError(b *testing.B) {
	structuredJSON := []byte(`{"error":{"message":"malformed request body: unexpected EOF","type":"invalid_request_error","code":"request_malformed","param":"arguments"}}`)
	plainText := []byte("upstream gateway timeout after 30000ms")

	b.Run("Structured", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(structuredJSON)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			err := parseAPIError(http.StatusBadRequest, structuredJSON)
			benchSinkAPIError = err
		}
	})

	b.Run("PlainText", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(plainText)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			err := parseAPIError(http.StatusGatewayTimeout, plainText)
			benchSinkAPIError = err
		}
	})
}

func BenchmarkChangesResponseDecoding(b *testing.B) {
	for _, count := range []int{1, 10, 50} {
		events := make([]ChangeEvent, count)
		for j := 0; j < count; j++ {
			events[j] = ChangeEvent{
				Kind:       "mutation",
				Seq:        uint64(j + 1),
				Tool:       "write_file",
				Tags:       []string{"fs:/srv/data", "tenant:default"},
				Witness:    "sha256:feedbeef010203",
				WorldVer:   uint64(100 + j),
				TrustEpoch: 42,
			}
		}
		raw, err := json.Marshal(ChangesResponse{Events: events, Cursor: uint64(count)})
		if err != nil {
			b.Fatal(err)
		}

		b.Run(fmt.Sprintf("%d_Events", count), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var resp ChangesResponse
				if err := json.Unmarshal(raw, &resp); err != nil {
					b.Fatal(err)
				}
				benchSinkChanges = resp
			}
		})
	}
}

func BenchmarkDecodeRuntimeWireEvent(b *testing.B) {
	wireEvent := RuntimeWireEvent{
		Schema: RuntimeEventWireSchema,
		Admission: RuntimeAdmission{
			Screened: true,
			Taint:    "clean",
			Screen:   "ctxmmu/1",
		},
		Event: RuntimeEvent{
			Schema:    RuntimeEventSchema,
			EventID:   "evt-001",
			TraceID:   "trc-001",
			Sequence:  1,
			Timestamp: time.Unix(1700000000, 0),
			Kind:      RuntimeKindTurnStarted,
			Payload:   json.RawMessage(`{"task":"audit"}`),
		},
	}
	ndjson, err := json.Marshal(wireEvent)
	if err != nil {
		b.Fatal(err)
	}
	sse := []byte("event: runtime\ndata: " + string(ndjson) + "\n\n")

	b.Run("NDJSON", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(ndjson)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ev, err := DecodeRuntimeNDJSON(ndjson)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkRuntime = ev
		}
	})

	b.Run("SSE", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(sse)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ev, err := DecodeRuntimeSSE(sse)
			if err != nil {
				b.Fatal(err)
			}
			benchSinkRuntime = ev
		}
	})
}

type staticChangesRoundTripper struct {
	payload []byte
}

func (s *staticChangesRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(s.payload)),
	}, nil
}

func BenchmarkConsumerDrain(b *testing.B) {
	const batchSize = 25
	events := make([]ChangeEvent, batchSize)
	for j := 0; j < batchSize; j++ {
		events[j] = ChangeEvent{
			Kind:       "mutation",
			Seq:        uint64(j + 1),
			Tool:       "write_file",
			Tags:       []string{"fs:/srv"},
			WorldVer:   1,
			TrustEpoch: 1,
		}
	}
	payload, _ := json.Marshal(ChangesResponse{Events: events, Cursor: batchSize})

	client := New("http://fak.internal", WithHTTPClient(&http.Client{
		Transport: &staticChangesRoundTripper{payload: payload},
	}))
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cs := NewConsumer(client)
		delivered, err := cs.Drain(ctx, func(_ context.Context, _ ChangeEvent) error {
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		if delivered != batchSize {
			b.Fatalf("delivered = %d, want %d", delivered, batchSize)
		}
	}
}

type staticResponseRoundTripper struct {
	payload []byte
}

func (s *staticResponseRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(s.payload)),
	}, nil
}

func BenchmarkClientAdjudicateRoundTrip(b *testing.B) {
	responseBody := []byte(`{"verdict":{"kind":"ALLOW","by":"fastpath"},"trace_id":"t-bench-01"}`)
	client := New("http://fak.internal",
		WithAPIKey("bench-key"),
		WithPrincipal("tenant-bench"),
		WithHTTPClient(&http.Client{
			Transport: &staticResponseRoundTripper{payload: responseBody},
		}),
	)
	req := SyscallRequest{
		Tool:      "read_file",
		Arguments: json.RawMessage(`{"path":"internal/kernel/kernel.go"}`),
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := client.Adjudicate(ctx, req)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkResponse = *resp
	}
}
