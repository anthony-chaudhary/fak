package gateway

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

type receiptRoundTripper func(*http.Request) (*http.Response, error)

func (f receiptRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type truncatedBody struct{ sent bool }

func (b *truncatedBody) Read(p []byte) (int, error) {
	if !b.sent {
		b.sent = true
		copy(p, "partial")
		return 7, nil
	}
	return 0, io.ErrUnexpectedEOF
}
func (*truncatedBody) Close() error { return nil }

func TestUpstreamFailureReceiptsDistinguishOrigins(t *testing.T) {
	tests := []struct {
		name      string
		rt        receiptRoundTripper
		layer     string
		status    int
		cause     string
		requestID string
	}{
		{"provider 502", func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 502, Header: http.Header{"X-Openai-Request-Id": {"prov-1"}}, Body: io.NopCloser(strings.NewReader("secret body"))}, nil
		}, "provider", 502, "", "prov-1"},
		{"unknown 502", func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 502, Header: http.Header{"Authorization": {"secret"}}, Body: io.NopCloser(strings.NewReader("secret body"))}, nil
		}, "unknown", 502, "", ""},
		{"local proxy 502", func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 502, Header: http.Header{"Via": {"1.1 local-envoy"}, "X-Proxy-Request-Id": {"proxy-1"}, "Set-Cookie": {"secret"}}, Body: io.NopCloser(strings.NewReader("secret body"))}, nil
		}, "local_proxy", 502, "", "proxy-1"},
		{"transport", func(*http.Request) (*http.Response, error) { return nil, io.ErrUnexpectedEOF }, "transport", 0, "unexpected EOF", ""},
		{"truncated 200", func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: &truncatedBody{}}, nil
		}, "transport", 200, "unexpected EOF", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []UpstreamFailureReceipt
			client := &http.Client{Transport: tt.rt}
			wrapUpstreamObserver(client, nil, nil, func(r UpstreamFailureReceipt) { got = append(got, r) })
			req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.example.test/v1/chat/completions?api_key=secret", nil)
			req.Header.Set("Authorization", "Bearer secret")
			req.Header.Set("X-Fak-Session-Id", "sess-1")
			req.Header.Set("X-Fak-Call-Id", "call-1")
			req.Header.Set("Traceparent", "00-trace-span-01")
			resp, _ := client.Do(req)
			if resp != nil {
				_, _ = io.ReadAll(resp.Body)
				_ = resp.Body.Close()
			}
			if len(got) != 1 {
				t.Fatalf("receipts=%d want 1", len(got))
			}
			r := got[0]
			if r.EmittingLayer != tt.layer || r.HTTPStatus != tt.status {
				t.Fatalf("origin/status=%s/%d want %s/%d", r.EmittingLayer, r.HTTPStatus, tt.layer, tt.status)
			}
			if tt.cause != "" && !strings.Contains(r.Cause, tt.cause) {
				t.Fatalf("cause=%q", r.Cause)
			}
			id := r.ProviderRequestID
			if tt.layer == "local_proxy" {
				id = r.ProxyRequestID
			}
			if id != tt.requestID {
				t.Fatalf("request id=%q want %q", id, tt.requestID)
			}
			if r.TargetID != "api.example.test" || strings.Contains(r.PathClass, "secret") || r.SessionID != "sess-1" || r.CallID != "call-1" {
				t.Fatalf("sanitization/correlation: %+v", r)
			}
			if _, ok := r.Diagnostic["Authorization"]; ok {
				t.Fatal("authorization leaked")
			}
			if _, ok := r.Diagnostic["Set-Cookie"]; ok {
				t.Fatal("cookie leaked")
			}
		})
	}
}

func TestUpstreamFailureReceiptAttemptAndRecoveryFields(t *testing.T) {
	var got []UpstreamFailureReceipt
	n := 0
	client := &http.Client{Transport: receiptRoundTripper(func(*http.Request) (*http.Response, error) {
		n++
		if n == 1 {
			return nil, io.ErrUnexpectedEOF
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("ok"))}, nil
	})}
	wrapUpstreamObserver(client, nil, nil, func(r UpstreamFailureReceipt) { got = append(got, r) })
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.example.test/v1/chat", nil)
		resp, _ := client.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
	}
	if len(got) != 1 || got[0].Attempt != 1 || got[0].RetryBudget < 2 || got[0].RetryDisposition != "eligible" {
		t.Fatalf("receipt=%+v", got)
	}
}
