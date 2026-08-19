package agent

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

func TestApplyTraceContext(t *testing.T) {
	ctx := WithTraceContext(context.Background(), "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "vendor=value")
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://provider.invalid", nil)
	ApplyTraceContext(req)
	if req.Header.Get("traceparent") != "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01" || req.Header.Get("tracestate") != "vendor=value" {
		t.Fatalf("headers=%v", req.Header)
	}
}
func TestApplyTraceContextBoundsTracestate(t *testing.T) {
	ctx := WithTraceContext(context.Background(), "tp", strings.Repeat("x", 513))
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://provider.invalid", nil)
	ApplyTraceContext(req)
	if req.Header.Get("traceparent") != "tp" || req.Header.Get("tracestate") != "" {
		t.Fatalf("headers=%v", req.Header)
	}
}
