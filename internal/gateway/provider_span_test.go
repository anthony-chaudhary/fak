package gateway

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"net/http"
	"testing"
	"time"
)

func TestProviderSpanContextCreatesChild(t *testing.T) {
	e := &otlpExporter{queue: make(chan otlpSpan, 2)}
	ctx := agent.WithTraceContext(context.Background(), "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "")
	ctx = providerSpanContext(ctx, e)
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://provider.invalid/v1", nil)
	finish := agent.BeginProviderCall(req)
	child, err := parseTraceparent(req.Header.Get(traceparentHeader))
	if err != nil {
		t.Fatal(err)
	}
	if child.TraceID != "4bf92f3577b34da6a3ce929d0e0e4736" || child.ParentID == "00f067aa0ba902b7" {
		t.Fatalf("child=%+v", child)
	}
	finish(200, nil)
	select {
	case span := <-e.queue:
		if span.ParentSpanID != "00f067aa0ba902b7" || span.SpanID != child.ParentID || span.TraceID != child.TraceID {
			t.Fatalf("span=%+v", span)
		}
	case <-time.After(time.Second):
		t.Fatal("no child span")
	}
}
