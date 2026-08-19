package agent

import (
	"context"
	"net/http"
	"strings"
)

type outboundTraceContext struct{ traceparent, tracestate string }
type outboundTraceContextKey struct{}

// WithTraceContext carries validated W3C context from an ingress adapter to provider calls.
func WithTraceContext(ctx context.Context, traceparent, tracestate string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, outboundTraceContextKey{}, outboundTraceContext{traceparent: strings.TrimSpace(traceparent), tracestate: strings.TrimSpace(tracestate)})
}

// ApplyTraceContext stamps provider requests from the validated context carrier.
func ApplyTraceContext(req *http.Request) {
	if req == nil {
		return
	}
	c, ok := req.Context().Value(outboundTraceContextKey{}).(outboundTraceContext)
	if !ok {
		return
	}
	if c.traceparent != "" {
		req.Header.Set("traceparent", c.traceparent)
	}
	if c.tracestate != "" && len(c.tracestate) <= 512 {
		req.Header.Set("tracestate", c.tracestate)
	}
}
