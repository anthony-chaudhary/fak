package faultlab

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
)

// BenchmarkFaultLab exercises fault injection and stream interception in a loop.
func BenchmarkFaultLab(b *testing.B) {
	fi := NewFaultInjector(WithSeed(42))
	rule := NewFaultRule("bench_rule", Truncation, "agent/stream/*")
	rule.TruncateBytes = 32
	if err := fi.RegisterRule(rule); err != nil {
		b.Fatalf("register rule: %v", err)
	}

	ctx := context.Background()
	payload := []byte(strings.Repeat("0123456789abcdef", 16))

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = fi.Inject(ctx, "agent/stream/call", payload)
		r := bytes.NewReader(payload)
		ir := fi.InterceptReader(ctx, "agent/stream/call", r)
		_, _ = io.ReadAll(ir)
	}
}
