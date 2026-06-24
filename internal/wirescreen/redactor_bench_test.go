package wirescreen

import (
	"context"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob" // CAS backend so Apply's witness pin is real
)

// redactor_bench_test.go measures the end-to-end pre-send redaction latency of the
// deterministic floor — the rung-5 "measure before you default it on" gate (issue #572).
// The concern the issue names is the TTFB a redactor adds before bytes leave the box;
// these two benchmarks bound it from both ends: Propose is the pure classify (the regex +
// Luhn scan, the direct analogue of the gated model arm's NER classify that is unmeasured
// until weights land), and Apply is the full pre-send cost (classify + placeholder rewrite
// + the CAS witness pin the quarantine path shares).
//
// The body below carries every pattern shape plus ordinary prose a compliance buyer would
// not want scrubbed but the floor must still scan end-to-end.

var benchBody = []byte("reach alice@example.com and charge 4111 1111 1111 1111, ssn 123-45-6789, " +
	"aws AKIAIOSFODNN7EXAMPLE, github ghp_012345678901234567890123456789012345, " +
	"slack xoxb-1234567890-abcdefghij, stripe sk_live_0123456789abcdef0123456789abcdef, " +
	"google AIzaSyA0123456789012345678901234567890a, bearer abcdefghijklmnopqrstuvwx, " +
	"plus a run of ordinary prose that a compliance buyer would not want scrubbed but the " +
	"floor must scan anyway to reach the secrets near the end of the body")

// BenchmarkPropose measures the classify-only cost (the regex + Luhn span scan), the
// direct analogue of the model NER classify the gated follow-on would add.
func BenchmarkPropose(b *testing.B) {
	r := piiRedactor{}
	ctx := context.Background()
	b.SetBytes(int64(len(benchBody)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.Propose(ctx, benchBody, "user")
	}
}

// BenchmarkApply measures the full end-to-end pre-send cost: classify + placeholder
// rewrite + the CAS witness pin (pageOut + PinResolved) that holds the original for a
// byte-exact Restore. This is the number the redactor adds to TTFB on the live wire.
func BenchmarkApply(b *testing.B) {
	r := piiRedactor{}
	ctx := context.Background()
	b.SetBytes(int64(len(benchBody)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Apply(ctx, r, benchBody, "user")
	}
}
