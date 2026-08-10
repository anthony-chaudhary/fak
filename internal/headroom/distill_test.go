package headroom

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

func goTestFixture() []byte {
	var b strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&b, "=== RUN   TestPass%03d\n--- PASS: TestPass%03d (0.00s)\n", i, i)
	}
	b.WriteString("=== RUN   TestAlpha\n--- FAIL: TestAlpha (0.01s)\n    alpha_test.go:17: got 4, want 5\n")
	b.WriteString("=== RUN   TestBeta\n--- FAIL: TestBeta (0.02s)\n    beta_test.go:29: panic: sentinel failure\n")
	b.WriteString("FAIL\nFAIL\texample.invalid/pkg\t0.031s\nok  \texample.invalid/other\t0.004s\n")
	return []byte(b.String())
}

func TestDistillGoTestKeepsFailuresAndDropsRoutinePasses(t *testing.T) {
	orig := goTestFixture()
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "Bash", Kind: KindText, Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.Codec != "go-test-distill" || len(out.Bytes) >= len(orig) {
		t.Fatalf("output = %#v", out)
	}
	got := string(out.Bytes)
	for _, required := range []string{"--- FAIL: TestAlpha", "alpha_test.go:17", "--- FAIL: TestBeta", "panic: sentinel failure", "\nFAIL\n", "FAIL\texample.invalid/pkg", "ok  \texample.invalid/other", "402 routine go test line(s) dropped"} {
		if !strings.Contains(got, required) {
			t.Errorf("missing %q in:\n%s", required, got)
		}
	}
	if strings.Contains(got, "--- PASS:") || strings.Contains(got, "=== RUN") {
		t.Fatalf("routine pass records survived:\n%s", got)
	}
}

func TestDistillGoTestNeverDropsErrorShapedUnknownLines(t *testing.T) {
	orig := []byte("--- PASS: TestOne (0.00s)\ncompiler: ERROR unusual diagnostic\n--- FAIL: TestTwo (0.01s)\n    continuation detail\nFAIL\n")
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "shell", Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, keep := range [][]byte{[]byte("ERROR unusual diagnostic"), []byte("--- FAIL: TestTwo"), []byte("continuation detail"), []byte("FAIL")} {
		if !bytes.Contains(out.Bytes, keep) {
			t.Errorf("dropped error evidence %q: %s", keep, out.Bytes)
		}
	}
}

func TestDistillRegistryDispatchesByToolAndKind(t *testing.T) {
	orig := goTestFixture()
	tests := []struct {
		name string
		in   Input
		want bool
	}{
		{name: "text shell matches", in: Input{Tool: "shell", Kind: KindText, Bytes: orig}, want: true},
		{name: "unknown kind is sniffed", in: Input{Tool: "Bash", Kind: KindUnknown, Bytes: orig}, want: true},
		{name: "wrong tool", in: Input{Tool: "Read", Kind: KindText, Bytes: orig}},
		{name: "wrong kind", in: Input{Tool: "Bash", Kind: KindJSON, Bytes: orig}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, got := applyDistillFilter(tt.in)
			if got != tt.want {
				t.Fatalf("matched = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestDistillFallsBackToNativeWhenNoToolFilterMatches(t *testing.T) {
	orig := []byte(strings.Repeat("same line repeated for native fallback\n", 20))
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "Read", Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || !strings.Contains(out.Codec, "line-dedup") {
		t.Fatalf("native fallback did not run: %#v", out)
	}
}

func TestDistillNeverWorse(t *testing.T) {
	orig := []byte("--- PASS: X (0.00s)\nFAIL\n")
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "go test", Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if out.Compressed && len(out.Bytes) >= len(orig) {
		t.Fatalf("compressed output grew: %d >= %d", len(out.Bytes), len(orig))
	}
}

func TestDistillAdmissionPreservesOriginalInCAS(t *testing.T) {
	orig := goTestFixture()
	if !Select(DistillName) {
		t.Fatal("distill compressor not registered")
	}
	t.Cleanup(func() { Select(NoopName) })

	call := &abi.ToolCall{Tool: "Bash"}
	result := &abi.Result{Call: call, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}}
	verdict := NewGate().Admit(context.Background(), call, result)
	if verdict.Kind != abi.VerdictTransform {
		t.Fatalf("verdict = %#v", verdict)
	}
	origin := verdict.Meta["origin"]
	if origin == "" {
		t.Fatalf("missing origin handle: %#v", verdict.Meta)
	}
	restored, err := blob.Default.PageIn(context.Background(), abi.Ref{Kind: abi.RefBlob, Digest: origin})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(restored.Inline, orig) {
		t.Fatal("origin did not restore the original byte-for-byte")
	}
	payload, ok := verdict.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload = %#v", verdict.Payload)
	}
	if !bytes.Contains(payload.NewArgs.Inline, []byte(`fak_context_restore {"origin":"`+origin+`"}`)) {
		t.Fatalf("inline hint omitted exact restore handle %q: %s", origin, payload.NewArgs.Inline)
	}
	if bytes.Contains(payload.NewArgs.Inline, []byte("--- PASS:")) || !bytes.Contains(payload.NewArgs.Inline, []byte("--- FAIL:")) {
		t.Fatalf("admitted rendering did not filter passes and preserve failures: %s", payload.NewArgs.Inline)
	}
}
