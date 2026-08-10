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

func golangCIFixture() []byte {
	var b strings.Builder
	b.WriteString("golangci-lint has version 2.1.0\n")
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&b, "level=info msg=\"analyzing package %03d\"\n", i)
	}
	b.WriteString("pkg/alpha.go:17:4: unchecked return value (errcheck)\n")
	b.WriteString("    value := call()\n    ^ return value ignored\n")
	b.WriteString("level=warning msg=\"runner exceeded soft budget\"\n")
	b.WriteString("pkg/beta.go:29:2: shadowed variable (govet)\n")
	b.WriteString("\tbeta := beta\n\t^ shadow\n")
	b.WriteString("level=error msg=\"issues found\"\n2 issues:\n")
	return []byte(b.String())
}

func TestDistillGolangCILintKeepsDiagnosticsAndDropsInfo(t *testing.T) {
	orig := golangCIFixture()
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "Bash", Kind: KindText, Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.Codec != "golangci-lint-distill" || len(out.Bytes) >= len(orig) {
		t.Fatalf("output = %#v", out)
	}
	got := string(out.Bytes)
	for _, want := range []string{"alpha.go:17:4", "^ return value ignored", "level=warning", "beta.go:29:2", "^ shadow", "level=error", "2 issues:", "200 routine golangci-lint line(s) dropped"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, "level=info") {
		t.Fatal("routine info survived")
	}
}

func TestDistillGolangCILintDispatchRequiresToolKindAndVocabulary(t *testing.T) {
	orig := golangCIFixture()
	for _, tt := range []struct {
		name string
		in   Input
		want bool
	}{
		{name: "shell text", in: Input{Tool: "shell", Kind: KindText, Bytes: orig}, want: true},
		{name: "named tool unknown kind", in: Input{Tool: "golangci-lint", Kind: KindUnknown, Bytes: []byte("golangci-lint\nlevel=info msg=ready\nlevel=info msg=done\n")}, want: true},
		{name: "wrong tool", in: Input{Tool: "Read", Kind: KindText, Bytes: orig}},
		{name: "wrong kind", in: Input{Tool: "Bash", Kind: KindJSON, Bytes: orig}},
		{name: "ambiguous shell text", in: Input{Tool: "Bash", Kind: KindText, Bytes: []byte("ordinary output")}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, got := applyDistillFilter(tt.in)
			if got != tt.want {
				t.Fatalf("matched=%v want=%v", got, tt.want)
			}
		})
	}
}

func TestDistillGolangCILintMutationPreservesFailureEvidence(t *testing.T) {
	base := golangCIFixture()
	mutations := [][]byte{
		[]byte("pkg/new.go:3:9: injected failure (staticcheck)\n    continuation\n"),
		[]byte("level=warning msg=\"injected warning\"\n"),
		[]byte("level=error msg=\"injected error\"\n"),
		[]byte("UNKNOWN sentinel must remain\n"),
	}
	for _, mutation := range mutations {
		input := append(append([]byte(nil), base...), mutation...)
		out, _, ok := applyGolangCILintFilter(input)
		if !ok || !bytes.Contains(out, bytes.TrimSpace(mutation)) {
			t.Fatalf("mutation lost: %q\n%s", mutation, out)
		}
	}
}

func TestDistillGolangCILintAdmissionRestoresOriginal(t *testing.T) {
	orig := golangCIFixture()
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
	restored, err := blob.Default.PageIn(context.Background(), abi.Ref{Kind: abi.RefBlob, Digest: origin})
	if err != nil || !bytes.Equal(restored.Inline, orig) {
		t.Fatalf("restore err=%v equal=%v", err, bytes.Equal(restored.Inline, orig))
	}
	payload, ok := verdict.Payload.(abi.TransformPayload)
	if !ok || !bytes.Contains(payload.NewArgs.Inline, []byte(`fak_context_restore {"origin":"`+origin+`"}`)) {
		t.Fatalf("missing inline restore handle: %#v", verdict.Payload)
	}
}
