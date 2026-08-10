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

func packageManagerFixture() []byte {
	var b strings.Builder
	b.WriteString("pnpm 10.0.0\n")
	for i := 0; i < 100; i++ {
		fmt.Fprintf(&b, "Progress: resolved %d, reused %d, downloaded 0, added 0\n", i, i)
		fmt.Fprintf(&b, "npm http fetch GET 200 https://registry.npmjs.org/pkg-%d 10ms (cache hit)\n", i)
	}
	b.WriteString("npm WARN deprecated old-pkg@1.0.0: unsupported\n")
	b.WriteString("npm error code ELIFECYCLE\nnpm error command failed\n    at lifecycle continuation\n")
	b.WriteString("ERR_PNPM_PEER_DEP_ISSUES Unmet peer dependencies\n    peer react@18 required by widget@2\n")
	b.WriteString("Tests: 2 failed, 8 passed\n  FAIL src/widget.test.ts\n    expected true, got false\n")
	b.WriteString("UNKNOWN package-manager sentinel\n")
	return []byte(b.String())
}

func TestDistillPackageManagerKeepsFailuresAndDropsProgress(t *testing.T) {
	orig := packageManagerFixture()
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "Bash", Kind: KindText, Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.Codec != "package-manager-distill" || len(out.Bytes) >= len(orig) {
		t.Fatalf("output = %#v", out)
	}
	got := string(out.Bytes)
	for _, want := range []string{"npm WARN deprecated", "npm error code ELIFECYCLE", "lifecycle continuation", "ERR_PNPM_PEER_DEP_ISSUES", "peer react@18", "Tests: 2 failed", "FAIL src/widget.test.ts", "expected true", "UNKNOWN package-manager sentinel", "200 routine npm/pnpm line(s) dropped"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(got, "Progress: resolved") || strings.Contains(got, "npm http fetch") {
		t.Fatal("routine progress survived")
	}
}

func TestDistillPackageManagerDispatchIsConservative(t *testing.T) {
	orig := packageManagerFixture()
	for _, tt := range []struct {
		name string
		in   Input
		want bool
	}{
		{name: "pnpm shell", in: Input{Tool: "shell", Kind: KindText, Bytes: orig}, want: true},
		{name: "named npm", in: Input{Tool: "npm", Kind: KindUnknown, Bytes: []byte("npm 10\nnpm http fetch one\nnpm http fetch two\n")}, want: true},
		{name: "wrong tool", in: Input{Tool: "Read", Kind: KindText, Bytes: orig}},
		{name: "wrong kind", in: Input{Tool: "Bash", Kind: KindJSON, Bytes: orig}},
		{name: "ambiguous shell", in: Input{Tool: "Bash", Kind: KindText, Bytes: []byte("ordinary output")}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, got := applyDistillFilter(tt.in)
			if got != tt.want {
				t.Fatalf("matched=%v want=%v", got, tt.want)
			}
		})
	}
}

func TestDistillPackageManagerMutationsPreserveEvidence(t *testing.T) {
	base := packageManagerFixture()
	for _, mutation := range [][]byte{
		[]byte("npm error injected sentinel\n    npm continuation\n"),
		[]byte("npm WARN injected warning\n"),
		[]byte("ERR_PNPM_INJECTED injected failure\n    pnpm continuation\n"),
		[]byte("UNKNOWN injected package record\n"),
	} {
		input := append(append([]byte(nil), base...), mutation...)
		out, _, ok := applyPackageManagerFilter(input)
		if !ok || !bytes.Contains(out, bytes.TrimSpace(mutation)) {
			t.Fatalf("mutation lost: %q", mutation)
		}
	}
}

func TestDistillPackageManagerAdmissionRestoresOriginal(t *testing.T) {
	orig := packageManagerFixture()
	if !Select(DistillName) {
		t.Fatal("distill compressor not registered")
	}
	t.Cleanup(func() { Select(NoopName) })
	call := &abi.ToolCall{Tool: "Bash"}
	result := &abi.Result{Call: call, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}}
	verdict := NewGate().Admit(context.Background(), call, result)
	origin := verdict.Meta["origin"]
	restored, err := blob.Default.PageIn(context.Background(), abi.Ref{Kind: abi.RefBlob, Digest: origin})
	if err != nil || !bytes.Equal(restored.Inline, orig) {
		t.Fatalf("restore err=%v equal=%v verdict=%#v", err, bytes.Equal(restored.Inline, orig), verdict)
	}
	payload, ok := verdict.Payload.(abi.TransformPayload)
	if !ok || !bytes.Contains(payload.NewArgs.Inline, []byte(`fak_context_restore {"origin":"`+origin+`"}`)) {
		t.Fatalf("missing inline restore handle: %#v", verdict.Payload)
	}
}
