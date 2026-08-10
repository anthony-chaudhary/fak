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

func gitStatusFixture() []byte {
	var b strings.Builder
	b.WriteString("On branch main\nYour branch is up to date with 'origin/main'.\n\nChanges not staged for commit:\n")
	for i := 0; i < 100; i++ {
		b.WriteString("  (use \"git add <file>...\" to update what will be committed)\n")
		b.WriteString("  (use \"git restore <file>...\" to discard changes in working directory)\n")
	}
	b.WriteString("\tmodified:   internal/headroom/distill.go\n\nUntracked files:\n\tnew-filter.go\n\nUnmerged paths:\n\tboth modified: conflict.go\nwarning: sentinel warning\n")
	return []byte(b.String())
}

func gitLogFixture() []byte {
	var b strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&b, "commit %040x (HEAD -> main)\nAuthor: Agent <agent@example.invalid>\nDate:   Sat Aug 9 12:%02d:00 2026 -0700\n\n    feat: subject %03d\n\n", i+1, i%60, i)
	}
	b.WriteString("warning: log sentinel\n")
	return []byte(b.String())
}

func TestDistillGitStatusPreservesEveryStateMutation(t *testing.T) {
	orig := gitStatusFixture()
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "Bash", Kind: KindText, Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.Codec != "git-status-distill" || len(out.Bytes) >= len(orig) {
		t.Fatalf("output=%#v", out)
	}
	got := string(out.Bytes)
	for _, want := range []string{"modified:   internal/headroom/distill.go", "Untracked files:", "new-filter.go", "Unmerged paths:", "both modified: conflict.go", "warning: sentinel warning", "200 routine git status advice line(s) dropped"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q", want)
		}
	}
}

func TestDistillGitLogPreservesCommitsAndSubjects(t *testing.T) {
	orig := gitLogFixture()
	out, err := (distillCompressor{}).Compress(context.Background(), Input{Tool: "git log", Kind: KindText, Bytes: orig, UpstreamHeadroom: &Presence{}})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.Codec != "git-log-distill" {
		t.Fatalf("output=%#v", out)
	}
	got := string(out.Bytes)
	for _, want := range []string{"commit 0000000000000000000000000000000000000001", "feat: subject 000", "feat: subject 199", "warning: log sentinel", "200 routine git log date line(s) dropped"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(got, "Date:   ") {
		t.Fatal("routine dates survived")
	}
}

func TestDistillGitDispatchAndPorcelainSafety(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   Input
		want bool
	}{
		{name: "status", in: Input{Tool: "shell", Kind: KindText, Bytes: gitStatusFixture()}, want: true},
		{name: "log", in: Input{Tool: "git", Kind: KindText, Bytes: gitLogFixture()}, want: true},
		{name: "wrong tool", in: Input{Tool: "Read", Kind: KindText, Bytes: gitStatusFixture()}},
		{name: "wrong kind", in: Input{Tool: "Bash", Kind: KindJSON, Bytes: gitStatusFixture()}},
		{name: "ambiguous", in: Input{Tool: "Bash", Kind: KindText, Bytes: []byte("ordinary output")}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, got := applyDistillFilter(tt.in)
			if got != tt.want {
				t.Fatalf("matched=%v want=%v", got, tt.want)
			}
		})
	}
}

func TestDistillGitPorcelainIsRecognizedButNeverLossilyChanged(t *testing.T) {
	input := Input{Tool: "Bash", Kind: KindText, Bytes: []byte(" M tracked.go\n?? untracked.go\nUU conflict.go\n")}
	if !matchesGitStatus(input) {
		t.Fatal("porcelain status not recognized")
	}
	if body, dropped, ok := applyGitStatusFilter(input.Bytes); ok || dropped != 0 || body != nil {
		t.Fatalf("porcelain unexpectedly transformed: ok=%v dropped=%d body=%q", ok, dropped, body)
	}
}
func TestDistillGitMutationsRemainVisible(t *testing.T) {
	base := gitStatusFixture()
	for _, mutation := range [][]byte{
		[]byte("\tboth added: injected-conflict.go\n"),
		[]byte("\tmodified: injected.go\n"),
		[]byte("\tuntracked-injected.go\n"),
		[]byte("fatal: injected git failure\n"),
	} {
		out, _, ok := applyGitStatusFilter(append(append([]byte(nil), base...), mutation...))
		if !ok || !bytes.Contains(out, bytes.TrimSpace(mutation)) {
			t.Fatalf("mutation lost: %q", mutation)
		}
	}
}

func TestDistillGitAdmissionRestoresOriginal(t *testing.T) {
	orig := gitStatusFixture()
	if !Select(DistillName) {
		t.Fatal("distill not registered")
	}
	t.Cleanup(func() { Select(NoopName) })
	call := &abi.ToolCall{Tool: "Bash"}
	verdict := NewGate().Admit(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}})
	origin := verdict.Meta["origin"]
	restored, err := blob.Default.PageIn(context.Background(), abi.Ref{Kind: abi.RefBlob, Digest: origin})
	if err != nil || !bytes.Equal(restored.Inline, orig) {
		t.Fatalf("restore err=%v verdict=%#v", err, verdict)
	}
	payload, ok := verdict.Payload.(abi.TransformPayload)
	if !ok || !bytes.Contains(payload.NewArgs.Inline, []byte(`fak_context_restore {"origin":"`+origin+`"}`)) {
		t.Fatalf("restore handle missing: %#v", verdict.Payload)
	}
}
