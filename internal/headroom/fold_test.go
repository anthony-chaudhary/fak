package headroom

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// TestFoldsScatteredDuplicates: a line repeated NON-consecutively across the blob is
// folded to its first occurrence + a recurrence marker, later copies elided, every
// unique line preserved in order.
func TestFoldsScatteredDuplicates(t *testing.T) {
	in := []byte(strings.Join([]string{
		"build start",
		"warning: deprecated API foo",
		"compiling a.go",
		"warning: deprecated API foo",
		"compiling b.go",
		"warning: deprecated API foo",
		"compiling c.go",
		"warning: deprecated API foo",
		"compiling d.go",
		"warning: deprecated API foo",
		"build done",
	}, "\n"))
	got, did := foldGlobalDuplicates(in)
	if !did {
		t.Fatal("expected scattered duplicates to fold")
	}
	s := string(got)
	if n := strings.Count(s, "warning: deprecated API foo"); n != 1 {
		t.Fatalf("folded line should appear exactly once, got %d:\n%s", n, s)
	}
	if !strings.Contains(s, "×4 more identical, elided") {
		t.Fatalf("missing recurrence marker: %q", s)
	}
	for _, want := range []string{"build start", "compiling a.go", "compiling d.go", "build done"} {
		if !strings.Contains(s, want) {
			t.Fatalf("unique line lost: %q missing from %q", want, s)
		}
	}
	if len(got) >= len(in) {
		t.Fatalf("fold did not shrink: %d -> %d", len(in), len(got))
	}
}

// TestGlobalFoldKeepsFirstOccurrenceOrder: the first occurrence of each folded line
// stays where it first appeared.
func TestGlobalFoldKeepsFirstOccurrenceOrder(t *testing.T) {
	in := []byte("alpha header line\nrepeated payload line\nbeta header line\nrepeated payload line\ngamma header line\nrepeated payload line\n")
	got, did := foldGlobalDuplicates(in)
	if !did {
		t.Fatal("expected a fold")
	}
	s := string(got)
	iAlpha := strings.Index(s, "alpha header line")
	iPayload := strings.Index(s, "repeated payload line")
	iBeta := strings.Index(s, "beta header line")
	if !(iAlpha < iPayload && iPayload < iBeta) {
		t.Fatalf("first-occurrence order not preserved: alpha=%d payload=%d beta=%d\n%s", iAlpha, iPayload, iBeta, s)
	}
}

// TestGlobalFoldSkipsShortLines: short structural lines are left alone — folding them
// hurts readability more than it saves.
func TestGlobalFoldSkipsShortLines(t *testing.T) {
	in := []byte("ok\nrow 1\nok\nrow 2\nok\nrow 3\nok\nok\n") // "ok" is below minFoldLen
	if _, did := foldGlobalDuplicates(in); did {
		t.Fatal("short lines must not fold")
	}
}

// TestGlobalFoldNoopWhenAllUnique: nothing recurs ⇒ no change, no codec.
func TestGlobalFoldNoopWhenAllUnique(t *testing.T) {
	in := []byte("the first distinct line\nthe second distinct line\nthe third distinct line\n")
	if _, did := foldGlobalDuplicates(in); did {
		t.Fatal("all-unique input must not fold")
	}
}

// TestGlobalFoldBelowThresholdNoop: a line seen only twice stays (floor is 3).
func TestGlobalFoldBelowThresholdNoop(t *testing.T) {
	in := []byte("a longer repeated line here\nunique middle line\na longer repeated line here\n")
	if _, did := foldGlobalDuplicates(in); did {
		t.Fatal("two occurrences are below the global-dup floor and must not fold")
	}
}

// TestNativeFoldsScatteredViaCompress: the folder reaches the real compressor path
// and labels the codec.
func TestNativeFoldsScatteredViaCompress(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 8; i++ {
		fmt.Fprintf(&sb, "processing item number %d of the batch\n", i)
		sb.WriteString("warning: rate limit approaching, backing off\n")
	}
	in := []byte(sb.String())
	out, err := nativeCompressor{}.Compress(context.Background(), Input{Bytes: in})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed {
		t.Fatal("scattered duplicate warnings should compress")
	}
	if !strings.Contains(out.Codec, "line-fold") {
		t.Fatalf("codec=%q, want line-fold", out.Codec)
	}
	if n := strings.Count(string(out.Bytes), "warning: rate limit approaching, backing off"); n != 1 {
		t.Fatalf("the scattered warning should fold to one copy, got %d", n)
	}
	// the unique per-item lines must all survive
	for i := 0; i < 8; i++ {
		if !strings.Contains(string(out.Bytes), fmt.Sprintf("processing item number %d", i)) {
			t.Fatalf("unique item line %d lost", i)
		}
	}
}

// TestGlobalFoldComposesWithConsecutive: a line that appears BOTH in a consecutive run
// and scattered elsewhere is collapsed by both passes — line-dedup then line-fold.
func TestGlobalFoldComposesWithConsecutive(t *testing.T) {
	lines := []string{
		"ERROR: boom happened here",
		"ERROR: boom happened here",
		"ERROR: boom happened here", // a consecutive run of 3
		"some other distinct output",
		"ERROR: boom happened here", // scattered
		"another distinct output line",
		"ERROR: boom happened here", // scattered
	}
	in := []byte(strings.Join(lines, "\n"))
	out, err := nativeCompressor{}.Compress(context.Background(), Input{Bytes: in})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed {
		t.Fatal("mixed consecutive + scattered duplicates should compress")
	}
	for _, want := range []string{"line-dedup", "line-fold"} {
		if !strings.Contains(out.Codec, want) {
			t.Fatalf("codec=%q, want it to contain %q", out.Codec, want)
		}
	}
	if n := strings.Count(string(out.Bytes), "ERROR: boom happened here"); n != 1 {
		t.Fatalf("the error line should survive exactly once, got %d:\n%s", n, out.Bytes)
	}
}
