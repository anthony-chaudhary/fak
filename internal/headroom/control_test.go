package headroom

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	// blob registers the content-addressed Ref backend so the gate's reversible-CCR
	// preserve/resolve round-trip is exercised for the control-strip path too.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

const esc = "\x1b"

// TestStripEscapeSequences covers the escape-sequence families directly: SGR
// color, a CSI cursor move, and an OSC window-title — all removed, the payload
// text (including a multi-byte rune) preserved.
func TestStripEscapeSequences(t *testing.T) {
	in := esc + "[32mPASS" + esc + "[0m " + esc + "[2K" + esc + "]0;title\x07ok — café\n"
	got, did := stripEscapeSequences([]byte(in))
	if !did {
		t.Fatal("expected escapes to be stripped")
	}
	want := "PASS ok — café\n"
	if string(got) != want {
		t.Fatalf("stripEscapeSequences = %q, want %q", got, want)
	}
	if strings.ContainsRune(string(got), 0x1b) {
		t.Fatalf("ESC survived: %q", got)
	}
}

// TestStripEscapeSequencesNoopOnCleanText: clean text allocates nothing and
// reports no transform, so a no-op never claims a codec.
func TestStripEscapeSequencesNoopOnCleanText(t *testing.T) {
	in := []byte("plain text\twith a tab\nand two lines\n")
	got, did := stripEscapeSequences(in)
	if did {
		t.Fatalf("clean text should not be transformed: %q", got)
	}
	if string(got) != string(in) {
		t.Fatalf("clean text changed: %q", got)
	}
}

// TestStripDropsBareControlKeepsTabsNewlines: bare C0 controls and DEL are
// dropped; tab and newline survive (the CR is left for cr-collapse).
func TestStripDropsBareControlKeepsTabsNewlines(t *testing.T) {
	in := "a\x00b\x07c\x7fd\te\nf"
	got, _ := stripEscapeSequences([]byte(in))
	want := "abcd\te\nf"
	if string(got) != want {
		t.Fatalf("strip = %q, want %q", got, want)
	}
}

// TestCollapseCarriageReturnRedraw: an in-place redraw keeps only its final frame;
// a CRLF line ending (lone trailing CR) is NOT treated as a redraw.
func TestCollapseCarriageReturnRedraw(t *testing.T) {
	in := "step: 1%\rstep: 50%\rstep: 100%\nnext line\r\ndone"
	got, did := collapseCarriageReturns([]byte(in))
	if !did {
		t.Fatal("expected a redraw collapse")
	}
	want := "step: 100%\nnext line\r\ndone"
	if string(got) != want {
		t.Fatalf("collapse = %q, want %q", got, want)
	}
}

// TestCollapseCarriageReturnNoopOnCRLF: a file that only carries CRLF endings has
// no interior CR, so cr-collapse never fires (the trim pass owns the trailing CR).
func TestCollapseCarriageReturnNoopOnCRLF(t *testing.T) {
	in := []byte("line one\r\nline two\r\nline three\r\n")
	got, did := collapseCarriageReturns(in)
	if did {
		t.Fatalf("CRLF endings must not register as a redraw: %q", got)
	}
}

// TestNativeStripsANSIColor: the native compressor shrinks a colorized log,
// labels the codec, and keeps the message text (and tabs) intact.
func TestNativeStripsANSIColor(t *testing.T) {
	in := []byte(esc + "[32mPASS" + esc + "[0m ok\tgithub.com/x/foo\t0.42s\n" +
		esc + "[31mFAIL" + esc + "[0m github.com/x/bar\t1.01s\n")
	out, err := nativeCompressor{}.Compress(context.Background(), Input{Bytes: in})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed || out.NewLen >= out.OrigLen {
		t.Fatalf("colorized log should compress: %d -> %d", out.OrigLen, out.NewLen)
	}
	if !strings.Contains(out.Codec, "ansi-strip") {
		t.Fatalf("codec=%q, want ansi-strip", out.Codec)
	}
	s := string(out.Bytes)
	if strings.ContainsRune(s, 0x1b) {
		t.Fatalf("ESC survived compression: %q", s)
	}
	for _, want := range []string{"PASS", "FAIL", "github.com/x/foo", "\t"} {
		if !strings.Contains(s, want) {
			t.Fatalf("information lost: %q missing from %q", want, s)
		}
	}
}

// TestNativeCollapsesProgressBar: a carriage-return progress bar (the dominant
// waste in install/download tool output) collapses to its final frame — the kind
// of large, honest saving the strip targets.
func TestNativeCollapsesProgressBar(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("downloading model.gguf ")
	for i := 0; i <= 100; i++ {
		fmt.Fprintf(&sb, "\rdownloading model.gguf %3d%% [%s]", i, strings.Repeat("#", i/5))
	}
	sb.WriteString("\ndone\n")
	in := []byte(sb.String())

	out, err := nativeCompressor{}.Compress(context.Background(), Input{Bytes: in})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed {
		t.Fatal("a progress bar should compress")
	}
	if !strings.Contains(out.Codec, "cr-collapse") {
		t.Fatalf("codec=%q, want cr-collapse", out.Codec)
	}
	// The collapse should be dramatic (final frame + a line), not marginal.
	if got := out.SavedRatio(); got < 0.80 {
		t.Fatalf("progress-bar saving = %.2f, want >= 0.80", got)
	}
	s := string(out.Bytes)
	if !strings.Contains(s, "100%") || !strings.Contains(s, "done") {
		t.Fatalf("final frame / trailing line lost: %q", s)
	}
	if strings.Contains(s, "  0%") || strings.Contains(s, " 50%") {
		t.Fatalf("intermediate frames survived: %q", s)
	}
}

// TestNativeControlComposesWithLineDedup: ansi-strip reveals identical lines that
// the existing line-dedup then collapses — the transforms compose, and the codec
// records both.
func TestNativeControlComposesWithLineDedup(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 50; i++ {
		sb.WriteString(esc + "[33mwarning: deprecated API in use" + esc + "[0m\n")
	}
	in := []byte(sb.String())
	out, err := nativeCompressor{}.Compress(context.Background(), Input{Bytes: in})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Compressed {
		t.Fatal("colorized repeated warnings should compress")
	}
	for _, want := range []string{"ansi-strip", "line-dedup"} {
		if !strings.Contains(out.Codec, want) {
			t.Fatalf("codec=%q, want it to contain %q", out.Codec, want)
		}
	}
	if !strings.Contains(string(out.Bytes), "identical lines elided") {
		t.Fatalf("missing elision marker: %q", out.Bytes)
	}
}

// TestGateStripsANSIAndPreservesOriginal: end-to-end through the result-admit gate
// — a colorized benign result is transformed smaller AND the exact original is
// recoverable from the shared CAS (the reversible-CCR promise) for the new path.
func TestGateStripsANSIAndPreservesOriginal(t *testing.T) {
	withSelected(t, NativeName)
	orig := []byte(esc + "[32mok" + esc + "[0m   github.com/x/a\t0.01s\n" +
		esc + "[32mok" + esc + "[0m   github.com/x/b\t0.02s\n" +
		esc + "[31mFAIL" + esc + "[0m github.com/x/c\t0.03s\n")
	r := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: orig, Len: int64(len(orig))}}
	v := NewGate().Admit(context.Background(), &abi.ToolCall{Tool: "Bash"}, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("native gate must Transform a colorized benign result, got %v", v.Kind)
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("Transform verdict carries no TransformPayload: %+v", v.Payload)
	}
	if int(tp.NewArgs.Len) >= len(orig) {
		t.Fatalf("rewritten payload not smaller: %d >= %d", tp.NewArgs.Len, len(orig))
	}
	digest := v.Meta["origin"]
	if digest == "" {
		t.Fatal("expected an origin digest (blob backend is imported in this test)")
	}
	got, err := abi.ActiveResolver().Resolve(context.Background(), abi.Ref{Kind: abi.RefBlob, Digest: digest, Len: int64(len(orig))})
	if err != nil {
		t.Fatalf("resolve preserved original: %v", err)
	}
	if string(got) != string(orig) {
		t.Fatal("preserved original did not round-trip")
	}
}
