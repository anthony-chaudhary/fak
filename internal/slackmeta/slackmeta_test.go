package slackmeta

import (
	"strings"
	"testing"
)

func TestNewClampsNoiseAndFormatsLine(t *testing.T) {
	s := New(4, 0, "facts vs wrapper")
	if s.Signal != 4 || s.Noise != 1 || s.Ratio != 4 {
		t.Fatalf("score = %+v, want signal=4 noise=1 ratio=4", s)
	}
	line := s.Line()
	if !strings.Contains(line, "S/N self-score 4.0") || !strings.Contains(line, "facts vs wrapper") {
		t.Fatalf("line missing score/basis: %q", line)
	}
}

func TestAppendTextIsIdempotent(t *testing.T) {
	s := New(2, 1, "")
	once := AppendText("hello", s)
	twice := AppendText(once, s)
	if once != twice {
		t.Fatalf("AppendText duplicated metadata:\nonce=%q\ntwice=%q", once, twice)
	}
	if !strings.Contains(once, "S/N self-score") {
		t.Fatalf("AppendText missing metadata: %q", once)
	}
}

func TestAppendContextIsIdempotent(t *testing.T) {
	blocks := []any{map[string]any{"type": "section", "text": map[string]any{"text": "hello"}}}
	once := AppendContext(blocks, New(1, 1, ""))
	twice := AppendContext(once, New(1, 1, ""))
	if len(once) != 2 || len(twice) != 2 {
		t.Fatalf("context append lengths = %d/%d, want 2/2", len(once), len(twice))
	}
}

func TestNonEmpty(t *testing.T) {
	if got := NonEmpty("x", " ", "", "y"); got != 2 {
		t.Fatalf("NonEmpty = %d, want 2", got)
	}
}
