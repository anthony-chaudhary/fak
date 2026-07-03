package randhex

import "testing"

func TestStringReturnsLowerHexWithExpectedLength(t *testing.T) {
	got, ok := String(16)
	if !ok {
		t.Fatal("String reported random read failure")
	}
	if len(got) != 32 {
		t.Fatalf("String(16) length = %d, want 32", len(got))
	}
	for _, r := range got {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			t.Fatalf("String(16) contains non-lower-hex rune %q in %q", r, got)
		}
	}
}

func TestStringZeroBytes(t *testing.T) {
	got, ok := String(0)
	if !ok {
		t.Fatal("String(0) reported random read failure")
	}
	if got != "" {
		t.Fatalf("String(0) = %q, want empty string", got)
	}
}
