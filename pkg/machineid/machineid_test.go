package machineid

import (
	"regexp"
	"testing"
)

var hex16 = regexp.MustCompile(`^[0-9a-f]{16}$`)

// fak-test:runtime fast est=1s
func TestHash(t *testing.T) {
	for _, raw := range []string{"", " ", "\n\t "} {
		if got := Hash(raw); got != "" {
			t.Fatalf("Hash(%q) = %q, want empty", raw, got)
		}
	}
	base := Hash("0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
	if !hex16.MatchString(base) {
		t.Fatalf("Hash = %q, want 16 lowercase hex", base)
	}
	for _, raw := range []string{"0F1E2D3C-4B5A-6978-8796-A5B4C3D2E1F0", "  0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0\n"} {
		if got := Hash(raw); got != base {
			t.Fatalf("Hash(%q) = %q, want %q (trim/upper equivalent)", raw, got, base)
		}
	}
	if other := Hash("another-machine"); other == base || !hex16.MatchString(other) {
		t.Fatalf("Hash(another-machine) = %q, want a distinct 16-hex id", other)
	}
	// Pinned: sha256("ABC")[:8], so the format never silently changes.
	if got, want := Hash("abc"), "b5d4045c3f466fa9"; got != want {
		t.Fatalf("Hash(abc) = %q, want %q", got, want)
	}
}

// fak-test:runtime fast est=1s
func TestIDIsHashOfSource(t *testing.T) {
	if got, want := ID(), Hash(Source()); got != want {
		t.Fatalf("ID() = %q, want Hash(Source()) = %q", got, want)
	}
}
