package agentsindex

import (
	"bytes"
	"os"
	"testing"
)

func TestDetectAndReconcileLineEndings(t *testing.T) {
	mixed := bytes.Repeat([]byte("dominant\r\n"), 3654)
	mixed = append(mixed, bytes.Repeat([]byte("stray\n"), 15)...)

	tests := []struct {
		name     string
		existing []byte
		incoming []byte
		policy   EOL
		want     []byte
	}{
		{"absent uses policy", nil, []byte("a\nb\n"), EOLCRLF, []byte("a\r\nb\r\n")},
		{"uniform existing wins", []byte("old\r\n"), []byte("new\nline\n"), EOLLF, []byte("new\r\nline\r\n")},
		{"mixed heals dominant", mixed, []byte("new\nline\n"), EOLLF, []byte("new\r\nline\r\n")},
		{"binary untouched", []byte("old\x00\r\n"), []byte("new\x00\n"), EOLCRLF, []byte("new\x00\n")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Reconcile(tt.existing, tt.incoming, tt.policy)
			if !bytes.Equal(got, tt.want) {
				t.Fatalf("Reconcile() = %q, want %q", got, tt.want)
			}
		})
	}
	if ending, isMixed := Detect(mixed); ending != EOLCRLF || !isMixed {
		t.Fatalf("Detect(realistic mixed) = %q, %v; want CRLF, true", ending, isMixed)
	}
}

func TestLineEndingContractHasNoPlatformDefault(t *testing.T) {
	for _, source := range [][]byte{mustRead(t, "lineendings.go"), mustRead(t, "agentsindex.go")} {
		if bytes.Contains(source, []byte("os.EOL")) {
			t.Fatal("write path must not use platform-default os.EOL")
		}
	}
}

func mustRead(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
