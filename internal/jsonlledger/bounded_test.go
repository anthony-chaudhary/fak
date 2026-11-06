package jsonlledger

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestAppendBoundedRotatesBeforeCrossingActiveBound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	if err := AppendBounded(path, []byte(`{"n":1}`), 20); err != nil {
		t.Fatal(err)
	}
	if err := AppendBounded(path, []byte(`{"n":222222}`), 20); err != nil {
		t.Fatal(err)
	}
	active, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sealed, err := os.ReadFile(path + ".1")
	if err != nil {
		t.Fatal(err)
	}
	if string(active) != "{\"n\":222222}\n" {
		t.Fatalf("active = %q", active)
	}
	if string(sealed) != "{\"n\":1}\n" {
		t.Fatalf("sealed = %q", sealed)
	}
	if int64(len(active)) > 20 {
		t.Fatalf("active crossed bound: %d", len(active))
	}
}

func TestReadTailIsBoundedAndDropsPartialFirstRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	content := []byte("first-row\nsecond-row\nthird-row\n")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}
	got := ReadTail(path, 22) // starts inside first-row
	if !bytes.Equal(got, []byte("second-row\nthird-row\n")) {
		t.Fatalf("tail = %q", got)
	}
	if len(got) > 22 {
		t.Fatalf("read %d bytes, bound 22", len(got))
	}
}

// One 24-byte file and five bounded reads; the runtime estimate is unmeasured.
// fak-test:runtime fast est=10ms lane=default
func TestReadTailPreservesCompleteFirstRow(t *testing.T) {
	t.Parallel()
	const first = "{\"n\":1}\n"
	const second = "{\"n\":2}\n"
	const third = "{\"n\":3}\n"
	const content = first + second + third
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name     string
		maxBytes int64
		want     string
	}{
		{"whole file", int64(len(content)), content},
		{"row boundary", int64(len(second + third)), second + third},
		{"last row boundary", int64(len(third)), third},
		{"newline boundary", int64(len("\n" + second + third)), second + third},
		{"partial first row", int64(len(second+third) - 1), third},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ReadTail(path, tc.maxBytes)
			if string(got) != tc.want {
				t.Errorf("ReadTail(%d) = %q, want %q", tc.maxBytes, got, tc.want)
			}
			if int64(len(got)) > tc.maxBytes {
				t.Errorf("read %d bytes, bound %d", len(got), tc.maxBytes)
			}
		})
	}
}
