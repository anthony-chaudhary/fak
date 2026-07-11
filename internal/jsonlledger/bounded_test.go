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
