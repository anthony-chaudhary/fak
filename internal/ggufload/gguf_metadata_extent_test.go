package ggufload

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// metaExtentFixture builds a minimal GGUF header whose single metadata string value
// declares a length that passes maxStringBytes but runs past the supplied mapping
// extent. When mapped <= 0 the header is well-formed and parses.
func metaExtentFixture(declaredLen int, body []byte) []byte {
	var b bytes.Buffer
	writeString := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString(Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0)) // tensors
	_ = binary.Write(&b, binary.LittleEndian, uint64(1)) // one KV
	writeString("x.arch")
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeString))
	_ = binary.Write(&b, binary.LittleEndian, uint64(declaredLen))
	b.Write(body)
	return b.Bytes()
}

// TestGGUFMetadataStringExceedsMappingRefused is the fak#13065 witness: a declared
// metadata string length that passes the maxStringBytes ceiling but exceeds the mapping
// extent must be refused with a typed error and must not over-read. The SAME bytes parse
// fine when the extent is unknown (the historical Read(io.Reader) contract).
func TestGGUFMetadataStringExceedsMappingRefused(t *testing.T) {
	body := []byte("short-body")
	raw := metaExtentFixture(1<<20, body) // declares 1 MiB, file is tiny

	// Unknown extent: the historical path has no extent to compare against, so it
	// proceeds to the bounded read and surfaces the short source as an EOF rather than
	// a mapped-extent refusal. The point is that the extent check only fires when an
	// extent is known.
	if _, err := Read(bytes.NewReader(raw)); err == nil || strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("Read with unknown extent must not report a mapped-extent refusal, got %v", err)
	}

	// Known extent: the declared range exceeds the mapping, so refuse with a typed error.
	_, err := ReadSize(bytes.NewReader(raw), int64(len(raw)))
	if err == nil {
		t.Fatal("ReadSize admitted a metadata string past the mapped extent")
	}
	if !strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("want a mapped-extent refusal, got %v", err)
	}
}

// TestGGUFMetadataStringWithinMappingAdmitted pins the positive arm: a well-formed header
// whose declared lengths fit the mapping still parses under ReadSize.
func TestGGUFMetadataStringWithinMappingAdmitted(t *testing.T) {
	raw := metaExtentFixture(len("good"), []byte("good"))
	got, err := ReadSize(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadSize refused a well-formed header: %v", err)
	}
	v, ok := got.String("x.arch")
	if !ok || v != "good" {
		t.Fatalf("x.arch = (%q,%v), want (\"good\",true)", v, ok)
	}
}

// TestGGUFMetadataArrayExceedsMappingRefused pins the array arm: a fixed-width array
// element count whose span exceeds the mapping is refused before allocation.
func TestGGUFMetadataArrayExceedsMappingRefused(t *testing.T) {
	var b bytes.Buffer
	writeString := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString(Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1))
	writeString("x.arr")
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeUint64)) // element type
	_ = binary.Write(&b, binary.LittleEndian, uint64(1<<20))      // count: 8 MiB span, file tiny
	raw := b.Bytes()

	if _, err := ReadSize(bytes.NewReader(raw), int64(len(raw))); err == nil {
		t.Fatal("ReadSize admitted an array span past the mapped extent")
	} else if !strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("want a mapped-extent refusal, got %v", err)
	}
}
