package ggufload

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// extentFixtureKV builds a minimal GGUF header (magic, version 3, zero tensors, one
// key/value pair) whose value payload is `body` preceded by an explicit declared
// length `declaredLen`. It returns the bytes and the absolute file offset at which
// the value's bytes begin, which is also rr.n when the extent check fires.
func extentFixtureKV(declaredLen int, body []byte) (raw []byte, off int) {
	var b bytes.Buffer
	b.WriteString(Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0)) // tensors
	_ = binary.Write(&b, binary.LittleEndian, uint64(1)) // one KV
	key := "x"
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(key)))
	b.WriteString(key)
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeString))
	off = b.Len() // start of the length field is not the value; value starts after it
	_ = binary.Write(&b, binary.LittleEndian, uint64(declaredLen))
	off = b.Len() // value bytes begin here = rr.n when checkMappedExtent runs
	b.Write(body)
	return b.Bytes(), off
}

// TestGGUFMetadataExtentIndependent independently verifies the fak#13065 mapping-extent
// bound on a GGUF metadata string with its OWN minimal fixture (not the existing
// TestGGUFMetadataStringExceedsMappingRefused): unknown extent must not refuse, a known
// extent that the declared length overruns must refuse, and the boundary off+n == mapped
// must be admitted while one byte more is refused. It also exercises the exported extent
// predicate and the int64-overflow guard on an array's fixed-width span.
func TestGGUFMetadataExtentIndependent(t *testing.T) {
	// (a)+(b): declared length far larger than the file, but under maxStringBytes.
	body := []byte("body")
	raw, off := extentFixtureKV(1<<20, body)
	if off >= len(raw) || off != 45 {
		t.Fatalf("fixture offset = %d (len %d), want 45", off, len(raw))
	}

	// (a) Unknown extent (mapped=0): the historical Read path must NOT report a
	// mapped-extent refusal; it proceeds to the bounded read and surfaces EOF.
	if _, err := Read(bytes.NewReader(raw)); err == nil || strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("Read with unknown extent must not report a mapped-extent refusal, got %v", err)
	}

	// (b) Known extent equal to the real file length: the declared range runs past it.
	_, err := ReadSize(bytes.NewReader(raw), int64(len(raw)))
	if err == nil {
		t.Fatal("ReadSize admitted a metadata string past the mapped extent")
	}
	if !strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("want a mapped-extent refusal, got %v", err)
	}

	// (c) Boundary: a declared length EXACTLY equal to the remaining mapped bytes is
	// admitted; one byte more is refused. Pad the file so the value's declared range
	// ends exactly at the mapping end (off+n == mapped).
	exact := []byte("exactly-fit")
	exactRaw, exactOff := extentFixtureKV(len(exact), exact)
	pad := exactOff + len(exact) - len(exactRaw)
	if pad < 0 {
		t.Fatalf("fixture longer than declared range: pad=%d", pad)
	}
	mapped := append(append([]byte{}, exactRaw...), bytes.Repeat([]byte{0}, pad)...)
	if len(mapped) != exactOff+len(exact) {
		t.Fatalf("mapping = %d, want off+n = %d", len(mapped), exactOff+len(exact))
	}
	if _, err := ReadSize(bytes.NewReader(mapped), int64(len(mapped))); err != nil {
		t.Fatalf("ReadSize refused off+n == mapped: %v", err)
	}
	over := append([]byte{}, mapped...)
	binary.LittleEndian.PutUint64(over[exactOff-8:], uint64(len(exact)+1))
	if _, err := ReadSize(bytes.NewReader(over), int64(len(mapped))); err == nil {
		t.Fatal("ReadSize admitted off+n == mapped+1")
	} else if !strings.Contains(err.Error(), "exceeds mapped extent") {
		t.Fatalf("want a mapped-extent refusal for one byte over, got %v", err)
	}

	// (d) Direct predicate.
	for _, tc := range []struct {
		off, n, mapped int64
		wantErr        bool
	}{
		{0, 10, 10, false},
		{0, 11, 10, true},
		{5, 6, 10, true},
		{0, 10, 0, false},
	} {
		err := checkMappedExtent(tc.off, tc.n, tc.mapped)
		if (err != nil) != tc.wantErr {
			t.Errorf("checkMappedExtent(%d,%d,%d) err=%v, wantErr=%v", tc.off, tc.n, tc.mapped, err, tc.wantErr)
		}
	}

	// (e) Array whose fixed-width element count overflows int64 must error, not panic.
	var b bytes.Buffer
	b.WriteString(Magic)
	_ = binary.Write(&b, binary.LittleEndian, uint32(Version))
	_ = binary.Write(&b, binary.LittleEndian, uint64(0))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1))
	b.WriteString("x")
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(&b, binary.LittleEndian, uint32(TypeUint64))
	_ = binary.Write(&b, binary.LittleEndian, uint64(1)<<60) // * 8 bytes overflows int64
	if _, err := Read(bytes.NewReader(b.Bytes())); err == nil {
		t.Fatal("array span with int64-overflowing count was admitted")
	} else if !strings.Contains(err.Error(), "overflows int64") {
		t.Fatalf("want an int64 overflow refusal, got %v", err)
	}
}
