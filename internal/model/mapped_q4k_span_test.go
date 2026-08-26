package model

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"
)

func TestMappedQ4KSpanOwnerNonPageAlignedRange(t *testing.T) {
	requireMappedQ4KSpanDarwin(t)
	path, payload := mappedQ4KSpanFixture(t)
	page := os.Getpagesize()
	offset := int64(page + 37)
	length := page + 91

	owner, err := openMappedQ4KSpan(path, offset, length)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()

	want := payload[int(offset) : int(offset)+length]
	if !bytes.Equal(owner.bytes(), want) {
		t.Fatal("logical span differs from the requested file range")
	}
	if len(owner.bytes()) != length || cap(owner.bytes()) != length {
		t.Fatalf("logical span len/cap = %d/%d, want %d/%d", len(owner.bytes()), cap(owner.bytes()), length, length)
	}
	wantMappedOffset := offset - offset%int64(page)
	if owner.logicalOffset != offset || owner.mappedOffset != wantMappedOffset {
		t.Fatalf("offsets logical=%d mapped=%d, want %d/%d", owner.logicalOffset, owner.mappedOffset, offset, wantMappedOffset)
	}
	if len(owner.mapped) == 0 || len(owner.mapped)%page != 0 {
		t.Fatalf("mapped length = %d, want a non-zero page multiple (%d)", len(owner.mapped), page)
	}
	if uintptr(unsafe.Pointer(&owner.mapped[0]))%uintptr(page) != 0 {
		t.Fatalf("mapped base %p is not page-aligned to %d", &owner.mapped[0], page)
	}
}

func TestMappedQ4KSpanOwnerRejectsInvalidRanges(t *testing.T) {
	requireMappedQ4KSpanDarwin(t)
	path, payload := mappedQ4KSpanFixture(t)
	cases := []struct {
		name   string
		path   string
		offset int64
		length int
	}{
		{name: "negative offset before open", path: filepath.Join(t.TempDir(), "missing.gguf"), offset: -1, length: 1},
		{name: "zero length", path: path, offset: 0, length: 0},
		{name: "past end", path: path, offset: int64(len(payload) - 4), length: 5},
		{name: "offset past end", path: path, offset: int64(len(payload) + 1), length: 1},
		{name: "end overflow", path: path, offset: math.MaxInt64 - 1, length: 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			owner, err := openMappedQ4KSpan(tc.path, tc.offset, tc.length)
			if owner != nil {
				_ = owner.Close()
				t.Fatal("invalid range returned an owner")
			}
			var rangeErr *mappedQ4KSpanRangeError
			if !errors.As(err, &rangeErr) {
				t.Fatalf("error = %v, want *mappedQ4KSpanRangeError", err)
			}
		})
	}
}

func TestMappedQ4KSpanOwnerBytesRemainValidUntilClose(t *testing.T) {
	requireMappedQ4KSpanDarwin(t)
	path, payload := mappedQ4KSpanFixture(t)
	offset := int64(23)
	length := os.Getpagesize() + 17

	owner, err := openMappedQ4KSpan(path, offset, length)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	if !bytes.Equal(owner.bytes(), payload[int(offset):int(offset)+length]) {
		t.Fatal("mapped bytes changed after path removal while owner remained live")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if owner.bytes() != nil {
		t.Fatal("closed owner still exposes mapped bytes")
	}
}

func TestMappedQ4KSpanOwnerDoubleClose(t *testing.T) {
	requireMappedQ4KSpanDarwin(t)
	path, _ := mappedQ4KSpanFixture(t)
	owner, err := openMappedQ4KSpan(path, 7, 33)
	if err != nil {
		t.Fatal(err)
	}
	retainedFile := owner.file
	if err := owner.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if _, err := retainedFile.Stat(); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("retained file after close: %v, want os.ErrClosed", err)
	}
	if owner.file != nil || owner.mapped != nil || owner.bytes() != nil {
		t.Fatal("first close retained file or mapped views")
	}
	if err := owner.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

func TestMappedQ4KSpanOwnerPortableUnavailable(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("Darwin has the read-only mapped-span implementation")
	}
	owner, err := openMappedQ4KSpan("unused.gguf", 0, 1)
	if owner != nil {
		t.Fatal("portable stub returned an owner")
	}
	var unavailable *mappedQ4KSpanUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("error = %v, want *mappedQ4KSpanUnavailableError", err)
	}
}

func mappedQ4KSpanFixture(t *testing.T) (string, []byte) {
	t.Helper()
	page := os.Getpagesize()
	payload := make([]byte, 3*page+211)
	for i := range payload {
		payload[i] = byte((i*29 + 17) % 251)
	}
	path := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	return path, payload
}

func requireMappedQ4KSpanDarwin(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("Darwin-only mmap behavior")
	}
}
