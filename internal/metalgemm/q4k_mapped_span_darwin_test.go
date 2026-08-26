//go:build darwin && arm64 && cgo

package metalgemm

import (
	"os"
	"syscall"
	"testing"
)

func TestQ4KMappedSpan(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const out, in, offset = 2, 256, 32
	page := os.Getpagesize()
	span, err := syscall.Mmap(-1, 0, page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(span)
	raw := q4kTestRaw(out, in, 0x9072)
	payload := span[offset : offset+len(raw)]
	copy(payload, raw)
	if err := syscall.Mprotect(span, syscall.PROT_READ); err != nil {
		t.Fatal(err)
	}

	w := UploadQ4KMappedSpan(span, offset, out, in)
	if w == nil || !w.NoCopy() {
		t.Fatalf("mapped span upload = %#v, want no-copy weight", w)
	}
	x := q4kTestVector(in, 9072)
	want := q4kVectorizedReference(raw, out, in, x)
	got := make([]float32, out)
	w.GEMV(x, got)
	if cosine, maxRel := q4kTestCosineMaxRel(want, got); cosine < 0.999999 || maxRel > 5e-3 {
		t.Fatalf("direct mapped-span parity: cosine=%g max_rel=%g", cosine, maxRel)
	}
	group := GEMVGroup([]*Q4KWeight{w}, x)
	if len(group) != 1 || len(group[0]) != out {
		t.Fatalf("group shape = %v", group)
	}
	if cosine, maxRel := q4kTestCosineMaxRel(want, group[0]); cosine < 0.999999 || maxRel > 5e-3 {
		t.Fatalf("grouped mapped-span parity: cosine=%g max_rel=%g", cosine, maxRel)
	}
	ResetQ4K()
	ResetQ4K()

	for name, bad := range map[string]func() *Q4KWeight{
		"unaligned base":   func() *Q4KWeight { return UploadQ4KMappedSpan(span[1:], offset, out, in) },
		"unaligned span":   func() *Q4KWeight { return UploadQ4KMappedSpan(span[:page-1], offset, out, in) },
		"unaligned offset": func() *Q4KWeight { return UploadQ4KMappedSpan(span, 1, out, in) },
		"short range":      func() *Q4KWeight { return UploadQ4KMappedSpan(span, page-32, out, in) },
	} {
		t.Run(name, func(t *testing.T) {
			if got := bad(); got != nil {
				t.Fatalf("invalid mapped span accepted: %#v", got)
			}
		})
	}
}
