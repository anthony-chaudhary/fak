//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
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
	payload := span[offset : offset+out*BlockQ4KBytes]
	for row := 0; row < out; row++ {
		block := payload[row*BlockQ4KBytes : (row+1)*BlockQ4KBytes]
		putF16(block[0:2], 1)
		putF16(block[2:4], 0)
		for i := 16; i < len(block); i++ {
			block[i] = 0x11
		}
	}
	if err := syscall.Mprotect(span, syscall.PROT_READ); err != nil {
		t.Fatal(err)
	}

	w := UploadQ4KMappedSpan(span, offset, out, in)
	if w == nil || !w.NoCopy() {
		t.Fatalf("mapped span upload = %#v, want no-copy weight", w)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i%7) / 7
	}
	want := float32(0)
	for _, v := range x {
		want += v
	}
	got := w.GEMV(x)
	if len(got) != out {
		t.Fatalf("GEMV len=%d, want %d", len(got), out)
	}
	for i, v := range got {
		if d := math.Abs(float64(v - want)); d > 0.05 {
			t.Fatalf("GEMV[%d]=%g, want %g (diff %g)", i, v, want, d)
		}
	}
	group := GEMVGroup([]*Q4KWeight{w}, x)
	if len(group) != 1 || len(group[0]) != out {
		t.Fatalf("group shape = %v", group)
	}
	for i, v := range group[0] {
		if d := math.Abs(float64(v - want)); d > 0.05 {
			t.Fatalf("group[%d]=%g, want %g (diff %g)", i, v, want, d)
		}
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
