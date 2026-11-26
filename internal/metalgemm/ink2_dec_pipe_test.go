//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

func ink2PipeTestMmap(n int) ([]byte, error) {
	return syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
}

func ink2PipeTestMunmap(b []byte) { _ = syscall.Munmap(b) }

// ink2PipeTestF16 widens IEEE half bits to float32.
func ink2PipeTestF16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp, frac := int(h>>10&0x1f), uint32(h&0x3ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		v := float32(math.Ldexp(float64(frac), -24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | frac<<13)
}

// ink2PipeTestQ5K is one Q5_K row the way GGML lays it out, with each value's
// d1*q - m1 rounded once.
func ink2PipeTestQ5K(raw []byte, cols int) []float32 {
	out := make([]float32, cols)
	for b := 0; b < cols/256; b++ {
		blk := raw[b*176 : (b+1)*176]
		d := ink2PipeTestF16(uint16(blk[0]) | uint16(blk[1])<<8)
		dmin := ink2PipeTestF16(uint16(blk[2]) | uint16(blk[3])<<8)
		sc, qh, ql := blk[4:16], blk[16:48], blk[48:176]
		for j := 0; j < 4; j++ {
			for hi := 0; hi < 2; hi++ {
				is := 2*j + hi
				var scale, m byte
				if is < 4 {
					scale, m = sc[is]&63, sc[is+4]&63
				} else {
					scale, m = sc[is+4]&0x0f|(sc[is-4]>>6)<<4, sc[is+4]>>4|(sc[is]>>6)<<4
				}
				d1, m1 := d*float32(scale), dmin*float32(m)
				for l := 0; l < 32; l++ {
					q := ql[32*j+l]
					if hi != 0 {
						q >>= 4
					} else {
						q &= 0x0f
					}
					if qh[l]>>is&1 != 0 {
						q += 16
					}
					out[b*256+64*j+32*hi+l] = float32(float64(d1)*float64(q) - float64(m1))
				}
			}
		}
	}
	return out
}

// ink2PipeTestHalf is a random finite half, subnormals included.
func ink2PipeTestHalf(rng *rand.Rand) (byte, byte) {
	for {
		h := uint16(rng.Intn(1 << 16))
		if h>>10&0x1f != 0x1f {
			return byte(h), byte(h >> 8)
		}
	}
}

// ink2PipeTestTable maps a random Q5_K table at an unaligned byte offset of a
// page-aligned anonymous mapping, the way a GGUF payload sits in its file.
func ink2PipeTestTable(t *testing.T, rows, cols, lead int) ([]byte, Ink2Weight) {
	t.Helper()
	rowBytes := cols / 256 * 176
	page := os.Getpagesize()
	size := (lead + rows*rowBytes + page - 1) &^ (page - 1)
	mapped, err := ink2PipeTestMmap(size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ink2PipeTestMunmap(mapped) })
	rng := rand.New(rand.NewSource(7))
	raw := mapped[lead : lead+rows*rowBytes]
	rng.Read(raw)
	for b := 0; b < len(raw)/176; b++ {
		raw[b*176], raw[b*176+1] = ink2PipeTestHalf(rng)
		raw[b*176+2], raw[b*176+3] = ink2PipeTestHalf(rng)
	}
	store, err := NewPackedGGUFStore(mapped)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)
	m, err := store.Matrix(lead, 13, rows, cols, 1)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.Ink2Weight(0)
	if err != nil {
		t.Fatal(err)
	}
	return raw, w
}

// fak-test:runtime integration est=2s lane=default
func TestInk2EmbedRowQ5KMatchesHostBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	const rows, cols = 600, 768
	raw, w := ink2PipeTestTable(t, rows, cols, 4*1001)
	rowBytes := cols / 256 * 176
	idx := ink2TestAlloc(t, d, rows)
	ids := make([]float32, rows)
	for i := range ids {
		ids[i] = math.Float32frombits(uint32(i))
	}
	if err := d.Write(idx, ids); err != nil {
		t.Fatal(err)
	}
	out := ink2TestAlloc(t, d, rows*(cols+1))
	for r := 0; r < rows; r++ {
		if err := d.EmbedRow(w, idx.At(r), 1000+r, out.At(r*(cols+1))); err != nil {
			t.Fatal(err)
		}
	}
	got, err := d.Read(out, rows*(cols+1))
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		want := ink2PipeTestQ5K(raw[r*rowBytes:(r+1)*rowBytes], cols)
		row := got[r*(cols+1) : (r+1)*(cols+1)]
		for i, v := range want {
			if math.Float32bits(row[i]) != math.Float32bits(v) {
				t.Fatalf("row %d value %d: got %v (%#x), want %v (%#x)", r, i, row[i], math.Float32bits(row[i]), v, math.Float32bits(v))
			}
		}
		if p := int32(math.Float32bits(row[cols])); p != int32(1000+r) {
			t.Fatalf("row %d position word %d, want %d", r, p, 1000+r)
		}
	}
	// an index past the table reads the last row rather than past the view
	if err := d.Write(idx, []float32{math.Float32frombits(rows + 5)}); err != nil {
		t.Fatal(err)
	}
	if err := d.EmbedRow(w, idx, 3, out); err != nil {
		t.Fatal(err)
	}
	last, err := d.Read(out, cols)
	if err != nil {
		t.Fatal(err)
	}
	want := ink2PipeTestQ5K(raw[(rows-1)*rowBytes:], cols)
	for i := range want {
		if math.Float32bits(last[i]) != math.Float32bits(want[i]) {
			t.Fatalf("clamped row value %d: got %v, want %v", i, last[i], want[i])
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2CopySpansCopiesWords(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(3))
	var src, dst []Ink2Span
	var want [][]uint32
	for i := 0; i < 19; i++ {
		n := 1 + rng.Intn(3000)
		s, o := ink2TestAlloc(t, d, n+7), ink2TestAlloc(t, d, n+9)
		words := make([]uint32, n)
		for j := range words {
			words[j] = rng.Uint32()
		}
		// NaN payloads and negative zero must survive a copy
		words[0] = 0x7fa00001
		words[n-1] = 0x80000000
		if err := d.Write(s.At(7), unsafe.Slice((*float32)(unsafe.Pointer(&words[0])), n)); err != nil {
			t.Fatal(err)
		}
		src, dst = append(src, Ink2Span{s.At(7), n}), append(dst, Ink2Span{o.At(9), n})
		want = append(want, words)
	}
	if err := d.CopySpans(src, dst); err != nil {
		t.Fatal(err)
	}
	for i, sp := range dst {
		got, err := d.ReadInts(sp.At, sp.N)
		if err != nil {
			t.Fatal(err)
		}
		for j, w := range want[i] {
			if uint32(got[j]) != w {
				t.Fatalf("span %d word %d: got %#x, want %#x", i, j, uint32(got[j]), w)
			}
		}
	}
	if err := d.CopySpans(src[:1], dst[:2]); err == nil {
		t.Fatal("mismatched span lists were accepted")
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2CommitOrdersLaterBuffersAndReadsEarlyOnes(t *testing.T) {
	d := ink2TestDevice(t)
	const n = 1 << 16
	a, b, c := ink2TestAlloc(t, d, n), ink2TestAlloc(t, d, n), ink2TestAlloc(t, d, n)
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i)
	}
	if err := d.Write(a, vals); err != nil {
		t.Fatal(err)
	}
	// b = a + a, committed; then c = b + b in a later buffer that must see b
	if err := d.Add(a, a, b, n); err != nil {
		t.Fatal(err)
	}
	first, err := d.Commit()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Add(b, b, c, n); err != nil {
		t.Fatal(err)
	}
	second, err := d.Commit()
	if err != nil {
		t.Fatal(err)
	}
	if second <= first {
		t.Fatalf("tickets %d then %d", first, second)
	}
	got, err := d.ReadIntsAfter(first, b, n)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if v := math.Float32frombits(uint32(got[i])); v != 2*float32(i) {
			t.Fatalf("b[%d] = %v after its ticket, want %v", i, v, 2*float32(i))
		}
	}
	cs, err := d.Read(c, n)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range cs {
		if v != 4*float32(i) {
			t.Fatalf("c[%d] = %v, want %v", i, v, 4*float32(i))
		}
	}
	// a ticket already waited for, or nothing committed, returns at once
	if _, err := d.ReadIntsAfter(second, c, 1); err != nil {
		t.Fatal(err)
	}
	if d.pending != 0 {
		t.Fatalf("device still pending %d dispatches after every buffer completed", d.pending)
	}
}
