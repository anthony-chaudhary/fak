package model

import (
	"bytes"
	"errors"
	"math"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41DeepInferIndexMXFP4Format(t *testing.T) {
	t.Parallel()
	// Literal source-derived bytes, not an encode/decode self-oracle or a
	// captured CUDA witness. Each scale group has a different exponent.
	values := make([]float32, 128)
	values[0], values[1], values[32], values[64], values[65] = 6, .75, .75, 7, .75
	before := append([]float32(nil), values...)
	var want V41DeepInferIndexMXFP4Row
	want.Payload[0], want.Payload[16], want.Payload[32] = 0x27, 0x07, 0x16
	want.Scales = [4]byte{0x7f, 0x7c, 0x80, 0x01}
	row, err := EncodeV41DeepInferIndexMXFP4Row(values)
	if err != nil || row != want {
		t.Fatalf("row=%+v want=%+v err=%v", row, want, err)
	}
	decoded, err := want.Decode()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range decoded {
		wantValue := before[i]
		if i == 1 || i == 65 {
			wantValue = 1
		} else if i == 64 {
			wantValue = 8
		}
		if math.Float32bits(v) != math.Float32bits(wantValue) || values[i] != before[i] {
			t.Fatalf("value %d decoded=%g want=%g input=%g", i, v, wantValue, values[i])
		}
	}
	zero, err := EncodeV41DeepInferIndexMXFP4Row(make([]float32, 128))
	if err != nil || zero.Scales != [4]byte{1, 1, 1, 1} || zero.Payload != [64]byte{} {
		t.Fatalf("real zero row=%+v err=%v", zero, err)
	}
	rows := make([]V41DeepInferIndexMXFP4Row, 33)
	for i := range rows {
		rows[i] = zero
	}
	rows[1], rows[32] = want, want
	pages, err := PackV41DeepInferIndexMXFP4Pages(rows, 32)
	if err != nil {
		t.Fatal(err)
	}
	image := pages.Bytes()
	wantImage := make([]byte, 4352)
	for i := range rows {
		base, token := (i/32)*2176, i%32
		copy(wantImage[base+2048+4*token:], []byte{1, 1, 1, 1})
	}
	for _, offsets := range [][4]int{{64, 80, 96, 2052}, {2176, 2192, 2208, 4224}} {
		wantImage[offsets[0]], wantImage[offsets[1]], wantImage[offsets[2]] = 0x27, 0x07, 0x16
		copy(wantImage[offsets[3]:], []byte{0x7f, 0x7c, 0x80, 0x01})
	}
	if !bytes.Equal(image, wantImage) {
		t.Fatal("block-wide payload/scales or tail padding differ from literal page image")
	}
	imported, err := ReadV41DeepInferIndexMXFP4Pages(image, 33, 32)
	if err != nil {
		t.Fatal(err)
	}
	// All directions of ownership: packed input, returned bytes, imported
	// bytes and returned row arrays must not alias the stored page.
	rows[1] = zero
	image[64] = 0xff
	for _, p := range []*V41DeepInferIndexMXFP4Pages{pages, imported} {
		got, err := p.Row(0, 1)
		if err != nil || got != want {
			t.Fatalf("owned row=%+v err=%v", got, err)
		}
		got.Payload[0] = 0xff
		got, _ = p.Row(0, 1)
		if got != want {
			t.Fatal("row accessor aliases storage")
		}
		for _, loc := range [][2]int{{-1, 0}, {0, -1}, {0, 32}, {1, 1}, {2, 0}, {int(^uint(0) >> 1), 0}} {
			if _, err := p.Row(loc[0], loc[1]); !errors.Is(err, ErrV41DeepInferIndexMXFP4) {
				t.Fatalf("location %v err=%v", loc, err)
			}
		}
	}
	// Descriptor alignment is independent of device-specific admission.
	aligned, err := PackV41DeepInferIndexMXFP4Pages([]V41DeepInferIndexMXFP4Row{want}, 3)
	if err != nil || len(aligned.Bytes()) != 224 || aligned.Bytes()[192] != 0x7f {
		t.Fatalf("aligned descriptor err=%v", err)
	}
	for _, n := range []int{0, 127, 129} {
		if _, err := EncodeV41DeepInferIndexMXFP4Row(make([]float32, n)); !errors.Is(err, ErrV41DeepInferIndexMXFP4) {
			t.Fatalf("width %d err=%v", n, err)
		}
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32, math.Float32frombits(0x7f7f0000)} {
		bad := append([]float32(nil), before...)
		bad[127] = v
		if got, err := EncodeV41DeepInferIndexMXFP4Row(bad); !errors.Is(err, ErrV41DeepInferIndexMXFP4) || got != (V41DeepInferIndexMXFP4Row{}) || math.Float32bits(bad[127]) != math.Float32bits(v) {
			t.Fatalf("non-finite/overflow input %g returned %+v err=%v", v, got, err)
		}
	}
	for _, sf := range []byte{0, 255} {
		bad := want
		bad.Scales[3] = sf
		if p, err := PackV41DeepInferIndexMXFP4Pages([]V41DeepInferIndexMXFP4Row{bad}, 32); !errors.Is(err, ErrV41DeepInferIndexMXFP4) || p != nil {
			t.Fatalf("invalid scale %d published page: %v", sf, err)
		}
	}
	badImage := append([]byte(nil), wantImage...)
	badImage[2052] = 0
	for _, tc := range []struct {
		data        []byte
		rows, block int
	}{
		{wantImage[:len(wantImage)-1], 33, 32}, {append(append([]byte(nil), wantImage...), 0), 33, 32},
		{badImage, 33, 32}, {nil, -1, 32}, {nil, 0, 0}, {nil, 1, int(^uint(0) >> 1)}, {nil, int(^uint(0) >> 1), 1},
	} {
		before := append([]byte(nil), tc.data...)
		if p, err := ReadV41DeepInferIndexMXFP4Pages(tc.data, tc.rows, tc.block); !errors.Is(err, ErrV41DeepInferIndexMXFP4) || p != nil || !bytes.Equal(tc.data, before) {
			t.Fatalf("invalid page rows=%d block=%d err=%v", tc.rows, tc.block, err)
		}
	}
	// Literal RNE ties and signed-zero nibble, independent of old unwired oracle.
	ties := make([]float32, 128)
	copy(ties, []float32{6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, .5, 3})
	tieRow, err := EncodeV41DeepInferIndexMXFP4Row(ties)
	wantTies := [64]byte{0x07, 0x22, 0x44, 0x66, 0xa8, 0xca, 0xec, 0x1e, 0x05}
	if err != nil || tieRow.Payload != wantTies || tieRow.Scales != [4]byte{127, 1, 1, 1} {
		t.Fatalf("RNE ties=%+v err=%v", tieRow, err)
	}
	tieDecoded, err := tieRow.Decode()
	if err != nil || math.Float32bits(tieDecoded[8]) != 0x80000000 {
		t.Fatalf("negative zero decode=%08x err=%v", math.Float32bits(tieDecoded[8]), err)
	}
	lateOverflow := want
	lateOverflow.Scales[3], lateOverflow.Payload[63] = 254, 0x70
	if got, err := lateOverflow.Decode(); !errors.Is(err, ErrV41DeepInferIndexMXFP4) || got != ([128]float32{}) {
		t.Fatalf("late overflow published partial row: %v", err)
	}
	preround := make([]float32, 128)
	preround[0], preround[1] = 6+6*0x1p-12, 1.5
	preRow, err := EncodeV41DeepInferIndexMXFP4Row(preround)
	if err != nil || preRow.Payload[0] != 0x37 || preRow.Scales[0] != 127 {
		t.Fatalf("BF16 preround=%+v err=%v", preRow, err)
	}
	if v41DeepInferIndexE2M1(7) != 7 || v41DeepInferIndexE2M1(-7) != 15 {
		t.Fatal("finite E2M1 saturation")
	}
}
