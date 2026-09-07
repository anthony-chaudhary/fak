package bench

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

func TestQuantTraceGoldenAndRejections(t *testing.T) {
	// Independent golden: d=1, dmin=0.5, scale=min=1 in all subblocks.
	// q4_K packs the last four scale/min pairs in low/high nibbles of bytes 12..15.
	raw := make([]byte, 3*144)
	for b := 0; b < 3; b++ {
		r := raw[b*144 : (b+1)*144]
		binary.LittleEndian.PutUint16(r, 0x3c00)
		binary.LittleEndian.PutUint16(r[2:], 0x3800)
		for i := 4; i < 12; i++ {
			r[i] = 1
		}
		for i := 12; i < 16; i++ {
			r[i] = 0x11
		}
		for i := 16; i < 144; i++ {
			r[i] = 0xa3
		}
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	indices := []int{0, 31, 32, 63, 64, 255, 256, 512, 767}
	r, err := TraceQ4K(raw, 3, 256, 2, x, indices)
	if err != nil {
		t.Fatal(err)
	}
	if !r.ExactRoundTrip || !r.ExactContraction || r.MaxContractionError != 0 || r.PackedBytes != 432 || r.ExpandedFloatBytes != 3072 || r.OutputBytes != 12 || r.OriginalFloatErrorMeasured {
		t.Fatalf("bad receipt: %+v", r)
	}
	packed, _ := RepackQ4KInterleaved(raw, 3, 256, 2)
	for _, s := range r.Samples {
		want := float32(2.5)
		code := byte(3)
		if s.Column%64 >= 32 {
			want = 9.5
			code = 10
		}
		if s.Value != want || s.Code != code || (packed[s.RepackedByte]>>s.NibbleShift)&15 != code || s.ScaleCode != 1 || s.MinCode != 1 {
			t.Fatalf("bad sample: %+v", s)
		}
	}
	if r.Samples[0].SourceByte != 16 || r.Samples[0].RepackedByte != 32 || r.Samples[2].SourceByte != 16 || r.Samples[2].NibbleShift != 4 || r.Samples[6].RepackedByte != 33 || r.Samples[7].RepackedByte != 304 {
		t.Fatalf("wrong offsets: %+v", r.Samples)
	}
	// Dot = 128*2.5 + 128*9.5 = 1536 for every row.
	if len(r.OutputPrefix) != 3 || r.OutputPrefix[0] != 1536 || r.OutputPrefix[2] != 1536 || r.OutputSHA256 != quantTraceFloatHash([]float32{1536, 1536, 1536}) {
		t.Fatal("golden contraction mismatch")
	}
	r2, _ := TraceQ4K(raw, 3, 256, 2, x, indices)
	a, _ := json.Marshal(r)
	b, _ := json.Marshal(r2)
	if string(a) != string(b) {
		t.Fatal("nondeterministic receipt")
	}
	for _, tc := range []struct {
		rows, cols, width int
		raw               []byte
		sample            []int
	}{{math.MaxInt, 256, 2, raw, nil}, {3, math.MaxInt, 2, raw, nil}, {3, 256, 0, raw, nil}, {3, 256, 257, raw, nil}, {3, 256, 2, raw[:431], nil}, {3, 256, 2, raw, []int{-1}}, {3, 256, 2, raw, []int{768}}} {
		if _, err := TraceQ4K(tc.raw, tc.rows, tc.cols, tc.width, x, tc.sample); err == nil {
			t.Fatalf("accepted malformed input: %+v", tc)
		}
	}
	bad := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint16(bad, 0x7c00)
	if _, err := TraceQ4K(bad, 3, 256, 2, x, nil); err == nil {
		t.Fatal("accepted infinite scale")
	}
	x[0] = float32(math.NaN())
	if _, err := TraceQ4K(raw, 3, 256, 2, x, nil); err == nil {
		t.Fatal("accepted nonfinite x")
	}
}
