package model

import (
	"math"
	"slices"
	"testing"
)

func convHistoryBitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func TestConvPrefillTailKeepsHistoryAndBackingOrder(t *testing.T) {
	for _, keep := range []int{1, 2, 3, 7} {
		for initial := 0; initial <= keep; initial++ {
			for _, rows := range []int{2*keep + 1, 2*keep + 2, 5*keep + 3} {
				const width = 33
				mixed := make([]float32, rows*width)
				for i := range mixed {
					mixed[i] = math.Float32frombits(uint32(i%127+1) | uint32(i%2)<<31)
				}
				for i, bits := range []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345} {
					mixed[len(mixed)-1-i] = math.Float32frombits(bits)
				}
				before := slices.Clone(mixed)
				got := &linearAttnLayerState{conv: make([][]float32, initial, keep+2)}
				want := &linearAttnLayerState{conv: make([][]float32, initial, keep+2)}
				for i := 0; i < initial; i++ {
					got.conv[i] = make([]float32, width, width+7)
					want.conv[i] = make([]float32, width, width+7)
				}
				oldGot, oldWant := slices.Clone(got.conv), slices.Clone(want.conv)
				for row := 0; row < rows; row++ {
					want.pushConvRow(mixed[row*width:(row+1)*width], keep)
				}
				if !got.pushConvPrefillTail(mixed, rows, width, keep) {
					t.Fatalf("admitted shape declined: keep%d initial%d rows%d", keep, initial, rows)
				}
				if len(got.conv) != len(want.conv) || cap(got.conv) != cap(want.conv) || !convHistoryBitsEqual(mixed, before) {
					t.Fatal("history shape/reservation or immutable input changed")
				}
				for i := range want.conv {
					if !convHistoryBitsEqual(got.conv[i], want.conv[i]) || cap(got.conv[i]) != cap(want.conv[i]) {
						t.Fatalf("history differs: keep%d initial%d rows%d slot%d", keep, initial, rows, i)
					}
					for j := range oldGot {
						if (&got.conv[i][0] == &oldGot[j][0]) != (&want.conv[i][0] == &oldWant[j][0]) {
							t.Fatal("changed original backing row permutation")
						}
					}
					for _, prior := range got.conv[:i] {
						if convPrefillRowsOverlap(got.conv[i], prior) {
							t.Fatal("published overlapping history rows")
						}
					}
				}
			}
		}
	}
}

func TestConvPrefillTailDeclinesBeforeWrites(t *testing.T) {
	const rows, width, keep = 12, 4, 3
	for _, name := range []string{"source alias", "history alias", "short history row", "oversized history", "short panel", "small panel", "zero width", "zero keep"} {
		mixed := make([]float32, rows*width)
		for i := range mixed {
			mixed[i] = float32(i + 1)
		}
		st := &linearAttnLayerState{conv: [][]float32{make([]float32, width), make([]float32, width)}}
		r, w, k := rows, width, keep
		switch name {
		case "source alias":
			st.conv[0] = mixed[width : 2*width]
		case "history alias":
			shared := make([]float32, width+1)
			st.conv[0], st.conv[1] = shared[:width], shared[1:]
		case "short history row":
			st.conv[0] = st.conv[0][:width-1]
		case "oversized history":
			st.conv = append(st.conv, make([]float32, width), make([]float32, width))
		case "short panel":
			mixed = mixed[:len(mixed)-1]
		case "small panel":
			r, mixed = keep+1, mixed[:(keep+1)*width]
		case "zero width":
			w = 0
		case "zero keep":
			k = 0
		}
		before := slices.Clone(mixed)
		oldRows := slices.Clone(st.conv)
		oldValues := make([][]float32, len(st.conv))
		for i, row := range st.conv {
			oldValues[i] = slices.Clone(row)
		}
		if st.pushConvPrefillTail(mixed, r, w, k) {
			t.Fatalf("accepted %s", name)
		}
		if !convHistoryBitsEqual(mixed, before) || len(st.conv) != len(oldRows) {
			t.Fatalf("decline mutated %s", name)
		}
		for i, row := range st.conv {
			if !convHistoryBitsEqual(row, oldValues[i]) || &row[0] != &oldRows[i][0] {
				t.Fatalf("decline changed history in %s", name)
			}
		}
	}
}

func TestConvPrefillTailMalformedFallbackKeepsPartialPanic(t *testing.T) {
	run := func(useBulk bool) (st *linearAttnLayerState, panicked bool) {
		st = &linearAttnLayerState{conv: [][]float32{make([]float32, 4)}}
		mixed := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
		defer func() { panicked = recover() != nil }()
		if !useBulk || !st.pushConvPrefillTail(mixed, 3, 4, 2) {
			for row := 0; row < 3; row++ {
				st.pushConvRow(mixed[row*4:(row+1)*4], 2)
			}
		}
		return
	}
	got, gotPanic := run(true)
	want, wantPanic := run(false)
	if !gotPanic || gotPanic != wantPanic || len(got.conv) != len(want.conv) {
		t.Fatal("changed malformed panel panic boundary")
	}
	for i := range got.conv {
		if !convHistoryBitsEqual(got.conv[i], want.conv[i]) {
			t.Fatal("changed history writes preceding malformed panel panic")
		}
	}
}
