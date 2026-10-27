package model

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=30ms
func TestKQuantInt8PrefillPanelsMatchTokenWorkers(t *testing.T) {
	restoreKQuantSDOT(t)
	setKQuantSDOTForTest(true)
	const out, in, P = 9, 512, kQuantInt8PrefillRows + 3
	for _, kind := range []kQuantKind{kindQ5K, kindQ6K, kindQ2K} {
		t.Run(kind.String(), func(t *testing.T) {
			nblk := in / kind.blockWeights()
			raw := make([]byte, out*nblk*kind.blockBytes())
			lcgBytes(raw, 0x14045)
			pinResidentQuantScales(raw, out, nblk, kind)
			for off := 0; off < len(raw); off += kind.blockBytes() {
				if kind == kindQ5K {
					binary.LittleEndian.PutUint16(raw[off+2:], 0x3800)
				} else if kind == kindQ2K {
					binary.LittleEndian.PutUint16(raw[off+82:], 0x3800)
				}
			}
			qt := quantizeKQuantFromRaw(raw, out, in, kind)
			X := make([]float32, P*in)
			for tok := 0; tok < P; tok++ {
				if tok == 3 || tok == kQuantInt8PrefillRows {
					continue // zero rows also exercise clearing the reused panel
				}
				for i := 0; i < in; i++ {
					X[tok*in+i] = float32((i*13+tok*7)%37-18) / 7
				}
			}
			rawBefore := append([]byte(nil), qt.raw...)
			xBefore := append([]float32(nil), X...)
			want := make([]float32, P*out)
			for tok := 0; tok < P; tok++ {
				kQuantMatRowsIntoWorkers(qt, X[tok*in:(tok+1)*in], want[tok*out:(tok+1)*out], 2)
			}

			priorHook := q8PanelBuildHook
			var built []int
			q8PanelBuildHook = func(rows, width int) {
				if width != in {
					t.Errorf("panel width=%d, want %d", width, in)
				}
				built = append(built, rows)
			}
			t.Cleanup(func() { q8PanelBuildHook = priorHook })
			const sentinel = float32(-1234.5)
			guarded := make([]float32, P*out+2)
			for i := range guarded {
				guarded[i] = sentinel
			}
			got := guarded[1 : len(guarded)-1]
			kQuantMatRowsIntoBatch(qt, X, P, got)
			if !slices.Equal(built, []int{kQuantInt8PrefillRows, 3}) {
				t.Fatalf("ordinary prefill panel rows=%v, want bounded rollover", built)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("output[%d]=%v, per-token workers=%v", i, got[i], want[i])
				}
			}
			if guarded[0] != sentinel || guarded[len(guarded)-1] != sentinel {
				t.Fatal("prefill panel wrote outside its output")
			}
			if !slices.Equal(qt.raw, rawBefore) || !slices.Equal(X, xBefore) {
				t.Fatal("prefill panel modified immutable weights or activations")
			}
			// Refusal leaves ownership and destination contents with the old path.
			refused := *qt
			refused.w3MLP = true
			if kQuantMatRowsIntoBatchInt8(&refused, X, P, got, 2) || !slices.Equal(got, want) {
				t.Fatal("W3-tagged route must decline without output mutation")
			}
			refused.w3MLP, refused.kind = false, kindIQ4NL
			if kQuantMatRowsIntoBatchInt8(&refused, X, P, got, 2) || !slices.Equal(got, want) {
				t.Fatal("unsupported format must decline without output mutation")
			}
		})
	}
}
