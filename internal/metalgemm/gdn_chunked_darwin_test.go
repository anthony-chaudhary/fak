//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"testing"
)

// chunkedGDNFixture exercises grouped heads with independent beta/decay values.
// In particular, strong decay must remain finite even when cumulative products
// underflow; a division by such a product is not a valid recurrence rewrite.
func chunkedGDNFixture(g oracleGDNGeometry, tokens int) oracleGDNPanel {
	p := oracleGDNFixture(g, tokens, .73)
	for h := 0; h < g.nV; h++ {
		p.aLog[h] = []float32{-5, -1.5, 1, 3}[h%4]
		p.dtBias[h] = float32(h-2) * .7
		for token := 0; token < tokens; token++ {
			i := token*g.nV + h
			p.b[i] = []float32{-12, -.8, 1.7, 12}[(token+h)%4]
			p.a[i] = []float32{-8, -.3, 2, 12}[(token/3+h)%4]
		}
	}
	return p
}

func chunkedGDNSeed(g oracleGDNGeometry) *oracleGDNState {
	s := newOracleGDNState(g)
	for i := range s.conv {
		s.conv[i] = .13 * float32(math.Sin(float64(i)*.17+.9))
	}
	for i := range s.recurrent {
		s.recurrent[i] = .02 * float32(math.Sin(float64(i)*.23+.5))
	}
	return s
}

func chunkedGDNPanelSlice(g oracleGDNGeometry, p oracleGDNPanel, first, last int) GDNPanel {
	n := nativeGDNPanel(p)
	n.Tokens = last - first
	n.Mixed = n.Mixed[first*g.convDim() : last*g.convDim()]
	n.Z = n.Z[first*g.valueDim() : last*g.valueDim()]
	n.B = n.B[first*g.nV : last*g.nV]
	n.A = n.A[first*g.nV : last*g.nV]
	return n
}

// Explicit finiteness checks prevent NaNs from bypassing cosine/max-abs tests.
func requireChunkedGDNParity(t *testing.T, name string, want, got []float32) {
	t.Helper()
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s[%d]=%g is not finite", name, i, v)
		}
	}
	requireGDNParity(t, name, want, got)
}

// fak-test:runtime medium est=2s lane=default
func TestGDNChunkedPrefillOutputAndStateParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer SetGDNForceBaseline(false)
	SetGDNForceChunkedOff(false)
	defer SetGDNForceChunkedOff(false)
	// Full production head dimensions, four value heads sharing two key heads.
	// One owner's persistent state is only 264 KiB, avoiding model allocation.
	g := oracleGDNGeometry{nK: 2, nV: 4, kHd: 128, vHd: 128, kernel: 3}
	for _, tokens := range []int{15, 16, 17, 64, 128, 129} {
		t.Run(fmt.Sprintf("T%d", tokens), func(t *testing.T) {
			p := chunkedGDNFixture(g, tokens)
			oracle := chunkedGDNSeed(g)
			want := oracleGDNRun(g, p, oracle)
			for _, serial := range []bool{true, false} {
				SetGDNForceBaseline(serial)
				state, err := NewGDNState(nativeGDNGeometry(g))
				if err != nil {
					t.Fatal(err)
				}
				defer state.Close()
				seed := chunkedGDNSeed(g)
				if err := state.Seed(seed.conv, seed.recurrent); err != nil {
					t.Fatal(err)
				}
				var got []float32
				for first := 0; first < tokens; first += GDNMaxPanelTokens {
					last := min(first+GDNMaxPanelTokens, tokens)
					out, accounting, accepted, err := state.Run(chunkedGDNPanelSlice(g, p, first, last))
					if err != nil || !accepted {
						t.Fatalf("serial=%v panel[%d:%d] accepted=%v err=%v", serial, first, last, accepted, err)
					}
					requireGDNAccounting(t, accounting, 1)
					wantChunks := 0
					if !serial && last-first >= 16 {
						wantChunks = (last - first + 7) / 8
					}
					if accounting.ChunkRecurrenceSteps != wantChunks {
						t.Fatalf("serial=%v panel[%d:%d] chunk dispatches=%d, want %d", serial, first, last, accounting.ChunkRecurrenceSteps, wantChunks)
					}
					t.Logf("serial=%v panel[%d:%d] completed chunk dispatches=%d", serial, first, last, accounting.ChunkRecurrenceSteps)
					got = append(got, out...)
				}
				label := fmt.Sprintf("serial=%v", serial)
				requireChunkedGDNParity(t, label+" output", want, got)
				conv, recurrent, err := state.Snapshot()
				if err != nil {
					t.Fatal(err)
				}
				requireChunkedGDNParity(t, label+" conv", oracle.conv, conv)
				requireChunkedGDNParity(t, label+" recurrent", oracle.recurrent, recurrent)
				state.Close()
			}
		})
	}
}

// BenchmarkGDNChunkedPrefill includes identical fused convolution, normalization,
// staging and terminal waits in both modes. Run with -benchtime=10x to bound GPU
// occupancy; this measures the native call rather than a model/token-rate claim.
func BenchmarkGDNChunkedPrefill(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	defer SetGDNForceBaseline(false)
	defer SetGDNForceChunkedOff(false)
	for _, heads := range []int{2, 16} {
		g := oracleGDNGeometry{nK: heads, nV: 2 * heads, kHd: 128, vHd: 128, kernel: 3}
		p := nativeGDNPanel(oracleGDNFixture(g, 64, .73))
		for _, mode := range []string{"serial", "packed", "chunked"} {
			b.Run(fmt.Sprintf("K%d_V%d_%s_T64", g.nK, g.nV, mode), func(b *testing.B) {
				SetGDNForceBaseline(mode == "serial")
				SetGDNForceChunkedOff(mode == "packed")
				state, err := NewGDNState(nativeGDNGeometry(g))
				if err != nil {
					b.Fatal(err)
				}
				defer state.Close()
				if _, accounting, accepted, err := state.Run(p); err != nil || !accepted {
					b.Fatalf("warmup accepted=%v err=%v", accepted, err)
				} else if (accounting.ChunkRecurrenceSteps > 0) != (mode == "chunked") {
					b.Fatalf("mode=%s completed chunks=%d", mode, accounting.ChunkRecurrenceSteps)
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, _, accepted, err := state.Run(p); err != nil || !accepted {
						b.Fatalf("run accepted=%v err=%v", accepted, err)
					}
				}
			})
		}
	}
}
