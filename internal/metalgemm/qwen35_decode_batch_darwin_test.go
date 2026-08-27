//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"testing"
)

func TestQwen35DecodeBatchIndependentStateSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	for _, batch := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d", batch), func(t *testing.T) {
			const hidden = 32
			geometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 8, ValueHeadDim: 8, ConvKernel: 3}
			convDim, valueDim := geometry.convDim(), geometry.valueDim()
			weights := Qwen35DecodeWeights{
				InQKV: qwen35BatchQ8Weight(t, convDim, hidden, 11),
				InZ:   qwen35BatchQ8Weight(t, valueDim, hidden, 13),
				InB:   qwen35BatchQ8Weight(t, geometry.NumValueHeads, hidden, 17),
				InA:   qwen35BatchQ8Weight(t, geometry.NumValueHeads, hidden, 19),
				Out:   qwen35BatchQ8Weight(t, hidden, valueDim, 23),
			}
			for _, w := range []*Q8Weight{weights.InQKV, weights.InZ, weights.InB, weights.InA, weights.Out} {
				defer w.Release()
			}
			panel := qwen35BatchPanel(geometry)
			inputs := make([][]float32, batch)
			batchStates, serialStates := make([]*GDNState, batch), make([]*GDNState, batch)
			for row := 0; row < batch; row++ {
				inputs[row] = qwen35BatchInput(hidden, row+1)
				var err error
				batchStates[row], err = NewGDNState(geometry)
				if err != nil {
					t.Fatal(err)
				}
				serialStates[row], err = NewGDNState(geometry)
				if err != nil {
					t.Fatal(err)
				}
				defer batchStates[row].Close()
				defer serialStates[row].Close()
				conv, recurrent := qwen35BatchSeed(geometry, row+3)
				if err := batchStates[row].Seed(conv, recurrent); err != nil {
					t.Fatal(err)
				}
				if err := serialStates[row].Seed(conv, recurrent); err != nil {
					t.Fatal(err)
				}
			}
			want := make([][]float32, batch)
			for row := range want {
				var accepted bool
				var err error
				want[row], _, accepted, err = RunQwen35Decode(Qwen35DecodeRequest{Input: inputs[row], Weights: weights, State: serialStates[row], Panel: panel})
				if err != nil || !accepted {
					t.Fatalf("serial row %d: accepted=%v err=%v", row, accepted, err)
				}
			}
			flat := make([]float32, 0, batch*hidden)
			for _, row := range inputs {
				flat = append(flat, row...)
			}
			got, receipt, accepted, err := RunQwen35DecodeBatch(Qwen35DecodeBatchRequest{Input: flat, Weights: weights, States: batchStates, Panel: panel})
			if err != nil || !accepted {
				t.Fatalf("batch: accepted=%v err=%v", accepted, err)
			}
			if receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 || receipt.ProjectionDispatches != 5 || receipt.InputUploads != 1 || receipt.FinalReadbacks != 1 || receipt.IntermediateReadbacks != 0 || receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 {
				t.Fatalf("receipt=%+v", receipt)
			}
			if len(got) != batch {
				t.Fatalf("output rows=%d want=%d", len(got), batch)
			}
			for row := range got {
				qwen35BatchClose(t, "output", got[row], want[row])
				gc, gr, err := batchStates[row].Snapshot()
				if err != nil {
					t.Fatal(err)
				}
				wc, wr, err := serialStates[row].Snapshot()
				if err != nil {
					t.Fatal(err)
				}
				qwen35BatchClose(t, "conv", gc, wc)
				qwen35BatchClose(t, "recurrent", gr, wr)
			}
		})
	}
}

func qwen35BatchQ8Weight(t *testing.T, out, in, seed int) *Q8Weight {
	t.Helper()
	codes := make([]int8, out*in)
	scales := make([]float32, out*(in/32))
	for i := range codes {
		codes[i] = int8((i+seed)%15) - 7
	}
	for i := range scales {
		scales[i] = .004 + float32((i+seed)%9)*.001
	}
	w := UploadQ8(codes, scales, out, in)
	if w == nil {
		t.Fatal("UploadQ8 returned nil")
	}
	return w
}
func qwen35BatchPanel(g GDNGeometry) GDNPanel {
	p := GDNPanel{Conv1D: make([]float32, g.convDim()*g.ConvKernel), ALog: make([]float32, g.NumValueHeads), DTBias: make([]float32, g.NumValueHeads), Norm: make([]float32, g.ValueHeadDim), RMSNormEpsilon: 1e-6}
	for i := range p.Conv1D {
		p.Conv1D[i] = .01 * float32((i%7)-3)
	}
	for i := range p.ALog {
		p.ALog[i] = -.7
		p.DTBias[i] = .2
	}
	for i := range p.Norm {
		p.Norm[i] = 1
	}
	return p
}
func qwen35BatchInput(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i+seed))) * .2
	}
	return x
}
func qwen35BatchSeed(g GDNGeometry, seed int) ([]float32, []float32) {
	c := make([]float32, g.convDim()*(g.ConvKernel-1))
	r := make([]float32, g.NumValueHeads*g.ValueHeadDim*g.KeyHeadDim)
	for i := range c {
		c[i] = .01 * float32((i+seed)%11-5)
	}
	for i := range r {
		r[i] = .005 * float32((i+seed)%13-6)
	}
	return c, r
}
func qwen35BatchClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want=%d", name, len(got), len(want))
	}
	var dot, gg, ww, max float64
	for i := range got {
		a, b := float64(got[i]), float64(want[i])
		dot += a * b
		gg += a * a
		ww += b * b
		d := math.Abs(a - b)
		if d > max {
			max = d
		}
	}
	cos := dot / (math.Sqrt(gg*ww) + 1e-30)
	if cos < .999999 || max > .0001 {
		t.Fatalf("%s cosine=%g maxabs=%g", name, cos, max)
	}
}
