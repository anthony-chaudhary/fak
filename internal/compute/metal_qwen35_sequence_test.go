//go:build darwin && arm64 && cgo

package compute

import (
	"errors"
	"math"
	"testing"
)

func TestMetalQwen35SequenceGraphUsesOneTerminalWait(t *testing.T) {
	c := metalOrSkip(t)
	const tokens, hidden, intermediate = 2, 4, 3
	weight := mtlMkResident(c, []int{hidden}, []float32{1, 1, 1, 1})
	x := mtlMkResident(c, []int{tokens, hidden}, []float32{1, -2, 3, -4, -1, 2, -3, 4})
	gateW := mtlMkResident(c, []int{intermediate, hidden}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})
	upW := mtlMkResident(c, []int{intermediate, hidden}, []float32{
		2, 0, 0, 0,
		0, 2, 0, 0,
		0, 0, 2, 0,
	})
	norm, _ := c.devTr([]int{tokens, hidden}, F32)
	gate, _ := c.devTr([]int{tokens, intermediate}, F32)
	up, _ := c.devTr([]int{tokens, intermediate}, F32)
	activated, _ := c.devTr([]int{tokens, intermediate}, F32)
	residual, _ := c.devTr([]int{tokens, intermediate}, F32)
	defer func() {
		for _, tensor := range []Tensor{weight, x, gateW, upW, norm, gate, up, activated, residual} {
			c.Free(tensor)
		}
		c.Recycle()
	}()

	metalMu.Lock()
	graph, err := beginMetalQwen35SequenceGraph()
	if err == nil {
		err = graph.rmsNorm(c.mb(x), c.mb(weight), c.mb(norm), tokens, hidden, 1e-5)
	}
	if err == nil {
		err = graph.projection(c.mb(gateW), c.mb(norm), c.mb(gate), intermediate, hidden, tokens)
	}
	if err == nil {
		err = graph.projection(c.mb(upW), c.mb(norm), c.mb(up), intermediate, hidden, tokens)
	}
	if err == nil {
		err = graph.swiGLU(c.mb(gate), c.mb(up), c.mb(activated), tokens*intermediate)
	}
	if err == nil {
		err = graph.residual(c.mb(residual), c.mb(activated), tokens*intermediate)
	}
	var receipt metalCommandReceipt
	if err == nil {
		receipt, err = graph.finish()
	} else if graph != nil {
		err = graph.fail(err)
	}
	metalMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 5 {
		t.Fatalf("receipt=%+v", receipt)
	}

	got := c.Read(residual)
	for row, input := range [][]float32{{1, -2, 3, -4}, {-1, 2, -3, 4}} {
		scale := float32(1 / math.Sqrt(7.5+1e-5))
		for col := 0; col < intermediate; col++ {
			gateValue := input[col] * scale
			upValue := 2 * input[col] * scale
			want := upValue * (gateValue / (1 + float32(math.Exp(float64(-gateValue)))))
			if diff := float64(got[row*intermediate+col] - want); math.Abs(diff) > 1e-5 {
				t.Fatalf("row=%d col=%d got=%g want=%g", row, col, got[row*intermediate+col], want)
			}
		}
	}
}

func TestMetalQwen35SequenceGraphRefusesReuseAfterFinish(t *testing.T) {
	metalOrSkip(t)
	metalMu.Lock()
	graph, err := beginMetalQwen35SequenceGraph()
	if err == nil {
		_, err = graph.finish()
	}
	if err == nil {
		err = graph.residual(nil, nil, 0)
	}
	metalMu.Unlock()
	if !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("reuse error=%v want %v", err, errMetalOwnerTerminal)
	}
}
