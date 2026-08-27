//go:build darwin && arm64 && cgo

package metalgemm

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

func TestQwen35DecodeBatchIndependentStateSingleFence(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	geometry := GDNGeometry{NumKeyHeads: 16, NumValueHeads: 48, KeyHeadDim: 128, ValueHeadDim: 128, ConvKernel: 4}
	const hidden = 5120
	weights := newQwen35DecodeBatchWeights(t, hidden, geometry)
	defer releaseQwen35DecodeBatchWeights(weights)
	t.Logf("exact Qwen3.8 geometry: hidden=%d nK=%d nV=%d kHd=%d vHd=%d convKernel=%d valueHeadRepeat=%d convHistoryRows=%d",
		hidden, geometry.NumKeyHeads, geometry.NumValueHeads, geometry.KeyHeadDim, geometry.ValueHeadDim,
		geometry.ConvKernel, geometry.NumValueHeads/geometry.NumKeyHeads, geometry.ConvKernel-1)

	for _, batch := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d", batch), func(t *testing.T) {
			baseline := GDNLiveBufferCount()
			serialStates := newQwen35DecodeBatchStates(t, geometry, batch)
			batchStates := newQwen35DecodeBatchStates(t, geometry, batch)
			defer closeQwen35DecodeBatchStates(serialStates)
			defer closeQwen35DecodeBatchStates(batchStates)
			seedQwen35DecodeBatchStatePairs(t, geometry, serialStates, batchStates)
			requireQwen35DecodeBatchDifferentNonzeroState(t, batchStates)
			input := qwen35DecodeBatchInput(batch, hidden)
			panel := qwen35DecodeBatchPanel(geometry)

			serialOutputs := make([][]float32, batch)
			serialConv := make([][]float32, batch)
			serialRecurrent := make([][]float32, batch)
			for row := 0; row < batch; row++ {
				got, receipt, accepted, err := RunQwen35Decode(Qwen35DecodeRequest{
					Input: input[row*hidden : (row+1)*hidden], Weights: weights,
					State: serialStates[row], Panel: panel,
				})
				if err != nil || !accepted {
					t.Fatalf("serial row %d: accepted=%v err=%v", row, accepted, err)
				}
				if receipt.ProjectionDispatches != 5 || receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 {
					t.Fatalf("serial row %d receipt=%+v", row, receipt)
				}
				serialOutputs[row] = append([]float32(nil), got...)
				serialConv[row], serialRecurrent[row], err = serialStates[row].Snapshot()
				if err != nil {
					t.Fatalf("serial row %d snapshot: %v", row, err)
				}
				serialStates[row].Close()
			}
			if got := GDNLiveBufferCount(); got != baseline+2*batch {
				t.Fatalf("B=%d live batch buffers before submit=%d, want %d", batch, got, baseline+2*batch)
			}
			requireQwen35DecodeBatchDistinctOwners(t, batchStates)

			outputs, receipt, accepted, err := RunQwen35DecodeBatch(Qwen35DecodeBatchRequest{
				Input: input, Weights: weights, States: batchStates, Panel: panel,
			})
			if err != nil || !accepted {
				t.Fatalf("B=%d batch: accepted=%v err=%v", batch, accepted, err)
			}
			requireQwen35DecodeBatchReceipt(t, receipt, batch, 1)
			for row := 0; row < batch; row++ {
				requireGDNParity(t, fmt.Sprintf("B=%d row=%d output", batch, row), serialOutputs[row], outputs[row])
				conv, recurrent, snapshotErr := batchStates[row].Snapshot()
				if snapshotErr != nil {
					t.Fatalf("B=%d row=%d snapshot: %v", batch, row, snapshotErr)
				}
				requireGDNParity(t, fmt.Sprintf("B=%d row=%d convolution", batch, row), serialConv[row], conv)
				requireGDNParity(t, fmt.Sprintf("B=%d row=%d recurrent", batch, row), serialRecurrent[row], recurrent)
			}
			if qwen35DecodeBatchMaxAbs(outputs[0], outputs[batch-1]) == 0 {
				t.Fatalf("B=%d independent rows produced identical outputs", batch)
			}
			if got := GDNLiveBufferCount(); got != baseline+2*batch {
				t.Fatalf("B=%d accepted batch changed owner count=%d, want %d", batch, got, baseline+2*batch)
			}
			for _, state := range batchStates {
				state.Close()
				state.Close()
			}
			if got := GDNLiveBufferCount(); got != baseline {
				t.Fatalf("B=%d exact-once owner cleanup left buffers=%d, baseline=%d", batch, got, baseline)
			}
		})
	}

	t.Run("reversed overlapping owners cannot deadlock", func(t *testing.T) {
		// Keep the deadlock witness small: exact Qwen3.8 multihead/offset parity is
		// covered above, while this subtest isolates acquisition ordering.
		smallGeometry := GDNGeometry{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 32, ValueHeadDim: 32, ConvKernel: 2}
		const smallHidden = 32
		baseline := GDNLiveBufferCount()
		smallWeights := newQwen35DecodeBatchWeights(t, smallHidden, smallGeometry)
		states := newQwen35DecodeBatchStates(t, smallGeometry, 2)
		for row, state := range states {
			seedQwen35DecodeBatchState(t, smallGeometry, state, row+1)
		}
		input := qwen35DecodeBatchInput(2, smallHidden)
		reversedInput := append(append([]float32(nil), input[smallHidden:]...), input[:smallHidden]...)

		// Before canonical acquisition, both calls reach this hook while holding
		// different first owners, then each waits forever for the other's owner.
		// With canonical acquisition, the second call cannot reach the hook until
		// the first terminal fence releases the common lowest owner.
		var hookMu sync.Mutex
		arrivals, windowClosed := 0, false
		releaseWindow := make(chan struct{})
		afterFirstRetain := func() {
			hookMu.Lock()
			if windowClosed {
				hookMu.Unlock()
				return
			}
			arrivals++
			if arrivals == 2 {
				windowClosed = true
				close(releaseWindow)
				hookMu.Unlock()
				return
			}
			hookMu.Unlock()
			select {
			case <-releaseWindow:
			case <-time.After(2 * time.Second):
				hookMu.Lock()
				if !windowClosed {
					windowClosed = true
					close(releaseWindow)
				}
				hookMu.Unlock()
			}
		}

		type callResult struct {
			receipt  Qwen35DecodeBatchReceipt
			accepted bool
			err      error
		}
		start := make(chan struct{})
		results := make(chan callResult, 2)
		launch := func(req Qwen35DecodeBatchRequest) {
			go func() {
				<-start
				_, receipt, accepted, err := RunQwen35DecodeBatch(req)
				results <- callResult{receipt: receipt, accepted: accepted, err: err}
			}()
		}
		panel := qwen35DecodeBatchPanel(smallGeometry)
		launch(Qwen35DecodeBatchRequest{Input: input, Weights: smallWeights, States: []*GDNState{states[0], states[1]}, Panel: panel, afterFirstOwnerRetainForTest: afterFirstRetain})
		launch(Qwen35DecodeBatchRequest{Input: reversedInput, Weights: smallWeights, States: []*GDNState{states[1], states[0]}, Panel: panel, afterFirstOwnerRetainForTest: afterFirstRetain})
		close(start)
		for call := 0; call < 2; call++ {
			select {
			case result := <-results:
				if result.err != nil || !result.accepted {
					t.Fatalf("overlapping call %d: accepted=%v receipt=%+v err=%v", call, result.accepted, result.receipt, result.err)
				}
				requireQwen35DecodeBatchReceipt(t, result.receipt, 2, 1)
			case <-time.After(15 * time.Second):
				t.Fatal("reversed overlapping owner acquisition did not terminate")
			}
		}
		closeQwen35DecodeBatchStates(states)
		releaseQwen35DecodeBatchWeights(smallWeights)
		if got := GDNLiveBufferCount(); got != baseline {
			t.Fatalf("overlapping acquisition cleanup left buffers=%d, baseline=%d", got, baseline)
		}
	})

	t.Run("pre-submit owner alias declines without mutation", func(t *testing.T) {
		baseline := GDNLiveBufferCount()
		state := newQwen35DecodeBatchStates(t, geometry, 1)[0]
		defer state.Close()
		seedQwen35DecodeBatchState(t, geometry, state, 3)
		beforeConv, beforeRecurrent, err := state.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		outputs, receipt, accepted, err := RunQwen35DecodeBatch(Qwen35DecodeBatchRequest{
			Input: qwen35DecodeBatchInput(2, hidden), Weights: weights,
			States: []*GDNState{state, state}, Panel: qwen35DecodeBatchPanel(geometry),
		})
		var declined *GDNDeclinedError
		if outputs != nil || accepted || !errors.As(err, &declined) || receipt.Committed || receipt.Commits != 0 {
			t.Fatalf("aliased pre-submit request outputs=%v accepted=%v receipt=%+v err=%T %v", outputs, accepted, receipt, err, err)
		}
		afterConv, afterRecurrent, err := state.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		requireGDNParity(t, "declined convolution", beforeConv, afterConv)
		requireGDNParity(t, "declined recurrent", beforeRecurrent, afterRecurrent)
		state.Close()
		if got := GDNLiveBufferCount(); got != baseline {
			t.Fatalf("declined request leaked buffers=%d, baseline=%d", got, baseline)
		}
	})

	t.Run("post-submit failure advances once without replay", func(t *testing.T) {
		const batch = 2
		baseline := GDNLiveBufferCount()
		serialStates := newQwen35DecodeBatchStates(t, geometry, batch)
		batchStates := newQwen35DecodeBatchStates(t, geometry, batch)
		defer closeQwen35DecodeBatchStates(serialStates)
		defer closeQwen35DecodeBatchStates(batchStates)
		seedQwen35DecodeBatchStatePairs(t, geometry, serialStates, batchStates)
		input := qwen35DecodeBatchInput(batch, hidden)
		panel := qwen35DecodeBatchPanel(geometry)
		serialConv := make([][]float32, batch)
		serialRecurrent := make([][]float32, batch)
		for row := 0; row < batch; row++ {
			if _, _, accepted, err := RunQwen35Decode(Qwen35DecodeRequest{
				Input: input[row*hidden : (row+1)*hidden], Weights: weights,
				State: serialStates[row], Panel: panel,
			}); err != nil || !accepted {
				t.Fatalf("failure oracle row %d: accepted=%v err=%v", row, accepted, err)
			}
			var snapshotErr error
			serialConv[row], serialRecurrent[row], snapshotErr = serialStates[row].Snapshot()
			if snapshotErr != nil {
				t.Fatal(snapshotErr)
			}
			serialStates[row].Close()
		}

		outputs, receipt, accepted, err := RunQwen35DecodeBatch(Qwen35DecodeBatchRequest{
			Input: input, Weights: weights, States: batchStates, Panel: panel,
			InjectPostSubmitFailureForTest: true,
		})
		var post *GraphPostSubmitError
		if outputs != nil || !accepted || !errors.As(err, &post) {
			t.Fatalf("post-submit outputs=%v accepted=%v err=%T %v", outputs, accepted, err, err)
		}
		requireQwen35DecodeBatchReceipt(t, receipt, batch, 0)
		if receipt.ReplayAttempts != 0 {
			t.Fatalf("post-submit replay attempts=%d, want zero", receipt.ReplayAttempts)
		}
		for row, state := range batchStates {
			conv, recurrent, snapshotErr := state.Snapshot()
			if snapshotErr != nil {
				t.Fatalf("post-submit row %d lease was not released: %v", row, snapshotErr)
			}
			requireGDNParity(t, fmt.Sprintf("post-submit row=%d convolution", row), serialConv[row], conv)
			requireGDNParity(t, fmt.Sprintf("post-submit row=%d recurrent", row), serialRecurrent[row], recurrent)
			state.Close()
			state.Close()
		}
		if got := GDNLiveBufferCount(); got != baseline {
			t.Fatalf("post-submit exact-once cleanup left buffers=%d, baseline=%d", got, baseline)
		}
	})
}

func newQwen35DecodeBatchWeights(t *testing.T, hidden int, geometry GDNGeometry) Qwen35DecodeWeights {
	t.Helper()
	upload := func(name string, out, in, phase int) *Q8Weight {
		codes := make([]int8, out*in)
		scales := make([]float32, out*(in/32))
		for i := range codes {
			codes[i] = int8((i*7+phase)%19 - 9)
		}
		for i := range scales {
			scales[i] = 0.0075 + float32((i+phase)%5)*0.001
		}
		weight := UploadQ8(codes, scales, out, in)
		if weight == nil {
			t.Fatalf("upload %s Q8 shape=%dx%d", name, out, in)
		}
		return weight
	}
	return Qwen35DecodeWeights{
		InQKV: upload("in_qkv", geometry.convDim(), hidden, 1),
		InZ:   upload("in_z", geometry.valueDim(), hidden, 3),
		InB:   upload("in_b", geometry.NumValueHeads, hidden, 5),
		InA:   upload("in_a", geometry.NumValueHeads, hidden, 7),
		Out:   upload("out", hidden, geometry.valueDim(), 9),
	}
}

func releaseQwen35DecodeBatchWeights(weights Qwen35DecodeWeights) {
	for _, weight := range []*Q8Weight{weights.InQKV, weights.InZ, weights.InB, weights.InA, weights.Out} {
		weight.Release()
	}
}

func newQwen35DecodeBatchStates(t *testing.T, geometry GDNGeometry, count int) []*GDNState {
	t.Helper()
	states := make([]*GDNState, count)
	for i := range states {
		state, err := NewGDNState(geometry)
		if err != nil {
			for _, opened := range states[:i] {
				opened.Close()
			}
			t.Fatalf("allocate batch state %d: %v", i, err)
		}
		states[i] = state
	}
	return states
}

func closeQwen35DecodeBatchStates(states []*GDNState) {
	for _, state := range states {
		state.Close()
	}
}

func seedQwen35DecodeBatchStatePairs(t *testing.T, geometry GDNGeometry, a, b []*GDNState) {
	t.Helper()
	for row := range a {
		seedQwen35DecodeBatchState(t, geometry, a[row], row+1)
		seedQwen35DecodeBatchState(t, geometry, b[row], row+1)
	}
}

func seedQwen35DecodeBatchState(t *testing.T, geometry GDNGeometry, state *GDNState, phase int) {
	t.Helper()
	conv := make([]float32, (geometry.ConvKernel-1)*geometry.convDim())
	recurrent := make([]float32, geometry.NumValueHeads*geometry.KeyHeadDim*geometry.ValueHeadDim)
	for i := range conv {
		conv[i] = float32(phase)*0.003 + float32(i%11-5)*0.0007
	}
	for i := range recurrent {
		recurrent[i] = float32(phase)*0.0004 + float32(i%17-8)*0.00003
	}
	if err := state.Seed(conv, recurrent); err != nil {
		t.Fatalf("seed phase %d: %v", phase, err)
	}
}

func qwen35DecodeBatchInput(batch, hidden int) []float32 {
	input := make([]float32, batch*hidden)
	for row := 0; row < batch; row++ {
		for col := 0; col < hidden; col++ {
			input[row*hidden+col] = float32(math.Sin(float64((row+1)*(col+3))))*.13 + float32(row+1)*.01
		}
	}
	return input
}

func qwen35DecodeBatchPanel(geometry GDNGeometry) GDNPanel {
	panel := GDNPanel{
		Tokens: 1, Conv1D: make([]float32, geometry.convDim()*geometry.ConvKernel),
		ALog: make([]float32, geometry.NumValueHeads), DTBias: make([]float32, geometry.NumValueHeads),
		Norm: make([]float32, geometry.ValueHeadDim), RMSNormEpsilon: 1e-5,
	}
	for i := range panel.Conv1D {
		panel.Conv1D[i] = float32(i%13-6) * .004
	}
	for i := range panel.ALog {
		panel.ALog[i] = -1.7 + float32(i)*.01
		panel.DTBias[i] = .2 + float32(i)*.005
	}
	for i := range panel.Norm {
		panel.Norm[i] = .9 + float32(i%7)*.02
	}
	return panel
}

func requireQwen35DecodeBatchDistinctOwners(t *testing.T, states []*GDNState) {
	t.Helper()
	seen := make(map[GDNStateHandle]struct{}, 2*len(states))
	for row, state := range states {
		conv, recurrent := state.Handles()
		for _, handle := range []GDNStateHandle{conv, recurrent} {
			if handle == 0 {
				t.Fatalf("row %d has zero owner handle", row)
			}
			if _, duplicate := seen[handle]; duplicate {
				t.Fatalf("row %d aliases owner handle %d", row, handle)
			}
			seen[handle] = struct{}{}
		}
	}
}

func requireQwen35DecodeBatchDifferentNonzeroState(t *testing.T, states []*GDNState) {
	t.Helper()
	var first []float32
	for row, state := range states {
		conv, recurrent, err := state.Snapshot()
		if err != nil {
			t.Fatalf("row %d initial snapshot: %v", row, err)
		}
		combined := append(append([]float32(nil), conv...), recurrent...)
		var nonzero bool
		for _, value := range combined {
			if value != 0 {
				nonzero = true
				break
			}
		}
		if !nonzero {
			t.Fatalf("row %d initial state is all zero", row)
		}
		if row == 0 {
			first = combined
		} else if qwen35DecodeBatchMaxAbs(first, combined) == 0 {
			t.Fatalf("row %d initial state aliases row 0 values", row)
		}
	}
}

func requireQwen35DecodeBatchReceipt(t *testing.T, receipt Qwen35DecodeBatchReceipt, batch, finalReadbacks int) {
	t.Helper()
	if receipt.Rows != batch || receipt.OwnerCount != batch || receipt.CommandBuffers != 1 || receipt.Commits != 1 || receipt.CompletionWaits != 1 ||
		!receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("B=%d terminal receipt=%+v", batch, receipt)
	}
	if receipt.ProjectionDispatches != 5 || receipt.Quantizers != 2 || receipt.GDNDispatches != 3*batch ||
		receipt.GDNEncoders != 1 || receipt.Encoders != 8 {
		t.Fatalf("B=%d dispatch receipt=%+v, want five shared projections and 3B independent GDN dispatches", batch, receipt)
	}
	if receipt.InputUploads != 1 || receipt.FinalReadbacks != finalReadbacks || receipt.IntermediateReadbacks != 0 ||
		receipt.StateH2DTransfers != 0 || receipt.StateD2HTransfers != 0 || receipt.ReplayAttempts != 0 {
		t.Fatalf("B=%d transfer receipt=%+v", batch, receipt)
	}
	if receipt.OwnerRetains != batch || receipt.OwnerReleases != batch {
		t.Fatalf("B=%d owner lifecycle receipt=%+v", batch, receipt)
	}
	t.Logf("B=%d shared decode receipt: %+v", batch, receipt)
}

func qwen35DecodeBatchMaxAbs(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var maxAbs float64
	for i := range a {
		if delta := math.Abs(float64(a[i] - b[i])); delta > maxAbs {
			maxAbs = delta
		}
	}
	return maxAbs
}
