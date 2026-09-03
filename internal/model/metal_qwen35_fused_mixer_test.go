//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestFusedLinearAttentionMixerCPUOracleParity(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	mixer, err := NewFusedLinearAttentionMixer(m, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixer failed: %v", err)
	}
	defer mixer.Close()

	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	keep := cfg.LinearConvKernelDim - 1

	convSeed := randomVecF(keep*convDim, 7701)
	recSeed := randomVecF(nV*kHd*vHd, 7702)

	if err := mixer.Seed(convSeed, recSeed); err != nil {
		t.Fatalf("mixer.Seed failed: %v", err)
	}

	cpu := m.NewSession()
	cpu.Cache.linear = newLinearAttnCache(cfg)
	cpuLayer := cpu.Cache.linear.layer(cfg, 0)
	cpuLayer.conv = make([][]float32, keep)
	for r := range cpuLayer.conv {
		cpuLayer.conv[r] = make([]float32, convDim)
		copy(cpuLayer.conv[r], convSeed[r*convDim:(r+1)*convDim])
	}
	for h := range cpuLayer.recurrent {
		copy(cpuLayer.recurrent[h], recSeed[h*kHd*vHd:(h+1)*kHd*vHd])
	}

	for step := 0; step < 6; step++ {
		input := randomVecF(cfg.HiddenSize, int64(7800+step))

		want := cpu.linearAttnStep(0, input, q8Kernel{m})
		got, receipt, stepErr := mixer.Step(input)
		if stepErr != nil {
			t.Fatalf("step %d failed: %v", step, stepErr)
		}

		if receipt.CommandBuffers != 1 || receipt.CompletionWaits != 1 || receipt.TransferCount != 2 {
			t.Fatalf("step %d receipt topology: %+v", step, receipt)
		}

		outCos, outMaxAbs, err := CosineSimilarityAndMaxAbs(want, got)
		if err != nil {
			t.Fatalf("step %d output comparison: %v", step, err)
		}
		if outCos < 0.999999 || outMaxAbs >= 0.0001 {
			t.Fatalf("step %d output parity failed: cosine=%g (want >=0.999999), maxAbs=%g (want <0.0001)", step, outCos, outMaxAbs)
		}

		wantGreedy := decodeMixerArgmax(want)
		gotGreedy := decodeMixerArgmax(got)
		if wantGreedy != gotGreedy {
			t.Fatalf("step %d greedy token mismatch: want %d, got %d", step, wantGreedy, gotGreedy)
		}

		gpuConv, gpuRecurrent, err := mixer.Snapshot()
		if err != nil {
			t.Fatalf("step %d snapshot failed: %v", step, err)
		}

		cpuConv := FlattenLinearConvState(cpuLayer, keep, convDim)
		convCos, convMaxAbs, err := CosineSimilarityAndMaxAbs(cpuConv, gpuConv)
		if err != nil {
			t.Fatalf("step %d conv comparison: %v", step, err)
		}
		if convCos < 0.999999 || convMaxAbs >= 0.0001 {
			t.Fatalf("step %d conv parity failed: cosine=%g, maxAbs=%g", step, convCos, convMaxAbs)
		}

		cpuRecurrent := FlattenLinearRecurrentState(cpuLayer)
		recCos, recMaxAbs, err := CosineSimilarityAndMaxAbs(cpuRecurrent, gpuRecurrent)
		if err != nil {
			t.Fatalf("step %d recurrent comparison: %v", step, err)
		}
		if recCos < 0.999999 || recMaxAbs >= 0.0001 {
			t.Fatalf("step %d recurrent parity failed: cosine=%g, maxAbs=%g", step, recCos, recMaxAbs)
		}
	}
}

func TestFusedLinearAttentionMixerValidateCPUOracleMethod(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	mixer, err := NewFusedLinearAttentionMixer(m, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixer failed: %v", err)
	}
	defer mixer.Close()

	cpu := m.NewSession()
	inputs := make([][]float32, 4)
	for i := range inputs {
		inputs[i] = randomVecF(cfg.HiddenSize, int64(8800+i))
	}

	parities, err := mixer.ValidateCPUOracleMultiStep(cpu, inputs)
	if err != nil {
		t.Fatalf("ValidateCPUOracleMultiStep failed: %v", err)
	}
	if len(parities) != len(inputs) {
		t.Fatalf("parities count=%d, want %d", len(parities), len(inputs))
	}

	for i, p := range parities {
		if !p.Passed {
			t.Fatalf("step %d parity failed: %+v", i, p)
		}
		if p.OutputCosine < 0.999999 || p.OutputMaxAbs >= 0.0001 {
			t.Fatalf("step %d output parity failed: cosine=%g, maxAbs=%g", i, p.OutputCosine, p.OutputMaxAbs)
		}
		if p.ConvCosine < 0.999999 || p.ConvMaxAbs >= 0.0001 {
			t.Fatalf("step %d conv parity failed: cosine=%g, maxAbs=%g", i, p.ConvCosine, p.ConvMaxAbs)
		}
		if p.RecurrentCosine < 0.999999 || p.RecurrentMaxAbs >= 0.0001 {
			t.Fatalf("step %d rec parity failed: cosine=%g, maxAbs=%g", i, p.RecurrentCosine, p.RecurrentMaxAbs)
		}
	}
}

func TestFusedLinearAttentionMixerOperationOrderAndTransfers(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	mixer, err := NewFusedLinearAttentionMixer(m, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixer failed: %v", err)
	}
	defer mixer.Close()

	input := randomVecF(cfg.HiddenSize, 9001)
	out, receipt, err := mixer.Step(input)
	if err != nil {
		t.Fatalf("mixer.Step failed: %v", err)
	}
	if len(out) != cfg.HiddenSize {
		t.Fatalf("output length=%d, want %d", len(out), cfg.HiddenSize)
	}

	expectedOrder := []string{
		"QKV_Z_B_A_PROJECTION",
		"CONVOLUTION_SHIFT",
		"QK_NORM",
		"RECURRENT_STATE_UPDATE",
		"GATED_RMSNORM",
		"OUTPUT_PROJECTION",
	}
	if len(receipt.OperationOrder) != len(expectedOrder) {
		t.Fatalf("receipt.OperationOrder length=%d, want %d", len(receipt.OperationOrder), len(expectedOrder))
	}
	for i, expected := range expectedOrder {
		if receipt.OperationOrder[i] != expected {
			t.Fatalf("operation [%d]: got %s, want %s", i, receipt.OperationOrder[i], expected)
		}
	}

	if receipt.CommandBuffers != 1 {
		t.Fatalf("CommandBuffers=%d, want 1", receipt.CommandBuffers)
	}
	if receipt.Commits != 1 {
		t.Fatalf("Commits=%d, want 1", receipt.Commits)
	}
	if receipt.CompletionWaits != 1 {
		t.Fatalf("CompletionWaits=%d, want 1", receipt.CompletionWaits)
	}
	if !receipt.Committed {
		t.Fatal("Committed=false, want true")
	}
	if !receipt.CompletedWait {
		t.Fatal("CompletedWait=false, want true")
	}

	if receipt.ProjectionDispatches != 5 {
		t.Fatalf("ProjectionDispatches=%d, want 5", receipt.ProjectionDispatches)
	}
	if receipt.Quantizers != 2 {
		t.Fatalf("Quantizers=%d, want 2", receipt.Quantizers)
	}
	if receipt.GDNEncoders != 1 {
		t.Fatalf("GDNEncoders=%d, want 1", receipt.GDNEncoders)
	}
	if receipt.Encoders != 8 {
		t.Fatalf("Encoders=%d, want 8", receipt.Encoders)
	}

	if receipt.InputUploads != 1 {
		t.Fatalf("InputUploads=%d, want 1", receipt.InputUploads)
	}
	if receipt.FinalReadbacks != 1 {
		t.Fatalf("FinalReadbacks=%d, want 1", receipt.FinalReadbacks)
	}
	if receipt.H2DTransfers != 1 {
		t.Fatalf("H2DTransfers=%d, want 1", receipt.H2DTransfers)
	}
	if receipt.D2HTransfers != 1 {
		t.Fatalf("D2HTransfers=%d, want 1", receipt.D2HTransfers)
	}
	if receipt.TransferCount != 2 {
		t.Fatalf("TransferCount=%d, want 2 (1 H2D in, 1 D2H out)", receipt.TransferCount)
	}

	if receipt.IntermediateTransfers != 0 {
		t.Fatalf("IntermediateTransfers=%d, want 0", receipt.IntermediateTransfers)
	}
	if receipt.IntermediateReadbacks != 0 {
		t.Fatalf("IntermediateReadbacks=%d, want 0", receipt.IntermediateReadbacks)
	}
	if receipt.StateH2DTransfers != 0 {
		t.Fatalf("StateH2DTransfers=%d, want 0", receipt.StateH2DTransfers)
	}
	if receipt.StateD2HTransfers != 0 {
		t.Fatalf("StateD2HTransfers=%d, want 0", receipt.StateD2HTransfers)
	}
}

func TestFusedLinearAttentionMixerSessionIsolation(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	s1 := m.NewSession()
	s2 := m.NewSession()
	defer s1.Close()
	defer s2.Close()

	mixer1, err := NewFusedLinearAttentionMixerForSession(s1, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixerForSession(s1) failed: %v", err)
	}
	defer mixer1.Close()

	mixer2, err := NewFusedLinearAttentionMixerForSession(s2, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixerForSession(s2) failed: %v", err)
	}
	defer mixer2.Close()

	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	keep := cfg.LinearConvKernelDim - 1

	s1Conv := randomVecF(keep*convDim, 9101)
	s1Rec := randomVecF(nV*kHd*vHd, 9102)
	s2Conv := randomVecF(keep*convDim, 9103)
	s2Rec := randomVecF(nV*kHd*vHd, 9104)

	if err := mixer1.Seed(s1Conv, s1Rec); err != nil {
		t.Fatalf("mixer1.Seed failed: %v", err)
	}
	if err := mixer2.Seed(s2Conv, s2Rec); err != nil {
		t.Fatalf("mixer2.Seed failed: %v", err)
	}

	beforeConv2, beforeRec2, err := mixer2.Snapshot()
	if err != nil {
		t.Fatalf("mixer2.Snapshot failed: %v", err)
	}

	for i := 0; i < 4; i++ {
		input := randomVecF(cfg.HiddenSize, int64(9200+i))
		if _, _, err := mixer1.Step(input); err != nil {
			t.Fatalf("mixer1.Step failed: %v", err)
		}
	}

	afterConv2, afterRec2, err := mixer2.Snapshot()
	if err != nil {
		t.Fatalf("mixer2.Snapshot after mixer1 stepped failed: %v", err)
	}

	for i := range beforeConv2 {
		if beforeConv2[i] != afterConv2[i] {
			t.Fatalf("mixer2 conv state mutated at index %d by mixer1 execution", i)
		}
	}
	for i := range beforeRec2 {
		if beforeRec2[i] != afterRec2[i] {
			t.Fatalf("mixer2 recurrent state mutated at index %d by mixer1 execution", i)
		}
	}

	var wg sync.WaitGroup
	var err1, err2 error
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			in := randomVecF(cfg.HiddenSize, int64(9300+i))
			if _, _, err := mixer1.Step(in); err != nil {
				err1 = err
				return
			}
		}
	}()

	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			in := randomVecF(cfg.HiddenSize, int64(9400+i))
			if _, _, err := mixer2.Step(in); err != nil {
				err2 = err
				return
			}
		}
	}()

	wg.Wait()
	if err1 != nil {
		t.Fatalf("concurrent mixer1 error: %v", err1)
	}
	if err2 != nil {
		t.Fatalf("concurrent mixer2 error: %v", err2)
	}
}

func TestFusedLinearAttentionMixerCleanTeardown(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	baseline := metalgemm.GDNLiveBufferCount()

	mixer, err := NewFusedLinearAttentionMixer(m, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixer failed: %v", err)
	}

	if got := metalgemm.GDNLiveBufferCount(); got <= baseline {
		t.Fatalf("expected live buffers > baseline (%d), got %d", baseline, got)
	}

	input := randomVecF(cfg.HiddenSize, 9501)
	if _, _, err := mixer.Step(input); err != nil {
		t.Fatalf("mixer.Step failed: %v", err)
	}

	if err := mixer.Close(); err != nil {
		t.Fatalf("mixer.Close failed: %v", err)
	}
	if err := mixer.Close(); err != nil {
		t.Fatalf("mixer.Close second idempotent call failed: %v", err)
	}

	if got := metalgemm.GDNLiveBufferCount(); got != baseline {
		t.Fatalf("after Close live buffers=%d, want baseline=%d", got, baseline)
	}

	if _, _, err := mixer.Step(input); err == nil {
		t.Fatal("expected error on Step after mixer.Close, got nil")
	}
}

func TestFusedLinearAttentionMixerDeclineAndFailure(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()

	if _, err := NewFusedLinearAttentionMixer(m, -1); err == nil {
		t.Fatal("expected error for negative layer index")
	}
	if _, err := NewFusedLinearAttentionMixer(m, cfg.NumLayers); err == nil {
		t.Fatal("expected error for layer index out of bounds")
	}
	if _, err := NewFusedLinearAttentionMixer(m, 3); err == nil {
		t.Fatal("expected error for non-linear attention layer (layer 3 is full_attention)")
	}

	mixer, err := NewFusedLinearAttentionMixer(m, 0)
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixer failed: %v", err)
	}
	defer mixer.Close()

	badInput := randomVecF(cfg.HiddenSize-1, 9601)
	if _, _, err := mixer.Step(badInput); err == nil {
		t.Fatal("expected error for input size mismatch")
	}

	failingMixer, err := NewFusedLinearAttentionMixerWithOptions(m, 0, FusedLinearAttentionMixerOptions{
		InjectPostSubmitFailureForTest: true,
	})
	if err != nil {
		t.Fatalf("NewFusedLinearAttentionMixerWithOptions failed: %v", err)
	}
	defer failingMixer.Close()

	_, receipt, runErr := failingMixer.Step(randomVecF(cfg.HiddenSize, 9602))
	var post *metalgemm.GraphPostSubmitError
	if runErr == nil || !errors.As(runErr, &post) {
		t.Fatalf("expected GraphPostSubmitError, got err=%v", runErr)
	}
	if !receipt.Committed || !receipt.CompletedWait {
		t.Fatalf("failing receipt expected Committed & CompletedWait, got: %+v", receipt)
	}
}
