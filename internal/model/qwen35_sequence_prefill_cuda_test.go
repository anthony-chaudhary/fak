//go:build cuda

package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func requiredCUDAQwen35SequenceBackend(t *testing.T) (cudaGDNParityBackend, Qwen35SequencePrefillBackend) {
	t.Helper()
	be := requiredCUDAGDNParityBackend(t)
	sequence, ok := be.(Qwen35SequencePrefillBackend)
	if !ok || sequence.Qwen35SequencePrefillPath() != compute.Qwen35SequencePrefillPath {
		t.Fatalf("registered CUDA backend %T does not expose complete sequence path %q", be, compute.Qwen35SequencePrefillPath)
	}
	return be, sequence
}

func TestCUDAQwen35SequencePrefillMatchesCPUAndPersistsDecodeState(t *testing.T) {
	be, sequence := requiredCUDAQwen35SequenceBackend(t)
	if compute.Qwen35SequenceParityCosineMin != Qwen35GDNParityCosineMin {
		t.Fatalf("sequence parity floor %.3f drifted from established GDN floor %.3f", compute.Qwen35SequenceParityCosineMin, Qwen35GDNParityCosineMin)
	}
	m := NewSynthetic(qwen35HybridTestCfg())
	prompt := []int{3, 7, 11, 5, 17, 19, 23}
	next := 29
	cpu := m.NewSession()
	wantPrefill := cpu.Prefill(prompt)
	wantStep := cpu.Step(next)

	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	defer be.Recycle()
	req := device.qwen35SequencePrefillRequest(prompt, true)
	stateHandles := make([][2]compute.Buffer, len(req.States))
	for layer, state := range req.States {
		stateHandles[layer] = [2]compute.Buffer{state.Conv.Buf(), state.Recurrent.Buf()}
	}
	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.ResetQwen35GDNOperationCount()
	result, err := sequence.Qwen35SequencePrefill(req)
	if err != nil {
		t.Fatalf("complete resident CUDA sequence: %v", err)
	}
	wantOps := uint64(3 * len(prompt))
	if got := be.Qwen35GDNOperationCount(); got != wantOps {
		t.Fatalf("resident GDN operations = %d, want %d", got, wantOps)
	}
	if result.Tokens != len(prompt) || device.halKV.Len() != len(prompt) {
		t.Fatalf("sequence result tokens/KV = %d/%d, want %d/%d", result.Tokens, device.halKV.Len(), len(prompt), len(prompt))
	}
	if !result.LastHidden.Ready() || !result.Logits.Ready() || result.LastHidden.Numel() != m.Cfg.HiddenSize || result.Logits.Numel() != m.Cfg.VocabSize {
		t.Fatalf("resident result malformed: hidden=%v ready=%v logits=%v ready=%v", result.LastHidden.Shape, result.LastHidden.Ready(), result.Logits.Shape, result.Logits.Ready())
	}
	if host, ok := be.Host(result.LastHidden); ok || host != nil {
		t.Fatal("last hidden unexpectedly became host-addressable")
	}
	if host, ok := be.Host(result.Logits); ok || host != nil {
		t.Fatal("logits unexpectedly became host-addressable")
	}
	wantControlBytes := uint64(len(prompt) * 4)
	if got := result.Transfers; got.H2DBytes != wantControlBytes || got.D2HBytes != 0 || got.ActivationH2DBytes != 0 || got.ActivationD2HBytes != 0 {
		t.Fatalf("sequence transfer witness = %+v, want control H2D=%d and zero activation/D2H", got, wantControlBytes)
	}
	if got := be.H2DXferBytes(); got != wantControlBytes {
		t.Fatalf("backend H2D bytes inside sequence = %d, want token-id control bytes %d", got, wantControlBytes)
	}
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("backend D2H bytes inside sequence = %d, want 0", got)
	}
	for layer, state := range req.States {
		if !req.Layers[layer].Linear {
			continue
		}
		if state.Conv.Buf() != stateHandles[layer][0] || state.Recurrent.Buf() != stateHandles[layer][1] {
			t.Fatalf("layer %d recurrent state handle changed", layer)
		}
	}
	gotPrefill := be.Read(result.Logits)
	prefillStats := compareQwen35GDNVector(t, "sequence prefill logits", wantPrefill, gotPrefill)
	if got, want := argmaxF32(gotPrefill), argmaxF32(wantPrefill); got != want {
		t.Fatalf("sequence prefill argmax = %d, want %d", got, want)
	}
	be.Recycle()
	gotStep := device.Step(next)
	stepStats := compareQwen35GDNVector(t, "decode after sequence prefill", wantStep, gotStep)
	if got, want := argmaxF32(gotStep), argmaxF32(wantStep); got != want {
		t.Fatalf("decode-after-prefill argmax = %d, want %d", got, want)
	}
	t.Logf("path=%s tokens=%d operations=%d control_h2d=%d activation_h2d=0 activation_d2h=0 prefill_cosine=%.9f prefill_max_abs=%.3e step_cosine=%.9f step_max_abs=%.3e state_identity=true kv_len=%d",
		sequence.Qwen35SequencePrefillPath(), len(prompt), wantOps, wantControlBytes,
		prefillStats.cosine, prefillStats.maxAbs, stepStats.cosine, stepStats.maxAbs, device.halKV.Len())
}

func TestCUDAQwen35SequenceMalformedRequestFailsBeforeMutation(t *testing.T) {
	be, sequence := requiredCUDAQwen35SequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	defer be.Recycle()
	req := device.qwen35SequencePrefillRequest([]int{3, 7}, true)
	req.TokenIDs[1] = m.Cfg.VocabSize
	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.ResetQwen35GDNOperationCount()
	_, err = sequence.Qwen35SequencePrefill(req)
	var sequenceErr *compute.Qwen35SequenceError
	if err == nil || !errors.As(err, &sequenceErr) {
		t.Fatalf("malformed request error = %v, want typed sequence refusal", err)
	}
	if device.halKV.Len() != 0 || be.Qwen35GDNOperationCount() != 0 || be.H2DXferBytes() != 0 || be.HostXferBytes() != 0 {
		t.Fatalf("malformed request mutated execution state: kv=%d ops=%d h2d=%d d2h=%d", device.halKV.Len(), be.Qwen35GDNOperationCount(), be.H2DXferBytes(), be.HostXferBytes())
	}
}
