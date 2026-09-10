//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"reflect"
	"testing"
)

// TestVulkanQwen35PrefixReplayTruncatesEveryActiveKVPlane exercises the
// compute-owned half of partial target commit directly. The fixture deliberately
// gives a one-attention-layer hybrid panel four backing slots: the active compact
// slot must expose only the accepted K/Kraw/V prefix, while the surplus unused
// slots must remain byte-for-byte untouched.
func TestVulkanQwen35PrefixReplayTruncatesEveryActiveKVPlane(t *testing.T) {
	be := vk(t)
	const (
		tokens        = 4
		startPos      = 2
		numKeyHeads   = 1
		numValueHeads = 1
		keyHeadDim    = 2
		valueHeadDim  = 2
		convKernel    = 3
	)
	const convDim = 2*numKeyHeads*keyHeadDim + numValueHeads*valueHeadDim
	const valueDim = numValueHeads * valueHeadDim
	const kvWidth = numKeyHeads * keyHeadDim

	makeValues := func(n int, scale float32) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = float32((i%13)-6) * scale
		}
		return values
	}
	upload := func(shape []int, values []float32, class MemoryClass, name string) Tensor {
		return be.UploadClass(NewF32(Default(), shape, values), F32, class, name)
	}

	for accepted := 1; accepted < tokens; accepted++ {
		t.Run(fmt.Sprintf("accepted_%d", accepted), func(t *testing.T) {
			layers := make([]Qwen35SequenceLayer, 4)
			states := make([]Qwen35SequenceState, len(layers))
			before := make([]Qwen35SequenceState, len(layers))
			projections := make([]vulkanQwen35ReplayProjection, 0, 3)
			owned := make([]*vulkanBuf, 0, 12)
			for layer := 0; layer < 3; layer++ {
				layers[layer].Linear = true
				layers[layer].GDNConv = upload([]int{convDim, convKernel}, makeValues(convDim*convKernel, .025), MemoryWeights, fmt.Sprintf("partial commit layer %d conv", layer))
				layers[layer].GDNALog = upload([]int{numValueHeads}, []float32{-1}, MemoryWeights, fmt.Sprintf("partial commit layer %d alog", layer))
				layers[layer].GDNDTBias = upload([]int{numValueHeads}, []float32{.1}, MemoryWeights, fmt.Sprintf("partial commit layer %d dt bias", layer))
				layers[layer].GDNNorm = upload([]int{valueHeadDim}, []float32{1, 1}, MemoryWeights, fmt.Sprintf("partial commit layer %d norm", layer))
				states[layer] = Qwen35SequenceState{
					Conv:      upload([]int{convKernel - 1, convDim}, makeValues((convKernel-1)*convDim, .02), MemoryKVCache, fmt.Sprintf("partial commit layer %d live conv", layer)),
					Recurrent: upload([]int{numValueHeads, keyHeadDim, valueHeadDim}, makeValues(numValueHeads*keyHeadDim*valueHeadDim, .02), MemoryKVCache, fmt.Sprintf("partial commit layer %d live recurrent", layer)),
				}
				before[layer] = Qwen35SequenceState{
					Conv:      upload([]int{convKernel - 1, convDim}, makeValues((convKernel-1)*convDim, .01), MemoryKVCache, fmt.Sprintf("partial commit layer %d before conv", layer)),
					Recurrent: upload([]int{numValueHeads, keyHeadDim, valueHeadDim}, makeValues(numValueHeads*keyHeadDim*valueHeadDim, .01), MemoryKVCache, fmt.Sprintf("partial commit layer %d before recurrent", layer)),
				}
				for _, tensor := range []Tensor{layers[layer].GDNConv, layers[layer].GDNALog, layers[layer].GDNDTBias, layers[layer].GDNNorm, states[layer].Conv, states[layer].Recurrent, before[layer].Conv, before[layer].Recurrent} {
					value := tensor
					t.Cleanup(func() { be.Free(value) })
				}
				mixed := upload([]int{tokens, convDim}, makeValues(tokens*convDim, .03), MemoryActivation, fmt.Sprintf("partial commit layer %d mixed", layer))
				z := upload([]int{tokens, valueDim}, makeValues(tokens*valueDim, .04), MemoryActivation, fmt.Sprintf("partial commit layer %d z", layer))
				beta := upload([]int{tokens, numValueHeads}, makeValues(tokens*numValueHeads, .05), MemoryActivation, fmt.Sprintf("partial commit layer %d beta", layer))
				alpha := upload([]int{tokens, numValueHeads}, makeValues(tokens*numValueHeads, .02), MemoryActivation, fmt.Sprintf("partial commit layer %d alpha", layer))
				projections = append(projections, vulkanQwen35ReplayProjection{layer: layer, mixed: mixed, z: z, beta: beta, alpha: alpha})
				owned = append(owned, mixed.buf.(*vulkanBuf), z.buf.(*vulkanBuf), beta.buf.(*vulkanBuf), alpha.buf.(*vulkanBuf))
			}

			kv := be.NewKV(KVConfig{NumLayers: len(layers), NumKVHeads: numKeyHeads, HeadDim: keyHeadDim, RopeTheta: 10000}).(*vulkanKV)
			t.Cleanup(kv.Free)
			rows := startPos + tokens
			wantRaw := make([]float32, 0, rows*kvWidth)
			wantKey := make([]float32, 0, rows*kvWidth)
			wantValue := make([]float32, 0, rows*kvWidth)
			for pos := 0; pos < rows; pos++ {
				rawHost := []float32{float32(pos*10 + 1), float32(pos*10 + 2)}
				keyHost := []float32{float32(pos*10 + 3), float32(pos*10 + 4)}
				valueHost := []float32{float32(pos*10 + 5), float32(pos*10 + 6)}
				wantRaw = append(wantRaw, rawHost...)
				wantKey = append(wantKey, keyHost...)
				wantValue = append(wantValue, valueHost...)
				raw := upload([]int{kvWidth}, rawHost, MemoryActivation, "partial commit raw row")
				key := upload([]int{kvWidth}, keyHost, MemoryActivation, "partial commit key row")
				value := upload([]int{kvWidth}, valueHost, MemoryActivation, "partial commit value row")
				kv.AppendKV(0, raw, key, value, pos)
				be.Free(raw)
				be.Free(key)
				be.Free(value)
			}
			spareK := append([]vslice(nil), kv.K[1:]...)
			spareRaw := append([]vslice(nil), kv.Kraw[1:]...)
			spareV := append([]vslice(nil), kv.V[1:]...)

			replay := &vulkanQwen35PrefixReplay{
				backend: be, kv: kv, startPos: startPos, tokens: tokens,
				kvWidth: kvWidth, convDim: convDim, valueDim: valueDim,
				numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
				keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
				convKernel: convKernel, rmsEpsilon: 1e-5,
				layers: layers, states: states, projections: projections, owned: owned,
			}
			t.Cleanup(replay.Close)
			if err := replay.CommitPrefix(accepted, before); err != nil {
				t.Fatal(err)
			}

			wantRows := startPos + accepted
			wantFloats := wantRows * kvWidth
			if kv.Len() != wantRows || !reflect.DeepEqual(kv.Pos(), []int{0, 1, 2, 3, 4}[:wantRows]) {
				t.Fatalf("committed KV length/positions=%d/%v, want %d contiguous rows", kv.Len(), kv.Pos(), wantRows)
			}
			if kv.K[0].len != wantFloats || kv.Kraw[0].len != wantFloats || kv.V[0].len != wantFloats {
				t.Fatalf("active K/Kraw/V lengths=%d/%d/%d, want %d", kv.K[0].len, kv.Kraw[0].len, kv.V[0].len, wantFloats)
			}
			if got := be.Read(kv.KeysView(0)); !reflect.DeepEqual(got, wantKey[:wantFloats]) {
				t.Fatalf("committed K=%v want=%v", got, wantKey[:wantFloats])
			}
			if got := kv.readVS(&kv.Kraw[0]); !reflect.DeepEqual(got, wantRaw[:wantFloats]) {
				t.Fatalf("committed Kraw=%v want=%v", got, wantRaw[:wantFloats])
			}
			if got := be.Read(kv.ValuesView(0)); !reflect.DeepEqual(got, wantValue[:wantFloats]) {
				t.Fatalf("committed V=%v want=%v", got, wantValue[:wantFloats])
			}
			if !reflect.DeepEqual(kv.K[1:], spareK) || !reflect.DeepEqual(kv.Kraw[1:], spareRaw) || !reflect.DeepEqual(kv.V[1:], spareV) {
				t.Fatal("partial commit mutated surplus unused KV backing slots")
			}
		})
	}
}
