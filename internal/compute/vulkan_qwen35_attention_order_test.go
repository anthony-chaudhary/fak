//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestVulkanQwen35AttentionPanelMatchesTokenOrder compares the sequence panel
// shader with the production one-token attention path over identical Q/K/V.
// Growing the token-path KV cache before each query gives both kernels the same
// causal prefix while isolating attention arithmetic from the rest of the model.
func TestVulkanQwen35AttentionPanelMatchesTokenOrder(t *testing.T) {
	if os.Getenv("FAK_VULKAN_ATTENTION_CONTEXT_SPLIT") == "1" {
		t.Fatal("attention order regression requires the production one-dispatch token path")
	}
	v := vk(t)

	const (
		nHeads  = 24
		nKV     = 4
		headDim = 256
		group   = nHeads / nKV
	)
	scale := float32(1 / math.Sqrt(headDim))
	for _, test := range []struct {
		name           string
		prefix, tokens int
		seed           uint64
	}{
		{name: "prefix0_queries3", prefix: 0, tokens: 3, seed: 0x350003},
		{name: "prefix17_queries4", prefix: 17, tokens: 4, seed: 0x351704},
		{name: "prefix129_queries3", prefix: 129, tokens: 3, seed: 0x359103},
		{name: "prefix257_queries2", prefix: 257, tokens: 2, seed: 0x357202},
	} {
		t.Run(test.name, func(t *testing.T) {
			qWidth := nHeads * headDim
			kvWidth := nKV * headDim
			totalPositions := test.prefix + test.tokens
			q := qwen35AttentionOrderValues(test.tokens*qWidth, test.seed, 0.25)
			k := qwen35AttentionOrderValues(totalPositions*kvWidth, test.seed^0x9e3779b97f4a7c15, 0.25)
			value := qwen35AttentionOrderValues(totalPositions*kvWidth, test.seed^0xd1b54a32d192ed03, 0.5)

			c := cpu()
			dq := v.Upload(NewF32(c, []int{test.tokens, qWidth}, q), F32)
			dk := v.Upload(NewF32(c, []int{totalPositions, kvWidth}, k), F32)
			dv := v.Upload(NewF32(c, []int{totalPositions, kvWidth}, value), F32)
			panelTensor, _ := v.devTr([]int{test.tokens, qWidth}, F32)
			defer v.Free(dq)
			defer v.Free(dk)
			defer v.Free(dv)
			defer v.Free(panelTensor)

			status := v.qwen35SequenceCausalAttentionForTest(
				v.vp(dq), v.vp(dk), v.vp(dv), v.vp(panelTensor),
				test.tokens, test.prefix, nHeads, nKV, headDim, scale,
			)
			if status != 0 {
				t.Fatalf("causal attention panel status=%d", status)
			}
			panel := v.Read(panelTensor)

			kv := v.NewKV(KVConfig{NumLayers: 1, NumKVHeads: nKV, HeadDim: headDim})
			defer kv.Free()
			appendPosition := func(position int) {
				lo, hi := position*kvWidth, (position+1)*kvWidth
				keyRow := v.Upload(NewF32(c, []int{kvWidth}, k[lo:hi]), F32)
				valueRow := v.Upload(NewF32(c, []int{kvWidth}, value[lo:hi]), F32)
				kv.AppendKV(0, keyRow, keyRow, valueRow, position)
				v.Free(keyRow)
				v.Free(valueRow)
			}
			for position := 0; position < test.prefix; position++ {
				appendPosition(position)
			}

			tokenOutput := make([]float32, test.tokens*qWidth)
			for token := 0; token < test.tokens; token++ {
				appendPosition(test.prefix + token)
				lo, hi := token*qWidth, (token+1)*qWidth
				queryRow := v.Upload(NewF32(c, []int{qWidth}, q[lo:hi]), F32)
				out := v.Attention(queryRow, kv, 0, true, group, scale)
				copy(tokenOutput[lo:hi], v.Read(out))
				v.Free(out)
				v.Free(queryRow)
			}

			metrics := qwen35AttentionOrderMetrics(panel, tokenOutput)
			encoded, err := json.Marshal(map[string]any{
				"schema":             "fak.vulkan.qwen35-attention-order.v1",
				"scope":              "native attention primitive; sequence panel versus growing-cache token path",
				"device":             v.Tier(),
				"prefix":             test.prefix,
				"tokens":             test.tokens,
				"query_heads":        nHeads,
				"kv_heads":           nKV,
				"head_dim":           headDim,
				"elements_compared":  len(panel),
				"bitwise_mismatches": metrics.bitwiseMismatches,
				"max_abs_delta":      metrics.maxAbsDelta,
				"all_finite":         metrics.allFinite,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(encoded))

			if len(panel) != test.tokens*qWidth || len(tokenOutput) != len(panel) {
				t.Fatalf("output lengths panel=%d token=%d want=%d", len(panel), len(tokenOutput), test.tokens*qWidth)
			}
			if !metrics.allFinite {
				t.Fatal("attention panel/token output or residual is non-finite")
			}
			if metrics.bitwiseMismatches != 0 {
				t.Fatalf("attention operation order diverged: bitwise mismatches=%d first=%d panel=%g token=%g max abs delta=%g",
					metrics.bitwiseMismatches, metrics.firstMismatch, metrics.firstPanel, metrics.firstToken, metrics.maxAbsDelta)
			}
		})
	}
}

func qwen35AttentionOrderValues(n int, seed uint64, amplitude float32) []float32 {
	values := make([]float32, n)
	state := seed
	for i := range values {
		state = state*6364136223846793005 + 1442695040888963407
		unit := float32(int32(uint32(state>>40))-1<<23) / float32(1<<23)
		values[i] = unit * amplitude
	}
	return values
}

type qwen35AttentionOrderResult struct {
	bitwiseMismatches int
	firstMismatch     int
	firstPanel        float32
	firstToken        float32
	maxAbsDelta       float64
	allFinite         bool
}

func qwen35AttentionOrderMetrics(panel, token []float32) qwen35AttentionOrderResult {
	result := qwen35AttentionOrderResult{firstMismatch: -1, allFinite: len(panel) == len(token)}
	if len(panel) != len(token) {
		return result
	}
	for i, panelValue := range panel {
		tokenValue := token[i]
		if math.IsNaN(float64(panelValue)) || math.IsInf(float64(panelValue), 0) ||
			math.IsNaN(float64(tokenValue)) || math.IsInf(float64(tokenValue), 0) {
			result.allFinite = false
		}
		delta := math.Abs(float64(panelValue) - float64(tokenValue))
		result.maxAbsDelta = math.Max(result.maxAbsDelta, delta)
		if math.Float32bits(panelValue) != math.Float32bits(tokenValue) {
			result.bitwiseMismatches++
			if result.firstMismatch < 0 {
				result.firstMismatch = i
				result.firstPanel = panelValue
				result.firstToken = tokenValue
			}
		}
	}
	if math.IsNaN(result.maxAbsDelta) || math.IsInf(result.maxAbsDelta, 0) {
		result.allFinite = false
	}
	return result
}
