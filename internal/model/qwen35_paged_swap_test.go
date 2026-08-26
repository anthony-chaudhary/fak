package model

import (
	"bytes"
	"math"
	"testing"
)

func TestQwenHybridPagedSwapRoundTrip(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23}
	base := m.NewSession()
	base.Prefill(prompt)

	blob, err := QwenHybridKVCacheToHost(base.Cache, 4)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := QwenHybridKVCacheFromHost(cfg, blob)
	if err != nil {
		t.Fatal(err)
	}
	for l := range base.Cache.K {
		assertQwenSwapBitsEqual(t, "K", base.Cache.K[l], restored.K[l])
		assertQwenSwapBitsEqual(t, "Kraw", base.Cache.Kraw[l], restored.Kraw[l])
		assertQwenSwapBitsEqual(t, "V", base.Cache.V[l], restored.V[l])
	}
	for l := range base.Cache.linear.layers {
		for i := range base.Cache.linear.layers[l].conv {
			assertQwenSwapBitsEqual(t, "conv", base.Cache.linear.layers[l].conv[i], restored.linear.layers[l].conv[i])
		}
		for i := range base.Cache.linear.layers[l].recurrent {
			assertQwenSwapBitsEqual(t, "recurrent", base.Cache.linear.layers[l].recurrent[i], restored.linear.layers[l].recurrent[i])
		}
	}
	if !bytes.Equal(blob, mustQwenSwap(t, base.Cache, 4)) {
		t.Fatal("serialization is not deterministic")
	}
	assertQwenSwapBitsEqual(t, "next logits", base.Step(29), (&Session{M: m, Cache: restored}).Step(29))

	// Compactness: fixed recurrent state is present in both representations, while
	// token-indexed pages exist only for the declared full-attention layer.
	stride := cfg.NumKVHeads * cfg.HeadDim
	blocks := (len(prompt) + 3) / 4
	densePageBytes := blocks * 4 * cfg.NumLayers * 3 * stride * 4
	fixedStateBytes := 0
	for _, layer := range base.Cache.linear.layers {
		for _, row := range layer.conv {
			fixedStateBytes += len(row) * 4
		}
		for _, row := range layer.recurrent {
			fixedStateBytes += len(row) * 4
		}
	}
	if len(blob) >= densePageBytes+fixedStateBytes {
		t.Fatalf("blob=%d, dense pages plus fixed state=%d", len(blob), densePageBytes+fixedStateBytes)
	}

	corrupt := append([]byte(nil), blob...)
	corrupt[len(corrupt)/2] ^= 1
	if _, err := QwenHybridKVCacheFromHost(cfg, corrupt); err == nil {
		t.Fatal("checksum corruption accepted")
	}
	wrong := cfg
	wrong.LinearValueHeadDim++
	if _, err := QwenHybridKVCacheFromHost(wrong, blob); err == nil {
		t.Fatal("config mismatch accepted")
	}

}

func mustQwenSwap(t *testing.T, c *KVCache, block int) []byte {
	t.Helper()
	b, err := QwenHybridKVCacheToHost(c, block)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func assertQwenSwapBitsEqual(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length %d != %d", label, len(want), len(got))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d] bits differ", label, i)
		}
	}
}
