package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvcache"
)

// fak-test:runtime fast est=50ms lane=default
func TestModelKVPrecisionNVFP4AndCrossModelLayout(t *testing.T) {
	prec, err := ParseKVPrecision("nvfp4")
	if err != nil || prec != KVPrecisionNVFP4 {
		t.Fatalf("ParseKVPrecision(nvfp4) = (%v, %v)", prec, err)
	}
	if got := KVVectorBytes(128, KVPrecisionNVFP4); got != 72 {
		t.Fatalf("KVVectorBytes(128, nvfp4) = %d, want 72", got)
	}

	for _, arch := range []string{"qwen3", "gemma3", "llama"} {
		cfg := Config{
			ModelType:  arch,
			NumLayers:  4,
			NumKVHeads: 4,
			HeadDim:    128,
		}
		desc := KVCacheLayoutDesc(cfg, KVPrecisionNVFP4, 256)
		if err := desc.Validate(); err != nil {
			t.Fatalf("%s LayoutDesc.Validate: %v", arch, err)
		}
		if ratio := desc.CompressionRatioVsFP32(); ratio < 7.0 {
			t.Fatalf("%s CompressionRatioVsFP32 = %.3f, want >= 7.0", arch, ratio)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestModelNVFP4KVSnapshotRoundTrip(t *testing.T) {
	for _, arch := range []string{"qwen3", "gemma3", "llama"} {
		t.Run(arch, func(t *testing.T) {
			cfg := Config{
				ModelType:  arch,
				NumLayers:  2,
				NumKVHeads: 2,
				HeadDim:    32,
			}
			c := NewKVCache(cfg)
			const nTok = 8
			c.pos = make([]int, nTok)
			tokens := make([]int, nTok)
			rowLen := cfg.NumKVHeads * cfg.HeadDim
			for l := 0; l < cfg.NumLayers; l++ {
				c.K[l] = make([]float32, nTok*rowLen)
				c.Kraw[l] = make([]float32, nTok*rowLen)
				c.V[l] = make([]float32, nTok*rowLen)
				for i := range c.K[l] {
					c.K[l][i] = float32(math.Sin(float64(i+l*7)*0.19)) * 4.0
					c.Kraw[l][i] = c.K[l][i]
					c.V[l][i] = float32(math.Cos(float64(i+l*11)*0.23)) * 3.0
				}
			}
			for i := 0; i < nTok; i++ {
				c.pos[i] = i
				tokens[i] = 1000 + i
			}

			snap, overflow, err := CaptureNVFP4Snapshot(c, tokens)
			if err != nil || overflow {
				t.Fatalf("CaptureNVFP4Snapshot: err=%v overflow=%v", err, overflow)
			}
			wire, err := kvcache.EncodeSnapshot(snap)
			if err != nil {
				t.Fatalf("EncodeSnapshot: %v", err)
			}
			fp32Bytes := c.cfg.NumLayers * 2 * nTok * rowLen * 4
			if float64(fp32Bytes)/float64(len(wire)) < 4.5 {
				t.Fatalf("expected compressed wire bytes (%d) << fp32 (%d)", len(wire), fp32Bytes)
			}

			expected := KVCacheLayoutDesc(cfg, KVPrecisionNVFP4, 0)
			decoded, err := kvcache.DecodeSnapshot(wire, &expected)
			if err != nil {
				t.Fatalf("DecodeSnapshot: %v", err)
			}
			restored, err := RestoreNVFP4Snapshot(decoded, cfg)
			if err != nil {
				t.Fatalf("RestoreNVFP4Snapshot: %v", err)
			}
			if restored.Len() != nTok {
				t.Fatalf("restored.Len() = %d, want %d", restored.Len(), nTok)
			}
			var dot, nA, nB float64
			for i := range c.K[0] {
				dot += float64(c.K[0][i]) * float64(restored.K[0][i])
				nA += float64(c.K[0][i]) * float64(c.K[0][i])
				nB += float64(restored.K[0][i]) * float64(restored.K[0][i])
			}
			if cos := dot / (math.Sqrt(nA) * math.Sqrt(nB)); cos < 0.985 {
				t.Fatalf("restored K cosine = %.5f, want >= 0.985", cos)
			}
		})
	}
}
