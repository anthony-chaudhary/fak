package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

var qwenSwapRestoreSink *KVCache

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
	restoredSession := &Session{M: m, Cache: restored}
	for _, token := range []int{29, 31, 37} {
		assertQwenSwapBitsEqual(t, "continuation logits", base.Step(token), restoredSession.Step(token))
	}

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

func TestQwenHybridPagedSwapRejectsTrailingAndInvalidGeometry(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cache := NewSynthetic(cfg).NewSession()
	cache.Prefill([]int{3, 7, 11, 5, 17})
	blob := mustQwenSwap(t, cache.Cache, 4)
	body := blob[:len(blob)-sha256.Size]

	trailingBody := append(append([]byte(nil), body...), 0xa5)
	if _, err := QwenHybridKVCacheFromHost(cfg, qwenSwapTestResign(trailingBody)); err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("valid-checksum trailing byte error=%v, want trailing-byte refusal", err)
	}

	badGeometry := append([]byte(nil), body...)
	d := qwenSwapDecoder{buf: badGeometry}
	d.bytes()
	d.u32()
	if err := d.config(cfg); err != nil {
		t.Fatalf("locate geometry field: %v", err)
	}
	binary.LittleEndian.PutUint64(badGeometry[d.off:], 0) // blockTokens must be positive.
	if _, err := QwenHybridKVCacheFromHost(cfg, qwenSwapTestResign(badGeometry)); err == nil || !strings.Contains(err.Error(), "geometry") {
		t.Fatalf("zero block geometry error=%v, want geometry refusal", err)
	}

	qwenSwapAssertRefusedWithoutPanic(t, cfg, qwenSwapTestResign(body[:len(body)-1]), "re-signed truncation")
	oversizedLength := append([]byte(nil), body...)
	binary.LittleEndian.PutUint64(oversizedLength, ^uint64(0))
	qwenSwapAssertRefusedWithoutPanic(t, cfg, qwenSwapTestResign(oversizedLength), "oversized byte length")
}

func TestQwenHybridPagedSwapAdversarialMalformedCacheRefusesWithoutPanic(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*KVCache) *KVCache
	}{
		{name: "nil cache", mutate: func(*KVCache) *KVCache { return nil }},
		{name: "missing K plane", mutate: func(c *KVCache) *KVCache { c.K = nil; return c }},
		{name: "missing Kraw plane", mutate: func(c *KVCache) *KVCache { c.Kraw = nil; return c }},
		{name: "missing V plane", mutate: func(c *KVCache) *KVCache { c.V = nil; return c }},
		{name: "missing linear state", mutate: func(c *KVCache) *KVCache { c.linear = nil; return c }},
		{name: "short linear layer inventory", mutate: func(c *KVCache) *KVCache {
			c.linear.layers = c.linear.layers[:len(c.linear.layers)-1]
			return c
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cache := NewSynthetic(qwen35HybridTestCfg()).NewSession().Cache
			cache = tc.mutate(cache)
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("QwenHybridKVCacheToHost panicked: %v", recovered)
				}
			}()
			if blob, err := QwenHybridKVCacheToHost(cache, 4); err == nil || blob != nil {
				t.Fatalf("malformed cache accepted: blob=%d bytes err=%v", len(blob), err)
			}
		})
	}
}

func TestQwenHybridPagedSwapV1WireCompatibility(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	session := NewSynthetic(cfg).NewSession()
	session.Prefill([]int{3, 7, 11, 5, 17})

	legacy, err := qwenSwapLegacyV1(session.Cache, 4)
	if err != nil {
		t.Fatal(err)
	}
	got := mustQwenSwap(t, session.Cache, 4)
	if !bytes.Equal(got, legacy) {
		t.Fatalf("bulk encoder changed the authoritative v1 bytes: got sha256=%x legacy sha256=%x", sha256.Sum256(got), sha256.Sum256(legacy))
	}
	restored, err := QwenHybridKVCacheFromHost(cfg, legacy)
	if err != nil {
		t.Fatalf("bulk decoder rejected authoritative v1 blob: %v", err)
	}
	for l := range session.Cache.K {
		assertQwenSwapBitsEqual(t, "v1 K", session.Cache.K[l], restored.K[l])
		assertQwenSwapBitsEqual(t, "v1 Kraw", session.Cache.Kraw[l], restored.Kraw[l])
		assertQwenSwapBitsEqual(t, "v1 V", session.Cache.V[l], restored.V[l])
	}
}

func TestQwenHybridPagedSwapRestoreDoesNotMaterializePadding(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	session := NewSynthetic(cfg).NewSession()
	session.Prefill([]int{3, 7, 11, 5, 17}) // Deliberately non-aligned.
	aligned := mustQwenSwap(t, session.Cache, 5)
	padded := mustQwenSwap(t, session.Cache, 4096)
	if len(padded) <= len(aligned) {
		t.Fatalf("fixture has no serialized padding: padded=%d aligned=%d", len(padded), len(aligned))
	}

	allocatedBytes := func(blob []byte) int64 {
		result := testing.Benchmark(func(b *testing.B) {
			b.ReportAllocs()
			var restored *KVCache
			var err error
			for i := 0; i < b.N; i++ {
				restored, err = QwenHybridKVCacheFromHost(cfg, blob)
				if err != nil {
					b.Fatal(err)
				}
			}
			qwenSwapRestoreSink = restored
		})
		return result.AllocedBytesPerOp()
	}
	alignedBytes := allocatedBytes(aligned)
	paddedBytes := allocatedBytes(padded)
	t.Logf("public restore allocations: aligned=%d B/op padded=%d B/op serialized_padding=%d bytes", alignedBytes, paddedBytes, len(padded)-len(aligned))
	const allocatorSlack = 32 << 10
	if extra := paddedBytes - alignedBytes; extra > allocatorSlack {
		t.Fatalf("public restore allocated %d extra bytes for serialized padding (aligned=%d padded=%d), want <= %d", extra, alignedBytes, paddedBytes, allocatorSlack)
	}
}

func TestNewQwenHybridSwapCachePreservesAuxiliaryInitialization(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{name: "glm", cfg: func() Config {
			cfg := qwen35HybridTestCfg()
			cfg.ModelType = "glm_moe_dsa"
			cfg.Architectures = []string{"GlmMoeDsaForCausalLM"}
			return cfg
		}()},
		{name: "msa", cfg: func() Config {
			cfg := qwen35HybridTestCfg()
			cfg.ModelType = "minimax_m3"
			cfg.Architectures = []string{"MiniMaxM3ForCausalLM"}
			return cfg
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.cfg.IsQwen35Hybrid() {
				t.Fatal("mixed-predicate fixture must remain Qwen hybrid")
			}
			want := NewKVCache(tc.cfg)
			got := newQwenHybridSwapCache(tc.cfg)
			if (got.glm == nil) != (want.glm == nil) || (got.msa == nil) != (want.msa == nil) {
				t.Fatalf("auxiliary constructors differ: got glm=%t msa=%t want glm=%t msa=%t", got.glm != nil, got.msa != nil, want.glm != nil, want.msa != nil)
			}
			if got.glm != nil && (len(got.glm.K) != len(want.glm.K) || len(got.glm.IndexK) != len(want.glm.IndexK)) {
				t.Fatalf("GLM auxiliary geometry differs: got K=%d index=%d want K=%d index=%d", len(got.glm.K), len(got.glm.IndexK), len(want.glm.K), len(want.glm.IndexK))
			}
			if got.msa != nil && len(got.msa.IndexK) != len(want.msa.IndexK) {
				t.Fatalf("MSA auxiliary geometry differs: got=%d want=%d", len(got.msa.IndexK), len(want.msa.IndexK))
			}
			for l := range got.linear.layers {
				if len(got.linear.layers[l].recurrent) != 0 {
					t.Fatalf("restore cache eagerly allocated Qwen recurrent state at layer %d", l)
				}
			}
		})
	}
}

func TestQwenSwapDecoderDirectRowsAllocateNoPaddedBlock(t *testing.T) {
	const (
		liveFloats   = 3
		paddedFloats = 4096
	)
	raw := make([]byte, paddedFloats*4)
	for i := 0; i < liveFloats; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)+0.25))
	}
	dst := make([]float32, liveFloats)
	allocs := testing.AllocsPerRun(100, func() {
		d := qwenSwapDecoder{buf: raw}
		d.f32rawInto(dst)
		d.skipF32(paddedFloats - liveFloats)
		if d.err != nil || d.remaining() != 0 {
			panic("direct row decode failed")
		}
	})
	if allocs != 0 {
		t.Fatalf("direct live-row decode allocated %.0f objects, want zero temporary padded blocks", allocs)
	}
	for i := range dst {
		if got, want := math.Float32bits(dst[i]), math.Float32bits(float32(i)+0.25); got != want {
			t.Fatalf("direct row[%d] bits=%08x want=%08x", i, got, want)
		}
	}
}

func qwenSwapTestResign(body []byte) []byte {
	out := make([]byte, len(body)+sha256.Size)
	copy(out, body)
	sum := sha256.Sum256(body)
	copy(out[len(body):], sum[:])
	return out
}

func qwenSwapAssertRefusedWithoutPanic(t *testing.T, cfg Config, blob []byte, label string) {
	t.Helper()
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("%s panicked: %v", label, recovered)
		}
	}()
	if restored, err := QwenHybridKVCacheFromHost(cfg, blob); err == nil || restored != nil {
		t.Fatalf("%s accepted: cache=%p err=%v", label, restored, err)
	}
}

// qwenSwapLegacyV1 is the frozen scalar encoder replaced by the bulk codec. It is kept
// only as a small-test oracle so format-v1 compatibility is independent of the new writer.
func qwenSwapLegacyV1(c *KVCache, blockTokens int) ([]byte, error) {
	var payload bytes.Buffer
	var encodeErr error
	write := func(v any) {
		if encodeErr == nil {
			encodeErr = binary.Write(&payload, binary.LittleEndian, v)
		}
	}
	integer := func(v int) {
		if v < 0 {
			encodeErr = errQwenSwapLegacyNegative
			return
		}
		write(uint64(v))
	}
	byteString := func(v []byte) {
		integer(len(v))
		if encodeErr == nil {
			_, encodeErr = payload.Write(v)
		}
	}
	ints := func(v []int) {
		integer(len(v))
		for _, x := range v {
			integer(x)
		}
	}
	floats := func(v []float32) {
		for _, x := range v {
			write(math.Float32bits(x))
		}
	}
	rows := func(v [][]float32) {
		integer(len(v))
		for _, row := range v {
			integer(len(row))
			floats(row)
		}
	}

	byteString([]byte("FAKQHPS1"))
	write(uint32(1))
	for _, v := range []int{c.cfg.NumLayers, c.cfg.NumKVHeads, c.cfg.HeadDim, c.cfg.LinearNumKeyHeads, c.cfg.LinearNumValueHeads, c.cfg.LinearKeyHeadDim, c.cfg.LinearValueHeadDim, c.cfg.LinearConvKernelDim, c.cfg.FullAttentionInterval} {
		integer(v)
	}
	integer(len(c.cfg.LayerTypes))
	for _, layerType := range c.cfg.LayerTypes {
		byteString([]byte(layerType))
	}
	integer(blockTokens)
	integer(c.Len())
	ints(c.pos)
	full := qwenFullAttentionLayers(c.cfg)
	ints(full)
	blocks := 0
	if c.Len() > 0 {
		blocks = (c.Len() + blockTokens - 1) / blockTokens
	}
	integer(blocks)
	stride := c.kvStride()
	for b := 0; b < blocks; b++ {
		for _, l := range full {
			for _, plane := range [][][]float32{c.K, c.Kraw, c.V} {
				for off := 0; off < blockTokens; off++ {
					pos := b*blockTokens + off
					if pos < c.Len() {
						floats(plane[l][pos*stride : (pos+1)*stride])
					} else {
						floats(make([]float32, stride))
					}
				}
			}
		}
	}
	integer(len(c.linear.layers))
	for _, state := range c.linear.layers {
		rows(state.conv)
		rows(state.recurrent)
	}
	if encodeErr != nil {
		return nil, encodeErr
	}
	sum := sha256.Sum256(payload.Bytes())
	return append(payload.Bytes(), sum[:]...), nil
}

var errQwenSwapLegacyNegative = &qwenSwapLegacyError{}

type qwenSwapLegacyError struct{}

func (*qwenSwapLegacyError) Error() string { return "negative Qwen hybrid swap value" }

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
