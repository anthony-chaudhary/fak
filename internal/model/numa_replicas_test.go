package model

import (
	"encoding/binary"
	"math"
	"os"
	"sync"
	"testing"
)

func makeTestQ4KRaw(out, in int) []byte {
	nblk := in / 256
	rowBytes := nblk * 144
	raw := make([]byte, out*rowBytes)
	for r := 0; r < out; r++ {
		for b := 0; b < nblk; b++ {
			blk := raw[r*rowBytes+b*144 : r*rowBytes+(b+1)*144]
			// d = 0.5 (f16: 0x3800)
			binary.LittleEndian.PutUint16(blk[0:2], 0x3800)
			// min = 0.0 (f16: 0x0000)
			binary.LittleEndian.PutUint16(blk[2:4], 0x0000)
			// scales: 12 bytes
			for s := 0; s < 12; s++ {
				blk[4+s] = byte(0x11 * ((s % 3) + 1))
			}
			// qs: 128 bytes (256 nibbles)
			for q := 0; q < 128; q++ {
				blk[16+q] = byte((r*7 + b*13 + q*17) & 0xFF)
			}
		}
	}
	return raw
}

func makeQwen38TestModel(t *testing.T) *Model {
	t.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.Name = "qwen38:27b"
	cfg.HiddenSize = 256
	cfg.NumHeads = 4
	cfg.NumKVHeads = 2
	cfg.HeadDim = 64
	cfg.IntermediateSize = 512
	cfg.LinearKeyHeadDim = 64
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 64
	cfg.LinearNumValueHeads = 4
	cfg.VocabSize = 256
	cfg.NumLayers = 4

	m := NewSynthetic(cfg)
	m.q4kw = make(map[string]*q4kTensor)

	// Add resident Q4_K weights for linear attention projections, attention, and MLP
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		if cfg.isLinearAttnLayer(l) {
			_, _, _, _, _, valDim, convDim := cfg.linearAttnDims()
			m.q4kw[p+"linear_attn.in_proj_qkv.weight"] = &q4kTensor{out: convDim, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(convDim, cfg.HiddenSize)}
			m.q4kw[p+"linear_attn.in_proj_z.weight"] = &q4kTensor{out: valDim, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(valDim, cfg.HiddenSize)}
			m.q4kw[p+"linear_attn.out_proj.weight"] = &q4kTensor{out: cfg.HiddenSize, in: valDim, nblk: valDim / 256, raw: makeTestQ4KRaw(cfg.HiddenSize, valDim)}
		} else {
			qRows := cfg.NumHeads * cfg.HeadDim
			if cfg.AttnOutputGate {
				qRows *= 2
			}
			m.q4kw[p+"self_attn.q_proj.weight"] = &q4kTensor{out: qRows, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(qRows, cfg.HiddenSize)}
			m.q4kw[p+"self_attn.k_proj.weight"] = &q4kTensor{out: cfg.NumKVHeads * cfg.HeadDim, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(cfg.NumKVHeads*cfg.HeadDim, cfg.HiddenSize)}
			m.q4kw[p+"self_attn.v_proj.weight"] = &q4kTensor{out: cfg.NumKVHeads * cfg.HeadDim, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(cfg.NumKVHeads*cfg.HeadDim, cfg.HiddenSize)}
			m.q4kw[p+"self_attn.o_proj.weight"] = &q4kTensor{out: cfg.HiddenSize, in: cfg.NumHeads * cfg.HeadDim, nblk: (cfg.NumHeads * cfg.HeadDim) / 256, raw: makeTestQ4KRaw(cfg.HiddenSize, cfg.NumHeads*cfg.HeadDim)}
		}
		m.q4kw[p+"mlp.gate_proj.weight"] = &q4kTensor{out: cfg.IntermediateSize, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(cfg.IntermediateSize, cfg.HiddenSize)}
		m.q4kw[p+"mlp.up_proj.weight"] = &q4kTensor{out: cfg.IntermediateSize, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(cfg.IntermediateSize, cfg.HiddenSize)}
		m.q4kw[p+"mlp.down_proj.weight"] = &q4kTensor{out: cfg.HiddenSize, in: cfg.IntermediateSize, nblk: cfg.IntermediateSize / 256, raw: makeTestQ4KRaw(cfg.HiddenSize, cfg.IntermediateSize)}
	}

	m.q4khead = &q4kTensor{out: cfg.VocabSize, in: cfg.HiddenSize, nblk: cfg.HiddenSize / 256, raw: makeTestQ4KRaw(cfg.VocabSize, cfg.HiddenSize)}
	return m
}

func TestNUMAWeightReplication_ApplyAndFree(t *testing.T) {
	m := makeQwen38TestModel(t)
	defer m.FreeNUMAReplicas()

	lbl := m.ApplyNUMAWeightReplicas("2")
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("expected NUMAReplicasEnabled = true, got label: %s", lbl)
	}

	if m.NUMADecodePool() == nil {
		t.Fatal("expected non-nil NUMADecodePool")
	}
	if len(m.NUMATopology()) != 2 {
		t.Fatalf("expected 2 topology nodes, got %d", len(m.NUMATopology()))
	}

	// Verify all tensors received replicas
	for name, qt := range m.q4kw {
		if qt.replicas == nil {
			t.Fatalf("tensor %s has nil replicas", name)
		}
		if qt.replicas.Len() != 2 {
			t.Fatalf("tensor %s has %d replicas, want 2", name, qt.replicas.Len())
		}
		if qt.numaPool == nil {
			t.Fatalf("tensor %s has nil numaPool", name)
		}
	}

	if m.q4khead.replicas == nil || m.q4khead.replicas.Len() != 2 {
		t.Fatal("q4khead has missing or incomplete replicas")
	}

	if err := m.FreeNUMAReplicas(); err != nil {
		t.Fatalf("FreeNUMAReplicas failed: %v", err)
	}
	if m.NUMAReplicasEnabled() {
		t.Fatal("expected NUMAReplicasEnabled = false after free")
	}
	if m.NUMADecodePool() != nil {
		t.Fatal("expected nil NUMADecodePool after free")
	}
	for name, qt := range m.q4kw {
		if qt.replicas != nil {
			t.Fatalf("tensor %s replicas not cleared after free", name)
		}
		if qt.numaPool != nil {
			t.Fatalf("tensor %s numaPool not cleared after free", name)
		}
	}
}

func TestNUMAWeightReplication_Batch1DecodeBitIdentity(t *testing.T) {
	m := makeQwen38TestModel(t)
	defer m.FreeNUMAReplicas()

	// 1. Reference decode without NUMA replicas
	sessRef := m.NewSession()
	sessRef.Q4K = true

	prompt := []int{3, 17, 42}
	_ = sessRef.Prefill(prompt)

	const decodeTokens = 8
	refLogits := make([][]float32, decodeTokens)
	refTokens := make([]int, decodeTokens)

	curToken := 42
	for i := 0; i < decodeTokens; i++ {
		logits := sessRef.Step(curToken)
		refLogits[i] = append([]float32(nil), logits...)
		next := argmaxF32(logits)
		refTokens[i] = next
		curToken = next
	}

	// 2. Enable NUMA replicas with 2 nodes
	lbl := m.ApplyNUMAWeightReplicas("2")
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("failed to apply NUMA replicas: %s", lbl)
	}

	// 3. NUMA-replicated decode with barrier-free per-node schedule
	sessNUMA := m.NewSession()
	sessNUMA.Q4K = true

	_ = sessNUMA.Prefill(prompt)

	numaLogits := make([][]float32, decodeTokens)
	numaTokens := make([]int, decodeTokens)

	curToken = 42
	for i := 0; i < decodeTokens; i++ {
		logits := sessNUMA.Step(curToken)
		numaLogits[i] = append([]float32(nil), logits...)
		next := argmaxF32(logits)
		numaTokens[i] = next
		curToken = next
	}

	// 4. Assert 100% BIT-IDENTICAL logits and tokens
	for i := 0; i < decodeTokens; i++ {
		if refTokens[i] != numaTokens[i] {
			t.Fatalf("step %d token mismatch: got %d, want %d", i, numaTokens[i], refTokens[i])
		}
		if len(refLogits[i]) != len(numaLogits[i]) {
			t.Fatalf("step %d logits len mismatch", i)
		}
		for j := range refLogits[i] {
			wantBits := math.Float32bits(refLogits[i][j])
			gotBits := math.Float32bits(numaLogits[i][j])
			if wantBits != gotBits {
				t.Fatalf("step %d logit %d bit mismatch: got 0x%08X (%f), want 0x%08X (%f)",
					i, j, gotBits, numaLogits[i][j], wantBits, refLogits[i][j])
			}
		}
	}
}

func TestNUMAWeightReplication_ThreadSafety(t *testing.T) {
	m := makeQwen38TestModel(t)
	defer m.FreeNUMAReplicas()

	lbl := m.ApplyNUMAWeightReplicas("2")
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("failed to apply NUMA replicas: %s", lbl)
	}

	const concurrency = 8
	var wg sync.WaitGroup
	wg.Add(concurrency)

	for c := 0; c < concurrency; c++ {
		go func(cid int) {
			defer wg.Done()
			s := m.NewSession()
			s.Q4K = true
			_ = s.Prefill([]int{cid % 256})
			for step := 0; step < 4; step++ {
				logits := s.Step((cid + step) % 256)
				if len(logits) != m.Cfg.VocabSize {
					t.Errorf("worker %d step %d got %d logits, want %d", cid, step, len(logits), m.Cfg.VocabSize)
				}
				for _, val := range logits {
					if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
						t.Errorf("worker %d got non-finite logit %f", cid, val)
						return
					}
				}
			}
		}(c)
	}

	wg.Wait()
}

func TestNUMAWeightReplication_EnvironmentVariable(t *testing.T) {
	orig := os.Getenv("FAK_NUMA_REPLICAS")
	defer os.Setenv("FAK_NUMA_REPLICAS", orig)

	m := makeQwen38TestModel(t)
	defer m.FreeNUMAReplicas()

	os.Setenv("FAK_NUMA_REPLICAS", "off")
	lbl := m.ApplyNUMAWeightReplicas("")
	if m.NUMAReplicasEnabled() {
		t.Fatalf("expected disabled with FAK_NUMA_REPLICAS=off, got: %s", lbl)
	}

	os.Setenv("FAK_NUMA_REPLICAS", "2")
	lbl = m.ApplyNUMAWeightReplicas("")
	if !m.NUMAReplicasEnabled() {
		t.Fatalf("expected enabled with FAK_NUMA_REPLICAS=2, got: %s", lbl)
	}
}
