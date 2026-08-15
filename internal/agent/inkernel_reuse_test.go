package agent

import (
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"

	"context"
)

// inkernel_reuse_test.go is the candidate-#13(+#14) witness suite for RadixAttention
// KV-prefix reuse wired onto the live in-kernel planner. It drives generateReused — the
// tokenizer-free reuse/decode core Complete factors out — directly over a synthetic model
// (the model.Model path with a real tokenizer OOMs under WSL; the numerics are proven by
// internal/model's oracle, the bit-exact KV reuse by its KV-prefix-reuse rung). The arms:
//
//   - PARITY: reuse-through-a-split decodes BIT-IDENTICAL tokens to a full re-prefill.
//   - POISON: a quarantine eviction drops the poisoned branch and forces a re-prefill,
//     while the benign sibling sharing the prefix is preserved (no replay).
//   - PERF:   a growing multi-turn conversation prefills far FEWER tokens with reuse on,
//     and the real wall-clock speedup is reported.
//   - RACE:   concurrent turns + probes + evictions are data-race-free (the tree mutex).

// synthIDs builds a deterministic token-id sequence in [0,vocab) (mirrors radixbench's
// lcgIDs); shared seeds produce LITERALLY the same ids, so prefixes are genuinely shared.
func synthIDs(vocab, n int, seed uint64) []int {
	ids := make([]int, n)
	st := uint64(2463534242) + seed
	for i := range ids {
		st = (st*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(st % uint64(vocab))
	}
	return ids
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// reusePlanner builds an InKernelPlanner over a small synthetic model. quant=false runs
// the f32 forward (the exact path internal/model proves KV-prefix reuse bit-identical on);
// quant=true runs the Q8_0 forward the SERVED path actually ships (Model.Quantize()), so a
// parity arm can witness reuse on the production path too. tree on => reuse enabled.
func reusePlanner(reuse, quant bool, cfg model.Config) *InKernelPlanner {
	m := model.NewSynthetic(cfg)
	if quant {
		m.Quantize()
	}
	p := &InKernelPlanner{m: m, modelID: "synthetic", quant: quant}
	if reuse {
		p.tree = radixkv.New(0)
	}
	return p
}

func tinyCfg() model.Config {
	return model.Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 63,
	}
}

func tinyHybridCfg() model.Config {
	return model.Config{
		HiddenSize:            32,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      8,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    8,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

func tinyGLMDsaCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"},
		QLoraRank: 32, KVLoraRank: 32, QKNopeHeadDim: 4, QKRopeHeadDim: 4, VHeadDim: 8,
		IndexNHeads: 4, IndexHeadDim: 8, IndexTopK: 2,
		IndexerTypes: []string{"full", "shared", "full"},
		EOSTokenID:   -1,
	}
}

// decode runs one turn through generateReused, collecting the generated token ids (via the
// emit seam) and returning them alongside the reused-prefix length. No token-id stops are
// passed, so decode always runs the full maxNew — a deterministic, comparable trace.
func decode(p *InKernelPlanner, ids []int, maxNew int) (gen []int, matched int) {
	_, _, matched, _, _, _ = p.generateReused(ids, maxNew, 0, 0, 0, map[int]bool{}, func(id int) bool {
		gen = append(gen, id)
		return false
	})
	return gen, matched
}

type countingBackend struct {
	compute.Backend
	mu           sync.Mutex
	mat          int
	batched      int
	deviceMemory bool
}

func (c *countingBackend) Caps() compute.Caps {
	caps := c.Backend.Caps()
	caps.DeviceMemory = c.deviceMemory
	return caps
}

func (c *countingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	c.mu.Lock()
	c.mat++
	c.mu.Unlock()
	return c.Backend.MatMul(w, x)
}

func (c *countingBackend) BatchedMatMul(w, X compute.Tensor, P int) compute.Tensor {
	c.mu.Lock()
	c.batched++
	c.mu.Unlock()
	return c.Backend.BatchedMatMul(w, X, P)
}

func (c *countingBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	data := append([]float32(nil), c.Backend.Read(t)...)
	host := compute.NewF32(compute.Default(), append([]int(nil), t.Shape...), data)
	return c.Backend.Upload(host, compute.F32), nil
}

func (*countingBackend) Qwen35GDNPath() string { return model.Qwen35GDNCUDAPath }

func (c *countingBackend) Qwen35GDNDecode(
	normalizedInput,
	_, _, _, _,
	_, _, _, _, _,
	convState, recurrentState compute.Tensor,
	_, _, _, _, _ int,
	_ float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	data := append([]float32(nil), c.Backend.Read(normalizedInput)...)
	host := compute.NewF32(compute.Default(), append([]int(nil), normalizedInput.Shape...), data)
	return c.Backend.Upload(host, compute.F32), convState, recurrentState, nil
}

func (c *countingBackend) ops() (mat, batched int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.mat, c.batched
}

func (c *countingBackend) reset() {
	c.mu.Lock()
	c.mat = 0
	c.batched = 0
	c.mu.Unlock()
}

func kvConfigFromModelConfig(cfg model.Config) compute.KVConfig {
	return compute.KVConfig{
		NumLayers:  cfg.NumLayers,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		RopeTheta:  cfg.RopeTheta,
	}
}

func TestInKernelDecodeSkipsUnusedFinalStep(t *testing.T) {
	cfg := tinyCfg()
	ids := synthIDs(cfg.VocabSize, 12, 407)

	run := func(maxNew int) (gen, mat, batched int) {
		be := &countingBackend{Backend: compute.Default()}
		p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic-counting-backend", false, be, false)
		p.quant = false
		got, _ := decode(p, ids, maxNew)
		mat, batched = be.ops()
		return len(got), mat, batched
	}

	gen0, mat0, batch0 := run(0)
	if gen0 != 0 {
		t.Fatalf("maxNew=0 generated %d tokens, want 0", gen0)
	}
	gen1, mat1, batch1 := run(1)
	if gen1 != 1 {
		t.Fatalf("maxNew=1 generated %d tokens, want 1", gen1)
	}
	if mat1 != mat0 || batch1 != batch0 {
		t.Fatalf("maxNew=1 should stop after sampling the first token without an unused Step: ops maxNew=0 mat=%d batch=%d; maxNew=1 mat=%d batch=%d", mat0, batch0, mat1, batch1)
	}
	gen2, mat2, batch2 := run(2)
	if gen2 != 2 {
		t.Fatalf("maxNew=2 generated %d tokens, want 2", gen2)
	}
	if mat2 <= mat1 && batch2 <= batch1 {
		t.Fatalf("maxNew=2 should compute one next-token Step between generated tokens: ops maxNew=1 mat=%d batch=%d; maxNew=2 mat=%d batch=%d", mat1, batch1, mat2, batch2)
	}
}

func TestInKernelKVMemoryStatsDeviceBackendReportsGeometryOnly(t *testing.T) {
	cfg := tinyCfg()
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	p := &InKernelPlanner{
		m:       model.NewSynthetic(cfg),
		modelID: "synthetic-device",
		backend: backend,
	}

	st := p.KVMemoryStats()
	wantPerToken := compute.EstimateKVStoreBytes(kvConfigFromModelConfig(cfg), 1)
	if st.Enabled {
		t.Fatalf("KVMemoryStats.Enabled = true on device backend; stats=%+v", st)
	}
	if st.Backend != backend.Name() || st.MemoryClass != string(compute.MemoryKVCache) || st.Scope != string(compute.MemoryScopeDevice) {
		t.Fatalf("KVMemoryStats labels = backend=%q class=%q scope=%q", st.Backend, st.MemoryClass, st.Scope)
	}
	if st.BytesPerToken != wantPerToken {
		t.Fatalf("BytesPerToken = %d, want %d", st.BytesPerToken, wantPerToken)
	}
	if st.ResidentTokens != 0 || st.ResidentBytes != 0 || st.LRUTokens != 0 || st.Nodes != 0 {
		t.Fatalf("device backend should report geometry only, got %+v", st)
	}
}

func TestInKernelRadixCostAwareVictimRuleFromEnv(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	t.Setenv("FAK_INKERNEL_RADIX_BUDGET", "20")
	t.Setenv("FAK_NATIVE_KV_VICTIM_RULE", "cost-aware")

	cfg := tinyCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	p.quant = false
	if p.tree == nil {
		t.Fatal("FAK_INKERNEL_RADIX=on should construct a radix tree")
	}
	if got := p.tree.Stats().EvictionPolicy; got != "cost-aware" {
		t.Fatalf("radix eviction policy = %q, want cost-aware", got)
	}

	a := synthIDs(cfg.VocabSize, 10, 210)
	b := synthIDs(cfg.VocabSize, 10, 211)
	c := synthIDs(cfg.VocabSize, 10, 212)
	p.generateReused(a, 0, 0, 0, 0, map[int]bool{}, nil)
	p.generateReused(a, 0, 0, 0, 0, map[int]bool{}, nil) // make a reused, but older than b.
	p.generateReused(b, 0, 0, 0, 0, map[int]bool{}, nil)
	p.generateReused(c, 0, 0, 0, 0, map[int]bool{}, nil) // 30 tokens > budget 20.

	if got := p.cachedPrefixLen(a); got != len(a) {
		t.Fatalf("cost-aware radix should keep reused a, cached %d/%d", got, len(a))
	}
	if got := p.cachedPrefixLen(b); got != 0 {
		t.Fatalf("cost-aware radix should evict one-shot b, cached %d", got)
	}
	if got := p.cachedPrefixLen(c); got != len(c) {
		t.Fatalf("new c prompt should survive its insert-time eviction pass, cached %d/%d", got, len(c))
	}
	st := p.tree.Stats()
	if st.Evictions != 1 || st.CostEvictions != 1 || st.ReuseHits == 0 {
		t.Fatalf("cost-aware radix stats = %+v, want one cost eviction fed by reuse hits", st)
	}
}

// TestInKernelReuseMatchesFullPrefill is the PARITY witness: a second turn that shares a
// long prefix with the first reuses that prefix's KV (through an edge split), and its
// greedy decode is BIT-IDENTICAL to the same turn run with reuse disabled (full prefill).
// It runs BOTH the f32 forward (the path internal/model proves reuse bit-exact on) and the
// Q8_0 forward (quant=true — the path the served gateway actually ships), so the production
// reuse path is witnessed too, not just the proven-bit-exact reference path.
func TestInKernelReuseMatchesFullPrefill(t *testing.T) {
	cfg := tinyCfg()
	sys := synthIDs(cfg.VocabSize, 40, 1) // the shared system/tool-schema prefix
	turn1 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 8, 2)...)
	turn2 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 10, 3)...)
	const maxNew = 12

	for _, quant := range []bool{false, true} {
		name := "f32"
		if quant {
			name = "q8"
		}
		t.Run(name, func(t *testing.T) {
			pon := reusePlanner(true, quant, cfg)
			decode(pon, turn1, maxNew) // prime the cache so turn2 can reuse `sys`
			gotON, matched := decode(pon, turn2, maxNew)
			if matched != len(sys) {
				t.Fatalf("turn2 reused %d tokens, want the shared prefix %d (reuse-through-split)", matched, len(sys))
			}

			poff := reusePlanner(false, quant, cfg)
			decode(poff, turn1, maxNew) // OFF: full prefill every turn
			gotOFF, matchedOFF := decode(poff, turn2, maxNew)
			if matchedOFF != 0 {
				t.Fatalf("reuse-disabled planner must never reuse, matched %d", matchedOFF)
			}

			if !eqInts(gotON, gotOFF) {
				t.Fatalf("[%s] reuse-through-prefix changed the decode (not bit-identical):\n on=%v\noff=%v", name, gotON, gotOFF)
			}
			t.Logf("PARITY[%s]: reuse-through-split (%d/%d reused) == full re-prefill, %d tokens identical", name, matched, len(turn2), len(gotON))
		})
	}
}

func TestInKernelReuseExactHitUsesCachedPromptLogits(t *testing.T) {
	cfg := tinyCfg()
	ids := synthIDs(cfg.VocabSize, 24, 404)
	const maxNew = 8

	pon := reusePlanner(true, false, cfg)
	gotPrime, primeMatched := decode(pon, ids, maxNew)
	if primeMatched != 0 {
		t.Fatalf("first turn unexpectedly reused %d tokens", primeMatched)
	}
	gotReplay, matched := decode(pon, ids, maxNew)
	if want := len(ids); matched != want {
		t.Fatalf("exact replay reused %d tokens, want %d (cached prompt-final logits avoid last-token refeed)", matched, want)
	}
	if !eqInts(gotReplay, gotPrime) {
		t.Fatalf("exact-hit cached-logits replay changed decode:\nprime=%v\nreplay=%v", gotPrime, gotReplay)
	}

	poff := reusePlanner(false, false, cfg)
	gotOFF, matchedOFF := decode(poff, ids, maxNew)
	if matchedOFF != 0 {
		t.Fatalf("reuse-disabled exact replay matched %d", matchedOFF)
	}
	if !eqInts(gotReplay, gotOFF) {
		t.Fatalf("exact-hit replay diverged from full prefill:\nreplay=%v\nfull=%v", gotReplay, gotOFF)
	}
}

func TestInKernelReuseExactHitCachedLogitsDoNotRequireTruncation(t *testing.T) {
	cfg := tinyHybridCfg()
	p := reusePlanner(true, false, cfg)
	ids := synthIDs(cfg.VocabSize, 18, 405)

	decode(p, ids, 1)
	gen, matched := decode(p, ids, 1)
	if matched != len(ids) {
		t.Fatalf("hybrid exact replay reused %d tokens, want full exact hit via cached logits", matched)
	}
	if len(gen) != 1 {
		t.Fatalf("hybrid exact replay generated %d tokens, want 1", len(gen))
	}
}

func TestInKernelBackendGLMDsaEnablesHostPrefixReuse(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend not registered")
	}
	cfg := tinyGLMDsaCfg()
	p := NewInKernelPlanner(model.NewSyntheticGLMDsa(cfg), nil, "glm-dsa-backend", false, backend, false)
	p.quant = false
	if p.tree == nil {
		t.Fatal("GLM-DSA backend planner should enable host radix KV reuse")
	}

	ids := synthIDs(cfg.VocabSize, 9, 406)
	gotPrime, primeMatched := decode(p, ids, 3)
	if primeMatched != 0 {
		t.Fatalf("first backend GLM turn unexpectedly reused %d tokens", primeMatched)
	}
	gotReplay, matched := decode(p, ids, 3)
	if want := len(ids); matched != want {
		t.Fatalf("backend GLM exact replay reused %d tokens, want %d", matched, want)
	}
	if !eqInts(gotReplay, gotPrime) {
		t.Fatalf("backend GLM exact replay changed decode:\nprime=%v\nreplay=%v", gotPrime, gotReplay)
	}

	stats := p.KVMemoryStats()
	if !stats.Enabled || stats.Backend != "radixkv" || stats.Scope != string(compute.MemoryScopeHost) {
		t.Fatalf("backend GLM KV stats should report enabled host radixkv, got %+v", stats)
	}
}

func TestInKernelBackendGLMDsaExactHitCachedLogitsAvoidsRefeed(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	be := &countingBackend{Backend: compute.Default()}
	cfg := tinyGLMDsaCfg()
	p := NewInKernelPlanner(model.NewSyntheticGLMDsa(cfg), nil, "glm-dsa-counting-backend", false, be, false)
	p.quant = false
	ids := synthIDs(cfg.VocabSize, 9, 407)

	decode(p, ids, 0) // prime KV + prompt-final logits
	be.reset()
	gen, matched := decode(p, ids, 1)
	if matched != len(ids) {
		t.Fatalf("exact replay reused %d tokens, want full prompt hit", matched)
	}
	if len(gen) != 1 {
		t.Fatalf("exact replay generated %d tokens, want 1", len(gen))
	}
	if mat, batched := be.ops(); mat != 0 || batched != 0 {
		t.Fatalf("exact replay with cached logits should not refeed the prompt or compute an unused final Step, got mat=%d batched=%d", mat, batched)
	}

	be.reset()
	gen, matched = decode(p, ids, 2)
	if matched != len(ids) || len(gen) != 2 {
		t.Fatalf("two-token exact replay matched=%d generated=%d, want full hit and 2 generated", matched, len(gen))
	}
	if mat, batched := be.ops(); mat == 0 && batched == 0 {
		t.Fatalf("two-token exact replay should compute a Step between generated tokens")
	}
}

func TestInKernelBackendRestoresDemotedPrefixFromHostDRAML2(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	t.Setenv("FAK_INKERNEL_RADIX_HOST_L2_BYTES", "1073741824")
	be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
	cfg := tinyHybridCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "qwen35-host-l2", false, be, false)
	p.quant = false
	ids := synthIDs(cfg.VocabSize, 9, 6851)
	run := func(maxNew int) (gen []int, matched int) {
		_, _, matched, _, _, _, err := p.generateReusedContext(context.Background(), ids, maxNew, 0, 0, 0, map[int]bool{}, func(id int) bool {
			gen = append(gen, id)
			return false
		})
		if err != nil {
			t.Fatalf("generateReused: %v", err)
		}
		return gen, matched
	}

	first, matched := run(1)
	if matched != 0 || len(first) != 1 {
		t.Fatalf("cold turn matched=%d generated=%d", matched, len(first))
	}
	resident, candidates := p.KVPrefixPressuredCandidates()
	if resident <= 0 || len(candidates) != 1 {
		t.Fatalf("native pressure source resident=%d candidates=%+v", resident, candidates)
	}
	staged := p.StageKVPrefixToHost(context.Background(), candidates[0].SpanDigest)
	if staged.Outcome != radixkv.SnapshotTransferOK || staged.BytesMoved <= 0 {
		t.Fatalf("host stage=%+v", staged)
	}
	if evicted := p.EvictHotKVPrefix(candidates[0].SpanDigest); evicted != len(ids) {
		t.Fatalf("hot eviction=%d, want %d", evicted, len(ids))
	}

	second, matched := run(1)
	if matched != len(ids) || !eqInts(second, first) {
		t.Fatalf("host-L2 replay matched=%d/%d tokens=%v want=%v", matched, len(ids), second, first)
	}
	stats := p.KVMemoryStats()
	if stats.L2Hits != 1 || stats.L1Misses != 2 || stats.L2StageBytes != staged.BytesMoved ||
		stats.L2RestoreBytes != staged.BytesMoved || stats.L2HostResidentBytes <= 0 {
		t.Fatalf("host-L2 counters=%+v", stats)
	}
	if stats.L2HostCapacityBytes != 1<<30 {
		t.Fatalf("host-L2 capacity=%d, want %d", stats.L2HostCapacityBytes, int64(1<<30))
	}
}

type inKernelRemoteSnapshotStore struct{ data map[string][]byte }

func (s *inKernelRemoteSnapshotStore) Put(_ context.Context, key string, payload []byte) error {
	if s.data == nil {
		s.data = map[string][]byte{}
	}
	s.data[key] = append([]byte(nil), payload...)
	return nil
}

func (s *inKernelRemoteSnapshotStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	payload, ok := s.data[key]
	return append([]byte(nil), payload...), ok, nil
}

func TestInKernelBackendWritesThroughAndRestoresQwenPrefixFromRemoteL3(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	t.Setenv("FAK_INKERNEL_RADIX_HOST_L2_BYTES", "1073741824")
	be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
	cfg := tinyHybridCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "qwen35-remote-l3", false, be, false)
	p.quant = false
	store := &inKernelRemoteSnapshotStore{}
	if err := p.ConfigureKVPrefixRemote(store); err != nil {
		t.Fatal(err)
	}
	ids := synthIDs(cfg.VocabSize, 9, 6852)
	run := func() (gen []int, matched int) {
		_, _, matched, _, _, _, err := p.generateReusedContext(context.Background(), ids, 1, 0, 0, 0, map[int]bool{}, func(id int) bool {
			gen = append(gen, id)
			return false
		})
		if err != nil {
			t.Fatalf("generateReused: %v", err)
		}
		return gen, matched
	}

	first, matched := run()
	if matched != 0 || len(first) != 1 {
		t.Fatalf("cold turn matched=%d generated=%d", matched, len(first))
	}
	_, candidates := p.KVPrefixPressuredCandidates()
	if len(candidates) != 1 {
		t.Fatalf("native pressure candidates=%+v", candidates)
	}
	digest := candidates[0].SpanDigest
	staged := p.StageKVPrefixToHost(context.Background(), digest)
	if staged.Outcome != radixkv.SnapshotTransferOK || staged.BytesMoved <= 0 || len(store.data) != 1 {
		t.Fatalf("remote write-through=%+v objects=%d", staged, len(store.data))
	}
	if evicted := p.EvictHotKVPrefix(digest); evicted != len(ids) {
		t.Fatalf("hot eviction=%d, want %d", evicted, len(ids))
	}
	p.mu.Lock()
	hostEvicted := p.tree.EvictHostSnapshot(digest)
	p.mu.Unlock()
	if hostEvicted != len(ids) {
		t.Fatalf("host eviction=%d, want %d", hostEvicted, len(ids))
	}

	second, matched := run()
	if matched != len(ids) || !eqInts(second, first) {
		t.Fatalf("remote-L3 replay matched=%d/%d tokens=%v want=%v", matched, len(ids), second, first)
	}
	stats := p.KVMemoryStats()
	if !stats.L3Enabled || stats.L3Hits != 1 || stats.L3HitTokens != len(ids) ||
		stats.L3StageBytes != staged.BytesMoved || stats.L3RestoreBytes != staged.BytesMoved ||
		stats.L3Faults != 0 || stats.L3ReferencedBytes <= 0 || stats.L2HostResidentBytes != 0 {
		t.Fatalf("remote-L3 counters=%+v", stats)
	}
}

func TestInKernelReuseHybridSplitFallsBackInsteadOfPanicking(t *testing.T) {
	cfg := tinyHybridCfg()
	p := reusePlanner(true, false, cfg)
	sys := synthIDs(cfg.VocabSize, 24, 1143)
	turn1 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 6, 1144)...)
	turn2 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 7, 1145)...)

	decode(p, turn1, 1)
	gen, matched := decode(p, turn2, 1)
	if matched != 0 {
		t.Fatalf("hybrid split reused %d tokens, want fallback re-prefill because GDN state cannot be truncated", matched)
	}
	if len(gen) != 1 {
		t.Fatalf("hybrid fallback generated %d tokens, want 1", len(gen))
	}
	if got := p.cachedPrefixLen(turn2); got != len(turn2) {
		t.Fatalf("hybrid turn not cached after fallback: %d/%d", got, len(turn2))
	}
}

// TestInKernelPoisonEvictionForcesReprefill is the POISON witness (#14): two turns sharing
// a system prefix are cached; quarantining one evicts ONLY its branch, so the next turn on
// the poisoned transcript RE-PREFILLS (cannot replay the poisoned KV) while the benign
// sibling stays fully cached.
func TestInKernelPoisonEvictionForcesReprefill(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	sys := synthIDs(cfg.VocabSize, 32, 10)
	good := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 8, 11)...)
	bad := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 8, 12)...) // a poisoned tool-result tail

	decode(p, good, 4) // cache the benign turn
	_, mBad := decode(p, bad, 4)
	if mBad != len(sys) {
		t.Fatalf("the poisoned turn should reuse the shared prefix, matched %d want %d", mBad, len(sys))
	}
	if got := p.cachedPrefixLen(good); got != len(good) {
		t.Fatalf("benign turn not fully cached: %d/%d", got, len(good))
	}
	if got := p.cachedPrefixLen(bad); got != len(bad) {
		t.Fatalf("poisoned turn not fully cached: %d/%d", got, len(bad))
	}

	freed := p.evictPoisonedIDs(bad) // the quarantine verdict
	if want := len(bad) - len(sys); freed != want {
		t.Fatalf("evicted %d tokens, want %d (the poisoned tail only)", freed, want)
	}

	if got := p.cachedPrefixLen(good); got != len(good) {
		t.Errorf("benign sibling was evicted! cached %d/%d", got, len(good))
	}
	if got := p.cachedPrefixLen(bad); got != len(sys) {
		t.Errorf("poisoned KV survived: cached %d, want %d (only the shared prefix)", got, len(sys))
	}
	if _, m := decode(p, bad, 4); m != len(sys) {
		t.Errorf("next turn on the poisoned transcript reused %d, want %d (must re-prefill the poison)", m, len(sys))
	}
	t.Logf("POISON: quarantine freed %d poisoned tokens; benign sibling preserved; poison re-prefills", freed)
}

// TestInKernelReuseMultiTurnPrefillSavings is the PERF witness: a growing multi-turn
// conversation prefills far fewer tokens with reuse ON (each turn reuses the full prior
// prefix), and the real wall-clock speedup is reported. The deterministic prefill-token
// saving is the non-flaky assertion; the milliseconds are the measured number.
func TestInKernelReuseMultiTurnPrefillSavings(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 128, NumLayers: 4, NumHeads: 8, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 256, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 255,
	}
	const turns = 6
	base := synthIDs(cfg.VocabSize, 320, 7) // a long static system+tool-schema prefix
	convo := make([][]int, turns)
	ctx := append([]int{}, base...)
	for i := 0; i < turns; i++ {
		if i > 0 {
			ctx = append(ctx, synthIDs(cfg.VocabSize, 16, uint64(700+i))...) // a short new user turn
		}
		convo[i] = append([]int{}, ctx...)
	}

	runConvo := func(reuse bool) (computed int, dur time.Duration) {
		p := reusePlanner(reuse, false, cfg)
		t0 := time.Now()
		for _, ids := range convo {
			_, _, matched, _, _, _ := p.generateReused(ids, 2, 0, 0, 0, map[int]bool{}, nil)
			computed += len(ids) - matched
		}
		return computed, time.Since(t0)
	}

	// warm the kernel so the first timed prefill isn't a cold outlier.
	reusePlanner(false, false, cfg).generateReused(synthIDs(cfg.VocabSize, 8, 1), 1, 0, 0, 0, map[int]bool{}, nil)

	computedON, durON := runConvo(true)
	computedOFF, durOFF := runConvo(false)

	if computedON >= computedOFF {
		t.Fatalf("reuse did not cut prefill work: computed ON=%d OFF=%d", computedON, computedOFF)
	}
	saved := 100 * (1 - float64(computedON)/float64(computedOFF))
	speedup := float64(durOFF) / float64(durON)
	t.Logf("PERF (%d turns, base=%d): prefill tokens computed ON=%d OFF=%d (%.0f%% saved); wall %s -> %s (%.2fx)",
		turns, len(base), computedON, computedOFF, saved, durOFF.Round(time.Microsecond), durON.Round(time.Microsecond), speedup)
}

func TestInKernelKVMemoryStatsReportsResidentFootprint(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	first := synthIDs(cfg.VocabSize, 12, 90)
	second := append(append([]int{}, first...), synthIDs(cfg.VocabSize, 5, 91)...)

	decode(p, first, 0)
	_, matched := decode(p, second, 0)
	if matched != len(first) {
		t.Fatalf("second turn reused %d tokens, want first prefix %d", matched, len(first))
	}

	stats := p.KVMemoryStats()
	if !stats.Enabled {
		t.Fatalf("KV memory stats should report enabled for a radix-backed planner: %+v", stats)
	}
	if stats.Backend != "radixkv" || stats.MemoryClass != string(compute.MemoryKVCache) || stats.Scope != string(compute.MemoryScopeHost) {
		t.Fatalf("unexpected KV memory labels: %+v", stats)
	}
	wantBytesPerToken := compute.EstimateKVStoreBytes(compute.KVConfig{
		NumLayers:  cfg.NumLayers,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		RopeTheta:  cfg.RopeTheta,
	}, 1)
	if stats.BytesPerToken != wantBytesPerToken {
		t.Fatalf("bytes/token = %d, want %d", stats.BytesPerToken, wantBytesPerToken)
	}
	if stats.ResidentTokens <= stats.LRUTokens {
		t.Fatalf("resident PrefixTokens should exceed LRU edge-token count for nested prefixes: %+v", stats)
	}
	if want := compute.EstimateKVStoreBytes(compute.KVConfig{
		NumLayers:  cfg.NumLayers,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		RopeTheta:  cfg.RopeTheta,
	}, stats.ResidentTokens); stats.ResidentBytes != want {
		t.Fatalf("resident bytes = %d, want %d from PrefixTokens=%d", stats.ResidentBytes, want, stats.ResidentTokens)
	}
	if total, _, known := compute.HostSystemMemoryInfo(); known {
		if !stats.CapacityKnown || stats.CapacityTotalBytes != total {
			t.Fatalf("host capacity known but KV stats did not report it: total=%d stats=%+v", total, stats)
		}
		if stats.HeadroomRatio != inKernelKVMemoryHeadroom {
			t.Fatalf("KV headroom = %g, want %g", stats.HeadroomRatio, inKernelKVMemoryHeadroom)
		}
		if stats.FitBudgetBytes <= 0 || stats.FitMarginBytes != stats.FitBudgetBytes-stats.ResidentBytes {
			t.Fatalf("invalid KV fit budget/margin: %+v", stats)
		}
	}
	if stats.Nodes == 0 || stats.Leaves == 0 || stats.MaxDepthTokens != len(second) {
		t.Fatalf("tree shape not reflected in KV memory stats: %+v", stats)
	}
}

// TestInKernelReuseConcurrentNoRace drives concurrent turns, probes, and evictions through
// the shared tree so `go test -race` proves the planner mutex serializes every tree access
// (a broken build/race here would otherwise accumulate silently — see the -race gate note).
func TestInKernelReuseConcurrentNoRace(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	sys := synthIDs(cfg.VocabSize, 24, 50)

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			ids := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 6, uint64(100+g))...)
			for i := 0; i < 12; i++ {
				p.generateReused(ids, 3, 0, 0, 0, map[int]bool{}, nil)
				_ = p.cachedPrefixLen(ids)
				if i%4 == 0 {
					p.evictPoisonedIDs(ids) // exercise eviction under contention
				}
			}
		}(g)
	}
	wg.Wait()

	// Post-contention: the planner must still be usable and the tree must not
	// have leaked or deadlocked under concurrent access.
	post := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 4, 200)...)
	gen, matched := decode(p, post, 3)
	if matched != len(sys) {
		t.Errorf("post-contention reuse: matched %d, want shared prefix %d", matched, len(sys))
	}
	if len(gen) != 3 {
		t.Errorf("post-contention decode produced %d tokens, want 3", len(gen))
	}
	if p.cachedPrefixLen(post) == 0 {
		t.Error("post-contention cache empty after concurrent operations")
	}
	t.Logf("RACE: planner functional post-contention, matched %d/%d, generated %d tokens", matched, len(post), len(gen))
}

// TestInKernelPlannerRadixReuseIsDefaultOnForInkernelEngine is the #14 DEFAULT witness:
// a plain in-kernel planner — the exact object `fak serve --engine inkernel` (the default
// engine, cmd/fak/serve.go `-engine`) boots via the gateway's in-kernel planner path
// (internal/gateway/gateway.go newInKernelPlanner → agent.NewInKernelPlanner) — turns on
// local radix/prefix KV reuse with NO env opt-in and NO benchmark harness, and actually
// clones/reuses a shared stable prefix across turns. The other reuse witnesses in this file
// either set FAK_INKERNEL_RADIX=on explicitly (TestInKernelRadixCostAwareVictimRuleFromEnv,
// TestInKernelBackendGLMDsaEnablesHostPrefixReuse) or bypass the constructor entirely
// (reusePlanner hand-builds p.tree); NONE pins that reuse is the DEFAULT, which is the whole
// of item 14. This proves both directions: default (no env) is ON, and FAK_INKERNEL_RADIX=off
// is the documented — and only — A/B disable (cold-path correctness: off falls back to a full
// prefill every turn, never a stale serve).
func TestInKernelPlannerRadixReuseIsDefaultOnForInkernelEngine(t *testing.T) {
	// Force a clean env so an ambient FAK_INKERNEL_RADIX from the shell cannot decide the
	// default: "" is not "off", so NewInKernelPlanner takes the default-on constructor path.
	t.Setenv("FAK_INKERNEL_RADIX", "")
	cfg := tinyCfg()

	// backend == nil is precisely the CPU in-kernel session path --engine inkernel serves.
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	if p.tree == nil {
		t.Fatal("radix/prefix reuse must be ON by default for an --engine inkernel session (no FAK_INKERNEL_RADIX set)")
	}
	p.quant = false

	// It genuinely clones/reuses a shared stable prefix across turns — no fanbench/ablate.
	sys := synthIDs(cfg.VocabSize, 40, 1)
	turn1 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 8, 2)...)
	turn2 := append(append([]int{}, sys...), synthIDs(cfg.VocabSize, 10, 3)...)
	decode(p, turn1, 4) // prime the shared prefix
	if _, matched := decode(p, turn2, 4); matched != len(sys) {
		t.Fatalf("default in-kernel session did not reuse the shared prefix: matched %d, want %d", matched, len(sys))
	}

	// FAK_INKERNEL_RADIX=off is the documented disable — verify it is the ONLY off switch.
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	off := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	if off.tree != nil {
		t.Fatal("FAK_INKERNEL_RADIX=off must disable reuse (the A/B tree-OFF arm)")
	}
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "fak-inkernel", ModelID: "synthetic", PromptTokens: int64(len(turn2)), CachedTokens: int64(len(sys))})
	if entry.Metrics.PrefillTokensSaved != int64(len(sys)) || entry.ID.Length != int64(len(turn2)) {
		t.Fatalf("kernel reuse metadata=%+v", entry.Metrics)
	}
	t.Logf("DEFAULT[#14]: --engine inkernel enables radix/prefix reuse with no env opt-in and no harness; reused %d-token shared prefix, off-arm disables it", len(sys))
}

func TestInKernelAuthenticatedPrefixPrivateUntilPromoted(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	p.quant = false
	ids := synthIDs(cfg.VocabSize, 12, 4774)
	ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
	ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
	run := func(ctx context.Context) (cacheable, matched int) {
		_, _, cacheable, matched, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 0, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return cacheable, matched
	}
	if cacheable, matched := run(ctxA); cacheable != 0 || matched != 0 {
		t.Fatalf("first tenant-A request cacheable=%d matched=%d", cacheable, matched)
	}
	if cacheable, matched := run(ctxA); cacheable != len(ids) || matched != len(ids) {
		t.Fatalf("same-tenant request cacheable=%d matched=%d, want %d/%d", cacheable, matched, len(ids), len(ids))
	}
	if cacheable, matched := run(ctxB); cacheable != 0 || matched != 0 {
		t.Fatalf("tenant-B observed tenant-A private hit: cacheable=%d matched=%d", cacheable, matched)
	}
	if err := p.scopedTree.Promote(radixkv.ScopeTenant, radixkv.CacheIdentity{Tenant: "tenant-a"}, ids); err != nil {
		t.Fatal(err)
	}
	if cacheable, matched := run(ctxB); cacheable != len(ids) || matched != len(ids) {
		t.Fatalf("promoted fleet request cacheable=%d matched=%d, want %d/%d", cacheable, matched, len(ids), len(ids))
	}
}

func TestInKernelSnapshotCheckpointCreatesStableSiblingBoundary(t *testing.T) {
	tests := []struct {
		name            string
		matched, prompt int
		want            int
	}{
		{name: "short prompt", prompt: 63, want: 0},
		{name: "one block plus suffix", prompt: 65, want: 64},
		{name: "exact block stays leaf", prompt: 64, want: 0},
		{name: "long prompt uses deepest boundary", prompt: 570, want: 512},
		{name: "restored boundary is not repeated", matched: 512, prompt: 570, want: 0},
		{name: "restored older prefix advances once", matched: 256, prompt: 570, want: 512},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inKernelSnapshotCheckpoint(tt.matched, tt.prompt); got != tt.want {
				t.Fatalf("checkpoint(%d, %d)=%d want %d", tt.matched, tt.prompt, got, tt.want)
			}
		})
	}
}

func TestInKernelBackendWithoutPrefixReuseSkipsCheckpointAdmission(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "no-device-prefix-reuse", false, be, false)
	p.quant = false
	if p.tree != nil {
		t.Fatal("generic device model must leave unsupported prefix reuse disabled")
	}
	ids := synthIDs(cfg.VocabSize, inKernelSnapshotCheckpointTokens+8, 6917)
	if _, _, _, _, _, _, err := p.generateReusedContext(context.Background(), ids, 1, 0, 0, 0, map[int]bool{}, nil); err != nil {
		t.Fatalf("device decode without prefix reuse: %v", err)
	}
}
