package agent

import (
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
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

func kvConfigFromModelConfig(cfg model.Config) compute.KVConfig {
	return compute.KVConfig{
		NumLayers:  cfg.NumLayers,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		RopeTheta:  cfg.RopeTheta,
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

	// Post-contention the shared tree must be coherent, not just race-free: the
	// common system prefix every goroutine generated is resident, and a probe
	// returns a match length bounded by the prefix it was asked about (the mutex
	// serialized every tree access, so no torn/over-long match survived).
	matched := p.cachedPrefixLen(sys)
	if matched <= 0 {
		t.Fatalf("system prefix not resident after concurrent turns: cachedPrefixLen=%d", matched)
	}
	if matched > len(sys) {
		t.Fatalf("match length %d exceeds probed prefix len %d (torn shared-tree state)", matched, len(sys))
	}
}
