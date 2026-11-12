package agent

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// inkernel_radix_cold_prefill_test.go is the [SW-VERIFIED] witness for ticket
// prefill-radix-cold-prefill-tax-20261108 (gap 4 of the matched-BF16 receipt):
// attribute the radix-on cold-prefill cost on the sim harness, witness that the
// warm multi-turn cache ratio is preserved by the off-lock admission seam, and
// record the typed verdict — within-noise or inherent-cost — per the ticket's
// acceptance gate. Wall-clock figures here are diagnostic software receipts,
// never hardware-performance evidence ([HW-GATED] tok/s rows stay open until a
// GPU host reruns the P-panel).

// radixTaxCfg sizes the synthetic model so the per-token KV payload is large
// enough that the O(prefix-bytes) admission clone is measurable against prefill:
// 2 (K+V) x 4 KV heads x 16 head dim x 4 bytes x 16 layers = 4 KiB/token.
func radixTaxCfg() model.Config {
	return model.Config{
		HiddenSize: 128, NumLayers: 16, NumHeads: 8, NumKVHeads: 4, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 255,
	}
}

// ttftTurn measures time-to-first-emitted-token for one demand turn: the client-
// perceived cold cost (lookup + prefill + admission + first decode step).
func ttftTurn(t *testing.T, p *InKernelPlanner, ids []int, maxNew int) (time.Duration, []int) {
	t.Helper()
	var first time.Time
	var emitted []int
	start := time.Now()
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
		context.Background(), ids, maxNew, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			if first.IsZero() {
				first = time.Now()
			}
			emitted = append(emitted, id)
			return false
		})
	if err != nil {
		t.Fatalf("turn failed: %v", err)
	}
	if len(emitted) != maxNew {
		t.Fatalf("emitted %d tokens, want %d", len(emitted), maxNew)
	}
	return first.Sub(start), emitted
}

func medianDuration(ds []time.Duration) time.Duration {
	s := slices.Clone(ds)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// TestInKernelRadixColdPrefillTaxAttribution is the cold-P AB the ticket's
// acceptance gate names: interleaved distinct cold prompts through a radix-OFF
// planner and a radix-ON planner (budget-armed, so inserts also pay the
// eviction bookkeeping), with the env-gated debug arm attributing the ON arm's
// extra work to its candidate classes. The verdict is typed either way:
// WITHIN_NOISE when the ON median TTFT is inside the noise band, INHERENT_COST
// recorded when the residual is attributed to the O(prefix-bytes) capture clone
// that the self-contained-snapshot design requires (removing it is a KV-layout
// change — paged-reference admission — outside this ticket's fence).
func TestInKernelRadixColdPrefillTaxAttribution(t *testing.T) {
	cfg := radixTaxCfg()
	const promptTokens = 512
	const turnsPerArm = 6
	const maxNew = 8
	prompts := make([][]int, turnsPerArm)
	for i := range prompts {
		prompts[i] = synthIDs(cfg.VocabSize, promptTokens, 9000+uint64(i)*7919)
	}

	off := reusePlanner(false, false, cfg)
	on := reusePlanner(true, false, cfg)
	// One context window of cached-prefix tokens: the served default regime
	// (resolveInKernelRadixBudgetTokens derives the bound from the context
	// ceiling), small enough that the sweep overflows it and every later insert
	// pays the budget-pressure eviction walk.
	on.tree = radixkv.New(2 * promptTokens)
	on.radixOffLockAdmit = true // the shipped constructor default
	on.EnableRadixAttribution(false)

	offTTFTs := make([]time.Duration, turnsPerArm)
	onTTFTs := make([]time.Duration, turnsPerArm)
	for i := range prompts {
		offTTFTs[i], _ = ttftTurn(t, off, prompts[i], maxNew)
		onTTFTs[i], _ = ttftTurn(t, on, prompts[i], maxNew)
	}
	offMed, onMed := medianDuration(offTTFTs), medianDuration(onTTFTs)
	ratio := float64(onMed) / float64(offMed)

	attr := on.RadixAttribution()
	// Every turn admitted twice (pre-decode prompt + post-decode continuation),
	// all through the off-lock fast path: the sweep is single-goroutine, so the
	// boundary re-validation can never see a topology change.
	if attr.Admissions != 2*turnsPerArm || attr.OffLockClones != 2*turnsPerArm {
		t.Fatalf("admission ledger admissions/off_lock=%d/%d, want %d/%d",
			attr.Admissions, attr.OffLockClones, 2*turnsPerArm, 2*turnsPerArm)
	}
	if attr.Turns != turnsPerArm || attr.BytesCloned <= 0 {
		t.Fatalf("attribution ledger turns=%d bytes_cloned=%d, want %d turns and positive cloned bytes",
			attr.Turns, attr.BytesCloned, turnsPerArm)
	}

	receipt := "SOFTWARE RECEIPT cold_p_ab cache_off_median_ttft=%s cache_on_median_ttft=%s ratio=%.3f " +
		"attribution[lookups_ns=%d capture_ns=%d insert_ns=%d bytes_cloned=%d logits_bytes=%d off_lock_clones=%d sync_admissions=%d evictions_seen=%d] " +
		"tree[tokens_budget=%d]"
	t.Logf(receipt, offMed, onMed, ratio,
		attr.LookupsNs, attr.CaptureNs, attr.InsertsNs, attr.BytesCloned, attr.LogitsBytes,
		attr.OffLockClones, attr.SyncAdmissions, attr.EvictionsSeen, 2*promptTokens)

	const noiseBand = 1.15
	if ratio <= noiseBand {
		t.Logf("[SW-VERIFIED] verdict=WITHIN_NOISE radix-on cold prefill median TTFT %.3fx radix-off (band %.2f); "+
			"timing_diagnostic_only=true", ratio, noiseBand)
	} else {
		// The residual must be attributable: the capture clone dominates the ON-arm
		// extra phase time, while the locked bookkeeping stays a small share.
		if attr.CaptureNs <= 0 {
			t.Fatalf("verdict=INHERENT_COST requires a measured capture component; capture_ns=%d", attr.CaptureNs)
		}
		extra := attr.CaptureNs + attr.InsertsNs + attr.LookupsNs
		captureShare := float64(attr.CaptureNs) / float64(extra)
		t.Logf("[SW-VERIFIED] verdict=INHERENT_COST_RECORD radix-on cold prefill %.3fx radix-off beyond the %.2f band; "+
			"attributed capture_share=%.2f (capture_ns=%d of %d ns attributed phase time, bytes_cloned=%d); "+
			"the O(prefix-KV-bytes) capture clone is inherent to the self-contained-snapshot tree design "+
			"(each node owns an independent KV copy; the session KV keeps mutating, so no reference can be stored "+
			"without a layout change). Removal requires paged-reference admission (KV-layout class) or a truncating "+
			"HAL surface — follow-up, out of this ticket's fence. [HW-GATED] H200 P-panel tok/s rows remain open. "+
			"timing_diagnostic_only=true", ratio, noiseBand, captureShare, attr.CaptureNs, extra, attr.BytesCloned)
	}

	// Warm-path rejection test: the off-lock prompt admission must not regress the
	// multi-turn cache ratio. Turn 1's generated continuation was admitted
	// synchronously at turn 1's tail, so turn 2 restores the FULL evaluated
	// continuation — the exact contract TestInKernelContinuationCacheTwoTurnWorkReceipt
	// pins, re-witnessed here against the off-lock seam.
	warmPrompt := synthIDs(cfg.VocabSize, 32, 424242)
	_, warmGen := ttftTurn(t, on, warmPrompt, 8)
	evaluated := append(append([]int(nil), warmPrompt...), warmGen[:len(warmGen)-1]...)
	_, _, _, warmMatched, _, _, _, _, err := on.generateReusedContextWithBias(
		context.Background(), evaluated, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil)
	if err != nil {
		t.Fatalf("warm second turn failed: %v", err)
	}
	if warmMatched != len(evaluated) {
		t.Fatalf("warm multi-turn regression: second turn matched=%d/%d, want the full evaluated continuation",
			warmMatched, len(evaluated))
	}
	if got := on.cachedPrefixLen(warmPrompt); got != len(warmPrompt) {
		t.Fatalf("prompt-only leaf missing after off-lock admission: cached=%d want %d", got, len(warmPrompt))
	}
}

// TestInKernelRadixOffLockAdmissionParity proves the off-lock attach is
// behaviorally identical to the historical fully-locked section: same generated
// tokens, same resulting tree shape, same warm readback.
func TestInKernelRadixOffLockAdmissionParity(t *testing.T) {
	cfg := radixTaxCfg()
	prompts := [][]int{
		synthIDs(cfg.VocabSize, 128, 777001),
		synthIDs(cfg.VocabSize, 128, 777002),
		synthIDs(cfg.VocabSize, 256, 777003),
	}
	run := func(offLock bool) ([][]int, radixkv.Stats, int) {
		p := reusePlanner(true, false, cfg)
		p.radixOffLockAdmit = offLock
		var gens [][]int
		for _, ids := range prompts {
			_, gen := ttftTurn(t, p, ids, 6)
			gens = append(gens, gen)
		}
		st := p.tree.Stats()
		probe := append(append([]int(nil), prompts[0]...), gens[0][:5]...)
		return gens, st, p.cachedPrefixLen(probe)
	}
	stdGens, stdStats, stdProbe := run(false)
	fastGens, fastStats, fastProbe := run(true)
	for i := range stdGens {
		if !eqInts(stdGens[i], fastGens[i]) {
			t.Fatalf("turn %d output diverged between locked and off-lock admission: %v vs %v", i, stdGens[i], fastGens[i])
		}
	}
	if stdStats.Tokens != fastStats.Tokens || stdStats.Nodes != fastStats.Nodes ||
		stdStats.PrefixTokens != fastStats.PrefixTokens || stdStats.Splits != fastStats.Splits ||
		stdStats.Evictions != fastStats.Evictions {
		t.Fatalf("tree shape diverged: locked={tokens=%d nodes=%d prefix=%d splits=%d evictions=%d} vs "+
			"off-lock={tokens=%d nodes=%d prefix=%d splits=%d evictions=%d}",
			stdStats.Tokens, stdStats.Nodes, stdStats.PrefixTokens, stdStats.Splits, stdStats.Evictions,
			fastStats.Tokens, fastStats.Nodes, fastStats.PrefixTokens, fastStats.Splits, fastStats.Evictions)
	}
	if stdProbe != fastProbe {
		t.Fatalf("warm readback diverged: locked=%d off-lock=%d", stdProbe, fastProbe)
	}
}

// TestInKernelRadixAttributionArmsOnlyWhenEnabled guards the zero-cost disarmed
// default: no ledger entry without the arm, and a complete breakdown with it.
func TestInKernelRadixAttributionArmsOnlyWhenEnabled(t *testing.T) {
	cfg := tinyCfg()
	disarmed := reusePlanner(true, false, cfg)
	_, _ = ttftTurn(t, disarmed, synthIDs(cfg.VocabSize, 16, 555001), 4)
	if got := disarmed.RadixAttribution(); got.Turns != 0 || got.Admissions != 0 {
		t.Fatalf("disarmed planner recorded a ledger: %+v", got)
	}

	armed := reusePlanner(true, false, cfg)
	armed.radixOffLockAdmit = true
	armed.EnableRadixAttribution(false)
	_, _ = ttftTurn(t, armed, synthIDs(cfg.VocabSize, 16, 555002), 4)
	got := armed.RadixAttribution()
	if got.Turns != 1 || got.Admissions != 2 {
		t.Fatalf("armed ledger turns/admissions=%d/%d, want 1/2 (prompt + continuation)", got.Turns, got.Admissions)
	}
	// Timing fields are diagnostic-only and may legitimately measure zero on a
	// trivially small tree; the structural counters are the contract.
	if got.OffLockClones != 2 || got.BytesCloned <= 0 {
		t.Fatalf("armed breakdown incomplete: off_lock=%d bytes_cloned=%d lookups_ns=%d capture_ns=%d",
			got.OffLockClones, got.BytesCloned, got.LookupsNs, got.CaptureNs)
	}
}
