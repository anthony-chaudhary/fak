// Command o1proof1b witnesses the ctxplan->kvmmu O(1) residency bridge
// (CLAIMS.md:141, issue #550) at a scale close to a real ~1.5B-parameter
// model, entirely on this machine: no network, no GPU, no HuggingFace
// export. It reuses model.NewSynthetic with the exact "qwen25-1.5b" shape
// cmd/sessionbench names (HiddenSize 1536, 28 layers, vocab 151936 — see
// cmd/sessionbench/main.go's syntheticShape) — a deterministic in-memory
// transformer with fixed-seed pseudo-random weights.
//
// Honest fence, same one CLAIMS.md and internal/model/synthetic.go already
// state for the shipped tiny witness: the logits are numerically
// meaningless (no real weights). What this proves is the CACHE MECHANICS —
// append/evict/re-RoPE/renumber — which are structurally correct for ANY
// weights of the declared shape, so exercising them at ~1.5B scale is a
// legitimate scale-up of the shipped proof, not a new claim. Numerics
// (real-weight correctness) are proven separately by internal/model's
// oracle tests against real HuggingFace exports, not by this program.
//
// Three phases, all against the SAME synthetic ~1.5B model instance (built
// once — weights are fixed, only session/KV state varies per phase):
//
//  1. Bounded (O(1)) scaling: append many turns, applying a ctxplan.Plan
//     after each one that keeps only the system span + the last K turns
//     resident (kvmmu.Context.ApplyPlan evicts the rest). Witnesses that KV
//     residency (Context.CacheLen) stays CONSTANT and per-turn append cost
//     stays FLAT as the number of turns M grows arbitrarily large.
//  2. Naive scaling: the same turns, same model, but nothing is ever
//     evicted — residency grows O(M), the contrast case.
//  3. Bit-exact + non-vacuity witness: the exact proof shape of
//     internal/kvmmu/planbridge_test.go's
//     TestApplyPlanEvictsElidedToO1Residency, run at this scale. A "kept
//     everything" reference and a "resident-only" reference MUST differ
//     (non-vacuity: the elided spans genuinely mattered), and the actual
//     bounded session's post-query distribution MUST be bit-identical
//     (max|Delta|=0) to the resident-only reference (eviction ==
//     never-having-seen the elided spans, byte for byte).
//
// Run: go run ./cmd/o1proof1b
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func qwen25_1_5b() model.Config {
	// Mirrors cmd/sessionbench/main.go's syntheticShape("qwen25-1.5b") exactly —
	// the closest existing named shape to "a 1B model or similar" (~1.5-1.65B
	// params: 28 layers x [attn ~5.5M + mlp ~41M] + tied 1536x151936 embedding).
	return model.Config{
		HiddenSize:        1536,
		NumLayers:         28,
		NumHeads:          12,
		NumKVHeads:        2,
		HeadDim:           128,
		IntermediateSize:  8960,
		VocabSize:         151936,
		RMSNormEps:        1e-6,
		RopeTheta:         1000000,
		TieWordEmbeddings: true,
		EOSTokenID:        151643,
		HiddenAct:         "silu",
		ModelType:         "qwen2",
	}
}

func lcgIDs(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func cat(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func maxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var mx float64
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		if d < 0 {
			d = -d
		}
		if d > mx {
			mx = d
		}
	}
	return mx
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// boundedRun appends sys + each turn, applying an over-budget elision plan
// after every turn so only sys + the last K turns ever stay resident. It
// returns the per-turn append wall-clock and the KV residency (CacheLen)
// observed right after each turn's plan is applied.
func boundedRun(m *model.Model, sysToks []int, turns [][]int, k int) (durs []time.Duration, cacheLens []int, c *kvmmu.Context) {
	s := m.NewSession()
	c = kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sysToks)

	var window []string // turn IDs currently resident, oldest first
	for i, tk := range turns {
		id := fmt.Sprintf("t%d", i)
		t0 := time.Now()
		c.Append(id, "turn", tk)
		window = append(window, id)

		plan := ctxplan.Plan{
			Objective: ctxplan.ObjGreedy,
			Selected:  []ctxplan.Selection{{ID: "sys", Step: 0}},
		}
		if len(window) > k {
			oldest := window[0]
			window = window[1:]
			plan.Elided = append(plan.Elided, ctxplan.Elision{
				ID: oldest, Step: i, Digest: "d-" + oldest, Reason: ctxplan.ElideOverBudget,
			})
		}
		for j, wid := range window {
			plan.Selected = append(plan.Selected, ctxplan.Selection{ID: wid, Step: i - len(window) + j + 1})
		}
		plan.Candidates = len(plan.Selected) + len(plan.Elided)

		if w := ctxplan.Audit(plan); !w.Faithful {
			panic(fmt.Sprintf("o1proof1b: turn %d plan not faithful: %+v", i, w))
		}
		c.ApplyPlan(plan)

		durs = append(durs, time.Since(t0))
		cacheLens = append(cacheLens, c.CacheLen())
	}
	return durs, cacheLens, c
}

// naiveRun appends sys + each turn and NEVER evicts — the O(M) contrast arm.
func naiveRun(m *model.Model, sysToks []int, turns [][]int) (durs []time.Duration, cacheLens []int, c *kvmmu.Context) {
	s := m.NewSession()
	c = kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sysToks)
	for i, tk := range turns {
		id := fmt.Sprintf("t%d", i)
		t0 := time.Now()
		c.Append(id, "turn", tk)
		durs = append(durs, time.Since(t0))
		cacheLens = append(cacheLens, c.CacheLen())
	}
	return durs, cacheLens, c
}

func main() {
	turnsBounded := flag.Int("turns-bounded", 128, "turns processed in the bounded (O(1)) scaling arm")
	turnsNaive := flag.Int("turns-naive", 48, "turns processed in the naive (O(M)) scaling arm")
	turnLen := flag.Int("turn-len", 24, "tokens per turn span")
	sysLen := flag.Int("sys-len", 8, "tokens in the pinned system span")
	queryLen := flag.Int("query-len", 8, "tokens in the final query span")
	window := flag.Int("window", 4, "K: turns kept resident in the bounded arm")
	exactResident := flag.Int("exact-resident-turns", 3, "prefix turns kept resident in the bit-exact witness (phase 3)")
	exactElided := flag.Int("exact-elided-turns", 10, "suffix turns appended then elided in the bit-exact witness (phase 3)")
	flag.Parse()

	fail := false
	failf := func(format string, a ...any) {
		fail = true
		fmt.Printf("FAIL: "+format+"\n", a...)
	}

	cfg := qwen25_1_5b()
	fmt.Printf("=== o1proof1b: O(1) residency-bridge witness at ~1.5B-parameter synthetic scale ===\n")
	fmt.Printf("config: hidden=%d layers=%d heads=%d/%d headdim=%d intermediate=%d vocab=%d (qwen2.5-1.5b shape, synthetic weights)\n",
		cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize, cfg.VocabSize)

	t0 := time.Now()
	m := model.NewSynthetic(cfg)
	fmt.Printf("model build: %.1fs (no download, no GPU, no network — deterministic in-memory weights)\n", time.Since(t0).Seconds())

	sysToks := lcgIDs(*sysLen, cfg.VocabSize, 1)
	queryToks := lcgIDs(*queryLen, cfg.VocabSize, 999)

	// ---- Phase 1: bounded (O(1)) scaling ----
	maxTurns := *turnsBounded
	if *turnsNaive > maxTurns {
		maxTurns = *turnsNaive
	}
	if n := *exactResident + *exactElided; n > maxTurns {
		maxTurns = n
	}
	allTurns := make([][]int, maxTurns)
	for i := range allTurns {
		allTurns[i] = lcgIDs(*turnLen, cfg.VocabSize, uint64(100+i))
	}

	// NOTE on scope: this phase's recency-window pattern (evict the OLDEST
	// turn once a newer one has already been appended on top of it) is a
	// RESIDENCY/memory-footprint proof only. It is honestly NOT a bit-exact
	// claim — a surviving turn's cached K/V was computed while the
	// now-evicted older turn was still present, so kvmmu.go's own doc comment
	// on ApplyPlan flags exactly this direction ("eliding an earlier span a
	// survivor already attended to shrinks residency but is not reported
	// bit-exact"). Phase 3 below proves bit-exactness in the direction the
	// bridge actually guarantees it.
	fmt.Printf("\n--- Phase 1: bounded arm, %d turns, window K=%d (residency/footprint claim only) ---\n", *turnsBounded, *window)
	bDurs, bLens, _ := boundedRun(m, sysToks, allTurns[:*turnsBounded], *window)
	wantResident := *sysLen + *window*(*turnLen)
	firstLen, lastLen := bLens[0], bLens[len(bLens)-1]
	firstMS, lastMS := ms(bDurs[0]), ms(bDurs[len(bDurs)-1])
	fmt.Printf("KV residency: turn 1 = %d, turn %d = %d (constant target = %d)\n", firstLen, *turnsBounded, lastLen, wantResident)
	fmt.Printf("per-turn append cost: turn 1 = %.1fms, turn %d = %.1fms\n", firstMS, *turnsBounded, lastMS)
	if lastLen != wantResident {
		failf("bounded arm residency after %d turns = %d, want %d (O(1) claim violated)", *turnsBounded, lastLen, wantResident)
	}
	// every turn once the window has filled must show the SAME residency.
	for i := *window; i < len(bLens); i++ {
		if bLens[i] != wantResident {
			failf("bounded arm residency at turn %d = %d, want constant %d", i+1, bLens[i], wantResident)
			break
		}
	}

	// ---- Phase 2: naive (O(M)) scaling, for contrast ----
	fmt.Printf("\n--- Phase 2: naive arm (no eviction), %d turns ---\n", *turnsNaive)
	nDurs, nLens, _ := naiveRun(m, sysToks, allTurns[:*turnsNaive])
	nFirstLen, nLastLen := nLens[0], nLens[len(nLens)-1]
	nFirstMS, nLastMS := ms(nDurs[0]), ms(nDurs[len(nDurs)-1])
	fmt.Printf("KV residency: turn 1 = %d, turn %d = %d (grows with M — the contrast case)\n", nFirstLen, *turnsNaive, nLastLen)
	fmt.Printf("per-turn append cost: turn 1 = %.1fms, turn %d = %.1fms\n", nFirstMS, *turnsNaive, nLastMS)
	wantNaiveLen := *sysLen + *turnsNaive*(*turnLen)
	if nLastLen != wantNaiveLen {
		failf("naive arm residency after %d turns = %d, want %d (unbounded growth expected)", *turnsNaive, nLastLen, wantNaiveLen)
	}
	if nLastLen <= lastLen {
		failf("naive arm residency (%d) did not exceed bounded arm residency (%d) — contrast is not demonstrated", nLastLen, lastLen)
	}

	// ---- Phase 3: bit-exact + non-vacuity witness (the load-bearing proof) ----
	// Mirrors internal/kvmmu/planbridge_test.go's
	// TestApplyPlanEvictsElidedToO1Residency shape exactly, scaled up: a fixed
	// PREFIX (sys + a few early turns) is appended and stays resident; a
	// SUFFIX of later turns is appended after it and then, in ONE plan,
	// elided — before anything else ever attends to them. Every elided span
	// is positionally AFTER every resident span, the direction kvmmu.go's
	// ApplyPlan doc comment documents as bit-exact (no resident span's cached
	// K/V ever attended to what gets evicted).
	fmt.Printf("\n--- Phase 3: bit-exact + non-vacuity witness, %d resident + %d elided turns ---\n", *exactResident, *exactElided)
	prefixTurns := allTurns[:*exactResident]
	suffixTurns := allTurns[*exactResident : *exactResident+*exactElided]

	s3 := m.NewSession()
	c3 := kvmmu.NewWithGate(s3, ctxmmu.New())
	c3.Append("sys", "system", sysToks)
	for i, tk := range prefixTurns {
		c3.Append(fmt.Sprintf("p%d", i), "turn", tk)
	}
	for i, tk := range suffixTurns {
		c3.Append(fmt.Sprintf("q%d", i), "turn", tk)
	}
	fullLen := c3.CacheLen()

	plan := ctxplan.Plan{Objective: ctxplan.ObjGreedy}
	plan.Selected = append(plan.Selected, ctxplan.Selection{ID: "sys", Step: 0})
	for i := range prefixTurns {
		plan.Selected = append(plan.Selected, ctxplan.Selection{ID: fmt.Sprintf("p%d", i), Step: i + 1})
	}
	for i := range suffixTurns {
		id := fmt.Sprintf("q%d", i)
		plan.Elided = append(plan.Elided, ctxplan.Elision{
			ID: id, Step: *exactResident + 1 + i, Digest: "d-" + id, Reason: ctxplan.ElideOverBudget,
		})
	}
	plan.Candidates = len(plan.Selected) + len(plan.Elided)
	if w := ctxplan.Audit(plan); !w.Faithful {
		failf("phase 3 plan not faithful: %+v", w)
	}

	evicted := c3.ApplyPlan(plan)
	gotResident := c3.CacheLen()
	wantExactResident := *sysLen + (*exactResident)*(*turnLen)
	if evicted != *exactElided {
		failf("phase 3 ApplyPlan evicted %d segments, want %d", evicted, *exactElided)
	}
	if gotResident != wantExactResident {
		failf("phase 3 bounded residency = %d, want %d", gotResident, wantExactResident)
	}
	if gotResident == fullLen {
		failf("phase 3 residency did not shrink (%d == pre-eviction %d)", gotResident, fullLen)
	}
	fmt.Printf("KV residency: %d before eviction, %d after (%d turns elided)\n", fullLen, gotResident, evicted)

	kept := cat(sysToks, cat(prefixTurns...), cat(suffixTurns...), queryToks)
	residentOnly := cat(sysToks, cat(prefixTurns...), queryToks)

	t1 := time.Now()
	lKept := m.NewSession().Prefill(kept)
	fmt.Printf("reference prefill (kept-everything, %d tokens): %.1fms\n", len(kept), ms(time.Since(t1)))

	t2 := time.Now()
	lResident := m.NewSession().Prefill(residentOnly)
	fmt.Printf("reference prefill (resident-only, %d tokens): %.1fms\n", len(residentOnly), ms(time.Since(t2)))

	if d := maxAbsDiff(lKept, lResident); d == 0 {
		failf("non-vacuity: kept-everything and resident-only references are IDENTICAL (max|Delta|=0) — the elided spans never mattered, so the bit-exact check below would be vacuous")
	} else {
		fmt.Printf("non-vacuity OK: kept-everything vs resident-only max|Delta| = %.3e (elided spans DO perturb the distribution)\n", d)
	}

	lGot, _ := c3.Append("usr", "user", queryToks)
	if d := maxAbsDiff(lGot, lResident); d != 0 {
		failf("bit-exactness: bounded-session post-query distribution != resident-only reference (max|Delta|=%.3e) — the elision-to-eviction bridge is NOT bit-exact at this scale", d)
	} else {
		fmt.Printf("bit-exact OK: bounded-session post-query distribution == resident-only reference (max|Delta|=0)\n")
	}

	fmt.Println()
	if fail {
		fmt.Println("VERDICT: NOT PROVEN — see FAIL lines above.")
		os.Exit(1)
	}
	fmt.Printf("VERDICT: O(1) residency bridge (CLAIMS.md:141) holds at ~1.5B-parameter synthetic scale, on this machine:\n")
	fmt.Printf("  - KV residency stays constant (%d positions) across %d turns while the naive arm grows to %d.\n", wantResident, *turnsBounded, nLastLen)
	fmt.Printf("  - The bounded session's post-query distribution is bit-identical to a resident-only reference,\n")
	fmt.Printf("    and non-vacuously different from a kept-everything reference — eviction == never-having-seen,\n")
	fmt.Printf("    byte for byte, at this scale.\n")
	fmt.Printf("  - Honest fence: logits are numerically meaningless (synthetic weights); this witnesses cache\n")
	fmt.Printf("    MECHANICS only. Real-weight numerics are proven separately by internal/model's oracle tests.\n")
}
