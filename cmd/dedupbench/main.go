// Command dedupbench measures fak's IN-BATCH COLD-PREFIX FUSION
// (fak#1914, internal/modelengine's SetInBatchPrefixDedup over
// radixkv.PrefixFlightGroup) against a serial warm-cache comparison arm.
//
// The mechanism under test. N cold admissions from one prompt family share a
// prefix P, then diverge. The FIRST admission becomes the leader and prefills its
// full prompt. Every follower JOINS THE IN-FLIGHT leader before computation starts,
// waits for publication, adopts a clone of the shared prefix KV, and prefills its
// own divergent suffix. The admission barrier makes this synthetic simultaneous
// workload deterministic; its scheduling overhead is included in wall-clock time.
//
// The warm-cache comparison. Run the leader to completion, insert its KV into a
// radix tree, then serve each follower serially from the warm prefix (clone +
// suffix prefill). This avoids recomputing the same shared prefix too, but has no
// in-flight registration. A whole-prompt recompute arm is reported only as
// CONTEXT, never as the headline (see BENCHMARK-GOVERNANCE.md anti-inflation).
//
// What the ratio means. For immediately divergent suffixes both arms compute one
// shared prefix + N suffixes. Actual token counts are reported per arm: accidental
// matches inside generated suffixes can change reuse with the chosen leader.
// In both arms followers adopt KV only after the leader finishes. The fusion cohort runs follower suffixes concurrently; the warm-cache
// comparison runs them serially. Their wall-clock ratio describes these synthetic
// workloads, including scheduling and cloning costs, and does not isolate a gain
// from prefix fusion alone. Every arm is correctness-gated: final last-token logits
// must be bit-identical to a fresh full prefill.
//
// Usage:
//
//	dedupbench                          # default envelope, human table
//	dedupbench -json                    # machine-readable receipt
//	dedupbench -check                   # correctness gate only; exit 0 iff both arms match a full prefill
//	dedupbench -n 16 -prefix 384 -suffix 48 -reps 5
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const workSchema = "fak.dedupbench.v1"

// minShared mirrors nativeInBatchPrefixDedupMinShared in internal/modelengine: the
// minimum shared token prefix for two concurrent cold admissions to coalesce.
const minShared = 32

type envelope struct {
	N         int `json:"n"`
	Prefix    int `json:"prefix_tokens"`
	Suffix    int `json:"suffix_tokens"`
	MinShared int `json:"min_shared_tokens"`
	Reps      int `json:"reps"`
}

type armResult struct {
	WallMS      float64 `json:"wall_ms_best"`
	PrefillTok  int     `json:"prefill_tokens"`
	Correctness bool    `json:"correctness_bit_identical"`
	Failure     string  `json:"failure,omitempty"`
	// CoalescedLeaders / CoalescedFollowers prove the FUSION arm actually coalesced
	// rather than accidentally serializing into N independent leaders. Both are 0 for
	// the non-fusion arms.
	CoalescedLeaders   int64 `json:"coalesced_leaders,omitempty"`
	CoalescedFollowers int64 `json:"coalesced_followers,omitempty"`
}

type receipt struct {
	Schema          string    `json:"schema"`
	Model           string    `json:"model"`
	Envelope        envelope  `json:"envelope"`
	Fusion          armResult `json:"fusion"`
	WarmCacheNBA    armResult `json:"warm_cache_next_best"`
	NaiveContext    armResult `json:"naive_recompute_context"`
	FusionVsWarmNBA float64   `json:"fusion_speedup_vs_warm_cache_nba"`
	Note            string    `json:"note"`
}

func main() {
	var (
		n      = flag.Int("n", 8, "fan-out: concurrent shared-prefix admissions")
		prefix = flag.Int("prefix", 256, "shared prefix length in tokens")
		suffix = flag.Int("suffix", 32, "per-lane divergent suffix length in tokens")
		reps   = flag.Int("reps", 3, "best-of-N repetitions per arm")
		layers = flag.Int("layers", 4, "model layer count (bigger = more compute-bound)")
		hidden = flag.Int("hidden", 128, "model hidden size")
		check  = flag.Bool("check", false, "correctness gate only; exit 0 iff both arms match a full prefill")
		asJSON = flag.Bool("json", false, "emit the receipt as JSON")
		seed   = flag.Uint64("seed", 1, "token LCG seed")
	)
	flag.Parse()

	if *n < 2 {
		fatalf("-n must be >= 2 (a coalescing arm needs at least one follower)")
	}
	if *prefix < minShared {
		fatalf("-prefix must be >= %d (nativeInBatchPrefixDedupMinShared)", minShared)
	}

	cfg := model.Config{
		HiddenSize:        *hidden,
		NumLayers:         *layers,
		NumHeads:          8,
		NumKVHeads:        4,
		HeadDim:           *hidden / 8,
		IntermediateSize:  *hidden * 2,
		VocabSize:         512,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	m := model.NewSynthetic(cfg)
	vocab := cfg.VocabSize

	shared := lcg(*prefix, vocab, *seed)
	prompts := make([][]int, *n)
	for i := range prompts {
		prompts[i] = append(append([]int(nil), shared...), lcg(*suffix, vocab, *seed+1000+uint64(i))...)
	}

	// Correctness oracle: a fresh full prefill of every prompt.
	oracle := make([][]float32, *n)
	for i, p := range prompts {
		s := m.NewSession()
		oracle[i] = s.Prefill(p)
	}

	env := envelope{N: *n, Prefix: *prefix, Suffix: *suffix, MinShared: minShared, Reps: *reps}

	fusion := runBest(m, *reps, func() (armResult, bool) { return armFusion(m, prompts, oracle) })
	nba := runBest(m, *reps, func() (armResult, bool) { return armWarmCacheNBA(m, prompts, oracle) })
	naive := runBest(m, *reps, func() (armResult, bool) { return armNaive(m, prompts, oracle) })

	ratio := 0.0
	if fusion.WallMS > 0 {
		ratio = nba.WallMS / fusion.WallMS
	}

	r := receipt{
		Schema:          workSchema,
		Model:           "synthetic(dedupbench)",
		Envelope:        env,
		Fusion:          fusion,
		WarmCacheNBA:    nba,
		NaiveContext:    naive,
		FusionVsWarmNBA: ratio,
		Note: "Synthetic concurrent fusion cohort vs serial warm-cache comparison. Followers in both arms " +
			"wait for the leader to finish before adopting KV. Fusion follower suffixes run concurrently; " +
			"warm-cache follower suffixes run serially. Actual prefill token counts are reported per arm; " +
			"extra suffix-prefix matches can change reuse with the chosen leader. " +
			"Wall-clock includes admission, scheduling and cloning; " +
			"the ratio does not isolate a gain from prefix fusion alone. Naive recompute is context only. " +
			"All arms best-of-N and bit-identical to fresh full prefill. [SW-VERIFIED] Go-receipt numbers " +
			"are not hardware evidence.",
	}

	if *check {
		ok := fusion.Correctness && nba.Correctness && naive.Correctness
		if !ok {
			fmt.Fprintln(os.Stderr, "dedupbench -check: correctness gate FAILED (an arm diverged from a full prefill)")
			os.Exit(1)
		}
		if *asJSON {
			emit(r)
		} else {
			fmt.Printf("dedupbench -check: OK (all arms bit-identical to full prefill; fusion leaders=%d followers=%d); concurrent-fusion/serial-warm-cache wall ratio = %.3fx\n",
				fusion.CoalescedLeaders, fusion.CoalescedFollowers, ratio)
		}
		return
	}

	if *asJSON {
		emit(r)
		return
	}
	fmt.Printf("envelope: N=%d prefix=%d suffix=%d minShared=%d reps=%d\n", env.N, env.Prefix, env.Suffix, env.MinShared, env.Reps)
	fmt.Printf("  fusion (in-flight coalesce) : %8.3f ms  prefill_tokens=%d  correct=%v\n", fusion.WallMS, fusion.PrefillTok, fusion.Correctness)
	fmt.Printf("  warm-cache (serial)         : %8.3f ms  prefill_tokens=%d  correct=%v\n", nba.WallMS, nba.PrefillTok, nba.Correctness)
	fmt.Printf("  naive recompute (context)   : %8.3f ms  prefill_tokens=%d  correct=%v\n", naive.WallMS, naive.PrefillTok, naive.Correctness)
	fmt.Printf("  concurrent fusion / serial warm-cache wall-clock ratio: %.3fx\n", ratio)
}

func emit(r receipt) {
	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(b))
}

// runBest executes an arm up to reps times and keeps the fastest correctness-passing
// result. A failed correctness gate is fatal (never silently dropped).
func runBest(m *model.Model, reps int, arm func() (armResult, bool)) armResult {
	if reps < 1 {
		reps = 1
	}
	var best armResult
	best.WallMS = 1e18
	for i := 0; i < reps; i++ {
		res, ok := arm()
		if !ok {
			fmt.Fprintf(os.Stderr, "dedupbench: arm failed its correctness gate: %s\n", res.Failure)
			os.Exit(1)
		}
		if res.WallMS < best.WallMS {
			best = res
		}
	}
	return best
}

// armFusion runs N concurrent admissions through the PrefixFlightGroup single-flight:
// one leader prefills its full prompt; every follower joins the in-flight leader,
// adopts a clone of its prefix KV, and prefills only its divergent suffix.
func armFusion(m *model.Model, prompts [][]int, oracle [][]float32) (armResult, bool) {
	g := radixkv.NewPrefixFlightGroup(nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	start := make(chan struct{})
	type out struct {
		logits     []float32
		prefillTok int
		err        error
	}
	outs := make([]out, len(prompts))

	t0 := time.Now()
	for i := range prompts {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			p := prompts[i]
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(
				ctx, "", p, minShared,
				func(ctx context.Context) (*model.KVCache, []float32, error) {
					// A start-channel release only makes admissions runnable. Hold
					// computation until every follower has registered, before the
					// model's spin workers can occupy all available scheduler Ps.
					// This establishes the simultaneous cold workload under test;
					// admission and its scheduling cost remain inside the timer.
					if err := waitForFusionFollowers(ctx, g, int64(len(prompts)-1)); err != nil {
						cancel()
						return nil, nil, err
					}
					s := m.NewSession()
					lg := s.Prefill(p)
					return s.Cache, lg, nil
				})
			if err != nil {
				outs[i] = out{err: err}
				return
			}
			if leader {
				// The leader computed its full prompt, including its own suffix.
				outs[i] = out{logits: logits, prefillTok: len(p)}
				return
			}
			// Follower: adopt the prefix clone and prefill only the divergent suffix.
			s := m.SessionFromPrefix(kv)
			lg := s.Prefill(p[matched:])
			outs[i] = out{logits: lg, prefillTok: len(p) - matched}
		}(i)
	}
	close(start)
	wg.Wait()

	result := armResult{
		WallMS:             msSince(t0),
		CoalescedLeaders:   g.Leaders(),
		CoalescedFollowers: g.Coalesced(),
	}
	for _, out := range outs {
		result.PrefillTok += out.prefillTok
	}
	// Integrity: the fusion arm MUST have coalesced exactly once (one leader) with
	// every other admission as a follower; otherwise this is not a fusion measurement.
	if result.CoalescedLeaders != 1 || result.CoalescedFollowers != int64(len(prompts)-1) {
		result.Failure = fmt.Sprintf("coalescing leaders=%d followers=%d, want 1/%d",
			result.CoalescedLeaders, result.CoalescedFollowers, len(prompts)-1)
		return result, false
	}
	for i := range prompts {
		if outs[i].err != nil {
			result.Failure = fmt.Sprintf("admission %d: %v", i, outs[i].err)
			return result, false
		}
		if !bitEqual(outs[i].logits, oracle[i]) {
			result.Failure = fmt.Sprintf("admission %d logits differ from full prefill", i)
			return result, false
		}
	}
	result.Correctness = true
	return result, true
}

// waitForFusionFollowers establishes a real in-flight registration for every
// follower, even when a short leader prefill would finish before Go schedules one.
// The group releases its lock before invoking the leader callback, so yielding
// here lets followers register without blocking on the leader's computation.
func waitForFusionFollowers(ctx context.Context, g *radixkv.PrefixFlightGroup, want int64) error {
	// Bound admission only; a legitimately slow prefill keeps its existing budget.
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	for g.Coalesced() < want {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("waiting for %d fusion followers: %w", want, err)
		}
		if g.Leaders() != 1 {
			return fmt.Errorf("fusion workload admitted %d leaders, want 1", g.Leaders())
		}
		runtime.Gosched()
	}
	return ctx.Err()
}

// armWarmCacheNBA is the serial warm-cache comparison: the leader completes and
// warms a radix tree, then each follower reuses the warm prefix (clone + suffix
// prefill). Both arms wait for leader publication; this arm serializes followers.
func armWarmCacheNBA(m *model.Model, prompts [][]int, oracle [][]float32) (armResult, bool) {
	tree := radixkv.New(0)
	t0 := time.Now()
	// Leader runs to completion and its KV warms the tree.
	lead := m.NewSession()
	leadLogits := lead.Prefill(prompts[0])
	if !bitEqual(leadLogits, oracle[0]) {
		return armResult{}, false
	}
	boundary, matched0 := tree.Lookup(prompts[0])
	tree.Done(tree.Insert(boundary, prompts[0][matched0:], lead.Cache))

	prefillTokens := len(prompts[0])
	for i := 1; i < len(prompts); i++ {
		p := prompts[i]
		b, matched := tree.Lookup(p)
		s := m.SessionFromPrefix(b.KV())
		lg := s.Prefill(p[matched:])
		if !bitEqual(lg, oracle[i]) {
			return armResult{}, false
		}
		prefillTokens += len(p) - matched
	}
	wall := msSince(t0)
	return armResult{WallMS: wall, PrefillTok: prefillTokens, Correctness: true}, true
}

// armNaive recomputes every prompt in full — the cold stateless-API pattern, a
// WORST-CASE CONTEXT reference only, never the headline baseline.
func armNaive(m *model.Model, prompts [][]int, oracle [][]float32) (armResult, bool) {
	t0 := time.Now()
	prefillTokens := 0
	for i, p := range prompts {
		s := m.NewSession()
		lg := s.Prefill(p)
		if !bitEqual(lg, oracle[i]) {
			return armResult{}, false
		}
		prefillTokens += len(p)
	}
	return armResult{WallMS: msSince(t0), PrefillTok: prefillTokens, Correctness: true}, true
}

func bitEqual(a, b []float32) bool {
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

func lcg(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	x := seed*6364136223846793005 + 1442695040888963407
	for i := range ids {
		x = x*6364136223846793005 + 1442695040888963407
		ids[i] = int((x >> 33) % uint64(vocab))
	}
	return ids
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Nanoseconds()) / 1e6 }

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "dedupbench: "+format+"\n", args...)
	os.Exit(2)
}
