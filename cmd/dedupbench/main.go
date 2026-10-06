// Command dedupbench measures fak's cold-prefix-family prefill strategies against a
// FAIR next-best alternative (NBA), per request, on one in-process model.
//
// The workload. N requests from one prompt family arrive together (one common release
// instant): a shared prefix P, then an equal-length divergent suffix S per request. No
// cache is warm. Every arm must end with N sessions whose last-token logits are
// bit-identical to a fresh full prefill of prefix+suffix (the -check gate).
//
// Arms (all follower suffix prefills are ONE shared-weight batched forward —
// Model.NewBatchFromPrefix + BatchSession.PrefillEach — so no arm is credited with
// goroutine CPU parallelism the others do not get):
//
//   - fusion: fak's in-batch cold-prefix fusion (fak#1914, PrefixFlightGroup). All N
//     admissions call CoalesceSharedPrefixNS; the first becomes the leader and
//     prefills its full prompt; the others JOIN the flight and BLOCK on the leader's
//     ready signal (internal/radixkv/singleflight.go), then adopt a clone of its prefix
//     KV. The joined followers are then prefilled together as one batch. Followers do
//     wait for the leader — there is no overlap of follower work with the leader's
//     prefix prefill.
//   - nba (warm-cache next-best, SGLang-style): the leader is prefilled, its KV warms a
//     radix tree, and the same-prefix waiting twins are admitted in ONE extend batch
//     from the warm prefix. Real SGLang pays one extra scheduler step between the
//     leader's extend and the twins' extend; -sched-step-ms MODELS that cost (default
//     0). It is added arithmetically to the NBA followers' TTFT and makespan and
//     reported as its own field, never folded in silently.
//   - cascade: one-pass shared-prefix prefill (Hydragen / FlashInfer cascade pattern,
//     Model.CascadePrefill): the P prefix rows and all N*S suffix rows go through one
//     shared-weight GEMM per layer, prefix KV computed once. Every request's logits
//     land at the same instant T(P+N*S) — the leader is WORSE than a two-phase scheme's
//     T(P+S), and this bench reports that.
//   - naive (context only, never the headline): every request prefills its full prompt
//     sequentially, no reuse.
//
// Metrics. Per request: TTFT = time from the common release to that request's prefill
// completion (its first-token logits exist). Per arm: leader TTFT, follower mean / p50
// / p99 TTFT, all-request mean TTFT, makespan. Reps are interleaved across arms after a
// discarded warm-up rep and summarized by MEDIAN with min/max spread (not best-of).
// Token accounting: fusion, nba and cascade all prefill P + N*S tokens; naive N*(P+S).
//
// Usage:
//
//	dedupbench                          # default envelope, human table
//	dedupbench -json                    # machine-readable receipt
//	dedupbench -check                   # correctness gate; exit 0 iff every arm matches a full prefill
//	dedupbench -n 8 -prefix 512 -suffix 32 -hidden 1024 -layers 2 -reps 7 -sched-step-ms 2
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const workSchema = "fak.dedupbench.v2"

// minShared mirrors nativeInBatchPrefixDedupMinShared in internal/modelengine: the
// minimum shared token prefix for two concurrent cold admissions to coalesce.
const minShared = 32

type envelope struct {
	N           int     `json:"n"`
	Prefix      int     `json:"prefix_tokens"`
	Suffix      int     `json:"suffix_tokens"`
	Hidden      int     `json:"hidden"`
	Layers      int     `json:"layers"`
	MinShared   int     `json:"min_shared_tokens"`
	Reps        int     `json:"reps"`
	SchedStepMS float64 `json:"sched_step_ms_modeled"`
}

// repResult is one arm's measurement for one rep.
type repResult struct {
	TTFT       []float64 // per request, ms from common release; index 0 = leader
	ModeledMS  float64   // modeled (not measured) cost already added to TTFT entries
	PrefillTok int
	Leaders    int64
	Followers  int64
}

type spread struct {
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

type armSummary struct {
	LeaderTTFT       spread  `json:"leader_ttft_ms"`
	FollowerMeanTTFT spread  `json:"follower_mean_ttft_ms"`
	FollowerP50TTFT  float64 `json:"follower_p50_ttft_ms"`
	FollowerP99TTFT  float64 `json:"follower_p99_ttft_ms"`
	AllMeanTTFT      spread  `json:"all_mean_ttft_ms"`
	Makespan         spread  `json:"makespan_ms"`
	ModeledMS        float64 `json:"modeled_sched_step_ms_included,omitempty"`
	PrefillTok       int     `json:"prefill_tokens"`
	Correctness      bool    `json:"correctness_bit_identical"`
	// CoalescedLeaders / CoalescedFollowers prove the FUSION arm actually coalesced.
	CoalescedLeaders   int64 `json:"coalesced_leaders,omitempty"`
	CoalescedFollowers int64 `json:"coalesced_followers,omitempty"`
}

// ratios are NBA_metric / arm_metric on medians: >1 means the arm is faster.
type ratios struct {
	LeaderTTFT       float64 `json:"leader_ttft"`
	FollowerMeanTTFT float64 `json:"follower_mean_ttft"`
	AllMeanTTFT      float64 `json:"all_mean_ttft"`
	Makespan         float64 `json:"makespan"`
}

type receipt struct {
	Schema       string     `json:"schema"`
	Model        string     `json:"model"`
	Envelope     envelope   `json:"envelope"`
	Fusion       armSummary `json:"fusion"`
	NBA          armSummary `json:"warm_cache_next_best"`
	Cascade      armSummary `json:"cascade"`
	NaiveContext armSummary `json:"naive_recompute_context"`
	FusionVsNBA  ratios     `json:"fusion_speedup_vs_nba"`
	CascadeVsNBA ratios     `json:"cascade_speedup_vs_nba"`
	Note         string     `json:"note"`
}

type arm struct {
	name string
	run  func() (repResult, bool)
}

func main() {
	var (
		n         = flag.Int("n", 8, "fan-out: concurrent shared-prefix admissions")
		prefix    = flag.Int("prefix", 256, "shared prefix length in tokens")
		suffix    = flag.Int("suffix", 32, "per-lane divergent suffix length in tokens")
		reps      = flag.Int("reps", 5, "measured repetitions per arm (median reported; one extra warm-up rep is discarded)")
		layers    = flag.Int("layers", 4, "model layer count")
		hidden    = flag.Int("hidden", 128, "model hidden size (bigger = weight-load dominated)")
		schedStep = flag.Float64("sched-step-ms", 0, "MODELED extra scheduler step the NBA pays before admitting the twins (added to NBA follower TTFT, reported separately)")
		check     = flag.Bool("check", false, "correctness gate; exit 0 iff every arm matches a full prefill")
		asJSON    = flag.Bool("json", false, "emit the receipt as JSON")
		seed      = flag.Uint64("seed", 1, "token LCG seed")
		modelDir  = flag.String("model", "", "optional real-weights f32 model dir (export_oracle.py layout: config.json, manifest.json, weights.f32); default is the synthetic model sized by -hidden/-layers")
	)
	flag.Parse()

	if *n < 2 {
		fatalf("-n must be >= 2 (a coalescing arm needs at least one follower)")
	}
	if *prefix < minShared {
		fatalf("-prefix must be >= %d (nativeInBatchPrefixDedupMinShared)", minShared)
	}
	if *modelDir == "" && *hidden%8 != 0 {
		fatalf("-hidden must be a multiple of 8 (8 heads)")
	}
	if *reps < 1 {
		*reps = 1
	}

	m := model.NewSynthetic(benchConfig(*hidden, *layers))
	modelName := "synthetic(dedupbench)"
	if *modelDir != "" {
		lm, err := model.Load(*modelDir)
		if err != nil {
			fatalf("load -model %s: %v", *modelDir, err)
		}
		m, modelName = lm, "f32:"+*modelDir
		*hidden, *layers = m.Cfg.HiddenSize, m.Cfg.NumLayers
	}
	shared, suffixes, prompts := buildFamily(*n, *prefix, *suffix, m.Cfg.VocabSize, *seed)

	oracle := make([][]float32, *n)
	for i, p := range prompts {
		oracle[i] = m.NewSession().Prefill(p)
	}

	arms := []arm{
		{"fusion", func() (repResult, bool) { return armFusion(m, prompts, oracle, *prefix) }},
		{"nba", func() (repResult, bool) { return armWarmCacheNBA(m, prompts, oracle, *prefix, *schedStep) }},
		{"cascade", func() (repResult, bool) { return armCascade(m, shared, suffixes, oracle) }},
		{"naive", func() (repResult, bool) { return armNaive(m, prompts, oracle) }},
	}
	samples := make(map[string][]repResult, len(arms))
	for rep := 0; rep <= *reps; rep++ { // rep 0 is the discarded warm-up
		for _, a := range arms {
			res, ok := a.run()
			if !ok {
				fmt.Fprintf(os.Stderr, "dedupbench: arm %s failed its correctness/integrity gate\n", a.name)
				os.Exit(1)
			}
			if rep > 0 {
				samples[a.name] = append(samples[a.name], res)
			}
		}
	}

	fusion := summarize(samples["fusion"])
	nba := summarize(samples["nba"])
	cascade := summarize(samples["cascade"])
	naive := summarize(samples["naive"])

	r := receipt{
		Schema: workSchema,
		Model:  modelName,
		Envelope: envelope{N: *n, Prefix: *prefix, Suffix: *suffix, Hidden: *hidden, Layers: *layers,
			MinShared: minShared, Reps: *reps, SchedStepMS: *schedStep},
		Fusion:       fusion,
		NBA:          nba,
		Cascade:      cascade,
		NaiveContext: naive,
		FusionVsNBA:  ratioOf(nba, fusion),
		CascadeVsNBA: ratioOf(nba, cascade),
		Note: "TTFT = common release -> that request's prefill completion. Ratios are NBA/arm on medians " +
			"(>1 = arm faster). fusion and nba both prefill the leader then ONE batched follower extend; nba's " +
			"sched_step_ms is MODELED, not measured. cascade lands every request at T(P+N*S), so its leader TTFT " +
			"is worse by construction. naive is context only. Every arm is correctness-gated bit-identical to a " +
			"fresh full prefill. [SW-VERIFIED] in-process CPU run (see model); not hardware evidence unless run on the target host with real weights.",
	}

	if *check {
		if *asJSON {
			emit(r)
		} else {
			fmt.Printf("dedupbench -check: OK (all arms bit-identical to full prefill; fusion leaders=%d followers=%d)\n",
				fusion.CoalescedLeaders, fusion.CoalescedFollowers)
		}
		return
	}
	if *asJSON {
		emit(r)
		return
	}
	e := r.Envelope
	fmt.Printf("envelope: N=%d prefix=%d suffix=%d hidden=%d layers=%d reps=%d sched-step-ms(modeled)=%.2f\n",
		e.N, e.Prefix, e.Suffix, e.Hidden, e.Layers, e.Reps, e.SchedStepMS)
	fmt.Printf("  %-8s %11s %13s %9s %9s %11s %21s %8s\n", "arm", "leader_ms", "follower_mean", "f_p50", "f_p99", "all_mean", "makespan[min,max]", "tokens")
	for _, row := range []struct {
		name string
		s    armSummary
	}{{"fusion", fusion}, {"nba", nba}, {"cascade", cascade}, {"naive*", naive}} {
		fmt.Printf("  %-8s %11.3f %13.3f %9.3f %9.3f %11.3f %9.3f[%.2f,%.2f] %8d\n", row.name,
			row.s.LeaderTTFT.Median, row.s.FollowerMeanTTFT.Median, row.s.FollowerP50TTFT, row.s.FollowerP99TTFT,
			row.s.AllMeanTTFT.Median, row.s.Makespan.Median, row.s.Makespan.Min, row.s.Makespan.Max, row.s.PrefillTok)
	}
	fmt.Printf("  cascade vs nba: follower-mean %.3fx  leader %.3fx  all-mean %.3fx  makespan %.3fx\n",
		r.CascadeVsNBA.FollowerMeanTTFT, r.CascadeVsNBA.LeaderTTFT, r.CascadeVsNBA.AllMeanTTFT, r.CascadeVsNBA.Makespan)
	fmt.Printf("  fusion  vs nba: follower-mean %.3fx  leader %.3fx  all-mean %.3fx  makespan %.3fx\n",
		r.FusionVsNBA.FollowerMeanTTFT, r.FusionVsNBA.LeaderTTFT, r.FusionVsNBA.AllMeanTTFT, r.FusionVsNBA.Makespan)
	fmt.Println("  (* naive is context only; nba sched step is modeled)")
}

func benchConfig(hidden, layers int) model.Config {
	return model.Config{
		HiddenSize:        hidden,
		NumLayers:         layers,
		NumHeads:          8,
		NumKVHeads:        4,
		HeadDim:           hidden / 8,
		IntermediateSize:  hidden * 2,
		VocabSize:         512,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// buildFamily returns the shared prefix, the n suffixes, and the n full prompts. Each
// suffix's first token is distinct across lanes, so every pair of prompts shares
// EXACTLY the prefix (a chance suffix collision would make the coalescing match length
// vary per follower and turn the rectangular follower batch ragged).
func buildFamily(n, prefix, suffix, vocab int, seed uint64) ([]int, [][]int, [][]int) {
	shared := lcg(prefix, vocab, seed)
	suffixes := make([][]int, n)
	prompts := make([][]int, n)
	for i := range prompts {
		suffixes[i] = lcg(suffix, vocab, seed+1000+uint64(i))
		suffixes[i][0] = i % vocab
		prompts[i] = append(append([]int(nil), shared...), suffixes[i]...)
	}
	return shared, suffixes, prompts
}

func emit(r receipt) {
	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(b))
}

// armFusion runs N concurrent admissions through the PrefixFlightGroup single-flight.
// The leader prefills its full prompt; each follower joins the flight, BLOCKS until the
// leader is ready, and receives a prefix-KV clone. Once every follower has joined, the
// followers' suffixes are prefilled together as ONE batched forward over that prefix.
func armFusion(m *model.Model, prompts [][]int, oracle [][]float32, prefix int) (repResult, bool) {
	g := radixkv.NewPrefixFlightGroup(nil)
	n := len(prompts)
	start := make(chan struct{})
	type joined struct {
		idx     int
		kv      *model.KVCache
		matched int
	}
	followers := make(chan joined, n)
	ttft := make([]float64, n)
	var leaderLogits []float32
	leaderIdx := -1
	var failed bool
	var mu sync.Mutex
	var wg sync.WaitGroup

	var t0 time.Time
	for i := range prompts {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			p := prompts[i]
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(
				context.Background(), "", p, minShared,
				func(context.Context) (*model.KVCache, []float32, error) {
					s := m.NewSession()
					lg := s.Prefill(p)
					return s.Cache, lg, nil
				})
			if err != nil {
				mu.Lock()
				failed = true
				mu.Unlock()
				return
			}
			if leader {
				done := msSince(t0)
				mu.Lock()
				leaderIdx, leaderLogits, ttft[0] = i, logits, done
				mu.Unlock()
				return
			}
			followers <- joined{idx: i, kv: kv, matched: matched}
		}(i)
	}
	t0 = time.Now()
	close(start)
	wg.Wait()
	close(followers)
	if failed || leaderIdx < 0 || !bitEqual(leaderLogits, oracle[leaderIdx]) {
		return repResult{}, false
	}

	var js []joined
	for j := range followers {
		js = append(js, j)
	}
	if len(js) != n-1 {
		return repResult{}, false
	}
	suffixes := make([][]int, len(js))
	for k, j := range js {
		if j.matched != prefix {
			return repResult{}, false
		}
		suffixes[k] = prompts[j.idx][j.matched:]
	}
	bs := m.NewBatchFromPrefix(js[0].kv, len(js))
	lg := bs.PrefillEach(suffixes)
	done := msSince(t0)
	for k, j := range js {
		if !bitEqual(lg[k], oracle[j.idx]) {
			return repResult{}, false
		}
		ttft[k+1] = done
	}

	leaders, coalesced := g.Leaders(), g.Coalesced()
	if leaders != 1 || coalesced != int64(n-1) {
		return repResult{}, false
	}
	return repResult{
		TTFT:       ttft,
		PrefillTok: len(prompts[leaderIdx]) + (n-1)*(len(prompts[0])-prefix),
		Leaders:    leaders,
		Followers:  coalesced,
	}, true
}

// armWarmCacheNBA models SGLang-style in-batch prefix caching fairly: the leader's
// extend runs, its KV warms a radix tree, and the deferred same-prefix twins are then
// admitted together in ONE batched extend from the warm prefix. schedStepMS is the
// modeled extra scheduler step between the two extends (0 = none).
func armWarmCacheNBA(m *model.Model, prompts [][]int, oracle [][]float32, prefix int, schedStepMS float64) (repResult, bool) {
	n := len(prompts)
	ttft := make([]float64, n)
	tree := radixkv.New(0)
	t0 := time.Now()
	lead := m.NewSession()
	leadLogits := lead.Prefill(prompts[0])
	ttft[0] = msSince(t0)
	if !bitEqual(leadLogits, oracle[0]) {
		return repResult{}, false
	}
	boundary, matched0 := tree.Lookup(prompts[0])
	tree.Done(tree.Insert(boundary, prompts[0][matched0:], lead.Cache))

	b, matched := tree.Lookup(prompts[1])
	if matched != prefix || b == nil || b.KV() == nil {
		return repResult{}, false
	}
	suffixes := make([][]int, n-1)
	for i := 1; i < n; i++ {
		suffixes[i-1] = prompts[i][matched:]
	}
	bs := m.NewBatchFromPrefix(b.KV(), n-1)
	lg := bs.PrefillEach(suffixes)
	done := msSince(t0) + schedStepMS
	tree.Done(b)
	for i := 1; i < n; i++ {
		if !bitEqual(lg[i-1], oracle[i]) {
			return repResult{}, false
		}
		ttft[i] = done
	}
	return repResult{
		TTFT:       ttft,
		ModeledMS:  schedStepMS,
		PrefillTok: len(prompts[0]) + (n-1)*(len(prompts[0])-prefix),
	}, true
}

// armCascade prefills the whole family in one shared-weight pass (Model.CascadePrefill).
// Every request, including the "leader", completes at the same instant.
func armCascade(m *model.Model, shared []int, suffixes [][]int, oracle [][]float32) (repResult, bool) {
	t0 := time.Now()
	_, lg, err := m.CascadePrefill(shared, suffixes)
	done := msSince(t0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dedupbench: cascade:", err)
		return repResult{}, false
	}
	ttft := make([]float64, len(suffixes))
	for i := range suffixes {
		if !bitEqual(lg[i], oracle[i]) {
			return repResult{}, false
		}
		ttft[i] = done
	}
	return repResult{TTFT: ttft, PrefillTok: len(shared) + len(suffixes)*len(suffixes[0])}, true
}

// armNaive recomputes every prompt in full, sequentially — the cold stateless-API
// pattern, a WORST-CASE CONTEXT reference only, never the headline baseline.
func armNaive(m *model.Model, prompts [][]int, oracle [][]float32) (repResult, bool) {
	ttft := make([]float64, len(prompts))
	t0 := time.Now()
	tok := 0
	for i, p := range prompts {
		lg := m.NewSession().Prefill(p)
		ttft[i] = msSince(t0)
		if !bitEqual(lg, oracle[i]) {
			return repResult{}, false
		}
		tok += len(p)
	}
	return repResult{TTFT: ttft, PrefillTok: tok}, true
}

func summarize(reps []repResult) armSummary {
	var leader, fmean, amean, mk, pooled []float64
	for _, r := range reps {
		leader = append(leader, r.TTFT[0])
		f := r.TTFT[1:]
		fmean = append(fmean, mean(f))
		amean = append(amean, mean(r.TTFT))
		mk = append(mk, maxOf(r.TTFT))
		pooled = append(pooled, f...)
	}
	s := armSummary{
		LeaderTTFT:       spreadOf(leader),
		FollowerMeanTTFT: spreadOf(fmean),
		FollowerP50TTFT:  percentile(pooled, 50),
		FollowerP99TTFT:  percentile(pooled, 99),
		AllMeanTTFT:      spreadOf(amean),
		Makespan:         spreadOf(mk),
		Correctness:      len(reps) > 0,
	}
	if len(reps) > 0 {
		s.PrefillTok = reps[0].PrefillTok
		s.ModeledMS = reps[0].ModeledMS
		s.CoalescedLeaders = reps[0].Leaders
		s.CoalescedFollowers = reps[0].Followers
	}
	return s
}

func ratioOf(base, arm armSummary) ratios {
	div := func(a, b float64) float64 {
		if b <= 0 {
			return 0
		}
		return a / b
	}
	return ratios{
		LeaderTTFT:       div(base.LeaderTTFT.Median, arm.LeaderTTFT.Median),
		FollowerMeanTTFT: div(base.FollowerMeanTTFT.Median, arm.FollowerMeanTTFT.Median),
		AllMeanTTFT:      div(base.AllMeanTTFT.Median, arm.AllMeanTTFT.Median),
		Makespan:         div(base.Makespan.Median, arm.Makespan.Median),
	}
}

func spreadOf(xs []float64) spread {
	if len(xs) == 0 {
		return spread{}
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return spread{Median: percentile(s, 50), Min: s[0], Max: s[len(s)-1]}
}

// percentile is the linear-interpolated q-th percentile (q in [0,100]).
func percentile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	pos := q / 100 * float64(len(s)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	return s[lo] + (s[hi]-s[lo])*(pos-float64(lo))
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	t := 0.0
	for _, x := range xs {
		t += x
	}
	return t / float64(len(xs))
}

func maxOf(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}

func bitEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
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
