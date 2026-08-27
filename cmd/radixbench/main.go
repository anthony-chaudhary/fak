// Command radixbench benchmarks fak's KV-cache prefix reuse against SGLang's
// RadixAttention (arXiv:2312.07104 / NeurIPS 2024) on the metric SGLang's own paper
// headlines: CACHE HIT RATE — the fraction of prompt tokens served from cache instead of
// recomputed. That metric is the right head-to-head axis precisely because it is
// HARDWARE- and MODEL-INDEPENDENT: it is a function of (workload, matching algorithm)
// only, so fak-on-CPU vs SGLang-on-GPU is a FAIR comparison on this axis (unlike raw
// tok/s, which the repo correctly refuses to compare across regimes — see
// MODEL-BASELINE-RESULTS.md). fak runs the SAME algorithm SGLang runs (a radix tree of
// token sequences + longest-prefix match + LRU-leaf eviction, internal/radixkv), so on
// the same workload it should reach the same hit rate SGLang reports (50%-99%, 96% of
// optimal) — and we show it reaches the DFS-optimal bound the paper proves.
//
// Each workload is driven through three engines and reported per-workload:
//
//   - baseline (no cache)   — every request prefills its FULL prompt. Total prefill work
//     = Σ len(request). This is the cold "stateless API / re-prompt each call" pattern — a
//     WORST-CASE REFERENCE, not a serving baseline anyone ships.
//   - declare-one-prefix    — fak's PRE-radix capability (model.NewBatchFromPrefix): reuse
//     ONE declared shared prefix (the longest prefix common to ALL requests). This is the
//     ceiling of "you tell the engine the shared prefix." The gap between this and radix is
//     exactly what the radix TREE adds: automatic discovery of EVERY shared subtree, not
//     just the one global prefix.
//   - radix (RadixAttention) — internal/radixkv: walk the tree, reuse the longest cached
//     prefix of THIS request (discovered, not declared, including mid-run divergences that
//     force an edge split), prefill only the suffix.
//
// Headline per workload = cache hit rate (radix, SGLang's own hardware-independent axis) and
// the cross-subtree win radix_reuse_over_declare_one — radix's AUTOMATIC discovery of every
// shared subtree vs a WARM single-declared-prefix cache (the serving-realistic baseline: you
// already reuse the one declared prefix). The prefill-token speedup (baseline_tokens /
// radix_tokens) is reported too, but it is measured vs the COLD no-cache baseline — a
// WORST-CASE REFERENCE only, not a serving baseline. We also report:
//   - bounded-cache LRU under FCFS vs CACHE-AWARE (lexicographic ≡ DFS) order, reproducing
//     the paper's result that cache-aware scheduling reaches the optimal hit rate when the
//     cache budget >= the max request length (and FCFS thrashes below it);
//   - a LIVE wall-clock confirmation (a real kernel prefill per request) that token savings
//     become time savings, bit-identical to a full recompute (radixkv's split-reuse is
//     proven == recompute in internal/radixkv's test);
//   - a POLICY-EVICTION witness — the capability an opportunistic LRU radix cache
//     structurally cannot offer: evict a named (e.g. poisoned) prefix on a verdict, not on
//     memory pressure. Same primitive as SGLang, opposite governance.
//
// Usage:
//
//	radixbench                                  # synthetic model (deterministic, no deps)
//	radixbench -hf <snapshot> -lean             # live wall-clock on a real model
//	radixbench -out experiments/radixattention/radixbench.json
//	radixbench -budget-sweep -trace keys.trace  # captured key-trace -> hit-rate/$-vs-budget curve (#3397)
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/benchids"
	"github.com/anthony-chaudhary/fak/internal/intlist"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// ---- token generation (shared prefixes are LITERALLY the same ids) ----

func lcgIDs(n, vocab int, seed uint64) []int {
	return benchids.LCG(n, vocab, seed)
}

// ---- workloads (the verified SGLang benchmark shapes) ----

type Workload struct {
	Name     string  `json:"name"`
	Desc     string  `json:"desc"`
	Params   string  `json:"params"`
	Requests [][]int `json:"-"`
	SGLang   string  `json:"sglang_published"` // the paper's reported behavior for this shape
}

// fewShot: N questions share one P-token preamble; each adds a Q-token question. The
// single-level prefix-sharing case (MMLU 5-shot). Hit rate -> P/(P+Q) asymptotically.
func fewShot(vocab, P, Q, N int) Workload {
	pre := lcgIDs(P, vocab, 1001)
	reqs := make([][]int, N)
	for i := 0; i < N; i++ {
		reqs[i] = intlist.Concat(pre, lcgIDs(Q, vocab, uint64(2000+i)))
	}
	return Workload{
		Name: "few-shot", Desc: "N questions share one few-shot preamble (MMLU 5-shot shape)",
		Params:   fmt.Sprintf("preamble=%d question=%d N=%d", P, Q, N),
		Requests: reqs, SGLang: "single-level prefix reuse; hit rate within 50-99% band",
	}
}

// multiTurn: one conversation of T turns; each turn's request is the whole history so far
// plus a new user message. Each turn reuses the ENTIRE previous context — the chat case
// where RadixAttention approaches 100% reuse on the growing prefix (short-output regime).
func multiTurn(vocab, base, turnLen, T int) Workload {
	ctx := lcgIDs(base, vocab, 3001)
	reqs := make([][]int, 0, T)
	for t := 0; t < T; t++ {
		if t > 0 {
			ctx = intlist.Concat(ctx, lcgIDs(turnLen, vocab, uint64(3100+t)))
		}
		reqs = append(reqs, append([]int(nil), ctx...))
	}
	return Workload{
		Name: "multi-turn-chat", Desc: "T-turn conversation; each turn reuses the full history",
		Params:   fmt.Sprintf("base=%d turn=%d turns=%d", base, turnLen, T),
		Requests: reqs, SGLang: "history reuse; large speedup for short outputs (Sec 6.2)",
	}
}

// treeOfThought: a reasoning tree — a root prompt that branches F ways at each of D levels,
// every node appending an S-token thought. Requests are the root-to-leaf paths (F^D of
// them); siblings share every ancestor. This is the branching / parallel-sampling case
// (ToT on GSM-8K) where reuse is MULTI-LEVEL — the case declare-one-prefix cannot capture.
func treeOfThought(vocab, root, seg, fanout, depth int) Workload {
	var paths [][]int
	var rec func(prefix []int, level, seed int)
	rec = func(prefix []int, level, seed int) {
		if level == depth {
			paths = append(paths, append([]int(nil), prefix...))
			return
		}
		for f := 0; f < fanout; f++ {
			child := intlist.Concat(prefix, lcgIDs(seg, vocab, uint64(seed*7919+f*131+level*17+1)))
			rec(child, level+1, seed*fanout+f+1)
		}
	}
	rec(lcgIDs(root, vocab, 4001), 0, 1)
	return Workload{
		Name: "tree-of-thought", Desc: "branching reasoning tree; siblings share every ancestor (ToT/GSM-8K shape)",
		Params:   fmt.Sprintf("root=%d seg=%d fanout=%d depth=%d leaves=%d", root, seg, fanout, depth, len(paths)),
		Requests: paths, SGLang: "multi-level branch reuse; within 50-99% band",
	}
}

// agents: C agents share a P-token system prefix (tool schemas), each running T steps that
// append a step-token segment. Requests are every agent's per-step prompt: TWO-LEVEL
// sharing (the system prefix across agents + each agent's own growing chain) — the
// ReAct/generative-agents shape.
func agents(vocab, P, step, C, T int) Workload {
	sys := lcgIDs(P, vocab, 5001)
	ctx := make([][]int, C)
	for a := range ctx {
		ctx[a] = append([]int(nil), sys...)
	}
	var reqs [][]int
	// Turn-major arrival order: the C agents run CONCURRENTLY, so their per-step prompts
	// INTERLEAVE in arrival time (A0,A1,..,A4 step0; then step1; ...). That interleaving is
	// what makes a bounded cache thrash under FCFS and what cache-aware (longest-shared-
	// prefix-first ≡ DFS) scheduling fixes — the paper's headline scheduling result.
	for t := 0; t < T; t++ {
		for a := 0; a < C; a++ {
			ctx[a] = intlist.Concat(ctx[a], lcgIDs(step, vocab, uint64(5100+a*97+t)))
			reqs = append(reqs, append([]int(nil), ctx[a]...))
		}
	}
	return Workload{
		Name: "agents", Desc: "C concurrent agents share a system prefix; each has a private growing context, arrivals interleaved (ReAct shape)",
		Params:   fmt.Sprintf("sys=%d step=%d agents=%d turns=%d", P, step, C, T),
		Requests: reqs, SGLang: "two-level reuse (shared system + per-agent chain); cache-aware scheduling recovers hit rate under interleaving",
	}
}

// ---- workload loading (an operator's OWN token-id prompt set, #322) ----

// workloadJSON is the on-disk shape of a bundled / operator-supplied workload: a named
// set of requests, each request a flat list of token ids. Shared prefixes ARE literally
// the same leading ids — that is what the radix tree discovers and reuses. Unlike the
// synthetic generators above (which the harness ships for the verified SGLang benchmark
// shapes), this lets an operator sweep the four shapes against THEIR OWN prompts.
type workloadJSON struct {
	Name     string  `json:"name"`
	Desc     string  `json:"desc"`
	Params   string  `json:"params"`
	SGLang   string  `json:"sglang_published"`
	Requests [][]int `json:"requests"`
}

// loadWorkload reads one workload JSON file into a Workload. Token ids are taken
// verbatim (the cache-hit metric is a function of the ids + matching algorithm only);
// the name defaults to the file's base name when the JSON omits one.
func loadWorkload(path string) (Workload, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return Workload{}, err
	}
	var f workloadJSON
	if err := json.Unmarshal(blob, &f); err != nil {
		return Workload{}, fmt.Errorf("%s: %w", path, err)
	}
	if len(f.Requests) == 0 {
		return Workload{}, fmt.Errorf("%s: workload has no requests", path)
	}
	name := f.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return Workload{
		Name: name, Desc: f.Desc, Params: f.Params,
		Requests: f.Requests, SGLang: f.SGLang,
	}, nil
}

// maxTokenID is the largest token id across every request — used to check a loaded
// workload fits the synthetic model's vocab before the live arm tries to embed it.
func maxTokenID(ws []Workload) int {
	m := -1
	for _, w := range ws {
		for _, r := range w.Requests {
			for _, id := range r {
				if id > m {
					m = id
				}
			}
		}
	}
	return m
}

// ---- accounting (model-independent: the apples-to-apples-with-SGLang metric) ----

func totalTokens(reqs [][]int) int {
	n := 0
	for _, r := range reqs {
		n += len(r)
	}
	return n
}

// radixMatched runs the requests through an (optionally bounded) radix tree in the given
// order and returns the tokens reused, plus eviction/split counts.
func radixMatched(reqs [][]int, budget int) (matched, evictions, splits int) {
	t := radixkv.New(budget)
	for _, r := range reqs {
		matched += t.MatchLen(r) // measured BEFORE this request mutates the tree
		b, m := t.Lookup(r)
		leaf := t.Insert(b, r[m:], nil)
		t.Done(leaf)
	}
	st := t.Stats()
	return matched, st.Evictions, st.Splits
}

// longestCommonPrefix is the single prefix the declare-one-prefix engine could reuse: the
// longest token run shared by EVERY request.
func longestCommonPrefix(reqs [][]int) int {
	if len(reqs) == 0 {
		return 0
	}
	lcp := len(reqs[0])
	for _, r := range reqs[1:] {
		n := lcp
		if len(r) < n {
			n = len(r)
		}
		j := 0
		for j < n && r[j] == reqs[0][j] {
			j++
		}
		lcp = j
	}
	return lcp
}

// declaredMatched is what fak's pre-radix NewBatchFromPrefix path reuses: the one global
// prefix, cloned to every request after the first.
func declaredMatched(reqs [][]int) (lcp, matched int) {
	lcp = longestCommonPrefix(reqs)
	if len(reqs) > 0 {
		matched = lcp * (len(reqs) - 1)
	}
	return
}

// lexLess orders token sequences lexicographically — the DFS pre-order of the radix tree,
// which the paper proves is equivalent to longest-shared-prefix-first (cache-aware)
// scheduling and achieves the OPTIMAL hit rate when budget >= max request length.
func lexLess(a, b []int) bool {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
}

func cacheAwareOrder(reqs [][]int) [][]int {
	cp := make([][]int, len(reqs))
	copy(cp, reqs)
	sort.SliceStable(cp, func(i, j int) bool { return lexLess(cp[i], cp[j]) })
	return cp
}

func maxReqLen(reqs [][]int) int {
	m := 0
	for _, r := range reqs {
		if len(r) > m {
			m = len(r)
		}
	}
	return m
}

// ---- budget sweep (#3397): replay a CAPTURED key-trace across budgets -> savings curve ----

// dollarsPerMTok prices the sweep's $-savings proxy: every matched (cache-hit) token is a
// prompt token NOT recomputed, valued at this many dollars per MILLION tokens. The default
// is a round frontier-model input-token list price; -price-per-mtok overrides it for your
// own contract. The savings column is a PROJECTION (price x tokens saved), not a bill.
var dollarsPerMTok = 3.0

// SweepRow is one point on the savings-vs-budget curve: the captured trace replayed
// through radixkv.New(BudgetTokens) — the same bounded replay radixMatched runs — read
// for hit rate and the dollarsPerMTok savings proxy.
type SweepRow struct {
	BudgetTokens  int     `json:"budget_tokens"` // radixkv LRU budget in cached tokens; 0 = unbounded
	TotalTokens   int     `json:"total_prompt_tokens"`
	MatchedTokens int     `json:"matched_tokens"`
	HitRate       float64 `json:"hit_rate"`
	Evictions     int     `json:"evictions"`
	SavingsUSD    float64 `json:"savings_usd_proxy"`
}

// sweepBudgets replays reqs through a FRESH bounded radix cache at each budget (in the
// order given) and returns the per-budget row: matched tokens, hit rate = matched/total,
// evictions, and the savings proxy. Pure accounting — no model, file, or network — so an
// operator's savings-vs-budget curve is a function of (their trace, the matching
// algorithm, the budget) only, exactly like the hit-rate headline above.
func sweepBudgets(reqs [][]int, budgets []int) []SweepRow {
	total := totalTokens(reqs)
	rows := make([]SweepRow, 0, len(budgets))
	for _, b := range budgets {
		matched, evictions, _ := radixMatched(reqs, b)
		rows = append(rows, SweepRow{
			BudgetTokens: b, TotalTokens: total, MatchedTokens: matched,
			HitRate: ratio(matched, total), Evictions: evictions,
			SavingsUSD: float64(matched) / 1e6 * dollarsPerMTok,
		})
	}
	return rows
}

// parseTrace reads a captured key-trace: one request per non-blank line, each line the
// request's token/block ids as space-separated non-negative integers, in arrival order.
// Lines whose first non-space character is '#' are comments; blank lines are skipped.
// This is the operator export format — a REAL logged key side-stream (the shape
// radixMatched consumes), not a synthetic workload.
func parseTrace(r io.Reader) ([][]int, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024) // one request line can carry many thousands of ids
	var reqs [][]int
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		fields := strings.Fields(text)
		req := make([]int, 0, len(fields))
		for _, f := range fields {
			id, err := strconv.Atoi(f)
			if err != nil || id < 0 {
				return nil, fmt.Errorf("trace line %d: %q is not a non-negative integer token id", line, f)
			}
			req = append(req, id)
		}
		reqs = append(reqs, req)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(reqs) == 0 {
		return nil, fmt.Errorf("trace has no requests (every line blank or a # comment)")
	}
	return reqs, nil
}

// parseBudgets parses the -budgets flag: comma-separated non-negative token budgets
// (0 = unbounded), swept in the order given.
func parseBudgets(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		b, err := strconv.Atoi(part)
		if err != nil || b < 0 {
			return nil, fmt.Errorf("-budgets: %q is not a non-negative token budget", part)
		}
		out = append(out, b)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("-budgets: no budgets given")
	}
	return out, nil
}

// defaultBudgetLadder is what -budget-sweep sweeps when -budgets is not given: anchored to
// the trace's own shape via L = max request length (the same anchor the per-workload
// bounded run uses for its single point, L + L/2), it climbs L/2, L, 3L/2, 2L, 4L, 8L and
// finishes at 0 (unbounded) — the ceiling the curve converges to, so the flat region that
// says "a bigger cache buys nothing more" is visible on every trace. Zero/duplicate rungs
// (short traces) are dropped.
func defaultBudgetLadder(reqs [][]int) []int {
	l := maxReqLen(reqs)
	seen := map[int]bool{}
	var out []int
	for _, b := range []int{l / 2, l, l + l/2, 2 * l, 4 * l, 8 * l} {
		if b <= 0 || seen[b] {
			continue
		}
		seen[b] = true
		out = append(out, b)
	}
	return append(out, 0) // unbounded ceiling
}

// runBudgetSweep is the -budget-sweep mode (#3397): load the captured trace, pick the
// budget ladder, replay, and emit the savings-vs-budget curve — human rows on stderr and
// the JSON report to -out or stdout, the same output idiom as the workload run. Everything
// here is wiring; the accounting lives in parseTrace + sweepBudgets.
func runBudgetSweep(tracePath, budgetsCSV, outPath string) error {
	f, err := os.Open(tracePath)
	if err != nil {
		return fmt.Errorf("trace open: %w", err)
	}
	defer f.Close()
	reqs, err := parseTrace(f)
	if err != nil {
		return fmt.Errorf("trace parse: %w", err)
	}
	var budgets []int
	if budgetsCSV != "" {
		if budgets, err = parseBudgets(budgetsCSV); err != nil {
			return err
		}
	} else {
		budgets = defaultBudgetLadder(reqs)
	}
	rows := sweepBudgets(reqs, budgets)

	fmt.Fprintf(os.Stderr, "budget-sweep: trace=%s requests=%d tokens=%d price=$%.2f/MTok\n",
		tracePath, len(reqs), totalTokens(reqs), dollarsPerMTok)
	for _, r := range rows {
		label := strconv.Itoa(r.BudgetTokens)
		if r.BudgetTokens == 0 {
			label = "unbounded"
		}
		fmt.Fprintf(os.Stderr, "  budget %-10s | HIT RATE %5.1f%% | matched %d/%d | evictions %d | saved $%.6f\n",
			label, 100*r.HitRate, r.MatchedTokens, r.TotalTokens, r.Evictions, r.SavingsUSD)
	}

	report := map[string]any{
		"app_version": appversion.Current(),
		"engine": "fak radixbench -budget-sweep — captured key-trace replayed through radixkv.New(budget) " +
			"across a budget ladder (#3397, LMCache cache_simulator technique, clean-room)",
		"trace":               tracePath,
		"requests":            len(reqs),
		"total_prompt_tokens": totalTokens(reqs),
		"dollars_per_mtok":    dollarsPerMTok,
		"budgets":             rows,
		"notes": "hit_rate = matched/total prompt tokens at that LRU token budget (0 = unbounded); " +
			"savings_usd_proxy = matched tokens x dollars_per_mtok / 1e6 — a PROJECTION of recompute avoided at " +
			"the configured price, not a bill. Replay is arrival-order (FCFS) over the captured trace, so the " +
			"curve reflects YOUR interleaving, not a synthetic shape.",
	}
	blob, _ := benchcli.MarshalReport(report)
	if outPath != "" {
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir: %v\n", err)
		}
		if err := os.WriteFile(outPath, blob, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", outPath, err)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", outPath)
	} else {
		fmt.Println(string(blob))
	}
	return nil
}

// ---- live wall-clock (a real kernel prefill per request) ----

func liveTiming(m *model.Model, quant bool, reqs [][]int, reps int) (baseMS, radixMS float64, liveMatched int) {
	if reps < 1 {
		reps = 1
	}
	baseMS, radixMS = 1e18, 1e18 // min over reps damps the run-to-run variance of a shared box

	// baseline: full prefill per request.
	for rep := 0; rep < reps; rep++ {
		t0 := time.Now()
		for _, r := range reqs {
			s := m.NewSession()
			s.Quant = quant
			s.Prefill(r)
		}
		baseMS = min64(baseMS, msSince(t0))
	}

	// radix: reuse the longest cached prefix, prefill only the suffix. The clone in
	// SessionFromPrefix is a COPY (not SGLang's zero-copy page share), so the suffix prefill
	// is charged in full AND the prefix copy is charged — a CONSERVATIVE live speedup.
	for rep := 0; rep < reps; rep++ {
		tree := radixkv.New(0)
		matchedThisRep := 0
		t1 := time.Now()
		for _, r := range reqs {
			b, matched := tree.Lookup(r)
			matchedThisRep += matched
			var s *model.Session
			if b.KV() != nil {
				s = m.SessionFromPrefix(b.KV()) // clone the cached prefix
			} else {
				s = m.NewSession()
			}
			s.Quant = quant
			s.Prefill(r[matched:]) // prefill ONLY the unmatched suffix
			leaf := tree.Insert(b, r[matched:], s.Cache)
			tree.Done(leaf)
		}
		radixMS = min64(radixMS, msSince(t1))
		liveMatched = matchedThisRep
	}
	return
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Nanoseconds()) / 1e6 }

// ---- policy-eviction witness (the differentiator LRU cannot offer) ----

// policyEvictionWitness builds a tree from a workload, then evicts a SPECIFIC prefix on a
// "verdict" (not LRU): it reports the tokens freed and confirms a benign sibling sharing
// the same ancestor is untouched. The KV-level bit-exactness of the corresponding session
// eviction (Evict == never-saw-it) is proven in internal/model; here we witness the
// radix-tree governance: removal is policy-driven, span-exact, sibling-preserving.
type policyWitness struct {
	Demonstrated bool   `json:"demonstrated"`
	FreedTokens  int    `json:"freed_tokens"`
	SiblingKept  bool   `json:"benign_sibling_kept"`
	Note         string `json:"note"`
}

func policyEvictionWitness(vocab int) policyWitness {
	pre := lcgIDs(16, vocab, 9001)
	good := intlist.Concat(pre, lcgIDs(8, vocab, 9100))
	bad := intlist.Concat(pre, lcgIDs(8, vocab, 9200)) // e.g. a poisoned tool-result span sharing the system prefix

	tree := radixkv.New(0)
	bg, mg := tree.Lookup(good)
	tree.Done(tree.Insert(bg, good[mg:], nil))
	bb, mb := tree.Lookup(bad)
	badLeaf := tree.Insert(bb, bad[mb:], nil)
	tree.Done(badLeaf)

	freed := tree.EvictNode(badLeaf) // verdict-driven eviction (NOT memory pressure)
	siblingKept := tree.MatchLen(good) == len(good)
	return policyWitness{
		Demonstrated: freed > 0 && siblingKept,
		FreedTokens:  freed,
		SiblingKept:  siblingKept,
		Note:         "verdict-driven span eviction (radixkv.EvictNode); shared system prefix + benign sibling preserved. LRU radix caches evict only under memory pressure.",
	}
}

// ---- per-workload report ----

type wlReport struct {
	Workload
	Requests       int     `json:"requests"`
	TotalTokens    int     `json:"total_prompt_tokens"`
	RadixMatched   int     `json:"radix_reused_tokens"`
	RadixComputed  int     `json:"radix_computed_tokens"`
	HitRate        float64 `json:"cache_hit_rate"`
	TokenSpeedup   float64 `json:"prefill_token_speedup"`
	Splits         int     `json:"radix_edge_splits"`
	DeclaredLCP    int     `json:"declare_one_prefix_lcp"`
	DeclaredReused int     `json:"declare_one_prefix_reused_tokens"`
	DeclaredHit    float64 `json:"declare_one_prefix_hit_rate"`
	RadixAddsX     float64 `json:"radix_reuse_over_declare_one"` // radix reused / declared reused
	// bounded-cache LRU under FCFS vs cache-aware (DFS) order
	Budget        int     `json:"bounded_budget_tokens"`
	MaxReqLen     int     `json:"max_request_len"`
	FCFSHit       float64 `json:"bounded_fcfs_hit_rate"`
	FCFSEvict     int     `json:"bounded_fcfs_evictions"`
	CacheAwareHit float64 `json:"bounded_cacheaware_hit_rate"`
	PctOfOptimal  float64 `json:"cacheaware_pct_of_optimal"`
	// live
	LiveBaseMS  float64 `json:"live_baseline_ms,omitempty"`
	LiveRadixMS float64 `json:"live_radix_ms,omitempty"`
	LiveSpeedup float64 `json:"live_prefill_speedup,omitempty"`
}

func analyze(w Workload, m *model.Model, quant bool, live bool, reps int) wlReport {
	base := totalTokens(w.Requests)
	matched, _, splits := radixMatched(w.Requests, 0) // unbounded => achievable (optimal) reuse
	computed := base - matched
	lcp, dMatched := declaredMatched(w.Requests)

	r := wlReport{
		Workload: w, Requests: len(w.Requests), TotalTokens: base,
		RadixMatched: matched, RadixComputed: computed,
		HitRate: ratio(matched, base), TokenSpeedup: safeDiv(float64(base), float64(computed)),
		Splits:      splits,
		DeclaredLCP: lcp, DeclaredReused: dMatched, DeclaredHit: ratio(dMatched, base),
		RadixAddsX: safeDiv(float64(matched), float64(dMatched)),
	}

	// bounded cache: budget just above the max single request so DFS can reach optimal but
	// FCFS (interleaved arrivals) must evict and re-fetch hot prefixes.
	r.MaxReqLen = maxReqLen(w.Requests)
	r.Budget = r.MaxReqLen + r.MaxReqLen/2
	fcfsM, fcfsE, _ := radixMatched(w.Requests, r.Budget)
	caM, _, _ := radixMatched(cacheAwareOrder(w.Requests), r.Budget)
	r.FCFSHit = ratio(fcfsM, base)
	r.FCFSEvict = fcfsE
	r.CacheAwareHit = ratio(caM, base)
	r.PctOfOptimal = safeDiv(float64(caM), float64(matched)) // vs unbounded optimal

	if live && m != nil {
		bMS, rMS, _ := liveTiming(m, quant, w.Requests, reps)
		r.LiveBaseMS, r.LiveRadixMS = bMS, rMS
		r.LiveSpeedup = safeDiv(bMS, rMS)
	}
	return r
}

func ratio(a, b int) float64 { return safeDiv(float64(a), float64(b)) }
func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

// ---- model loading (synthetic by default; -hf for a real checkpoint) ----

func syntheticModel() (*model.Model, string, int) {
	cfg := model.Config{
		HiddenSize: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 128, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 255,
	}
	return model.NewSynthetic(cfg), "synthetic-llama (64h/4L/8q-2kv, vocab 256) — WIRING witness; numerics proven by internal/model oracle", cfg.VocabSize
}

func main() {
	dir := flag.String("dir", "", "fak export dir for the LIVE arm (config.json + weights.f32; loaded via model.Load)")
	hf := flag.String("hf", "", "HuggingFace snapshot dir for the LIVE wall-clock arm (config.json + model.safetensors)")
	lean := flag.Bool("lean", false, "memory-lean quantize-at-load (requires -hf)")
	quant := flag.Bool("quant", false, "use the Q8_0 quantized lane for the live arm")
	live := flag.Bool("live", true, "run the live wall-clock arm (a real kernel prefill per request)")
	reps := flag.Int("reps", 3, "live-timing reps; the min over reps is reported (damps shared-box variance)")
	only := flag.String("only", "", "run only workloads whose name contains this substring (e.g. few-shot); empty = all")
	workload := flag.String("workload", "", "comma-separated JSON workload files (each: {name,desc,sglang_published,requests:[[ids]...]}) to sweep INSTEAD of the synthetic shapes — your OWN prompt set (#322)")
	scale := flag.Int("scale", 1, "token-size multiplier for the workloads (1 = quick synthetic; larger for real models)")
	out := flag.String("out", "", "write JSON report here (default stdout)")
	budgetSweep := flag.Bool("budget-sweep", false, "replay -trace across a budget ladder and PROJECT hit-rate + $-savings per budget INSTEAD of running the workload benchmark (#3397)")
	trace := flag.String("trace", "", "captured key-trace file for -budget-sweep: one request per line, space-separated non-negative integer token/block ids; # comments and blank lines skipped")
	budgets := flag.String("budgets", "", "comma-separated radixkv token budgets to sweep (0 = unbounded); implies -budget-sweep; default = a ladder anchored to the trace's max request length (L/2,L,3L/2,2L,4L,8L,unbounded)")
	price := flag.Float64("price-per-mtok", dollarsPerMTok, "dollars per million prompt tokens for the sweep's $-savings proxy")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)
	*hf = pathutil.ExpandTilde(*hf)
	*trace = pathutil.ExpandTilde(*trace)

	// -budget-sweep (or its implying flags) is a pure-accounting mode over a CAPTURED
	// trace: no model, no live arm — parse, sweep, report, done.
	if *budgetSweep || *trace != "" || *budgets != "" {
		dollarsPerMTok = *price
		if *trace == "" {
			fmt.Fprintln(os.Stderr, "budget-sweep: -trace <file> is required (the captured key-trace to replay)")
			os.Exit(1)
		}
		if err := runBudgetSweep(*trace, *budgets, *out); err != nil {
			fmt.Fprintf(os.Stderr, "budget-sweep: %v\n", err)
			os.Exit(1)
		}
		return
	}

	var m *model.Model
	var name string
	var vocab int
	if *dir != "" {
		var err error
		m, err = model.Load(*dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "dir load: %v\n", err)
			os.Exit(1)
		}
		if *quant {
			m.Quantize()
		}
		name = filepath.Base(*dir)
		vocab = m.Cfg.VocabSize
	} else if *hf != "" {
		cfg, err := benchcli.ReadHFConfig(*hf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "hf config: %v\n", err)
			os.Exit(1)
		}
		if *lean {
			m, err = model.LoadSafetensorsQuantDir(*hf, cfg)
		} else {
			m, err = model.LoadSafetensors(filepath.Join(*hf, "model.safetensors"), cfg)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "hf load: %v\n", err)
			os.Exit(1)
		}
		if *quant || *lean {
			m.Quantize()
			*quant = true
		}
		name = filepath.Base(*hf)
		vocab = cfg.VocabSize
	} else {
		m, name, vocab = syntheticModel()
	}

	s := *scale
	var all []Workload
	if *workload != "" {
		for _, p := range strings.Split(*workload, ",") {
			p = strings.TrimSpace(pathutil.ExpandTilde(p))
			if p == "" {
				continue
			}
			w, err := loadWorkload(p)
			if err != nil {
				fmt.Fprintf(os.Stderr, "workload load: %v\n", err)
				os.Exit(1)
			}
			all = append(all, w)
		}
	} else {
		all = []Workload{
			fewShot(vocab, 256*s, 16*s, 16),
			multiTurn(vocab, 64*s, 32*s, 8),
			treeOfThought(vocab, 64*s, 16*s, 3, 3),
			agents(vocab, 128*s, 24*s, 5, 6),
		}
	}
	var workloads []Workload
	for _, w := range all {
		if *only == "" || strings.Contains(w.Name, *only) {
			workloads = append(workloads, w)
		}
	}

	// A loaded workload carries its own token ids; the default synthetic model only
	// embeds ids < vocab. If an operator's ids exceed it, disable the live arm (a real
	// synthetic prefill would index out of range) — the hit-rate accounting is
	// model-independent, so the headline still stands. Pass -hf/-dir for live timing.
	if *workload != "" && *dir == "" && *hf == "" && *live {
		if mx := maxTokenID(workloads); mx >= vocab {
			fmt.Fprintf(os.Stderr, "note: workload token id %d >= synthetic vocab %d; live timing disabled "+
				"(hit-rate accounting is model-independent). Pass -hf/-dir for live timing on your own tokens.\n", mx, vocab)
			*live = false
		}
	}

	// warm the kernel once so the first live timing isn't a cold outlier.
	if *live {
		ws := m.NewSession()
		ws.Quant = *quant
		ws.Prefill(lcgIDs(8, vocab, 7))
	}

	var reports []wlReport
	fmt.Fprintf(os.Stderr, "model: %s  (vocab %d, GOMAXPROCS %d)\n", name, vocab, runtime.GOMAXPROCS(0))
	for _, w := range workloads {
		fmt.Fprintf(os.Stderr, "workload %-16s %s ...\n", w.Name, w.Params)
		r := analyze(w, m, *quant, *live, *reps)
		reports = append(reports, r)
		fmt.Fprintf(os.Stderr,
			"  reqs=%d tokens=%d | HIT RATE %.1f%% | HEADLINE radix reuses %.2fx vs a WARM declare-one-prefix cache (the cross-subtree win) | "+
				"bounded: FCFS %.1f%% -> cache-aware %.1f%% (%.0f%% of optimal) | no-cache speedup %.2fx (worst-case ref)",
			r.Requests, r.TotalTokens, 100*r.HitRate, r.RadixAddsX,
			100*r.FCFSHit, 100*r.CacheAwareHit, 100*r.PctOfOptimal, r.TokenSpeedup)
		if *live {
			fmt.Fprintf(os.Stderr, " | LIVE %.0f->%.0f ms (%.2fx)", r.LiveBaseMS, r.LiveRadixMS, r.LiveSpeedup)
		}
		fmt.Fprintln(os.Stderr)
		runtime.GC()
	}

	witness := policyEvictionWitness(vocab)
	fmt.Fprintf(os.Stderr, "policy-eviction witness: freed %d tokens on a verdict, benign sibling kept=%v\n",
		witness.FreedTokens, witness.SiblingKept)

	report := map[string]any{
		"app_version": appversion.Current(),
		"engine":      "fak radixbench — RadixAttention-style prefix cache (internal/radixkv) over the kernel-owned KV cache",
		"model":       name,
		"quant":       *quant,
		"scale":       s,
		"go_threads":  runtime.GOMAXPROCS(0),
		"headline": "cache hit rate (SGLang's axis) + radix_reuse_over_declare_one — radix's automatic cross-subtree discovery " +
			"vs a WARM single-declared-prefix cache (the serving-realistic baseline). prefill_token_speedup is vs the COLD no-cache " +
			"baseline — a worst-case REFERENCE only, not a serving baseline anyone ships.",
		"axis": "cache hit rate (hardware/model-independent) — the metric SGLang's paper headlines; " +
			"fak runs the same radix-tree + longest-prefix + LRU-leaf algorithm, so on the same workload it reaches the same reuse",
		"sglang_published": map[string]string{
			"cache_hit_rate":         "50%-99% across benchmarks (CONFIRMED vs NeurIPS 2024 PDF)",
			"cacheaware_pct_optimal": "96% of optimal on average (DFS-order optimality)",
			"throughput":             "up to 6.4x (Llama-2 7B-70B fp16 vs Guidance/vLLM/LMQL)",
			"latency":                "up to 3.7x",
			"source":                 "arXiv:2312.07104 / NeurIPS 2024 (724be4472168f31ba1c9ac630f15dec8)",
		},
		"workloads":       reports,
		"policy_eviction": witness,
		"notes": "baseline=full prefill per request (COLD no-cache — worst-case reference, not a serving baseline); " +
			"declare-one-prefix=fak's pre-radix single-declared-prefix reuse (the WARM serving-realistic baseline); " +
			"radix=internal/radixkv automatic longest-prefix discovery (incl. mid-run split). Live arm is a real kernel " +
			"prefill per request, bit-identical to recompute (proven in internal/radixkv). Cache-aware order = lexicographic " +
			"(== DFS == longest-shared-prefix-first), reproducing the paper's optimal-hit-rate-at-budget>=maxlen result.",
	}
	blob, _ := benchcli.MarshalReport(report)
	if *out != "" {
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir: %v\n", err)
		}
		if err := os.WriteFile(*out, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", *out, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", *out)
	} else {
		fmt.Println(string(blob))
	}
}
