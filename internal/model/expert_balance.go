package model

import "fmt"

// expert_balance.go — LOAD-AWARE expert-parallel placement: the native realization of
// DeepSeek-EPLB's balanced_packing (eplb.py:5-41) for fak's EP path, filed as #3886.
//
// WHY THIS EXISTS. ExpertParallelPlan (expert_parallel.go:68) delegates to NewTPPlan
// (tensor_parallel.go:70), which tiles the experts into near-even contiguous bands BY
// COUNT: rank r gets ≈ NumExperts/R experts regardless of how many TOKENS each expert
// actually draws. That is load-OBLIVIOUS. Real routed traffic (GLM-5.2 / DeepSeek group-
// limited routing) is skewed — a few "hot" experts concentrate most of the picks — so
// the count-even plan piles the hot experts onto whichever rank owns their band and caps
// per-token latency at that hottest rank. The GLM EP note names this symptom without
// fixing it ("load-imbalanced (all picks on one rank) ranks=2 == ranks=1, max|Δ|=0",
// docs/notes/GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md): the count plan reduces
// correctly under imbalance, it never REBALANCES.
//
// WHAT EPLB DOES AND WHAT WE BORROW. EPLB's balanced_packing is an LPT (longest-processing-
// time) greedy that assigns experts heaviest-first to the least-loaded rank, minimizing the
// MAX per-rank LOAD rather than per-rank count. Its assignment is in general NON-contiguous
// (any expert may land on any rank via a log2phy map). fak's EP reduce path, however, pins a
// load-bearing invariant: bands must be CONTIGUOUS ascending runs, because glmRoute returns
// picks in expert-ascending order and expertParallelPartials sums each band's picks in that
// order — so the cross-rank AllReduceSum is a purely associative REGROUPING of the monolith's
// ascending sum (bit-exact vs expertParallelReference; ~1e-6 vs the monolith), never a reorder
// of unrelated terms (expert_parallel.go:24-44). A non-contiguous LPT assignment cannot be
// expressed as a TPPlan at all (TPPlan.Validate requires gap-free ascending coverage) and would
// break that invariant. So the P1 slice #3886 scopes is the CONTIGUITY-PRESERVING realization of
// balanced_packing: partition [0,NumExperts) into `ranks` CONTIGUOUS bands whose per-band summed
// LOAD is as even as possible. That still plugs straight into the existing ExpertParallelDelta /
// expertParallelPartials path (it returns an ordinary TPPlan), changes only WHICH rank owns an
// expert — never the expert math — and keeps the ranks=1 bit-exactness (a single band [0,n)).
//
// Hot-expert REPLICATION (EPLB replicate_experts, redundant physical slots + log2phy) is the
// explicitly-deferred P2 of #3886 — it needs a redundant-residency mechanism (gated on #3174/
// #3212) and must always be followed by a pack, never shipped alone (EPLB commit e1100fe). This
// file lands only the load-balanced placement P1; it coins no redundant-slot machinery.
//
// WHY OPTIMAL, NOT GREEDY. Because the contiguity constraint makes the problem the classic
// "split an array into k non-empty contiguous parts minimizing the largest part-sum", which has
// an EXACT O(R·n²) dynamic program. We take the optimum rather than EPLB's greedy LPT: it is
// deterministic (ties broken by the smallest split index), needs no wall clock, and gives the
// strongest possible witness — the returned plan's max-per-rank load is provably ≤ any other
// contiguous plan's, in particular ≤ the count-even NewTPPlan bands. Load-oblivious equal bands
// stay the default (ExpertParallelPlan is unchanged); this is opt-in when a load vector exists.

// LoadBalancedExpertBands partitions the experts [0,len(load)) across `ranks` CONTIGUOUS bands
// so the maximum per-band summed load is minimized, returning an ordinary TPPlan the existing
// EP path (expertParallelPartials / ExpertParallelDelta) consumes unchanged. load[e] is the
// measured (or synthetic-skewed) load of expert e — the routed-pick histogram fak already
// computes from glmRoute picks (a token count), so it is a non-negative integer per expert.
//
// It is the load-aware counterpart of ExpertParallelPlan's count-even NewTPPlan bands: same
// contiguous-ascending shape (so the ranks=1 plan is the single band [0,n) and stays bit-exact
// vs the monolith, and every band is an ascending run so the reduce regrouping is associative-
// only), but band WIDTHS are chosen to balance LOAD, not count. On a skewed load vector its
// max-per-rank load is strictly below the count-even plan's; on a flat load vector the two
// coincide (equal load makes count-even already max-optimal).
//
// It fails closed exactly where NewTPPlan does — ranks must be in [1, len(load)] so no band is
// empty — and additionally rejects a negative load entry (a pick histogram is a count, never
// negative; a negative would be a miscomputed load vector, so refuse rather than balance garbage).
func LoadBalancedExpertBands(load []int, ranks int) (TPPlan, error) {
	n := len(load)
	if n <= 0 {
		return TPPlan{}, fmt.Errorf("model: LoadBalancedExpertBands load is empty (not an MoE config?)")
	}
	if ranks <= 0 {
		return TPPlan{}, fmt.Errorf("model: LoadBalancedExpertBands ranks = %d, want > 0", ranks)
	}
	if ranks > n {
		return TPPlan{}, fmt.Errorf("model: LoadBalancedExpertBands ranks = %d > experts = %d (would leave a rank with no work)", ranks, n)
	}
	for e, w := range load {
		if w < 0 {
			return TPPlan{}, fmt.Errorf("model: LoadBalancedExpertBands load[%d] = %d, want >= 0 (a routed-pick histogram is a count)", e, w)
		}
	}

	// prefix[i] = sum of load[0..i). Band [a,b)'s load is prefix[b]-prefix[a] in O(1).
	prefix := make([]int, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + load[i]
	}
	bandLoad := func(a, b int) int { return prefix[b] - prefix[a] }

	// DP over "first i experts split into k non-empty contiguous bands".
	//   best[k][i]  = the minimized largest band-load for that sub-partition (i >= k).
	//   cut[k][i]   = the split index j so band k-1 is [j,i) — used to reconstruct boundaries.
	// best[1][i] is just band [0,i). best[k][i] = min over j in [k-1,i-1] of
	// max(best[k-1][j], load([j,i))). Ties break to the SMALLEST j (strict <), so the plan is
	// deterministic — the same load vector always yields byte-identical bands, matching the
	// numeric discipline the rest of the EP path relies on.
	const inf = int(^uint(0) >> 1)
	best := make([][]int, ranks+1)
	cut := make([][]int, ranks+1)
	for k := 0; k <= ranks; k++ {
		best[k] = make([]int, n+1)
		cut[k] = make([]int, n+1)
		for i := range best[k] {
			best[k][i] = inf
		}
	}
	for i := 1; i <= n; i++ {
		best[1][i] = bandLoad(0, i)
		cut[1][i] = 0
	}
	for k := 2; k <= ranks; k++ {
		// A k-band split of [0,i) needs i >= k experts; the last band [j,i) needs j >= k-1.
		for i := k; i <= n; i++ {
			for j := k - 1; j < i; j++ {
				if best[k-1][j] == inf {
					continue
				}
				cand := best[k-1][j]
				if tail := bandLoad(j, i); tail > cand {
					cand = tail
				}
				if cand < best[k][i] {
					best[k][i] = cand
					cut[k][i] = j
				}
			}
		}
	}

	// Reconstruct the band boundaries by walking cut[] back from (ranks, n).
	bounds := make([]int, ranks+1)
	bounds[ranks] = n
	i := n
	for k := ranks; k >= 1; k-- {
		j := cut[k][i]
		bounds[k-1] = j
		i = j
	}

	shards := make([]TPShard, ranks)
	for r := 0; r < ranks; r++ {
		shards[r] = TPShard{Rank: r, Lo: bounds[r], Hi: bounds[r+1]}
	}
	p := TPPlan{Dim: n, Shards: shards}
	if err := p.Validate(); err != nil {
		return TPPlan{}, err
	}
	return p, nil
}

// RankLoads returns the per-rank summed load a plan induces over `load`: RankLoads[r] is the
// total load of the experts in band Shards[r].[Lo,Hi). It is the quantity a load-balanced plan
// flattens — the per-rank token work whose MAX caps per-token latency at the hottest rank. It
// fails closed if the plan does not tile exactly [0,len(load)) (Dim mismatch or a malformed
// plan), so a plan built for a different expert count cannot be silently mis-summed.
func RankLoads(load []int, p TPPlan) ([]int, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if p.Dim != len(load) {
		return nil, fmt.Errorf("model: RankLoads plan.Dim = %d, want experts = %d", p.Dim, len(load))
	}
	loads := make([]int, len(p.Shards))
	for r, s := range p.Shards {
		sum := 0
		for e := s.Lo; e < s.Hi; e++ {
			sum += load[e]
		}
		loads[r] = sum
	}
	return loads, nil
}

// Balancedness mirrors the vLLM EPLB baseline's balancedness metric (mean per-rank load /
// max per-rank load, VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md) so a native plan's balance is
// directly comparable to the external floor. It is in (0,1]: 1.0 is perfectly even (every
// rank equally loaded), and a hot-expert-on-one-rank count plan drives it toward 1/R. A
// LoadBalancedExpertBands plan has Balancedness >= the count-even plan's on the same load.
//
// It fails closed on an empty plan or a total load of zero (no traffic yet — balancedness is
// undefined, and a fabricated 1.0 would falsely claim a balanced serve).
func Balancedness(load []int, p TPPlan) (float64, error) {
	loads, err := RankLoads(load, p)
	if err != nil {
		return 0, err
	}
	total, max := 0, 0
	for _, l := range loads {
		total += l
		if l > max {
			max = l
		}
	}
	if max == 0 {
		return 0, fmt.Errorf("model: Balancedness total load is 0 (no routed traffic to balance)")
	}
	mean := float64(total) / float64(len(loads))
	return mean / float64(max), nil
}
