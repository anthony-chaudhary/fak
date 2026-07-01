package model

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// expert_parallel_ranklocal_test.go — the SHARDED (multi-process) EP gates: each rank holds ONLY
// its expert band and the LIVE forward computes only that band, reducing across a real cross-
// process collective (DistComm). Where expert_parallel_test.go proves the single-process all-band
// path (one model holding every expert, every band's partial computed locally), these prove the
// residency win #971 needs: no process holds the full expert set, yet the tokens are bit-exact vs
// full-model EP. The proof runs on one box over a loopback socket (runGroup), a genuine cross-
// process exchange — the host rung below the device-NCCL line (dist_collective.go's honesty note).

// modelWithExpertBandForTest returns a SHALLOW copy of src holding only the routed experts in
// [lo,hi) — the resident state a rank sees after a sharded load (ggufload.WithExpertShard). It
// clones each weight store, dropping every `mlp.experts.<e>.*` key whose expert e is outside the
// band; all non-expert tensors (dense, router, attention, the REPLICATED shared expert) stay, so
// the rank still runs the router and the shared-expert term. hasWeight on a peer-band expert then
// returns false — exactly the fail-closed condition expertParallelRankPartial guards.
func modelWithExpertBandForTest(src *Model, lo, hi int) *Model {
	cp := *src // shallow copy; we replace the weight-store maps with band-filtered clones
	keep := func(name string) bool {
		e, ok := expertIndexOfTestKey(name)
		if !ok {
			return true // not an expert-indexed tensor (dense / router / shared) — replicated
		}
		return e >= lo && e < hi
	}
	if src.manifest != nil {
		m := make(map[string]tensorMeta, len(src.manifest))
		for k, v := range src.manifest {
			if keep(k) {
				m[k] = v
			}
		}
		cp.manifest = m
	}
	if src.q8w != nil {
		m := make(map[string]*q8Tensor, len(src.q8w))
		for k, v := range src.q8w {
			if keep(k) {
				m[k] = v
			}
		}
		cp.q8w = m
	}
	if src.q4kw != nil {
		m := make(map[string]*q4kTensor, len(src.q4kw))
		for k, v := range src.q4kw {
			if keep(k) {
				m[k] = v
			}
		}
		cp.q4kw = m
	}
	if src.kqw != nil {
		m := make(map[string]*kQuantTensor, len(src.kqw))
		for k, v := range src.kqw {
			if keep(k) {
				m[k] = v
			}
		}
		cp.kqw = m
	}
	return &cp
}

// expertIndexOfTestKey pulls the expert index e out of a `…mlp.experts.<e>.<suffix>` weight name,
// reporting false for any name that is not a routed-expert tensor (so it is kept on every rank).
func expertIndexOfTestKey(name string) (int, bool) {
	const marker = "mlp.experts."
	i := strings.Index(name, marker)
	if i < 0 {
		return 0, false
	}
	rest := name[i+len(marker):]
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return 0, false
	}
	e, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return 0, false
	}
	return e, true
}

// glmFixtureRoutedLayer loads the tiny GLM-DSA fixture (with shared expert) and returns the model,
// config, kernel, a deterministic activation, and the first routed MoE layer — the shared setup
// the rank-local gates below reuse.
func glmFixtureRoutedLayer(t *testing.T) (*Model, Config, matKernel, []float32, int) {
	t.Helper()
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	mat := residentKernel{m}
	layer := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if m.hasWeight(routerName(l)) && m.hasWeight(expertName(l, 0, "gate_proj.weight")) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatalf("no routed MoE layer in the glm fixture")
	}
	xn := make([]float32, cfg.HiddenSize)
	for i := range xn {
		xn[i] = float32((float64(i)*0.11 + 0.3))
	}
	return m, cfg, mat, xn, layer
}

// TestRankLocalEPForwardMatchesFullEP is the keystone sharded-EP gate: per-rank band-only models,
// driven THROUGH THE LIVE DISPATCH (ffnForLayer -> glmMoeEPFFN{rankLocal}.apply -> the rank-local
// delta -> distCommCollective -> DistComm over a loopback socket), reduce to the SAME bytes as the
// single-process full-model EP delta. It is the through-the-dispatch, multi-process twin of
// TestGlmMoeEPFFNMatchesMonolith: it proves a serve where no process holds the full expert set
// still produces the correct routed+shared token.
func TestRankLocalEPForwardMatchesFullEP(t *testing.T) {
	full, cfg, mat, xn, layer := glmFixtureRoutedLayer(t)

	// ranks=1 stays on the monolith by design (ffnForLayer gates EP on epRanks>1, the bit-exact
	// no-op), and a sharded rank-1 serve is degenerate (one process holding everything). The
	// meaningful sharded counts are 2..NumExperts — each rank owning a real sub-band.
	var rankCounts []int
	for r := 2; r <= cfg.NumExperts; r++ {
		rankCounts = append(rankCounts, r)
	}
	if len(rankCounts) == 0 {
		t.Skipf("fixture NumExperts=%d has no multi-rank sharding", cfg.NumExperts)
	}
	for _, ranks := range rankCounts {
		plan, err := ExpertParallelPlan(cfg.NumExperts, ranks)
		if err != nil {
			t.Fatalf("ExpertParallelPlan(%d): %v", ranks, err)
		}
		// Oracle: the single-process full-model EP delta at this rank count (routed reduced in rank
		// order + shared once), which TestExpertParallelGLMSharedExpert already ties to glmMoeFFN.
		picks := glmRoute(full, layer, xn, mat)
		want, err := full.expertParallelGLMMoEDelta(layer, xn, mat, picks, plan, LocalCollective{})
		if err != nil {
			t.Fatalf("ranks=%d full EP oracle: %v", ranks, err)
		}

		results, errs := runGroup(t, ranks, func(g *DistComm) ([]float32, error) {
			// Each rank holds ONLY its band and drives the LIVE dispatch on that band-only model.
			s := plan.Shards[g.Rank()]
			local := modelWithExpertBandForTest(full, s.Lo, s.Hi)
			local.SetExpertParallelRanks(ranks)
			local.SetExpertParallelRank(g.Rank())
			local.SetExpertParallelCollective(NewDistCommCollective(g))
			lmat := residentKernel{local}
			kind := local.ffnForLayer(layer)
			ep, ok := kind.(glmMoeEPFFN)
			if !ok {
				return nil, fmt.Errorf("rank %d ffnForLayer = %T, want glmMoeEPFFN", g.Rank(), kind)
			}
			if !ep.rankLocal || ep.rank != g.Rank() {
				return nil, fmt.Errorf("rank %d dispatched rankLocal=%v rank=%d, want rankLocal=true rank=%d", g.Rank(), ep.rankLocal, ep.rank, g.Rank())
			}
			return ep.apply(local, layer, xn, lmat), nil
		})
		for r, err := range errs {
			if err != nil {
				t.Fatalf("ranks=%d rank %d: %v", ranks, r, err)
			}
		}
		// AllReduce returns the SAME reduced vector on every rank; each must equal the oracle.
		for r := 0; r < ranks; r++ {
			distAssertBitsEqual(t, fmt.Sprintf("rank-local EP ranks=%d rank=%d", ranks, r), results[r], want)
		}
		t.Logf("sharded EP ranks=%d: per-rank band-only models through the live dispatch == full-model EP (bit-exact)", ranks)
	}
}

// TestRankLocalEPFailsClosedOnMissingBand pins the fail-closed contract that keeps a sharded serve
// honest: a rank whose model LACKS a band it is asked to compute must surface an error, NOT fall
// back to the monolith (which would call expertSwiGLU on peer experts the rank does not hold and
// mis-serve). Here rank 0 is handed a model holding only rank 1's band but told it is rank 0, so
// every pick it owns is absent — the rank-local forward must panic with context rather than return
// a silently-wrong delta.
func TestRankLocalEPFailsClosedOnMissingBand(t *testing.T) {
	full, cfg, _, xn, layer := glmFixtureRoutedLayer(t)
	if cfg.NumExperts < 2 {
		t.Skipf("need >=2 experts to build a missing-band rank, have %d", cfg.NumExperts)
	}
	plan, err := ExpertParallelPlan(cfg.NumExperts, 2)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(2): %v", err)
	}
	// A model holding ONLY rank 1's band, but configured as rank 0 — so rank 0's owned picks are
	// all absent. (If rank 0 happens to route no tokens to its band, skip: nothing to miss.)
	wrong := modelWithExpertBandForTest(full, plan.Shards[1].Lo, plan.Shards[1].Hi)
	wrong.SetExpertParallelRanks(2)
	wrong.SetExpertParallelRank(0)
	wrong.SetExpertParallelCollective(LocalCollective{}) // identity at the reduce; the miss is upstream
	mat := residentKernel{wrong}
	picks := glmRoute(wrong, layer, xn, mat)
	ownsAPick := false
	for _, pk := range picks {
		if pk.expert >= plan.Shards[0].Lo && pk.expert < plan.Shards[0].Hi {
			ownsAPick = true
			break
		}
	}
	if !ownsAPick {
		t.Skip("router picked no expert in rank 0's band; no missing-band pick to witness")
	}

	kind := wrong.ffnForLayer(layer)
	ep, ok := kind.(glmMoeEPFFN)
	if !ok || !ep.rankLocal {
		t.Fatalf("ffnForLayer = %T (rankLocal=%v), want a rank-local glmMoeEPFFN", kind, ok && ep.rankLocal)
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("rank-local EP apply on a model missing its band did NOT panic — a silent monolith fallback would mis-serve")
		}
	}()
	_ = ep.apply(wrong, layer, xn, mat)
}
