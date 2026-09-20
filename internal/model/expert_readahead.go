package model

import (
	"fmt"
	"sort"
)

// expert_readahead.go Ã¢â‚¬â€ R3.5 of the activated-expert offload ladder (#5614 lineage, epic #5606):
// cross-LAYER gate prediction. R3 stages layer L's activated set the instant L's own router names
// it. This rung stages the PREDICTED activated set of layer L+1 while layer L still computes, so
// L+1's page-ins can be issued one whole layer earlier.
//
// The prediction, and why it is cheap. A transformer layer stack is L(0)..L(N-1); at the top of
// layer L the residual stream xn is fed to L's own router, and the expert loop then runs. But L+1's
// router consumes L's OUTPUT, which is not available until L's experts retire Ã¢â‚¬â€ so L+1's true
// activation is genuinely unknown at L's entry, and a predictor is required (#4300 owns that
// question). What IS knowable for free is a lower bound on the signal: the gate path is a single
// [E,H]x[H] GEMV, no expert GEMMs, and the top-k of a router applied to a slightly-stale hidden
// state is a usable ordering for a HINT. This rung applies L+1's router gate to L's current xn and
// stages its top-k as hints. It is a true prefetch: zero expert GEMMs, one extra gate GEMV per
// layer, and a misprediction costs only wasted page-in bandwidth, never correctness.
//
// Three invariants make this safe to leave in the tree default-off:
//
//   - A HINT IS NOT A DEMAND. The predicted set is staged through the SAME rules as R3's
//     activated-set prefetch: prefix that fits the budget, `prefetching=true` so no heat is earned
//     under the R4 value-aware policy, and no fallback to permanent halW residency on a ring
//     refusal. A mispredict therefore cannot change logits Ã¢â‚¬â€ it stages weights a later GEMM may or
//     may not read, and an expert that is not resident when demanded takes the unchanged miss path.
//     This is verified by construction (see prefetchNextLayerGateExperts).
//   - PER-SESSION AND DEFAULT OFF. The feature hangs off one Session field,
//     Session.CrossLayerGatePrefetch, whose zero value is false. Nothing in predictNextLayerExperts
//     or prefetchNextLayerGateExperts runs until a caller sets the field on ITS session. Enabling
//     one session therefore leaves every other session on the same model byte-for-byte on the
//     pre-rung path — the property a process-wide gate could not provide.
//   - HINTS DO NOT FEED THE SAME-LAYER METER. r.prefetched is incremented (these really were
//     staged ahead of a GEMM) but r.activatedExperts / r.activatedCovered are NOT Ã¢â‚¬â€ those count
//     the activated set of the layer being computed, and a cross-layer prediction is neither
//     activated nor covered on this layer's account.
//
// Session-owned state (promoted from prototype). The opt-in is the Session field
// CrossLayerGatePrefetch (kv.go), a sibling of ExpertPrefetch, and the precision/recall ledger
// lives ON the Session (crossLayerStats + crossLayerPending, guarded by crossLayerMu) rather than
// in a package-level map. There is no package-level state to leak: a session that never enables
// the knob allocates no ledger, and Session.Close clears the ledger unconditionally, so nothing
// outlives the session that owns it. The knobs stay inert without a ring, as before.

// CrossLayerPrefetchStats is the precision/recall ledger for next-layer gate prediction. Every
// counter is in EXPERT SLOTS, not experts: a layer predicts k experts and realizes k, so a layer
// contributes k to Predicted and k to Actual, and a perfect prediction contributes k to Hits.
// Counting slots (not distinct experts) is what makes Precision and Recall the standard confusion
// ratios and what lets a layer whose top-k is fully predicted and fully realized score 1.0.
//
// It is a GAUGE over the session's whole life, not a window: Reset it to start a measurement. The
// ledger is stored ON the Session (crossLayerStats) and released at Close; CrossLayerPrefetchStatsFor
// reads it under the session's own lock.
type CrossLayerPrefetchStats struct {
	Predicted int `json:"predicted"` // total predicted expert slots across layers
	Actual    int `json:"actual"`    // total realized activated expert slots
	Hits      int `json:"hits"`      // predicted intersect actual (per slot)
}

// Precision is Hits/Predicted Ã¢â‚¬â€ of the slots the predictor named, the share that were really
// activated by the next layer. 0 when nothing was predicted (the honest reading for "no
// measurement", not a perfect score).
func (s CrossLayerPrefetchStats) Precision() float64 {
	if s.Predicted == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Predicted)
}

// Recall is Hits/Actual Ã¢â‚¬â€ of the slots the next layer really activated, the share the predictor
// named. 0 when nothing was realized. Read alongside Precision and the two totals: a predictor
// that names everything scores Recall 1.0 and Precision ~k/E.
func (s CrossLayerPrefetchStats) Recall() float64 {
	if s.Actual == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Actual)
}

// CrossLayerPrefetchStatsFor returns the next-layer gate-prediction ledger for s. A nil session, or
// one that has never predicted, reports the zero ledger (Precision/Recall 0). Safe to call from
// another goroutine: the read is guarded by the session's own ledger lock.
func CrossLayerPrefetchStatsFor(s *Session) CrossLayerPrefetchStats {
	if s == nil {
		return CrossLayerPrefetchStats{}
	}
	s.crossLayerMu.Lock()
	defer s.crossLayerMu.Unlock()
	return s.crossLayerStats
}

// ResetCrossLayerPrefetchStats clears the ledger and any pending prediction for s, so a measurement
// can start from zero. A nil session is a no-op.
func ResetCrossLayerPrefetchStats(s *Session) {
	if s == nil {
		return
	}
	s.crossLayerMu.Lock()
	s.crossLayerStats = CrossLayerPrefetchStats{}
	s.crossLayerPending = nil
	s.crossLayerMu.Unlock()
}

// predictNextLayerExperts applies layer+1's router gate to `layer`'s hidden state (the cheap gate
// path ONLY Ã¢â‚¬â€ mat.mul of the router weight, no expert GEMMs) and returns the predicted top-k picks.
//
// It returns nil when the cross-layer opt-in is off for THIS session (s == nil or
// !s.CrossLayerGatePrefetch), the next layer has no router (layer+1 >= NumLayers), the model is
// GPT-OSS (whose experts have no three-projection ring staging), or mat is not a device-HAL-capable
// routed-expert kernel (routedExpertKQuantActive) â€” i.e. on every path where staging would upload
// bytes nothing reads.
//
// The picks carry their gate logit in weight: softmax is deliberately NOT applied, because the
// predicted set is used only for staging and the weights are irrelevant. The tie-break matches
// `route`'s torch.topk order (value desc, then lower expert index) so a prediction is a stable
// function of the gate vector.
func predictNextLayerExperts(s *Session, m *Model, layer int, xn any, mat matKernel) []routePick {
	if s == nil || !s.CrossLayerGatePrefetch || m == nil {
		return nil
	}
	if layer+1 >= m.Cfg.NumLayers || m.Cfg.isGPTOSS() || !routedExpertKQuantActive(mat) {
		return nil
	}
	E, K := m.Cfg.NumExperts, m.Cfg.NumExpertsPerTok
	if E <= 0 || K <= 0 {
		return nil
	}
	logits := mat.mul(routerName(layer+1), xn, E, m.Cfg.HiddenSize)
	if len(logits) < E {
		return nil
	}
	idx := make([]int, E)
	for e := range idx {
		idx[e] = e
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return logits[idx[a]] > logits[idx[b]]
	})
	if K > E {
		K = E
	}
	picks := make([]routePick, K)
	for i := 0; i < K; i++ {
		picks[i] = routePick{expert: idx[i], weight: logits[idx[i]]}
	}
	return picks
}

// prefetchNextLayerGateExperts stages the predicted top-k of layer+1 as HINTS during `layer`'s
// compute. It reuses the exact staging rules of prefetchActivatedExperts: the longest prefix that
// fits the ring budget, `prefetching=true` so the staging earns recency but no heat, and never a
// fallback to permanent halW residency. It also records the prediction into the ledger's pending
// store keyed by the TARGET layer (layer+1), to be reconciled by recordNextLayerActual when that
// layer's real router runs.
//
// A mispredict cannot change logits by construction: nothing here writes a delta, mutates a pick,
// or touches the demand path â€” an expert staged by a wrong guess is simply resident a little early,
// and an expert not staged takes the unchanged miss path. The only shared state touched is the ring
// (recency) and the session ledger (precision/recall). It is inert for a nil session, a session that
// did not opt in, an empty prediction, a session with no ring budget, or a nil ring.
func (s *Session) prefetchNextLayerGateExperts(layer int, predicted []routePick) {
	if s == nil || !s.CrossLayerGatePrefetch || len(predicted) == 0 || s.ExpertRingBytes <= 0 || s.ExpertPrefetch == ExpertPrefetchOnDemand {
		return
	}
	target := layer + 1
	r := s.routedExpertRing(expertName(target, predicted[0].expert, "gate_proj.weight"))
	if r == nil {
		return
	}

	plans := make([]activatedExpertPlan, 0, len(predicted))
	for _, pk := range predicted {
		ws, ok := s.activatedExpertWeights(target, pk.expert)
		if !ok {
			continue
		}
		p := activatedExpertPlan{expert: pk.expert}
		for i, w := range ws {
			key, mk, dt, b, ok := w.ringStaging()
			if !ok || b <= 0 {
				p.bytes = -1
				break
			}
			p.stage[i].key, p.stage[i].mk, p.stage[i].dt, p.stage[i].bytes = key, mk, dt, b
			p.bytes += b
		}
		if p.bytes <= 0 {
			continue
		}
		plans = append(plans, p)
	}
	if len(plans) == 0 {
		return
	}

	recordCrossLayerPrediction(s, target, predicted)

	// Under a shared ring every ring touch runs inside one span; reentrant, and a no-op privately.
	release := s.ringEnter(r)
	defer release()

	// NOTE: unlike the same-layer meter, the cross-layer prediction does NOT touch
	// r.activatedExperts / r.activatedCovered Ã¢â‚¬â€ those describe THIS layer's activated set, and a
	// prediction about the next layer is not part of it.
	r.prefetching = true
	defer func() { r.prefetching = false }()

	budget := r.budget()
	var reserved int64
	for _, p := range plans {
		if reserved+p.bytes > budget {
			break
		}
		pinned := r.isExpertPinned(target, p.expert)
		staged := 0
		for _, d := range p.stage {
			if r.isResident(d.key) {
				staged++
				continue
			}
			if _, ok := r.stage(d.key, d.mk, d.dt, d.bytes, pinned); !ok {
				break
			}
			r.prefetched++
			staged++
		}
		if staged < len(p.stage) {
			break
		}
		reserved += p.bytes
	}
}

// recordNextLayerActual folds the REALIZED activated set of layer+1 (the picks its own router
// returns) into the precision/recall ledger, matched against the prediction made when layer was
// entered. It is a no-op unless a prediction for that exact layer is pending. It does not touch the
// ring: this is a pure accounting call, so it cannot perturb the forward.
func (s *Session) recordNextLayerActual(layer int, actual []routePick) {
	if s == nil {
		return
	}
	actualSet := make(map[int]struct{}, len(actual))
	for _, pk := range actual {
		actualSet[pk.expert] = struct{}{}
	}

	s.crossLayerMu.Lock()
	defer s.crossLayerMu.Unlock()
	byLayer := s.crossLayerPending
	predicted, ok := byLayer[layer]
	if !ok {
		return
	}
	delete(byLayer, layer)
	if len(byLayer) == 0 {
		s.crossLayerPending = nil
	}

	st := s.crossLayerStats
	st.Predicted += len(predicted)
	st.Actual += len(actualSet)
	for e := range predicted {
		if _, hit := actualSet[e]; hit {
			st.Hits++
		}
	}
	s.crossLayerStats = st
}

// recordCrossLayerPrediction stores a prediction for `target` layer as a set of experts, replacing
// any earlier prediction for the same target (a layer entered twice â€” e.g. across tokens â€” keeps
// only the most recent guess). Stored on the session's own pending map, under its ledger lock.
func recordCrossLayerPrediction(s *Session, target int, predicted []routePick) {
	if s == nil {
		return
	}
	set := make(map[int]struct{}, len(predicted))
	for _, pk := range predicted {
		set[pk.expert] = struct{}{}
	}
	s.crossLayerMu.Lock()
	defer s.crossLayerMu.Unlock()
	if s.crossLayerPending == nil {
		s.crossLayerPending = map[int]map[int]struct{}{}
	}
	s.crossLayerPending[target] = set
}

// expertSliceBounds returns the [start, end) byte range of expert e inside a fused expert
// tensor whose data begins at `base` and gives each of `expertCount` experts `stride`
// contiguous bytes, validated against the file's data region [dataBase, size). It is the
// shared bounds check for readExpertSlice and willneedExpertSlice.
func (sf *safetensorsFile) expertSliceBounds(base int64, e, expertCount int, stride int64) (start, end int64, err error) {
	if e < 0 || expertCount <= 0 || e >= expertCount {
		return 0, 0, fmt.Errorf("model: expert %d out of range [0,%d)", e, expertCount)
	}
	if base < 0 || stride <= 0 {
		return 0, 0, fmt.Errorf("model: invalid expert slice base=%d stride=%d", base, stride)
	}
	start = base + int64(e)*stride
	end = start + stride
	if end < start || start < sf.dataBase || end > sf.size {
		return 0, 0, fmt.Errorf("model: expert slice [%d,%d) outside data region [%d,%d)", start, end, sf.dataBase, sf.size)
	}
	return start, end, nil
}

// readExpertSlice reads exactly expert e's byte sub-range out of a fused expert tensor, never
// the whole E*stride layer. On the mmap path it returns a zero-copy three-index-capped slice
// into the mapped region; otherwise it ReadAts a single expert-sized buffer.
func (sf *safetensorsFile) readExpertSlice(base int64, e, expertCount int, stride int64) ([]byte, error) {
	start, end, err := sf.expertSliceBounds(base, e, expertCount, stride)
	if err != nil {
		return nil, err
	}
	if sf.data != nil {
		return sf.data[start:end:end], nil
	}
	b := make([]byte, end-start)
	if _, err := sf.r.ReadAt(b, start); err != nil {
		return nil, fmt.Errorf("model: expert slice read: %w", err)
	}
	return b, nil
}

// willneedExpertSlice issues a best-effort MADV_WILLNEED readahead over expert e's byte
// sub-range so the kernel warms those pages while the caller computes another expert's GEMM.
// It fires only on the mmap path (there is a mapped region to advise), swallows every error
// (a failed hint only forfeits the overlap, never correctness), and returns whether a hint
// was actually issued Î“Ã‡Ã¶ for a bytes/token witness, never for control flow.
func (sf *safetensorsFile) willneedExpertSlice(base int64, e, expertCount int, stride int64) bool {
	if sf.data == nil {
		return false
	}
	start, end, err := sf.expertSliceBounds(base, e, expertCount, stride)
	if err != nil || end > int64(len(sf.data)) {
		return false
	}
	return madviseWillneed(sf.data, int(start), int(end-start))
}

// crossLayerGatePrefetch is the per-MoE-layer entry the FFN calls beside prefetchActivatedExperts
// (moe.go). It does the two halves of a cross-layer prefetch in one place:
//
//  1. RECONCILE — fold THIS layer's realized picks (already in hand) into the precision/recall
//     ledger against the prediction made when the previous layer was entered.
//  2. PREDICT + STAGE — apply this layer's successor gate to this layer's hidden state and stage
//     the predicted top-k as hints, so layer+1's page-ins are issued one whole layer early.
//
// It is the call-site wrapper the moe.go seams use, so the prediction never has to reach into a
// matKernel private field twice. It derives the session from the mat kernel and consults THAT
// session's own CrossLayerGatePrefetch, so it is inert for a nil model, a session that did not opt
// in, or a kernel with no session (residentKernel / splitKernel / f32Kernel â€” none of which reaches
// the ring). Keeping this wrapper's signature intact lets moe.go call it unchanged.
func crossLayerGatePrefetch(m *Model, layer int, picks []routePick, xn any, mat matKernel) {
	if m == nil {
		return
	}
	var sess *Session
	switch mk := mat.(type) {
	case sessionQ4KKernel:
		sess = mk.s
	case backendKernel:
		sess = mk.s
	}
	if sess == nil || !sess.CrossLayerGatePrefetch {
		return
	}
	sess.recordNextLayerActual(layer, picks)
	sess.prefetchNextLayerGateExperts(layer, predictNextLayerExperts(sess, m, layer, xn, mat))
}
