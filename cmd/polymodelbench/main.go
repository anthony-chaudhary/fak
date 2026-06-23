// Command polymodelbench is the runnable witness for the poly-model serving design
// (docs/serving/polymodel-prefill-share-plan.md): it hosts MANY real (synthetic)
// models in one process, drives the serial decode lane over real model.Session
// decode, and proves the cache-led multi-token-prediction loop is LOSSLESS — all
// CPU-only, deterministic, no GPU, no weight download.
//
// Three checks, each a hard assertion (`-selfcheck` exits non-zero on any failure):
//
//  1. HOST MANY — admit N synthetic models into a polymodel.Pool under a weight-byte
//     budget that forces eviction, and confirm the budget is never exceeded and the
//     pinned drafter is never evicted. This is the "host 10s of models" bookkeeping.
//  2. DECODE ONE — schedule prefill (compute-bound, emitted once each) + decode
//     (serialized) for the resident models, then EXECUTE the decode steps as real
//     model.Session.Step calls, confirming at most one model decodes per step.
//  3. CACHE-LED MTP — run greedy speculative decoding (a draft model proposes, the
//     target verifies, polymodel.AcceptGreedy decides, model.KVCache.Evict rolls back
//     the rejected draft tokens) and confirm it produces TOKEN-IDENTICAL output to
//     plain greedy decoding. Losslessness only holds if the bit-exact rollback is
//     correct, so this is the end-to-end witness for the speculative KV path.
//
// The throughput-optimal single-pass batched verify (the GPU lever) is NOT done
// here — the verify runs as sequential target Steps, which is correctness-faithful
// but not the speedup. That, and real multi-model residency on a backend, are the
// sequenced GAPs in the plan doc. This command proves the design's CORRECTNESS core,
// never a tokens/sec number.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run all checks and exit non-zero on any failure")
	flag.Parse()
	quiet := *selfcheck

	ok := true
	ok = hostMany(quiet) && ok
	ok = decodeOne(quiet) && ok
	ok = cacheLedMTP(quiet) && ok

	if !ok {
		fmt.Fprintln(os.Stderr, "polymodelbench: FAIL")
		os.Exit(1)
	}
	fmt.Println("polymodelbench: OK — host-many, decode-one, lossless cache-led MTP all verified")
}

// ---------------------------------------------------------------------------
// Check 1 — HOST MANY: prefill-warm multi-model residency under a budget.
// ---------------------------------------------------------------------------

func hostMany(quiet bool) bool {
	logf(quiet, "== 1. HOST MANY (residency under a weight-byte budget) ==")
	specs := modelZoo(10)
	var total int64
	for _, s := range specs {
		total += s.bytes
	}
	budget := total * 55 / 100 // ~half must page out via LRU
	pool := polymodel.NewPool(budget)
	pinned := smallest(specs)

	ok := true
	for _, s := range specs {
		m := polymodel.Model{ID: polymodel.ModelID(s.name), Family: "synthetic", WeightBytes: s.bytes, Pinned: s.name == pinned}
		evicted, err := pool.Admit(m)
		if err != nil {
			logf(quiet, "  admit %-4s (%4d KB): refused (%v)", s.name, s.bytes/1024, err)
			continue
		}
		if pool.Used() > pool.Budget() {
			fmt.Fprintf(os.Stderr, "  INVARIANT VIOLATED: used %d > budget %d after %s\n", pool.Used(), pool.Budget(), s.name)
			ok = false
		}
		for _, e := range evicted {
			if e == polymodel.ModelID(pinned) {
				fmt.Fprintf(os.Stderr, "  INVARIANT VIOLATED: pinned drafter %s was evicted\n", pinned)
				ok = false
			}
		}
		logf(quiet, "  admit %-4s (%4d KB): evicted %v; warm=%d used=%d/%d KB", s.name, s.bytes/1024, evicted, pool.Len(), pool.Used()/1024, pool.Budget()/1024)
	}
	if !pool.Has(polymodel.ModelID(pinned)) {
		fmt.Fprintf(os.Stderr, "  INVARIANT VIOLATED: pinned drafter %s is not resident at the end\n", pinned)
		ok = false
	}
	logf(quiet, "  -> %d models hosted across the run, %d warm now; budget never exceeded, pinned survived: %v", len(specs), pool.Len(), ok)
	return ok
}

// ---------------------------------------------------------------------------
// Check 2 — DECODE ONE: the serial decode lane over real model.Session decode.
// ---------------------------------------------------------------------------

func decodeOne(quiet bool) bool {
	logf(quiet, "== 2. DECODE ONE (serial lane over real Session.Step) ==")
	names := []string{"a", "b", "c"}
	cfgs := map[string]model.Config{
		"a": cfg(48, 3, 3, 1, 16, 96),
		"b": cfg(32, 2, 2, 1, 16, 64),
		"c": cfg(64, 4, 4, 2, 16, 128),
	}
	prompt := bytesToIDs([]byte("the cache is the lever"))
	sessions := map[polymodel.ModelID]*model.Session{}
	logits := map[polymodel.ModelID][]float32{}
	weights := map[polymodel.ModelID]int64{}
	var reqs []polymodel.Request
	for i, n := range names {
		m := model.NewSynthetic(cfgs[n])
		s := m.NewSession()
		id := polymodel.ModelID(n)
		logits[id] = s.Prefill(prompt) // the compute-bound, shareable half — once per model
		sessions[id] = s
		weights[id] = estimateBytes(cfgs[n])
		reqs = append(reqs, polymodel.Request{Model: id, Prefill: len(prompt), Decode: 4, Priority: 3 - i, Seq: uint64(i)})
	}

	steps, st := polymodel.Schedule(reqs, 2) // quantum 2 → models interleave on the lane
	if st.MaxConcurrentDecode != 1 {
		fmt.Fprintf(os.Stderr, "  INVARIANT VIOLATED: MaxConcurrentDecode=%d, want 1\n", st.MaxConcurrentDecode)
		return false
	}

	var laneOrder []polymodel.ModelID
	for _, step := range steps {
		if step.Phase != polymodel.Decode {
			continue
		}
		s := sessions[step.Model]
		for t := 0; t < step.Tokens; t++ { // real decode work, one model at a time
			logits[step.Model] = s.Step(argmax(logits[step.Model]))
		}
		laneOrder = append(laneOrder, step.Model)
	}
	bw := polymodel.DecodeBandwidthBytes(steps, weights)
	logf(quiet, "  prefill tokens=%d decode tokens=%d decode steps=%d (max concurrent decoders=%d)", st.PrefillTokens, st.DecodeTokens, st.DecodeSteps, st.MaxConcurrentDecode)
	logf(quiet, "  lane order (one model per step): %v", laneOrder)
	logf(quiet, "  decode HBM traffic = %d KB (only the decoding model pays; warm residency is free)", bw/1024)
	logf(quiet, "  -> serial decode lane drove %d real decode steps, never two models at once", st.DecodeSteps)
	return true
}

// ---------------------------------------------------------------------------
// Check 3 — CACHE-LED MTP: greedy speculative decode == plain greedy decode.
// ---------------------------------------------------------------------------

func cacheLedMTP(quiet bool) bool {
	logf(quiet, "== 3. CACHE-LED MTP (greedy speculative == greedy, via bit-exact KV rollback) ==")
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("speculative decoding is lossless when verified greedily"))
	const N, K = 24, 4

	want := greedyDecode(target, prompt, N)
	ok := true

	// 3a. Ensemble path: a real co-resident draft model proposes (the "idle models
	// become the speculation ensemble" idea). Whatever the acceptance, output must be
	// token-identical to greedy.
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64)) // cheaper, different weights
	gotA, draftedA, acceptedA, evictedA := specDecodeModel(target, draft, prompt, N, K)
	ok = assertLossless(quiet, "3a real-draft-model", gotA, want, N) && ok
	accA := rate(acceptedA, draftedA)
	logf(quiet, "  3a real draft model: proposed %d, accepted %d (%.0f%%), rolled-back %d; E(K=%d)=%.2f",
		draftedA, acceptedA, accA*100, evictedA, K, polymodel.EffectiveTokensPerVerify(K, accA))

	// 3b. Rollback stress: an ADVERSARIAL proposer (a deterministic counter, independent
	// of the target) forces rejections nearly every round, so the bit-exact Evict
	// rollback path runs hard. Output must STILL be token-identical to greedy, and we
	// assert rejections actually happened — otherwise the witness would be vacuous.
	adversary := func(round, j, last int) int { return (round*13 + j*7 + 1) % 256 }
	gotB, draftedB, acceptedB, evictedB := specDecodeProposer(target, prompt, N, K, adversary)
	ok = assertLossless(quiet, "3b adversarial-draft", gotB, want, N) && ok
	if evictedB == 0 {
		fmt.Fprintln(os.Stderr, "  VACUOUS WITNESS: adversarial draft caused 0 rollbacks — the Evict path was never exercised")
		ok = false
	}
	logf(quiet, "  3b adversarial draft: proposed %d, accepted %d (%.0f%%), rolled-back %d KV spans via bit-exact Evict",
		draftedB, acceptedB, rate(acceptedB, draftedB)*100, evictedB)

	if ok {
		logf(quiet, "  -> cache-led MTP is lossless on a real model — even when every round rolls back rejected drafts")
	}
	return ok
}

func assertLossless(quiet bool, label string, got, want []int, n int) bool {
	if len(got) < n || len(want) < n {
		fmt.Fprintf(os.Stderr, "  %s: short decode got %d want %d\n", label, len(got), len(want))
		return false
	}
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			fmt.Fprintf(os.Stderr, "  %s LOSSLESS VIOLATED at token %d: speculative=%d greedy=%d\n  (the KV rollback of a rejected draft was not bit-exact)\n", label, i, got[i], want[i])
			return false
		}
	}
	logf(quiet, "  %s: speculative output TOKEN-IDENTICAL to greedy ✓", label)
	return true
}

func rate(num, den int) float64 {
	if den == 0 {
		return 0
	}
	return float64(num) / float64(den)
}

// greedyDecode is plain autoregressive greedy decoding — the reference output.
func greedyDecode(m *model.Model, prompt []int, n int) []int {
	s := m.NewSession()
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		t := argmax(logits)
		out = append(out, t)
		logits = s.Step(t)
	}
	return out
}

// specDecode is greedy speculative decoding with bit-exact KV rollback. The draft
// proposes k tokens; the target verifies them (here as sequential Steps — the
// single-pass batched verify is the GPU lever, not this witness); polymodel.AcceptGreedy
// decides the accepted prefix; the rejected draft positions are removed from BOTH
// sessions with model.KVCache.Evict (the bit-exact rollback). Greedy speculation is
// provably lossless, so the output must equal greedyDecode — which only holds if Evict
// is bit-exact. Returns the output tokens plus draft/accept counts.
func specDecodeModel(target, draft *model.Model, prompt []int, n, k int) (out []int, drafted, accepted, evicted int) {
	ts := target.NewSession()
	ds := draft.NewSession()
	tl := ts.Prefill(prompt) // target's next-token distribution at the shared context
	dl := ds.Prefill(prompt) // draft's, threaded so it always reflects committed context
	out = make([]int, 0, n)

	for len(out) < n {
		pT := ts.Cache.Len()
		pD := ds.Cache.Len()

		// 1. Draft proposes k tokens greedily from its own model, threading dl.
		drafts := make([]int, 0, k)
		for j := 0; j < k; j++ {
			dj := argmax(dl)
			drafts = append(drafts, dj)
			dl = ds.Step(dj)
		}
		drafted += k

		// 2. Target verifies: argmax at the current position (from tl) and after each
		//    drafted token fed sequentially → k+1 argmaxes.
		targetArgmax := make([]int, 0, k+1)
		targetArgmax = append(targetArgmax, argmax(tl))
		for j := 0; j < k; j++ {
			targetArgmax = append(targetArgmax, argmax(ts.Step(drafts[j])))
		}

		// 3. Accept the longest matching prefix.
		res := polymodel.AcceptGreedy(drafts, targetArgmax)
		accepted += res.Accepted

		// 4. Roll back the rejected draft positions on BOTH sessions (bit-exact Evict).
		if res.EvictKV > 0 {
			ts.Cache.Evict(pT+res.Accepted, res.EvictKV)
			ds.Cache.Evict(pD+res.Accepted, res.EvictKV)
			evicted += res.EvictKV
		}

		// 5. Emit the committed tokens (accepted drafts + the correction/bonus) and
		//    advance both sessions by the correction so the next round shares context.
		correction := targetArgmax[res.Accepted]
		for j := 0; j < res.Accepted && len(out) < n; j++ {
			out = append(out, drafts[j])
		}
		if len(out) < n {
			out = append(out, correction)
		}
		tl = ts.Step(correction)
		dl = ds.Step(correction)
	}
	return out, drafted, accepted, evicted
}

// specDecodeProposer is speculative decode where the draft is a PROPOSER FUNCTION
// rather than a model — used to stress the rollback path with an adversarial draft
// that forces rejections. Only the target has a KV cache; the rejected draft tokens
// are rolled back from it with the bit-exact model.KVCache.Evict. In production the
// proposer is a co-resident small model (polymodel.PickDrafter); here a deterministic
// function lets the witness GUARANTEE rejections happen. Returns output + counts.
func specDecodeProposer(target *model.Model, prompt []int, n, k int, propose func(round, j, last int) int) (out []int, drafted, accepted, evicted int) {
	ts := target.NewSession()
	tl := ts.Prefill(prompt)
	out = make([]int, 0, n)
	last := prompt[len(prompt)-1]

	for round := 0; len(out) < n; round++ {
		pT := ts.Cache.Len()

		drafts := make([]int, 0, k)
		prev := last
		for j := 0; j < k; j++ {
			dj := ((propose(round, j, prev) % 256) + 256) % 256 // valid token id
			drafts = append(drafts, dj)
			prev = dj
		}
		drafted += k

		targetArgmax := make([]int, 0, k+1)
		targetArgmax = append(targetArgmax, argmax(tl))
		for j := 0; j < k; j++ {
			targetArgmax = append(targetArgmax, argmax(ts.Step(drafts[j])))
		}

		res := polymodel.AcceptGreedy(drafts, targetArgmax)
		accepted += res.Accepted
		if res.EvictKV > 0 {
			ts.Cache.Evict(pT+res.Accepted, res.EvictKV)
			evicted += res.EvictKV
		}

		correction := targetArgmax[res.Accepted]
		for j := 0; j < res.Accepted && len(out) < n; j++ {
			out = append(out, drafts[j])
		}
		if len(out) < n {
			out = append(out, correction)
		}
		last = correction
		tl = ts.Step(correction)
	}
	return out, drafted, accepted, evicted
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type modelSpec struct {
	name  string
	bytes int64
}

func modelZoo(n int) []modelSpec {
	specs := make([]modelSpec, 0, n)
	for i := 0; i < n; i++ {
		h := 32 + (i%4)*16 // 32,48,64,80 cycling
		layers := 2 + i%4  // 2..5
		c := cfg(h, layers, h/16, 1, 16, h*2)
		specs = append(specs, modelSpec{name: fmt.Sprintf("m%d", i), bytes: estimateBytes(c)})
	}
	return specs
}

func smallest(specs []modelSpec) string {
	best := specs[0]
	for _, s := range specs[1:] {
		if s.bytes < best.bytes {
			best = s
		}
	}
	return best.name
}

// cfg builds a small, valid PreNorm synthetic config with a byte-total vocab, so any
// input byte is a valid token id and a draft's token is valid for the target.
func cfg(hidden, layers, nHeads, nKV, headDim, inter int) model.Config {
	return model.Config{
		HiddenSize:        hidden,
		NumLayers:         layers,
		NumHeads:          nHeads,
		NumKVHeads:        nKV,
		HeadDim:           headDim,
		IntermediateSize:  inter,
		VocabSize:         256,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1, // never early-stop; decode a fixed length
	}
}

// estimateBytes is the approximate resident f32 footprint of a synthetic config:
// embedding + per-layer attention + MLP + norms, ×4 bytes. A residency proxy (the
// Pool reasons about bytes), not a measured allocation.
func estimateBytes(c model.Config) int64 {
	h, l := int64(c.HiddenSize), int64(c.NumLayers)
	qkv := int64(c.NumHeads*c.HeadDim) + 2*int64(c.NumKVHeads*c.HeadDim)
	attn := qkv*h + h*int64(c.NumHeads*c.HeadDim) // q,k,v + o
	mlp := 3 * int64(c.IntermediateSize) * h      // gate, up, down
	perLayer := attn + mlp + 2*h                  // + two norms
	params := int64(c.VocabSize)*h + l*perLayer + h
	return params * 4
}

func bytesToIDs(b []byte) []int {
	ids := make([]int, len(b))
	for i, c := range b {
		ids[i] = int(c)
	}
	return ids
}

func argmax(v []float32) int {
	if len(v) == 0 {
		return 0
	}
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func logf(quiet bool, format string, a ...any) {
	if quiet {
		return
	}
	fmt.Printf(format+"\n", a...)
}
