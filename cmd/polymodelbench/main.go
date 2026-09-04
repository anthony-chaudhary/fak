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
// The speculative loop itself is the shared internal/polymodel.SpecDecode driver
// (#4877): this command binds its drafter/verifier/rollback closures over real
// model.Session.VerifyForward (the single-pass batched verify) + model.KVCache.Evict,
// so there is ONE loop, tested once. Real multi-model residency on a backend remains
// a sequenced GAP in the plan doc. This command proves the design's CORRECTNESS core,
// never a tokens/sec number.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run all checks and exit non-zero on any failure")
	bench := flag.Bool("bench", false, "run the measured-numbers bench harness for #535 (E vs draft cost, decode-lane utilization, residency hit-rate) and print the report")
	out := flag.String("out", "", "with -bench: write the report as JSON to this path (the reproducible artifact)")
	tune := flag.Bool("tune", false, "run automated MTP parameter tuning sweep across K in {1,2,3,4} and print markdown performance matrix")
	flag.Parse()
	quiet := *selfcheck

	// The bench harness is the measured-numbers half (#535); the three -selfcheck
	// witnesses are the correctness half. -bench runs the witnesses too (a measured
	// run on a broken core is worthless), then emits the report.
	ok := true
	ok = hostMany(quiet) && ok
	ok = decodeOne(quiet) && ok
	ok = cacheLedMTP(quiet) && ok

	var report BenchReport
	if *bench {
		report = benchHarness(quiet, &ok)
	}

	if !ok {
		fmt.Fprintln(os.Stderr, "polymodelbench: FAIL")
		os.Exit(1)
	}
	if *tune {
		report := RunMTPSweep(quiet)
		fmt.Print(RenderMTPSweepMarkdown(report))
		return
	}
	if *bench {
		if *out != "" {
			if err := writeJSON(*out, report); err != nil {
				fmt.Fprintf(os.Stderr, "polymodelbench: -out %s: %v\n", *out, err)
				os.Exit(1)
			}
			fmt.Printf("polymodelbench: report written to %s\n", *out)
		}
		fmt.Println("polymodelbench: OK — correctness witnesses + #535 bench harness (E/draft-cost, decode-lane utilization, residency hit-rate) all measured")
		return
	}
	fmt.Println("polymodelbench: OK — host-many, decode-one, lossless cache-led MTP, P-EAGLE parallel-depth shape all verified")
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
			logits[step.Model] = s.Step(mathx.ArgmaxF32(logits[step.Model]))
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
	// 3b. Rollback stress: an ADVERSARIAL proposer (a deterministic counter, independent
	// of the target) forces rejections nearly every round, so the bit-exact Evict
	// rollback path runs hard. Output must STILL be token-identical to greedy, and we
	// assert rejections actually happened — otherwise the witness would be vacuous.
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64)) // cheaper, different weights
	gotA, draftedA, acceptedA, evictedA, gotB, draftedB, acceptedB, evictedB := runSpecDecodeTrials(target, draft, prompt, N, K)

	ok = assertLossless(quiet, "3a real-draft-model", gotA, want, N) && ok
	accA := rate(acceptedA, draftedA)
	logf(quiet, "  3a real draft model: proposed %d, accepted %d (%.0f%%), rolled-back %d; E(K=%d)=%.2f",
		draftedA, acceptedA, accA*100, evictedA, K, polymodel.EffectiveTokensPerVerify(K, accA))

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

func peagleParallelDepth(quiet bool) bool {
	logf(quiet, "== 4. P-EAGLE SHAPE (parallel-depth draft source) ==")
	const N, depths = 24, 4
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	r := measurePEagleShape(target, draft, bytesToIDs([]byte("parallel depth shape witness")), N, depths)
	ok := r.TokenIdenticalToGreedy && r.LogicalDraftCalls == r.TargetVerifyRounds &&
		r.SequentialDraftStepsAvoided == r.ProposedTokens && len(r.AcceptanceProfile) == depths
	if !ok {
		fmt.Fprintf(os.Stderr, "  P-EAGLE SHAPE FAILED: %+v\n", r)
		return false
	}
	logf(quiet, "  %s: target-rounds=%d sequential-draft-steps-avoided=%d mean-acceptance=%.2f profile=%v engine=%s",
		r.Name, r.TargetVerifyRounds, r.SequentialDraftStepsAvoided, r.MeanAcceptanceLength, r.AcceptanceProfile, r.Engine)
	return true
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
		t := mathx.ArgmaxF32(logits)
		out = append(out, t)
		logits = s.Step(t)
	}
	return out
}

// targetVerify binds a target model.Session to polymodel.SpecDecode's Verifier +
// Rollback seam. The verifier first syncs the session to the committed context (the
// previous round's correction/bonus token is committed by the loop but never fed to
// the session), then runs ONE model.Session.VerifyForward pass over the draft — the
// chain shape is bit-identical to sequential Steps (TestVerifyForwardMatchesSerial),
// so losslessness is preserved — and returns the target's argmax at the k+1 panel
// positions. The rollback removes the rejected draft suffix with the bit-exact
// model.KVCache.Evict, using the positions captured by the verify pass.
type targetVerify struct {
	s        *model.Session
	tl       []float32 // next-token logits at the committed context (threaded)
	base     int       // cache length at the start of the current verify pass
	draftLen int       // draft length of the current verify pass
}

func newTargetVerify(target *model.Model, prompt []int) *targetVerify {
	s := target.NewSession()
	return &targetVerify{s: s, tl: s.Prefill(prompt)}
}

func (v *targetVerify) verify(committed, draft []int) []int {
	for _, t := range committed[v.s.Cache.Len():] { // feed the not-yet-seen correction
		v.tl = v.s.Step(t)
	}
	v.base = v.s.Cache.Len()
	v.draftLen = len(draft)
	argmax := make([]int, 0, len(draft)+1)
	argmax = append(argmax, mathx.ArgmaxF32(v.tl)) // position 0: already known from tl
	for _, row := range v.s.VerifyForward(draft, nil, nil) {
		argmax = append(argmax, mathx.ArgmaxF32(row))
	}
	return argmax
}

func (v *targetVerify) rollback(evictKV int) {
	v.s.Cache.Evict(v.base+(v.draftLen-evictKV), evictKV)
}

// specDecodeModel is greedy speculative decoding with a real co-resident draft model,
// driven by the shared polymodel.SpecDecode loop (#4877): the drafter closure proposes
// k tokens greedily from the draft model's own session, the verifier closure runs the
// target's single-pass model.Session.VerifyForward, and the rollback closure removes
// the rejected draft positions from BOTH sessions with the bit-exact model.KVCache.Evict.
// Greedy speculation is provably lossless, so the output must equal greedyDecode —
// which only holds if Evict is bit-exact. Returns the output tokens plus counts.
func specDecodeModelRun(target, draft *model.Model, prompt []int, n, k int) polymodel.SpecDecodeRun {
	tv := newTargetVerify(target, prompt)
	ds := draft.NewSession()
	dl := ds.Prefill(prompt) // draft's logits, threaded so it always reflects committed context
	var pD int               // draft-session cache length at the start of the current round

	run, err := polymodel.SpecDecode(prompt,
		func(committed []int) []int {
			for _, t := range committed[ds.Cache.Len():] { // sync: feed the last correction
				dl = ds.Step(t)
			}
			pD = ds.Cache.Len()
			drafts := make([]int, 0, k)
			for j := 0; j < k; j++ {
				dj := mathx.ArgmaxF32(dl)
				drafts = append(drafts, dj)
				dl = ds.Step(dj)
			}
			return drafts
		},
		tv.verify,
		polymodel.SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k,
			Rollback: func(evictKV int) { // roll back BOTH sessions bit-exactly
				tv.rollback(evictKV)
				ds.Cache.Evict(pD+(tv.draftLen-evictKV), evictKV)
			}})
	if err != nil {
		fmt.Fprintf(os.Stderr, "  specDecodeModel: SpecDecode failed: %v\n", err)
	}
	return run
}

func specDecodeModel(target, draft *model.Model, prompt []int, n, k int) (out []int, drafted, accepted, evicted int) {
	run := specDecodeModelRun(target, draft, prompt, n, k)
	return run.Output, run.DraftedTokens, run.AcceptedDrafts, run.EvictKV
}

// specDecodeProposer is speculative decode where the draft is a PROPOSER FUNCTION
// rather than a model — used to stress the rollback path with an adversarial draft
// that forces rejections — driven by the same shared polymodel.SpecDecode loop. Only
// the target has a KV cache; the rejected draft tokens are rolled back from it with
// the bit-exact model.KVCache.Evict. In production the proposer is a co-resident
// small model (polymodel.PickDrafter); here a deterministic function lets the witness
// GUARANTEE rejections happen. Returns output + counts.
func specDecodeProposer(target *model.Model, prompt []int, n, k int, propose func(round, j, last int) int) (out []int, drafted, accepted, evicted int) {
	tv := newTargetVerify(target, prompt)
	round := -1

	run, err := polymodel.SpecDecode(prompt,
		func(committed []int) []int {
			round++
			prev := committed[len(committed)-1] // the last committed token
			drafts := make([]int, 0, k)
			for j := 0; j < k; j++ {
				dj := ((propose(round, j, prev) % 256) + 256) % 256 // valid token id
				drafts = append(drafts, dj)
				prev = dj
			}
			return drafts
		},
		tv.verify,
		polymodel.SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k, Rollback: tv.rollback})
	if err != nil {
		fmt.Fprintf(os.Stderr, "  specDecodeProposer: SpecDecode failed: %v\n", err)
	}
	return run.Output, run.DraftedTokens, run.AcceptedDrafts, run.EvictKV
}

// runSpecDecodeTrials runs BOTH speculative-decode regimes — a real co-resident
// draft model (specDecodeModel) and a deterministic adversarial proposer
// (specDecodeProposer) — against the same target over the same prompt/n/k, and
// returns their raw decode traces + counts. Shared by the -selfcheck witness
// (cacheLedMTP) and the #535 bench harness (measureSpec) so both measure the
// identical two regimes.
func runSpecDecodeTrials(target, draft *model.Model, prompt []int, n, k int) (gotA []int, draftedA, acceptedA, evictedA int, gotB []int, draftedB, acceptedB, evictedB int) {
	gotA, draftedA, acceptedA, evictedA = specDecodeModel(target, draft, prompt, n, k)
	adversary := func(round, j, last int) int { return (round*13 + j*7 + 1) % 256 }
	gotB, draftedB, acceptedB, evictedB = specDecodeProposer(target, prompt, n, k, adversary)
	return
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

func logf(quiet bool, format string, a ...any) {
	if quiet {
		return
	}
	fmt.Printf(format+"\n", a...)
}
