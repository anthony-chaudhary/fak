package modelengine

// shadow_perf.go — Continual Engine phase 2 (shadow-perf harness): extends the
// scrubbed-shadow correctness child (shadow_traffic.go) from token-divergence
// checking to two-tier PERF COMPARE of candidate vs incumbent throughput on
// scrubbed real-traffic replay.
//
// This file is the perf half of the shadow contract; the correctness half
// stays in shadow_traffic.go and its seams are reused here, not redefined:
// scrubbedRequest/shadowScrub (privacy), shadowReferenceDecode (the reference
// decode), shadowJudge (the bit-exact differential oracle), shadowServeLive
// (the user path this harness must never touch).
//
// Design anchors (docs/notes/CONTINUAL-ENGINE-INTERNAL-STRUCTURE-2026-11-04.md
// §3, §6):
//   - NEVER random inputs. TKCC Fig 12/13: random data activates 13–38% more
//     MoE experts and turned a 20µs kernel into 62ms at 70K context — a 1.3×
//     end-to-end lie. The corpus is ALWAYS the scrubbed projection of live
//     requests (real traffic shape); the fixed probe set is pinned.
//   - Two tiers. Fast tier = layer subset / short spans, sandboxed — the crash
//     absorber (a candidate fault is caught cheaply before the slow tier pays
//     for it). Slow tier = full layers on the full-span real-traffic replay;
//     it is the authoritative measurement.
//   - Split prefill/decode. The closed phase labels prefill_tok_s /
//     decode_tok_s are measured separately and never blended into one TPS
//     (PERF-FEEDBACK-LOOP doctrine forbids mixing).
//   - Correctness gate BEFORE perf counts. The bit-exact oracle (the
//     VerifyForward twin, internal/model/verify.go:43) runs on the fixed probe
//     set first; a divergent or crashing candidate produces ZERO perf samples.
//   - Isolation invariant holds: the candidate never reaches the user path;
//     ShadowPerfEvaluate is pure with respect to serving.
//   - Privacy invariant holds: every artifact is built from the scrubbed
//     projection (digest + shape only), scrubbed BEFORE persistence.
//   - Result pinning (§6.1): a measurement is pinned to
//     (weights-epoch, forward-gen, placement-gen, traffic-window). Coordinates
//     owned by landed seams (phase 0 ForwardSlot, phase 1 WeightDelta) are
//     supplied through ShadowPerfPins when those seams exist in the build;
//     otherwise they carry an explicit typed-unpinned marker with the reason —
//     never a fabricated value.
//
// Runtime/resource cost: pure in-process CPU simulation, tens of milliseconds
// per evaluate, no model download, accelerator, network, or live users. All
// results this package witnesses are [SW-VERIFIED] class only; no hardware or
// performance claims are made. Tier: PR (runs in the package unit gate).

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// shadowPerfReceiptSchema is the JSON contract emitted by ShadowPerfEvaluate.
// The private gate plane (fak-private cmd/fak-strix delta shadow-perf) parses
// it; the boundary rule keeps the two sides coupled by schema string only.
const shadowPerfReceiptSchema = "fak.modelengine-shadow-perf-receipt.v1"

// Closed tier labels.
const (
	// shadowPerfTierFast is the sandboxed screen: layer subset, short spans.
	// It exists to absorb candidate crashes cheaply, not to rank candidates.
	shadowPerfTierFast = "fast"
	// shadowPerfTierSlow is the authoritative tier: full layers, full spans,
	// real-traffic replay.
	shadowPerfTierSlow = "slow"
)

// Closed phase labels (PERF-FEEDBACK-LOOP): prefill and decode throughput are
// separate measurements. A blended TPS would hide exactly the regressions this
// harness exists to catch.
const (
	shadowPerfPhasePrefill = "prefill"
	shadowPerfPhaseDecode  = "decode"
)

// Tunable defaults. Every value ships auto-tunable through ShadowPerfOptions;
// these are the committed starting points, not user surface.
const (
	shadowPerfDefaultModelLayers = 32 // full simulated depth (slow tier)
	shadowPerfDefaultABBAReps    = 4  // ABBA blocks per tier
	shadowPerfDefaultFastSpanCap = 32 // fast-tier token cap per side
	// shadowPerfWorkPerUnit is the deterministic stream-draw count per
	// (layer × token) work unit. It sets the base scale of each timed region;
	// shadowPerfInnerReps then auto-sizes the inner pass count so every region
	// lands near shadowPerfTargetRegionDraws regardless of tier or corpus.
	shadowPerfWorkPerUnit = 128
	// shadowPerfMinUnits floors the draw count of a single inner pass so even a
	// degenerate (tiny corpus / shallow depth) region does sub-tick work. The
	// floor is a constant shared by all arms, so it cancels in every ratio the
	// gate consumes.
	shadowPerfMinUnits = 16384
	// shadowPerfTargetRegionDraws is the auto-tune target: the reference arm's
	// timed region aims at this many stream draws so its duration (~5ms class
	// on CI hosts) resolves well above host clock granularity — measured on
	// this class of host, sub-millisecond regions carry ±30% noise that
	// swamps the efficiency signal. This is an auto-tuned internal, not user
	// surface (AUTOTUNE-DEFAULT): the pass count adapts to the modeled load
	// instead of shipping a static size.
	shadowPerfTargetRegionDraws = 2_000_000
	// shadowPerfMaxInnerReps caps the adaptation so a near-zero load cannot
	// balloon a region into seconds.
	shadowPerfMaxInnerReps = 64
)

// shadowPerfComputeWitness is the package-level sink for measured work. The
// harness' compute load is otherwise a pure function whose result would be
// dead-code-eliminated whenever a caller never reads the per-tier witness —
// which would silently turn the timed loops into no-ops and the samples into
// timer noise. An atomic store is opaque to elimination, so the measured work
// always actually runs. The sink is content-free (stream checksums only) and
// is not part of any verdict.
var shadowPerfComputeWitness atomic.Uint64

// shadowPerfEngine models one engine generation under measurement: a
// correctness transform (which must be bit-exact against the incumbent for any
// perf count to count) plus a deterministic compute load that encodes its
// per-(layer,token) efficiency. Efficiency is a percent of incumbent work:
// 100 == parity, lower == faster, higher == slower. The load is content-free:
// only the scrubbed canonical bytes enter the fold, so the measurement tracks
// modeled compute (depth × tokens × efficiency), never request content.
type shadowPerfEngine struct {
	name       string
	backend    string
	efficiency uint64
	decode     func(r scrubbedRequest) []string
}

// shadowPerfEngineRef is the receipt-safe identity of an engine under test.
type shadowPerfEngineRef struct {
	Name    string `json:"name"`
	Backend string `json:"backend"`
}

// shadowPerfCoord is one coordinate of the result pin (§6.1): a generation
// counter observed from a landed seam, or an explicit unpinned marker when the
// seam is absent from this build. Unpinned coordinates carry their reason in
// Source; a pin never carries a fabricated value.
type shadowPerfCoord struct {
	Value  uint64 `json:"value"`
	Pinned bool   `json:"pinned"`
	Source string `json:"source,omitempty"`
}

func shadowPerfUnpinned(reason string) shadowPerfCoord {
	return shadowPerfCoord{Pinned: false, Source: reason}
}

func shadowPerfObserved(value uint64, source string) shadowPerfCoord {
	return shadowPerfCoord{Value: value, Pinned: true, Source: source}
}

// shadowPerfPin pins a measurement to the engine-generation tuple
// (weights-epoch, forward-gen, placement-gen, traffic-window). TrafficWindow
// is always computed from the scrubbed corpus; the three generation
// coordinates are pinned only when a landed seam supplies them.
type shadowPerfPin struct {
	WeightsEpoch  shadowPerfCoord `json:"weights_epoch"`
	ForwardGen    shadowPerfCoord `json:"forward_gen"`
	PlacementGen  shadowPerfCoord `json:"placement_gen"`
	TrafficWindow string          `json:"traffic_window"`
}

// ShadowPerfPins carries observed pin coordinates from landed seams. Zero
// value leaves every coordinate typed-unpinned with the documented reason.
type ShadowPerfPins struct {
	WeightsEpoch    shadowPerfCoord
	HasWeightsEpoch bool
	ForwardGen      shadowPerfCoord
	HasForwardGen   bool
	PlacementGen    shadowPerfCoord
	HasPlacementGen bool
}

// ShadowPerfOptions configures one ShadowPerfEvaluate run. Zero fields fall
// back to the committed defaults below; the slow tier is always uncapped.
type ShadowPerfOptions struct {
	ModelLayers  int // full simulated depth (slow tier); default 32
	FastLayerCap int // fast-tier layer subset; default max(1, ModelLayers/8)
	FastSpanCap  int // fast-tier token cap per side; default 32
	ABBAReps     int // ABBA blocks per tier; default 4
	Pins         ShadowPerfPins
}

func (o *ShadowPerfOptions) withDefaults() {
	if o.ModelLayers <= 0 {
		o.ModelLayers = shadowPerfDefaultModelLayers
	}
	if o.FastLayerCap <= 0 {
		cap := o.ModelLayers / 8
		if cap < 1 {
			cap = 1
		}
		o.FastLayerCap = cap
	}
	if o.FastSpanCap <= 0 {
		o.FastSpanCap = shadowPerfDefaultFastSpanCap
	}
	if o.ABBAReps <= 0 {
		o.ABBAReps = shadowPerfDefaultABBAReps
	}
}

// shadowPerfVerifyGate records the outcome of the bit-exact oracle on the
// fixed probe set. It runs BEFORE any perf count counts.
type shadowPerfVerifyGate struct {
	Passed          bool   `json:"passed"`
	Probes          int    `json:"probes"`
	Detail          string `json:"detail,omitempty"`
	DivergenceIndex int    `json:"divergence_index,omitempty"`
}

// shadowPerfBlock is one ABBA block's evidence: the outer incumbent pair and
// the inner candidate pair, per closed phase label. The private gate plane
// computes the paired median shift over these blocks.
type shadowPerfBlock struct {
	BaselinePrefill  [2]float64 `json:"baseline_prefill"`
	CandidatePrefill [2]float64 `json:"candidate_prefill"`
	BaselineDecode   [2]float64 `json:"baseline_decode"`
	CandidateDecode  [2]float64 `json:"candidate_decode"`
}

// ShadowPerfTierResult is one tier's split-phase sample sets, ABBA-scheduled.
// The closed metric labels appear verbatim as field names: prefill_tok_s and
// decode_tok_s, baseline (incumbent) and candidate, never blended.
type ShadowPerfTierResult struct {
	Tier                 string            `json:"tier"`
	Layers               int               `json:"layers"`
	SpanCap              int               `json:"span_cap"` // 0 = uncapped
	ABBAReps             int               `json:"abba_reps"`
	Crashed              bool              `json:"crashed"`
	CrashReason          string            `json:"crash_reason,omitempty"`
	BaselinePrefillTokS  []float64         `json:"baseline_prefill_tok_s"`
	BaselineDecodeTokS   []float64         `json:"baseline_decode_tok_s"`
	CandidatePrefillTokS []float64         `json:"candidate_prefill_tok_s"`
	CandidateDecodeTokS  []float64         `json:"candidate_decode_tok_s"`
	Blocks               []shadowPerfBlock `json:"blocks,omitempty"`
	ComputeWitness       uint64            `json:"compute_witness,omitempty"`
}

// ShadowPerfReceipt is the persisted shadow-perf result. It is built from the
// scrubbed projection only (scrub before persistence): no raw prompt byte and
// no end-user id can appear in it.
type ShadowPerfReceipt struct {
	Schema             string                 `json:"schema"`
	GeneratedAt        time.Time              `json:"generated_at"`
	Candidate          shadowPerfEngineRef    `json:"candidate"`
	Incumbent          shadowPerfEngineRef    `json:"incumbent"`
	VerifyGate         shadowPerfVerifyGate   `json:"verify_gate"`
	Pin                shadowPerfPin          `json:"pin"`
	Tiers              []ShadowPerfTierResult `json:"tiers"`
	IsolationPreserved bool                   `json:"isolation_preserved"`
	PerfCountsValid    bool                   `json:"perf_counts_valid"`
}

// --- fixture engines --------------------------------------------------------

// shadowPerfIncumbent is the live (reference) generation: the production
// decode at full efficiency.
func shadowPerfIncumbent() shadowPerfEngine {
	return shadowPerfEngine{name: "incumbent", backend: "production-serve", efficiency: 100, decode: shadowReferenceDecode}
}

// shadowPerfFaithfulFastCandidate models a lossless engine change — a
// kernel/scheduler refactor that is output-identical to the incumbent
// (bit-exact, so it passes the verify gate) but does 40% of the
// per-(layer,token) work. The gap is deliberately TKCC-class (large) so the
// ordering witness clears host scheduling noise with wide margin; small-delta
// detection is the private gate's statistical job, not this fixture's.
func shadowPerfFaithfulFastCandidate() shadowPerfEngine {
	return shadowPerfEngine{name: "candidate-faithful-fast", backend: "native-refactor", efficiency: 40, decode: shadowReferenceDecode}
}

// shadowPerfRegressingCandidate is output-identical but does 2.5× the
// per-(layer,token) work — the reject-path fixture: correct, measurably worse.
func shadowPerfRegressingCandidate() shadowPerfEngine {
	return shadowPerfEngine{name: "candidate-regressing", backend: "native-slow-path", efficiency: 250, decode: shadowReferenceDecode}
}

// shadowPerfCrashCandidate models a candidate whose new kernel path faults on
// long contexts (IMA-class): bit-exact on the fixed probe set (short prompts),
// fatal beyond promptLen 96. The fast tier must absorb it before the slow
// tier pays for it.
func shadowPerfCrashCandidate() shadowPerfEngine {
	return shadowPerfEngine{
		name: "candidate-crash", backend: "native-kernel-v2", efficiency: 100,
		decode: func(r scrubbedRequest) []string {
			if r.promptLen > 96 {
				panic("simulated IMA: kernel fault on long context")
			}
			return shadowReferenceDecode(r)
		},
	}
}

// shadowPerfCorpusFixture is the pinned production-like corpus: three
// scrubbed-realistic requests of increasing length, including one past the
// crash candidate's fault threshold. Real-traffic shape, never random.
func shadowPerfCorpusFixture() []liveRequest {
	return []liveRequest{
		{id: "corp-1", userID: "enduser-c1-PII-SECRET", prompt: "real traffic shape one " + strings.Repeat("chart summary segment ", 4), model: "fak-fixture-7b", tokenizer: "fak-bpe-v1", seed: 0xcafe0001, maxTokens: 8},
		{id: "corp-2", userID: "enduser-c2-PII-SECRET", prompt: "real traffic shape two " + strings.Repeat("replay segment ", 6), model: "fak-fixture-7b", tokenizer: "fak-bpe-v1", seed: 0xcafe0002, maxTokens: 16},
		{id: "corp-3", userID: "enduser-c3-PII-SECRET", prompt: "real traffic shape three " + strings.Repeat("long-context replay segment ", 8), model: "fak-fixture-7b", tokenizer: "fak-bpe-v1", seed: 0xcafe0003, maxTokens: 32},
	}
}

// shadowPerfProbeSet is the FIXED probe set the bit-exact oracle runs on
// before any perf count counts (the VerifyForward twin, design §3). Pinned:
// two short requests, stable across runs. Deliberately shorter than the crash
// candidate's fault threshold so semantic and crash defects stay distinct
// failure classes.
func shadowPerfProbeSet() []liveRequest {
	return []liveRequest{
		{id: "probe-a", userID: "probe-user-a", prompt: "probe alpha: deterministic short chart summary", model: "fak-fixture-7b", tokenizer: "fak-bpe-v1", seed: 0x5eed0001, maxTokens: 8},
		{id: "probe-b", userID: "probe-user-b", prompt: "probe beta: deterministic medium replay window", model: "fak-fixture-7b", tokenizer: "fak-bpe-v1", seed: 0x5eed0002, maxTokens: 16},
	}
}

// --- deterministic compute load ----------------------------------------------

// shadowPerfUnits converts (layer × token) work into stream draws at the
// engine's efficiency percent, floored at shadowPerfMinUnits so every timed
// region resolves above timer granularity. The floor is a constant shared by
// all arms, so it cancels in every ratio the gate consumes. Integer arithmetic
// keeps the load exact and reproducible.
func shadowPerfUnits(layerTokens uint64, efficiencyPercent uint64) uint64 {
	u := layerTokens * efficiencyPercent / 100 * shadowPerfWorkPerUnit
	if u < shadowPerfMinUnits {
		u = shadowPerfMinUnits
	}
	return u
}

// shadowPerfCorpusSeed derives one deterministic stream seed from the whole
// scrubbed corpus, so a timed region folds every request's digest exactly once.
func shadowPerfCorpusSeed(corpus []scrubbedRequest) uint64 {
	var s uint64
	for _, r := range corpus {
		s ^= r.seed
	}
	return s
}

// shadowPerfWork runs `units` deterministic stream draws folded with the
// request's scrubbed canonical bytes and returns a checksum witness. It exists
// so measured wall time tracks modeled compute while staying content-free:
// only the scrubbed digest enters the fold, and the witness proves the work
// actually ran (no dead-code elimination of the measured loop).
func shadowPerfWork(st *shadowStream, units uint64) uint64 {
	var sum uint64
	for i := uint64(0); i < units; i++ {
		sum ^= st.draw()
	}
	return sum
}

// shadowPerfInnerReps auto-sizes the inner pass count for one timed region so
// the REFERENCE (incumbent-parity, efficiency 100) arm lands near
// shadowPerfTargetRegionDraws: small structural loads get more passes, large
// loads get one. Crucially the pass count depends ONLY on the structural load
// (depth × tokens), never on the arm's efficiency — so every arm runs the SAME
// number of passes and its total work stays proportional to its efficiency.
// Normalizing total work across arms would erase the very throughput difference
// this harness measures; pinning the pass count to the reference load keeps the
// signal intact while giving every region timer-resolvable duration.
func shadowPerfInnerReps(layerTokens uint64) int {
	refDraws := shadowPerfUnits(layerTokens, 100)
	if refDraws == 0 {
		return 1
	}
	reps := shadowPerfTargetRegionDraws / refDraws
	if reps < 1 {
		reps = 1
	}
	if reps > shadowPerfMaxInnerReps {
		reps = shadowPerfMaxInnerReps
	}
	return int(reps)
}

// shadowPerfArmResult is one timed block: all corpus requests through one arm
// (incumbent or candidate) at one tier, split into the closed phase labels.
// Token counts cover every inner pass actually executed in the region.
type shadowPerfArmResult struct {
	PrefillTokS    float64
	DecodeTokS     float64
	PrefillTokens  int
	DecodeTokens   int
	ComputeWitness uint64
}

// shadowPerfMeasureArm times one arm over the whole corpus. Prefill and decode
// are timed in separate regions and reported under their closed labels — never
// one blended number. Each region's load is depth × tokens × efficiency
// (per-pass, floored at shadowPerfMinUnits), repeated for the auto-tuned inner
// pass count so the region resolves above host clock granularity; the decode
// region additionally runs the engine's ACTUAL forward pass (eng.decode) per
// request per pass, so candidate faults surface inside the timed region and
// are absorbed by the caller's sandbox. spanCap 0 means uncapped (the slow
// tier).
func shadowPerfMeasureArm(depth, spanCap int, eng shadowPerfEngine, corpus []scrubbedRequest) shadowPerfArmResult {
	var res shadowPerfArmResult

	// Prefill region: process every request's prompt through the layer stack.
	var prePassTokens int
	var preLayerTokens uint64
	for _, r := range corpus {
		p := r.promptLen
		if spanCap > 0 && p > spanCap {
			p = spanCap
		}
		prePassTokens += p
		preLayerTokens += uint64(depth) * uint64(p)
	}
	preReps := shadowPerfInnerReps(preLayerTokens)
	prePassDraws := shadowPerfUnits(preLayerTokens, eng.efficiency)
	res.PrefillTokens = prePassTokens * preReps
	t0 := time.Now()
	st := shadowNewStream(shadowPerfCorpusSeed(corpus))
	for _, r := range corpus {
		st.mix(scrubbedCanonicalBytes(r))
	}
	for rep := 0; rep < preReps; rep++ {
		res.ComputeWitness ^= shadowPerfWork(&st, prePassDraws)
	}
	preElapsed := time.Since(t0)

	// Decode region: generate every request's response. The engine's actual
	// forward pass runs inside the timed region — its faults are the findings
	// the sandbox exists to absorb — while the modeled layer load sets the
	// region's duration.
	var decPassTokens int
	var decLayerTokens uint64
	for _, r := range corpus {
		d := r.maxTokens
		if spanCap > 0 && d > spanCap {
			d = spanCap
		}
		decPassTokens += d
		decLayerTokens += uint64(depth) * uint64(d)
	}
	decReps := shadowPerfInnerReps(decLayerTokens)
	decPassDraws := shadowPerfUnits(decLayerTokens, eng.efficiency)
	res.DecodeTokens = decPassTokens * decReps
	t1 := time.Now()
	st2 := shadowNewStream(shadowPerfCorpusSeed(corpus) ^ 0xd00de00d)
	for _, r := range corpus {
		st2.mix(scrubbedCanonicalBytes(r))
	}
	for rep := 0; rep < decReps; rep++ {
		for _, r := range corpus {
			_ = eng.decode(r) // actual forward pass; may fault — absorbed upstream
		}
		res.ComputeWitness ^= shadowPerfWork(&st2, decPassDraws)
	}
	decElapsed := time.Since(t1)

	if preElapsed.Seconds() > 0 {
		res.PrefillTokS = float64(res.PrefillTokens) / preElapsed.Seconds()
	}
	if decElapsed.Seconds() > 0 {
		res.DecodeTokS = float64(res.DecodeTokens) / decElapsed.Seconds()
	}
	shadowPerfComputeWitness.Add(res.ComputeWitness)
	return res
}

// shadowPerfSandbox wraps a candidate arm block in recover(): a candidate
// crash is absorbed here — recorded, never propagated. The fast tier is the
// crash absorber; the slow tier's candidate arm is sandboxed too, as insurance
// against faults the short spans cannot trigger.
func shadowPerfSandbox(tier string, f func() shadowPerfArmResult) (res shadowPerfArmResult, crashed bool, crashErr error) {
	defer func() {
		if r := recover(); r != nil {
			crashed = true
			crashErr = fmt.Errorf("candidate fault absorbed in %s tier sandbox: %v", tier, r)
		}
	}()
	res = f()
	return res, false, nil
}

// shadowPerfABBAArms is one ABBA block: incumbent, candidate, candidate,
// incumbent. Each candidate pair is bracketed by incumbent pairs within the
// block, so drift between the pairs cancels in the paired median
// (bench_abba.go / bench_h2h_refresh.go methodology).
var shadowPerfABBAArms = []bool{false, true, true, false}

// shadowPerfRunTier schedules reps ABBA blocks for one tier and collects the
// split-phase sample sets. One throwaway ABBA block runs first as warm-up so
// both arms' code paths settle (frequency ramp, stream state, page faults)
// before any sample counts — the house pattern from bench_abba.go ("one warm
// prompt ... the rest are measured"); its samples are discarded wholesale and
// never enter the receipt. On a candidate crash — warm-up or measured — the
// in-progress block is discarded and the tier stops: the crash is the finding.
func shadowPerfRunTier(tier string, depth, spanCap, reps int, ref, cand shadowPerfEngine, corpus []scrubbedRequest) ShadowPerfTierResult {
	tr := ShadowPerfTierResult{Tier: tier, Layers: depth, SpanCap: spanCap, ABBAReps: reps}
	measure := func(isCand bool) (shadowPerfArmResult, bool, error) {
		if !isCand {
			return shadowPerfMeasureArm(depth, spanCap, ref, corpus), false, nil
		}
		return shadowPerfSandbox(tier, func() shadowPerfArmResult {
			return shadowPerfMeasureArm(depth, spanCap, cand, corpus)
		})
	}
	crash := func(crashErr error) {
		tr.Crashed = true
		tr.CrashReason = fmt.Sprintf("%v", crashErr)
	}
	for _, isCand := range shadowPerfABBAArms { // warm-up block: discarded
		if _, crashed, crashErr := measure(isCand); crashed {
			crash(crashErr)
			return tr
		}
	}
	for rep := 0; rep < reps; rep++ {
		var basePair, candPair [2]shadowPerfArmResult
		for i, isCand := range shadowPerfABBAArms {
			res, crashed, crashErr := measure(isCand)
			if crashed {
				crash(crashErr)
				return tr
			}
			switch i {
			case 0:
				basePair[0] = res
			case 1:
				candPair[0] = res
			case 2:
				candPair[1] = res
			case 3:
				basePair[1] = res
			}
		}
		tr.BaselinePrefillTokS = append(tr.BaselinePrefillTokS, basePair[0].PrefillTokS, basePair[1].PrefillTokS)
		tr.BaselineDecodeTokS = append(tr.BaselineDecodeTokS, basePair[0].DecodeTokS, basePair[1].DecodeTokS)
		tr.CandidatePrefillTokS = append(tr.CandidatePrefillTokS, candPair[0].PrefillTokS, candPair[1].PrefillTokS)
		tr.CandidateDecodeTokS = append(tr.CandidateDecodeTokS, candPair[0].DecodeTokS, candPair[1].DecodeTokS)
		tr.Blocks = append(tr.Blocks, shadowPerfBlock{
			BaselinePrefill:  [2]float64{basePair[0].PrefillTokS, basePair[1].PrefillTokS},
			CandidatePrefill: [2]float64{candPair[0].PrefillTokS, candPair[1].PrefillTokS},
			BaselineDecode:   [2]float64{basePair[0].DecodeTokS, basePair[1].DecodeTokS},
			CandidateDecode:  [2]float64{candPair[0].DecodeTokS, candPair[1].DecodeTokS},
		})
		tr.ComputeWitness ^= basePair[0].ComputeWitness ^ basePair[1].ComputeWitness ^
			candPair[0].ComputeWitness ^ candPair[1].ComputeWitness
	}
	return tr
}

// --- verify gate, pin, entry point -------------------------------------------

// shadowPerfVerifyGateOf runs the bit-exact oracle (shadowJudge) on the fixed
// probe set, candidate against incumbent. Any divergence — or a candidate
// panic during verification — refuses the candidate and, by construction, no
// perf count will ever count for it.
func shadowPerfVerifyGateOf(probes []scrubbedRequest, ref, cand shadowPerfEngine) (g shadowPerfVerifyGate) {
	g.Probes = len(probes)
	defer func() {
		if r := recover(); r != nil {
			g.Passed = false
			g.Detail = fmt.Sprintf("candidate panicked during the verify gate: %v", r)
		}
	}()
	for i, p := range probes {
		refToks := ref.decode(p)
		candToks := cand.decode(p)
		prov := shadowProvenanceOf("shadowperf/probe-"+strconv.Itoa(i)+"-"+cand.name, p, cand.backend, shadowBaseline(refToks))
		v := shadowJudge(refToks, candToks, prov)
		if !v.Pass {
			g.Passed = false
			g.Detail = v.Detail
			if v.Artifact != nil && v.Artifact.Divergence != nil {
				g.DivergenceIndex = v.Artifact.Divergence.Index
			}
			return g
		}
	}
	g.Passed = true
	g.Detail = fmt.Sprintf("candidate bit-exact against incumbent on %d probes", len(probes))
	return g
}

// shadowPerfBuildPin assembles the result pin: the traffic window is always
// computed from the ordered scrubbed corpus digests; the generation
// coordinates are pinned only when a landed seam supplied them.
func shadowPerfBuildPin(pins ShadowPerfPins, corpus []scrubbedRequest) shadowPerfPin {
	p := shadowPerfPin{
		WeightsEpoch: shadowPerfUnpinned("weight-epoch seam (phase 1 WeightDelta) not present in this base"),
		ForwardGen:   shadowPerfUnpinned("forward-gen seam (phase 0 ForwardSlot) not present in this base"),
		PlacementGen: shadowPerfUnpinned("placement-gen seam (phase 3 PlacementDelta) not started"),
	}
	if pins.HasWeightsEpoch {
		p.WeightsEpoch = pins.WeightsEpoch
	}
	if pins.HasForwardGen {
		p.ForwardGen = pins.ForwardGen
	}
	if pins.HasPlacementGen {
		p.PlacementGen = pins.PlacementGen
	}
	h := sha256.New()
	for _, r := range corpus {
		_, _ = h.Write([]byte(r.promptDigest))
		_, _ = h.Write([]byte("|"))
	}
	p.TrafficWindow = "sha256:" + hex.EncodeToString(h.Sum(nil))
	return p
}

// ShadowPerfEvaluate is the phase-2 entry point: candidate vs incumbent TPS on
// scrubbed real-traffic replay, two tiers, split phases, gated by the bit-exact
// oracle BEFORE any perf count counts. It scrubs the live requests first
// (privacy: scrub before persistence), runs the verify gate on the fixed probe
// set, then the sandboxed fast tier (crash absorber) and the authoritative
// slow tier. It never touches the user path: the response a user receives is
// produced exclusively by shadowServeLive on the incumbent.
func ShadowPerfEvaluate(live []liveRequest, ref, cand shadowPerfEngine, opts ShadowPerfOptions) ShadowPerfReceipt {
	opts.withDefaults()

	scrubbed := make([]scrubbedRequest, len(live))
	for i, l := range live {
		scrubbed[i] = shadowScrub(l)
	}

	recv := ShadowPerfReceipt{
		Schema:      shadowPerfReceiptSchema,
		GeneratedAt: time.Now().UTC(),
		Candidate:   shadowPerfEngineRef{Name: cand.name, Backend: cand.backend},
		Incumbent:   shadowPerfEngineRef{Name: ref.name, Backend: ref.backend},
		Pin:         shadowPerfBuildPin(opts.Pins, scrubbed),
	}

	// 1. Bit-exact oracle on the fixed probe set, BEFORE any perf count counts.
	probeLive := shadowPerfProbeSet()
	probeScrubbed := make([]scrubbedRequest, len(probeLive))
	for i, l := range probeLive {
		probeScrubbed[i] = shadowScrub(l)
	}
	recv.VerifyGate = shadowPerfVerifyGateOf(probeScrubbed, ref, cand)
	if !recv.VerifyGate.Passed {
		recv.IsolationPreserved = true
		recv.PerfCountsValid = false
		return recv
	}

	// 2. Fast tier: layer subset, short spans, sandboxed — the crash absorber.
	fast := shadowPerfRunTier(shadowPerfTierFast, opts.FastLayerCap, opts.FastSpanCap, opts.ABBAReps, ref, cand, scrubbed)
	recv.Tiers = append(recv.Tiers, fast)
	if fast.Crashed {
		recv.IsolationPreserved = true
		recv.PerfCountsValid = false
		return recv
	}

	// 3. Slow tier: full layers, full spans, real-traffic replay — authoritative.
	slow := shadowPerfRunTier(shadowPerfTierSlow, opts.ModelLayers, 0, opts.ABBAReps, ref, cand, scrubbed)
	recv.Tiers = append(recv.Tiers, slow)
	recv.IsolationPreserved = true
	recv.PerfCountsValid = !slow.Crashed
	return recv
}
