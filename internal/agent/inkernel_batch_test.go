package agent

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_batch_test.go is the equivalence witness for the opt-in continuous-batch decode
// wiring on the resident chat serve (#401 / #3079 L2). The extracted per-token step
// (decodeLane.decodeOne) is shared by BOTH the serial driver (Session.Step, the default) and
// the multi-lane batched driver (BatchSession.StepBatchActive, FAK_INKERNEL_BATCH=on); the
// forward is their ONLY difference, and model.StepBatchActive is bit-for-bit identical to
// serial Session.Step per lane. So this suite pins two properties GPU-free on a synthetic
// model — no GLM checkpoint, no GPU:
//
//   - EQUIVALENCE: N concurrent lanes co-batched through one shared StepBatchActive per step
//     each emit a token sequence BIT-IDENTICAL to the same prompt decoded serially, even when
//     lanes retire at different depths (the ragged active-mask path).
//   - SERVE WIRING / DEFAULT-SAFE: on the live serve path (generateReused), flipping
//     FAK_INKERNEL_BATCH on reroutes decode through StepBatchActive as a batch of one and the
//     served tokens are unchanged — the non-negotiable "default off == today" bar, plus its
//     B==1 on-arm, both witnessed on the same code the gateway drives.

// batchLaneSink collects a lane's emitted token ids through the decode emit seam.
type batchLaneSink struct{ toks []int }

func (c *batchLaneSink) emit(id int) bool { c.toks = append(c.toks, id); return false }

// newDecodeLaneOver builds a decodeLane over a fresh session prefilled with prompt, greedy
// (temp 0) with the given per-lane generation budget. Each lane owns an independent Session
// (its own KV) and its own seeded rng, so co-batching cannot cross-contaminate a lane.
func newDecodeLaneOver(m *model.Model, prompt []int, maxNew int, seed int64) (*decodeLane, *batchLaneSink) {
	sink := &batchLaneSink{}
	s := m.NewSession()
	s.Quant = false
	ln := &decodeLane{
		s:      s,
		logits: s.Prefill(prompt),
		rng:    rand.New(rand.NewSource(seed)),
		stops:  map[int]bool{},
		emit:   sink.emit,
		maxNew: maxNew,
	}
	return ln, sink
}

// TestInKernelBatchedDecodeMatchesSerial is the required equivalence witness: N lanes driven
// concurrently through the batched StepBatchActive loop each decode BIT-IDENTICALLY to the same
// prompt decoded serially through Session.Step. Distinct prompt lengths put the lanes at
// distinct absolute positions (per-user RoPE / per-user cache-length attention exercised inside
// one shared step), and staggered maxNew retires lanes at different steps so the ragged
// active-mask compaction is genuinely exercised — yet every surviving lane stays bit-exact.
func TestInKernelBatchedDecodeMatchesSerial(t *testing.T) {
	cfg := tinyCfg() // dense PreNorm f32 => the batched StepBatch fast-path GEMM engages (not the serial fallback)
	m := model.NewSynthetic(cfg)
	const N = 5
	const seed = 0

	prompts := make([][]int, N)
	maxNew := make([]int, N)
	for i := 0; i < N; i++ {
		prompts[i] = synthIDs(cfg.VocabSize, 6+i*3, uint64(1000+i))
		maxNew[i] = 4 + i*2 // staggered depths => lanes finish on different steps
	}

	// SERIAL reference: each lane decoded alone through Session.Step (the default path).
	serial := make([][]int, N)
	for i := 0; i < N; i++ {
		ln, sink := newDecodeLaneOver(m, prompts[i], maxNew[i], seed)
		inKernelDecodeSerial(context.Background(), ln)
		serial[i] = sink.toks
	}

	// BATCHED: the SAME N prompts co-batched through one shared StepBatchActive per step.
	lanes := make([]*decodeLane, N)
	sinks := make([]*batchLaneSink, N)
	for i := 0; i < N; i++ {
		lanes[i], sinks[i] = newDecodeLaneOver(m, prompts[i], maxNew[i], seed)
	}
	inKernelDecodeLanesBatched(context.Background(), lanes, m, false)

	for i := 0; i < N; i++ {
		if len(serial[i]) != maxNew[i] {
			t.Fatalf("lane %d serial produced %d tokens, want maxNew %d", i, len(serial[i]), maxNew[i])
		}
		if !eqInts(sinks[i].toks, serial[i]) {
			t.Fatalf("lane %d batched decode NOT bit-identical to serial:\n batched=%v\n serial =%v", i, sinks[i].toks, serial[i])
		}
	}
	t.Logf("EQUIVALENCE: %d co-batched lanes (staggered depths %v) each bit-identical to serial Session.Step decode", N, maxNew)
}

// TestInKernelStochasticSamplingIsBatchCompositionInvariant verifies that a logical
// request's seeded stochastic stream does not depend on which other requests share a
// decode batch, their order, or an early-retiring neighbor.
//
// fak-test:runtime fast est=1s lane=default
func TestInKernelStochasticSamplingIsBatchCompositionInvariant(t *testing.T) {
	cfg := tinyCfg()
	m := model.NewSynthetic(cfg)
	type request struct {
		prompt []int
		seed   int64
		maxNew int
	}

	sharedPrompt := synthIDs(cfg.VocabSize, 13, 8101)
	requests := []request{
		{prompt: sharedPrompt, seed: 101, maxNew: 12},
		{prompt: sharedPrompt, seed: 202, maxNew: 12},
		{prompt: synthIDs(cfg.VocabSize, 9, 8102), seed: 303, maxNew: 8},
		{prompt: synthIDs(cfg.VocabSize, 17, 8103), seed: 404, maxNew: 1},
	}

	run := func(order []int) map[int][]int {
		lanes := make([]*decodeLane, len(order))
		sinks := make([]*batchLaneSink, len(order))
		for batchIndex, requestIndex := range order {
			r := requests[requestIndex]
			lanes[batchIndex], sinks[batchIndex] = newDecodeLaneOver(m, r.prompt, r.maxNew, r.seed)
			lanes[batchIndex].temp = 5.0
			lanes[batchIndex].topP = 0.95
			lanes[batchIndex].topK = 16
		}
		inKernelDecodeLanesBatched(context.Background(), lanes, m, false)

		got := make(map[int][]int, len(order))
		for batchIndex, requestIndex := range order {
			if lanes[batchIndex].err != nil {
				t.Fatalf("request %d decode failed: %v", requestIndex, lanes[batchIndex].err)
			}
			if len(sinks[batchIndex].toks) != requests[requestIndex].maxNew {
				t.Fatalf("request %d produced %d tokens, want %d", requestIndex, len(sinks[batchIndex].toks), requests[requestIndex].maxNew)
			}
			got[requestIndex] = sinks[batchIndex].toks
		}
		return got
	}

	singletons := make(map[int][]int, len(requests))
	for i := range requests {
		singletons[i] = run([]int{i})[i]
	}
	if eqInts(singletons[0], singletons[1]) {
		t.Fatalf("distinct request seeds did not activate stochastic sampling: seed 101 and 202 both produced %v", singletons[0])
	}
	t.Logf("STOCHASTIC: identical prompts with seeds 101 and 202 diverged: %v != %v", singletons[0], singletons[1])

	assertMatchesSingleton := func(name string, got map[int][]int, requestIDs ...int) {
		t.Helper()
		for _, requestID := range requestIDs {
			if !eqInts(got[requestID], singletons[requestID]) {
				t.Fatalf("%s changed request %d's seeded stream:\n got =%v\n want=%v", name, requestID, got[requestID], singletons[requestID])
			}
		}
	}

	assertMatchesSingleton("original batch", run([]int{0, 1, 2}), 0, 1, 2)
	assertMatchesSingleton("permuted batch", run([]int{2, 0, 1}), 0, 1, 2)
	assertMatchesSingleton("batch with early-retiring neighbor", run([]int{3, 0, 1, 2}), 0, 1, 2, 3)
}

// TestInKernelRequestKeyedSamplingIsBatchCompositionInvariant proves the request
// key, rather than a lane-local mutable RNG, owns each stochastic target draw.
//
// fak-test:runtime fast est=1s lane=default
func TestInKernelRequestKeyedSamplingIsBatchCompositionInvariant(t *testing.T) {
	cfg := tinyCfg()
	m := model.NewSynthetic(cfg)
	type request struct {
		prompt []int
		seed   int64
		maxNew int
	}
	sharedPrompt := synthIDs(cfg.VocabSize, 13, 9101)
	requests := []request{
		{prompt: sharedPrompt, seed: 101, maxNew: 12},
		{prompt: sharedPrompt, seed: -1, maxNew: 12},
		{prompt: synthIDs(cfg.VocabSize, 9, 9102), seed: 303, maxNew: 8},
		{prompt: synthIDs(cfg.VocabSize, 17, 9103), seed: 0, maxNew: 1},
	}

	run := func(order []int) map[int][]int {
		lanes := make([]*decodeLane, len(order))
		sinks := make([]*batchLaneSink, len(order))
		for batchIndex, requestIndex := range order {
			r := requests[requestIndex]
			lanes[batchIndex], sinks[batchIndex] = newDecodeLaneOver(m, r.prompt, r.maxNew, 777)
			lanes[batchIndex].temp = 5
			lanes[batchIndex].topP = 0.95
			lanes[batchIndex].topK = 16
			lanes[batchIndex].sampleKey = uint64(r.seed)
			lanes[batchIndex].sampleKeyed = true
			lanes[batchIndex].sampleBase = len(r.prompt)
		}
		inKernelDecodeLanesBatched(context.Background(), lanes, m, false)
		got := make(map[int][]int, len(order))
		for batchIndex, requestIndex := range order {
			if lanes[batchIndex].err != nil {
				t.Fatalf("request %d decode failed: %v", requestIndex, lanes[batchIndex].err)
			}
			got[requestIndex] = sinks[batchIndex].toks
		}
		return got
	}

	singletons := make(map[int][]int, len(requests))
	for i := range requests {
		singletons[i] = run([]int{i})[i]
		if replay := run([]int{i})[i]; !eqInts(replay, singletons[i]) {
			t.Fatalf("request %d did not replay: got %v want %v", i, replay, singletons[i])
		}
	}
	if eqInts(singletons[0], singletons[1]) {
		t.Fatalf("distinct request keys produced the same active stream: %v", singletons[0])
	}
	assert := func(name string, got map[int][]int, ids ...int) {
		t.Helper()
		for _, id := range ids {
			if !eqInts(got[id], singletons[id]) {
				t.Fatalf("%s changed request %d: got %v want %v", name, id, got[id], singletons[id])
			}
		}
	}
	assert("original", run([]int{0, 1, 2}), 0, 1, 2)
	assert("permuted", run([]int{2, 0, 1}), 0, 1, 2)
	assert("early retire", run([]int{3, 0, 1, 2}), 0, 1, 2, 3)
}

// fak-test:runtime fast est=5ms lane=default
func TestInKernelRequestKeyedSampleUsesAbsolutePosition(t *testing.T) {
	draw := func(sampleBase int) int {
		got := -1
		lane := &decodeLane{
			logits:      make([]float32, 16),
			temp:        1,
			stops:       map[int]bool{},
			emit:        func(token int) bool { got = token; return false },
			maxNew:      1,
			sampleKey:   17,
			sampleKeyed: true,
			sampleBase:  sampleBase,
		}
		lane.decodeOne(context.Background())
		if lane.err != nil {
			t.Fatal(lane.err)
		}
		return got
	}
	if got := draw(3); got != 13 {
		t.Fatalf("absolute position 3 selected %d, want 13", got)
	}
	if got := draw(11); got != 0 {
		t.Fatalf("absolute position 11 selected %d, want 0", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInKernelForcedThinkingBypassesRequestCoin(t *testing.T) {
	constraint := &nativeThinkingConstraint{
		budget:  NewThinkBudget(0, true),
		closeID: 5,
		decodeToken: func(token int) (string, error) {
			if token == 5 {
				return thinkClose, nil
			}
			return "reasoning", nil
		},
	}
	got := -1
	lane := &decodeLane{
		logits:      make([]float32, 16),
		temp:        1,
		stops:       map[int]bool{},
		emit:        func(token int) bool { got = token; return false },
		maxNew:      1,
		thinking:    constraint,
		sampleKey:   17,
		sampleKeyed: true,
		sampleBase:  3,
	}
	lane.decodeOne(context.Background())
	if lane.err != nil {
		t.Fatal(lane.err)
	}
	if got != 5 || !constraint.forced {
		t.Fatalf("forced close selected %d forced=%v, want 5/true", got, constraint.forced)
	}
	if len(lane.sampling.probs) != 0 {
		t.Fatal("forced close entered the probability sampler")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelRequestSeedReplaysAfterOOMRetry(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	ids := synthIDs(cfg.VocabSize, 8, 9191)
	seed := int64(-9)
	run := func(injectOOM bool) ([]int, bool) {
		measurement := &nativeInferenceMeasurement{startedAt: time.Now()}
		backend := compute.Default()
		var fault *receiptOOMRetryBackend
		if injectOOM {
			fault = &receiptOOMRetryBackend{Backend: backend, measurement: measurement}
			backend = fault
		}
		p := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "seeded-oom-retry", backend: backend}
		ctx := withNativeTargetSampleSeed(context.Background(), &seed)
		var emitted []int
		_, err := p.generateReusedWithOOMRetry(ctx, ids, 6, 5, 0.95, 16, nil, 0, 0, map[int]bool{}, func(token int) bool {
			emitted = append(emitted, token)
			return false
		}, func() {
			emitted = emitted[:0]
			measurement.reset()
		}, measurement)
		if err != nil {
			t.Fatal(err)
		}
		return emitted, fault != nil && fault.failed
	}
	want, _ := run(false)
	got, faulted := run(true)
	if !faulted {
		t.Fatal("OOM retry fixture did not inject a first-attempt failure")
	}
	if !eqInts(got, want) {
		t.Fatalf("seeded OOM replay changed stream: got %v want %v", got, want)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestInKernelSeededCompleteReplaysWithSchema(t *testing.T) {
	p := nativeStructuredTestPlanner(t)
	temperature, topP, topK := 0.8, 0.8, 2
	messages := []Message{{Role: RoleUser, Content: "Answer with the requested object."}}
	run := func(seed int64) string {
		completion, err := p.Complete(context.Background(), messages, nil,
			WithSeed(&seed), WithResponseFormat(nativeStructuredTestFormat),
			WithTemperature(&temperature), WithTopP(&topP), WithTopK(&topK))
		if err != nil {
			t.Fatal(err)
		}
		nativeStructuredCheckAnswer(t, completion.Message.Content)
		return completion.Message.Content
	}
	if first, replay := run(0), run(0); first != replay {
		t.Fatalf("explicit zero seed did not replay through Complete: %q != %q", first, replay)
	}
	_ = run(-1)
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelSeededRequestBypassesSpeculativeProposals(t *testing.T) {
	probe := &speculativeRequestProbe{tokens: []int{1, 2}}
	p, ids := speculativeRequestPlanner(t, probe)
	seed := int64(-1)
	ctx := withNativeTargetSampleSeed(context.Background(), &seed)
	_, err := p.generateReusedRecovering(ctx, ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if probe.calls != 0 {
		t.Fatalf("seeded target-only request made %d speculative proposal calls", probe.calls)
	}
}

// TestInKernelBatchServePathByteIdenticalToSerial pins the acceptance bar on the LIVE serve
// path: generateReused (what /v1/chat/completions and /v1/messages drive) decodes byte-for-byte
// the same tokens whether FAK_INKERNEL_BATCH is off (serial Session.Step) or on (rerouted
// through StepBatchActive as a batch of one). Off is the non-negotiable "identical to today";
// on at B==1 is StepBatchActive == Seqs[0].Step, so it must not move a single token.
func TestInKernelBatchServePathByteIdenticalToSerial(t *testing.T) {
	cfg := tinyCfg()
	ids := synthIDs(cfg.VocabSize, 20, 909)
	const maxNew = 10

	poff := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "synthetic", quant: false}
	genOff, _ := decode(poff, ids, maxNew)

	pon := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "synthetic", quant: false, batchDecode: true}
	genOn, _ := decode(pon, ids, maxNew)

	if len(genOff) != maxNew {
		t.Fatalf("serial decode produced %d tokens, want %d", len(genOff), maxNew)
	}
	if !eqInts(genOn, genOff) {
		t.Fatalf("FAK_INKERNEL_BATCH reroute changed the served tokens (B==1 must be identical):\n on =%v\n off=%v", genOn, genOff)
	}
	t.Logf("SERVE WIRING: FAK_INKERNEL_BATCH on (B=1 StepBatchActive) byte-identical to the default serial decode, %d tokens", len(genOff))
}
