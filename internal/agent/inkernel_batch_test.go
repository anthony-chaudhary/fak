package agent

import (
	"context"
	"math/rand"
	"testing"

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
