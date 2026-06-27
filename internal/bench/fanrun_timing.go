package bench

// fanrun_timing.go — the OPTIONAL prefill-elision wall-clock half of fanrun.
//
// This is the only part of fanrun that touches a model, and it is gated behind -model-dir:
// the counter+geometry headline (RunFanoutLive in fanrun.go) is fully model-free. When a
// small CPU model is present, this measures the SAME reuse-vs-no-reuse prefix lever
// cmd/fleetserve measures, scoped to the fan-out width N: the master-goal prefix is
// prefilled ONCE and its KV cloned into N sub-agents (reuse), vs N independent full prefix
// prefills (no-reuse). Best (min) wall-clock over reps — least-contended sampling, the
// MODEL-BASELINE methodology. The model here is a FLOP PROXY for the prefill the kernel
// skips; it is NOT the planner that drives the agent loop (that is the deterministic
// MockPlanner). Two separate honest measurements, never blended.

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// maxTimingWidth caps the no-reuse ablation: N independent full prefix prefills at N=1024
// would dominate the run with redundant work the reuse arm exists to delete. The reuse arm
// and the geometry already carry the full-N story; the timing arm proves the per-clone win
// at a tractable width and the (N-1)*P geometry extrapolates it honestly.
const maxTimingWidth = 64

// timingDecodeTail is the short decode run after prefill so the comparison is dominated by
// the prefill cost (the lever), not the shared batched decode.
const timingDecodeTail = 4

var (
	timingModelOnce sync.Once
	timingModel     *model.Model
	timingModelErr  error
	timingVocab     int
)

// loadTimingModel loads + quantizes the prefill-timing model once per process.
func loadTimingModel(dir string, quant bool) (*model.Model, int, error) {
	timingModelOnce.Do(func() {
		m, err := model.Load(pathutil.ExpandTilde(dir))
		if err != nil {
			timingModelErr = err
			return
		}
		if quant {
			m.Quantize()
		}
		timingModel = m
		timingVocab = m.Cfg.VocabSize
	})
	return timingModel, timingVocab, timingModelErr
}

// applyPrefillTiming fills the cell's prefill-elision wall-clock fields from a real
// reuse-vs-no-reuse measurement. On any failure it records a structured skip reason rather
// than fabricating a number (the longctx-probe discipline).
func applyPrefillTiming(cell *FanrunCell, opts FanrunOptions, N int) {
	m, vocab, err := loadTimingModel(opts.ModelDir, opts.Quant)
	if err != nil {
		cell.PrefillTimingSkipped = fmt.Sprintf("MODEL_LOAD_FAILED: %v", err)
		return
	}
	C := N
	if C > maxTimingWidth {
		C = maxTimingWidth
		cell.PrefillTimingSkipped = fmt.Sprintf("timing measured at C=%d (capped from N=%d); reuse arm + (N-1)*P geometry carry full N", C, N)
	}

	prefix := lcgIDs(opts.Prefix, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)

	var reuse, noReuse []time.Duration
	for r := 0; r < opts.Reps; r++ {
		// ---- REUSE: prefill the master-goal prefix ONCE, clone its KV into C sub-agents ----
		t0 := time.Now()
		base := m.NewSession()
		base.Quant = opts.Quant
		base.Prefill(prefix)
		bs := m.NewBatchFromPrefixReserve(base.Cache, C, timingDecodeTail)
		bs.SetQuant(opts.Quant)
		decodeBatch(bs, ids0, timingDecodeTail, vocab)
		reuse = append(reuse, time.Since(t0))

		// ---- NO-REUSE: C independent full prefix prefills, same batched decode ----
		prompts := make([][]int, C)
		for b := range prompts {
			prompts[b] = prefix
		}
		n0 := time.Now()
		nbs := m.NewBatchSession(C)
		nbs.SetQuant(opts.Quant)
		nbs.PrefillEachNoLogits(prompts)
		nbs.Reserve(timingDecodeTail)
		decodeBatch(nbs, ids0, timingDecodeTail, vocab)
		noReuse = append(noReuse, time.Since(n0))

		runtime.GC()
	}

	cell.PrefillReuseTotalMs = minDurMs(reuse)
	cell.PrefillNoReuseTotalMs = minDurMs(noReuse)
	if cell.PrefillReuseTotalMs > 0 {
		cell.PrefillReuseSpeedup = cell.PrefillNoReuseTotalMs / cell.PrefillReuseTotalMs
	}
}

// decodeBatch runs a short batched decode tail (mirrors fleetserve.runTurns' decode loop,
// no result-ingest turns — pure decode after prefill).
func decodeBatch(bs *model.BatchSession, ids0 []int, steps, vocab int) {
	ids := append([]int(nil), ids0...)
	for s := 0; s < steps; s++ {
		bs.StepBatch(ids)
		for i := range ids {
			ids[i] = (ids[i]*48271 + 1) % vocab
		}
	}
}

// lcgIDs generates n deterministic pseudo-token ids in [0,vocab) (the fleetserve LCG, so
// the timing prefix is reproducible across runs and matches fleetserve's distribution).
func lcgIDs(n, vocab, seed int) []int {
	ids := make([]int, n)
	state := uint64(2463534242 + seed)
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		if vocab <= 0 {
			vocab = 1
		}
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}
