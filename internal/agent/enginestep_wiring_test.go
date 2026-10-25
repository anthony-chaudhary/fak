package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// These tests prove the enginestep recorder is reachable from the live native
// serving loop (InKernelPlanner.Complete), not merely present. enginestep.Default
// is process-global, so every assertion is a DELTA between two snapshots.

func engineStepPhaseCount(s enginestep.Snapshot, p enginestep.Phase) uint64 {
	return s.Phases[string(p)].Count
}

func engineStepExpectedDecodeSteps(t *testing.T, comp *Completion) uint64 {
	t.Helper()
	gen := comp.Usage.CompletionTokens
	if gen < 1 {
		t.Fatalf("completion generated %d tokens, want >=1 (finish=%s)", gen, comp.FinishReason)
	}
	// decodeOne skips the unused final Step at maxNew; a token-ID stop is sampled
	// but never emitted, so every emitted token was stepped.
	if comp.FinishReason == "length" {
		return uint64(gen - 1)
	}
	return uint64(gen)
}

func assertEngineStepRequestDeltas(t *testing.T, before, after enginestep.Snapshot) {
	t.Helper()
	if got := engineStepPhaseCount(after, enginestep.PhaseRequest) - engineStepPhaseCount(before, enginestep.PhaseRequest); got != 1 {
		t.Fatalf("request phase delta = %d, want 1", got)
	}
	if engineStepPhaseCount(after, enginestep.PhaseSample) <= engineStepPhaseCount(before, enginestep.PhaseSample) {
		t.Fatalf("sample phase count did not increase: %d -> %d", engineStepPhaseCount(before, enginestep.PhaseSample), engineStepPhaseCount(after, enginestep.PhaseSample))
	}
	if after.RequestsActive != before.RequestsActive {
		t.Fatalf("requests_active = %d after Complete, want prior %d", after.RequestsActive, before.RequestsActive)
	}
	if after.PrefillChunks <= before.PrefillChunks || after.PrefillTokens <= before.PrefillTokens {
		t.Fatalf("prefill chunks/tokens did not increase: %d/%d -> %d/%d", before.PrefillChunks, before.PrefillTokens, after.PrefillChunks, after.PrefillTokens)
	}
	if engineStepPhaseCount(after, enginestep.PhasePrefill) <= engineStepPhaseCount(before, enginestep.PhasePrefill) {
		t.Fatal("prefill phase count did not increase")
	}
	if engineStepPhaseCount(after, enginestep.PhaseDecode) <= engineStepPhaseCount(before, enginestep.PhaseDecode) {
		t.Fatal("decode phase count did not increase")
	}
	sawChunk := false
	for _, rec := range after.Recent {
		if rec.Kind == enginestep.KindPrefillChunk && rec.Seq > engineStepMaxSeq(before) && rec.Tokens > 0 {
			sawChunk = true
		}
	}
	if !sawChunk {
		t.Fatal("no new prefill_chunk record with tokens>0 in the recent ring")
	}
}

// fak-test:runtime fast est=2s lane=default
func TestEngineStepWiringSerialCompleteFeedsDefault(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-enginestep-serial", false, nil, false)
	p.maxNew = 4
	p.batchDecode = false

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "engine step"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")

	want := engineStepExpectedDecodeSteps(t, comp)
	b, a := before.Decode[enginestep.PathSerial], after.Decode[enginestep.PathSerial]
	if got := a.Tokens - b.Tokens; got != want {
		t.Fatalf("serial decode tokens delta = %d, want %d (completion tokens %d, finish %s)", got, want, comp.Usage.CompletionTokens, comp.FinishReason)
	}
	if got := a.Steps - b.Steps; got != want {
		t.Fatalf("serial decode steps delta = %d, want %d", got, want)
	}
	if got := after.Decode[enginestep.PathBatched].Tokens - before.Decode[enginestep.PathBatched].Tokens; got != 0 {
		t.Fatalf("batched decode tokens delta = %d on the serial path, want 0", got)
	}
	assertEngineStepRequestDeltas(t, before, after)
}

// fak-test:runtime fast est=2s lane=default
func TestEngineStepWiringCoalescedCohortFeedsDefault(t *testing.T) {
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		out := make([][]float32, len(ids))
		for i, on := range active {
			if on {
				out[i] = bs.Seqs[i].Step(ids[i])
			}
		}
		return out, 0, 0, true
	})
	defer restoreProbe()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.LayerTypes = []string{"linear_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-enginestep-coalesce", true, nil, false)
	p.maxNew = 4
	p.batchDecode = true
	p.metal = true // synthetic coordinator gate, as in TestInKernelPlannerCoalescesConcurrentQwenTurns
	p.coalesceReadyHook = func() {}
	var cohorts []int
	p.coalesceBatchHook = func(n int) { cohorts = append(cohorts, n) }
	if !p.coalescesQwenDecode() {
		t.Fatal("fixture does not engage the decode coalescer")
	}

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "cohort"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	if len(cohorts) != 1 {
		t.Fatalf("coordinator cohorts = %v, want one", cohorts)
	}

	if got := after.Cohorts.Steps - before.Cohorts.Steps; got < 1 {
		t.Fatalf("cohort count delta = %d, want >=1", got)
	}
	if got := engineStepPhaseCount(after, enginestep.PhaseCohortWait) - engineStepPhaseCount(before, enginestep.PhaseCohortWait); got != 1 {
		t.Fatalf("cohort_wait phase delta = %d, want 1", got)
	}
	// The CPU serial path deliberately skips devMu; the cohort path always takes it.
	if got := engineStepPhaseCount(after, enginestep.PhaseDeviceWait) - engineStepPhaseCount(before, enginestep.PhaseDeviceWait); got < 1 {
		t.Fatalf("device_wait phase delta = %d on the cohort path, want >=1", got)
	}
	if after.CoalesceQueueDepth != 0 {
		t.Fatalf("coalesce queue depth = %d after the cohort drained, want 0", after.CoalesceQueueDepth)
	}
	want := engineStepExpectedDecodeSteps(t, comp)
	b, a := before.Decode[enginestep.PathBatched], after.Decode[enginestep.PathBatched]
	if got := a.Tokens - b.Tokens; got != want {
		t.Fatalf("batched decode tokens delta = %d, want %d (completion tokens %d, finish %s)", got, want, comp.Usage.CompletionTokens, comp.FinishReason)
	}
	if want > 0 && a.MaxLanes < 1 {
		t.Fatalf("batched max lanes = %v, want >=1", a.MaxLanes)
	}
	sawCohort, sawBatched := false, false
	for _, rec := range after.Recent {
		if rec.Seq <= engineStepMaxSeq(before) {
			continue
		}
		switch {
		case rec.Kind == enginestep.KindCohort && rec.Lanes >= 1:
			sawCohort = true
		case rec.Kind == enginestep.KindDecodeStep && rec.Path == enginestep.PathBatched && rec.Lanes >= 1:
			sawBatched = true
		}
	}
	if !sawCohort || (want > 0 && !sawBatched) {
		t.Fatalf("recent ring cohort=%v batched_step=%v, want both", sawCohort, sawBatched)
	}
	assertEngineStepRequestDeltas(t, before, after)
}

func engineStepMaxSeq(s enginestep.Snapshot) uint64 {
	var max uint64
	for _, rec := range s.Recent {
		if rec.Seq > max {
			max = rec.Seq
		}
	}
	return max
}

// fak-test:runtime fast est=2s lane=default
func TestEngineStepWiringBatchedSingleLaneFeedsDefault(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-enginestep-batched", false, nil, false)
	p.maxNew = 4
	p.batchDecode = true // opt-in B=1 StepBatchActive path, outside the coalescer
	if p.coalescesQwenDecode() {
		t.Fatal("fixture unexpectedly engages the coalescer")
	}

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "batched lane"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")

	want := engineStepExpectedDecodeSteps(t, comp)
	b, a := before.Decode[enginestep.PathSerial], after.Decode[enginestep.PathSerial]
	if got := a.Tokens - b.Tokens; got != want {
		t.Fatalf("serial decode tokens delta = %d, want %d (completion tokens %d, finish %s)", got, want, comp.Usage.CompletionTokens, comp.FinishReason)
	}
	if got := a.Steps - b.Steps; got != want {
		t.Fatalf("serial decode steps delta = %d, want %d (one lane per step)", got, want)
	}
	if got := after.Decode[enginestep.PathBatched].Steps - before.Decode[enginestep.PathBatched].Steps; got != 0 {
		t.Fatalf("batched decode steps delta = %d for scalar fallback, want 0", got)
	}
	for _, rec := range after.Recent {
		if rec.Seq > engineStepMaxSeq(before) && rec.Kind == enginestep.KindDecodeStep {
			if rec.Path != enginestep.PathSerial || rec.Lanes != 1 || engineStepReason(t, rec) != "singleton" {
				t.Fatalf("singleton fallback record = %+v, want serial/1/singleton", rec)
			}
		}
	}
	if got := after.Cohorts.Steps - before.Cohorts.Steps; got != 0 {
		t.Fatalf("cohort delta = %d outside the coalescer, want 0", got)
	}
	assertEngineStepRequestDeltas(t, before, after)
}

// Read the wire field so the behavioral witness also compiles against the parent
// implementation, which has no reason field yet.
func engineStepReason(t *testing.T, rec enginestep.StepRecord) string {
	t.Helper()
	raw, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Reason
}

// fak-test:runtime fast est=200ms lane=default
func TestEngineStepWiringScalarFallbackExecution(t *testing.T) {
	for _, unsupported := range []bool{false, true} {
		name := "shared_dense"
		if unsupported {
			name = "unsupported_moe"
		}
		t.Run(name, func(t *testing.T) {
			cfg := tinyCfg()
			var m *model.Model
			if unsupported {
				cfg.NumExperts, cfg.NumExpertsPerTok = 2, 1
				m = model.NewSyntheticMoE(cfg)
			} else {
				m = model.NewSynthetic(cfg)
			}
			// Six active lanes become five, then only the final nonzero-index lane.
			// The final sampled token does not need another forward.
			budgets := []int{2, 3, 3, 3, 3, 4}
			lanes := make([]*decodeLane, len(budgets))
			sinks := make([]*batchLaneSink, len(budgets))
			wantTokens := make([][]int, len(budgets))
			for i, budget := range budgets {
				prompt := synthIDs(cfg.VocabSize, 3+i, uint64(100+i))
				serial, want := newDecodeLaneOver(m, prompt, budget, 0)
				inKernelDecodeSerial(context.Background(), serial)
				lanes[i], sinks[i] = newDecodeLaneOver(m, prompt, budget, 0)
				wantTokens[i] = want.toks
			}
			before := enginestep.Default.Snapshot(enginestep.MaxRecent, enginestep.KindDecodeStep)
			panels, macs := inKernelDecodeLanesBatched(context.Background(), lanes, m, false)
			after := enginestep.Default.Snapshot(enginestep.MaxRecent, enginestep.KindDecodeStep)
			for i, sink := range sinks {
				if !eqInts(sink.toks, wantTokens[i]) {
					t.Errorf("lane %d tokens = %v, want serial %v", i, sink.toks, wantTokens[i])
				}
			}
			wantSerial, wantBatched, wantBatchTokens := uint64(1), uint64(2), uint64(11)
			if unsupported {
				wantSerial, wantBatched, wantBatchTokens = 12, 0, 0
				if panels != 0 || macs != 0 {
					t.Fatalf("scalar fallback shared receipt = %d/%d, want 0/0", panels, macs)
				}
			} else if panels == 0 || macs == 0 {
				t.Fatal("dense control did not execute shared batch work")
			}
			for path, want := range map[string][2]uint64{
				enginestep.PathSerial:  {wantSerial, wantSerial},
				enginestep.PathBatched: {wantBatched, wantBatchTokens},
			} {
				b, a := before.Decode[path], after.Decode[path]
				if a.Steps-b.Steps != want[0] || a.Tokens-b.Tokens != want[1] {
					t.Errorf("%s steps/tokens = %d/%d, want %d/%d", path, a.Steps-b.Steps, a.Tokens-b.Tokens, want[0], want[1])
				}
			}
			var rows []enginestep.StepRecord
			for _, rec := range after.Recent {
				if rec.Seq > engineStepMaxSeq(before) {
					rows = append(rows, rec)
				}
			}
			if uint64(len(rows)) != wantSerial+wantBatched {
				t.Fatalf("decode rows = %d, want %d", len(rows), wantSerial+wantBatched)
			}
			for i, rec := range rows {
				path, reason, count := enginestep.PathSerial, "singleton", 1
				if i < len(rows)-1 {
					if unsupported {
						reason = "unsupported_config"
					} else {
						path, reason, count = enginestep.PathBatched, "", 6-i
					}
				}
				// A completed tiny CPU forward can fall within one Windows clock tick.
				// Exact rows, tokens, paths, reasons and token parity above prove execution;
				// zero records below-resolution time without inventing an epsilon.
				if rec.Path != path || rec.Lanes != count || rec.Tokens != count || engineStepReason(t, rec) != reason || rec.DurationNS < 0 {
					t.Errorf("decode row %d = %+v, want %s/%d/%s with a non-negative measured forward duration", i, rec, path, count, reason)
				}
			}
		})
	}
}
