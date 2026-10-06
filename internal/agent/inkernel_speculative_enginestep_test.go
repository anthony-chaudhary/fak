package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

func engineStepSpecRounds(s enginestep.Snapshot) (rounds, drafted, accepted uint64) {
	if s.Speculative == nil {
		return 0, 0, 0
	}
	return s.Speculative.Rounds, s.Speculative.DraftTokens, s.Speculative.AcceptedTokens
}

func engineStepNewSpecRecords(before, after enginestep.Snapshot) []enginestep.StepRecord {
	floor := engineStepMaxSeq(before)
	var out []enginestep.StepRecord
	for _, rec := range after.Recent {
		if rec.Seq > floor && rec.Kind == enginestep.KindDecodeStep && rec.Path == enginestep.PathSpeculative {
			out = append(out, rec)
		}
	}
	return out
}

// fak-test:runtime fast est=1s lane=default
func TestEngineStepSpeculativeVerifiedRoundFeedsDefault(t *testing.T) {
	probe := &speculativeRequestProbe{}
	p, prompt := speculativeRequestPlanner(t, probe)
	probe.tokens = speculativeAcceptedDraft(t, p.m, prompt, 3)

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	res, err := p.generateReusedRecovering(context.Background(), prompt, 3, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	if res.gen != 3 || p.SpeculativeEngine().Stats().VerificationRounds != 1 {
		t.Fatalf("generated/rounds = %d/%d, want 3/1", res.gen, p.SpeculativeEngine().Stats().VerificationRounds)
	}

	br, bd, ba := engineStepSpecRounds(before)
	ar, ad, aa := engineStepSpecRounds(after)
	if ar-br != 1 || ad-bd != 3 || aa-ba != 3 {
		t.Fatalf("spec rounds/drafted/accepted delta = %d/%d/%d, want 1/3/3", ar-br, ad-bd, aa-ba)
	}
	bs, as := before.Decode[enginestep.PathSpeculative], after.Decode[enginestep.PathSpeculative]
	if as.Steps-bs.Steps != 1 || as.Tokens-bs.Tokens != 3 {
		t.Fatalf("decode[speculative] steps/tokens delta = %d/%d, want 1/3", as.Steps-bs.Steps, as.Tokens-bs.Tokens)
	}
	recs := engineStepNewSpecRecords(before, after)
	if len(recs) != 1 || recs[0].Proposed != 3 || recs[0].Accepted != 3 || recs[0].Tokens != 3 || recs[0].Lanes != 1 {
		t.Fatalf("new speculative ring records = %+v, want one proposed=3 accepted=3 tokens=3", recs)
	}
	if engineStepPhaseCount(after, enginestep.PhasePrefill) <= engineStepPhaseCount(before, enginestep.PhasePrefill) {
		t.Fatal("prefill phase count did not increase on the speculative path")
	}
	if engineStepPhaseCount(after, enginestep.PhaseDecode)-engineStepPhaseCount(before, enginestep.PhaseDecode) != 1 {
		t.Fatal("decode phase delta != 1 on the speculative path")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestEngineStepSpeculativeRejectedRoundCountsCorrection(t *testing.T) {
	probe := &speculativeRequestProbe{}
	p, prompt := speculativeRequestPlanner(t, probe)
	accepted := speculativeAcceptedDraft(t, p.m, prompt, 2)
	probe.tokens = []int{accepted[0], (accepted[1] + 1) % p.m.Cfg.VocabSize}

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	res, err := p.generateReusedRecovering(context.Background(), prompt, 2, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	if res.gen != 2 {
		t.Fatalf("generated = %d, want 2", res.gen)
	}
	recs := engineStepNewSpecRecords(before, after)
	if len(recs) != 1 || recs[0].Proposed != 2 || recs[0].Accepted != 1 || recs[0].Tokens != 2 {
		t.Fatalf("new speculative ring records = %+v, want one proposed=2 accepted=1 tokens=2 (accepted+correction)", recs)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestEngineStepSpeculativeProposalFallbackIsSerial(t *testing.T) {
	probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	p, prompt := speculativeRequestPlanner(t, probe)

	before := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	res, err := p.generateReusedRecovering(context.Background(), prompt, 3, 0, 0, 0, nil, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	after := enginestep.Default.Snapshot(enginestep.MaxRecent, "")
	if res.gen != 3 {
		t.Fatalf("generated = %d, want 3", res.gen)
	}
	br, _, _ := engineStepSpecRounds(before)
	if ar, _, _ := engineStepSpecRounds(after); ar != br {
		t.Fatal("a fallback-only request recorded a speculative round")
	}
	if recs := engineStepNewSpecRecords(before, after); len(recs) != 0 {
		t.Fatalf("fallback-only request appended speculative records: %+v", recs)
	}
	if after.Decode[enginestep.PathSerial].Steps <= before.Decode[enginestep.PathSerial].Steps {
		t.Fatalf("serial decode steps did not increase on fallback: %d -> %d", before.Decode[enginestep.PathSerial].Steps, after.Decode[enginestep.PathSerial].Steps)
	}
}
