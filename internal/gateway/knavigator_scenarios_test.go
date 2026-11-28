package gateway

import "testing"

type fakeKNavHarness struct {
	maxBatch   int
	maxQueue   int
	batch      int
	queue      int
	seenPrefix map[string]bool
}

func (f *fakeKNavHarness) Admit(req KNavRequest) KNavDecision {
	hit := req.PrefixGroup != "" && f.seenPrefix[req.PrefixGroup]
	if req.PrefixGroup != "" {
		f.seenPrefix[req.PrefixGroup] = true
	}
	if f.batch < f.maxBatch {
		f.batch++
		return KNavDecision{Admitted: true, PrefixHit: hit}
	}
	if f.queue < f.maxQueue {
		f.queue++
		return KNavDecision{Queued: true, PrefixHit: hit, Reason: "batch_full"}
	}
	return KNavDecision{PrefixHit: hit, Reason: "queue_full"}
}

func (f *fakeKNavHarness) Snapshot() KNavSchedulerState {
	return KNavSchedulerState{QueueDepth: f.queue, ActiveBatch: f.batch}
}

// fak-test:runtime fast est=5ms lane=default
func TestKNavigatorScenarios(t *testing.T) {
	h := &fakeKNavHarness{maxBatch: 2, maxQueue: 2, seenPrefix: make(map[string]bool)}
	sc := KNavWorkloadScenario{
		Name: "burst-shared-prefix",
		Steps: []KNavStep{
			{Label: "warmup", Count: 3, PromptTokens: 512, MaxNewTokens: 64, Priority: 1, PrefixGroup: "sys-a"},
			{Label: "overflow", Count: 2, PromptTokens: 1024, MaxNewTokens: 128, Priority: 0, PrefixGroup: "sys-b"},
		},
	}
	rep := RunKNavScenario(h, sc)
	if rep.Submitted != 5 || rep.Admitted != 2 || rep.Queued != 2 || rep.Rejected != 1 {
		t.Fatalf("unexpected report: %+v", rep)
	}
	if rep.PrefixHits != 3 { // 2 repeats of sys-a + 1 repeat of sys-b
		t.Fatalf("PrefixHits=%d, want 3", rep.PrefixHits)
	}
	if rep.PeakBatchSize != 2 || rep.PeakQueue != 2 {
		t.Fatalf("peaks=(batch=%d, queue=%d), want (2, 2)", rep.PeakBatchSize, rep.PeakQueue)
	}
}
