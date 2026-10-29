package engine

import "testing"

func TestBatcherStarvationEnvReachesBatcher(t *testing.T) {
	t.Setenv(envBatcherWaitingServedRatio, "")
	t.Setenv(envBatcherMaxWaitingDecodeSteps, "")
	if c := DefaultBatchingEngineConfig().Batcher; c.WaitingServedRatio != 0 || c.MaxWaitingDecodeSteps != 0 {
		t.Fatalf("default starvation bound = %v/%d, want 0/0 (prefill-first)", c.WaitingServedRatio, c.MaxWaitingDecodeSteps)
	}

	t.Setenv(envBatcherWaitingServedRatio, "0.5")
	t.Setenv(envBatcherMaxWaitingDecodeSteps, "8")
	e, err := NewBatchingEngine(DefaultBatchingEngineConfig())
	if err != nil {
		t.Fatalf("NewBatchingEngine: %v", err)
	}
	defer e.Close()
	if got := e.cb.cfg; got.WaitingServedRatio != 0.5 || got.MaxWaitingDecodeSteps != 8 {
		t.Fatalf("batcher starvation bound = %v/%d, want 0.5/8", got.WaitingServedRatio, got.MaxWaitingDecodeSteps)
	}

	t.Setenv(envBatcherWaitingServedRatio, "-1")
	t.Setenv(envBatcherMaxWaitingDecodeSteps, "x")
	if c := DefaultBatchingEngineConfig().Batcher; c.WaitingServedRatio != 0 || c.MaxWaitingDecodeSteps != 0 {
		t.Fatalf("malformed env starvation bound = %v/%d, want 0/0", c.WaitingServedRatio, c.MaxWaitingDecodeSteps)
	}
}
