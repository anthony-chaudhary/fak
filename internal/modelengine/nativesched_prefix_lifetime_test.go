package modelengine

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=4s lane=default
func TestNativeSchedulerPrefixMetadataRetiresOnlyItsOwner(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(48)
	ctl := m.NewSession()
	ctl.Quant, ctl.Q4K = true, true
	ctl.Prefill(prompt[:16])
	tree := radixkv.New(0)
	boundary, matched := tree.Lookup(prompt[:16])
	tree.Done(tree.Insert(boundary, prompt[matched:16], ctl.Cache))
	ctl.Close()
	s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"cancel": prompt, "survive": prompt}))
	// Isolate ownership from machine-dependent worker/capacity coupling.
	s.coupler = nil
	s.SetRadixKV(tree)
	if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
		t.Fatal(err)
	}
	nativeSchedulerBeginManualDrain(t, s)
	defer nativeSchedulerEndManualDrain(s)
	canceled := nativeSchedulerAdmitLane(t, s, "cancel")
	survivor := nativeSchedulerAdmitLane(t, s, "survive")
	nativeSchedulerDriveIteration(t, s)
	st := s.getPrefixState()
	st.mu.RLock()
	holders, lanes := len(st.holderLookups), len(st.laneLookups)
	st.mu.RUnlock()
	if holders != 2 || lanes != 2 {
		t.Fatalf("active ownership = %d holders/%d lanes, want 2/2", holders, lanes)
	}
	canceled.Cancel()
	for i := 0; i < 5 && !canceled.terminal; i++ {
		nativeSchedulerDriveIteration(t, s)
	}
	if !canceled.terminal {
		t.Fatal("canceled lane did not retire")
	}
	st.mu.RLock()
	_, oldLane := st.laneLookups[canceled]
	_, liveLane := st.laneLookups[survivor]
	holders, lanes = len(st.holderLookups), len(st.laneLookups)
	st.mu.RUnlock()
	if oldLane || !liveLane || holders != 1 || lanes != 1 {
		t.Fatalf("retirement changed wrong owner: old=%v live=%v holders=%d lanes=%d", oldLane, liveLane, holders, lanes)
	}
	if got := tree.Stats().ProtectedTokens; got != 16 {
		t.Fatalf("survivor's prefix lease lost: protected=%d", got)
	}
	prefixDedupDrainAll(t, s, survivor)
	if _, err := survivor.Result(); err != nil {
		t.Fatalf("surviving lane: %v", err)
	}
	st.mu.RLock()
	holders, lanes = len(st.holderLookups), len(st.laneLookups)
	st.mu.RUnlock()
	if holders != 0 || lanes != 0 {
		t.Fatalf("terminal metadata retained: holders=%d lanes=%d", holders, lanes)
	}
	if got := tree.Stats().ProtectedTokens; got != 0 {
		t.Fatalf("terminal prefix lease retained: %d", got)
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=2s lane=default
func TestNativeSchedulerLateAdmissionKeepsPrefixOwnerAfterClose(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(48)
	tree := radixkv.New(0)
	boundary, matched := tree.Lookup(prompt)
	tree.Done(tree.Insert(boundary, prompt[matched:], nil))
	s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"late": prompt}))
	s.SetRadixKV(tree)
	if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
		t.Fatal(err)
	}
	owner := s.getPrefixState()
	// Zero-value lazy initialization must be scheduler-local as well.
	var other NativeScheduler
	if other.getPrefixState() != nil || other.getOrCreatePrefixState() == owner {
		t.Fatal("prefix state is not scheduler-owned")
	}
	entered, release := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	s.SetSessionProfilerFactory(func(NativeSessionLifecycle) *model.PhaseProfiler { close(entered); <-release; return nil })
	type result struct {
		req abi.EngineRequest
		err error
	}
	done := make(chan result, 1)
	go func() { r, err := s.Admit(context.Background(), inlineCall("late", `{}`)); done <- result{r, err} }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("late admission did not acquire session")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	err := s.CloseAndWait(ctx)
	close(release)
	if err != nil {
		t.Fatalf("close before any lane publication: %v", err)
	}
	select {
	case got := <-done:
		if got.req != nil || got.err != errSchedClosed {
			t.Fatalf("late admission = (%v, %v)", got.req, got.err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("late admission did not release owner")
	}
	if s.getPrefixState() != owner {
		t.Fatal("Close detached the owner needed by the late admission")
	}
	if s.PrefixStats().Hits != 1 {
		t.Fatal("late admission did not exercise a real post-Close prefix lease")
	}
	owner.mu.RLock()
	holders, lanes := len(owner.holderLookups), len(owner.laneLookups)
	owner.mu.RUnlock()
	if holders != 0 || lanes != 0 {
		t.Fatalf("late owner retained: holders=%d lanes=%d", holders, lanes)
	}
	tree.SetRetention(0)
	if tree.Stats().Tokens != 0 {
		t.Fatal("late admission leaked radix lease")
	}
}
