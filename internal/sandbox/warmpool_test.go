package sandbox

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type warmFakeInstance struct {
	spec   Spec
	id     int
	mu     sync.Mutex
	closes int
}

func (w *warmFakeInstance) Execute(context.Context, ExecutionRequest) (ExecutionResult, error) {
	return ExecutionResult{}, nil
}
func (w *warmFakeInstance) Reset(context.Context) error { return nil }
func (w *warmFakeInstance) Spec() Spec                  { return w.spec }
func (w *warmFakeInstance) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closes++
	return nil
}
func (w *warmFakeInstance) closeCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.closes
}

// warmFakeProvisioner records every instance it hands out; fail decides per call index.
type warmFakeProvisioner struct {
	mu    sync.Mutex
	calls int
	made  []*warmFakeInstance
	fail  func(call int) (returnInstance bool, fail bool)
}

func (f *warmFakeProvisioner) provision(_ context.Context, spec Spec) (Instance, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	inst := &warmFakeInstance{spec: spec, id: f.calls}
	if f.fail != nil {
		if withInst, fail := f.fail(f.calls); fail {
			if withInst {
				f.made = append(f.made, inst)
				return inst, errors.New("provision half-failed")
			}
			return nil, errors.New("provision failed")
		}
	}
	f.made = append(f.made, inst)
	return inst, nil
}

type warmFakeClock struct{ t time.Time }

func (c *warmFakeClock) now() time.Time          { return c.t }
func (c *warmFakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func warmTestSpec(workspace string) Spec {
	return Spec{Tier: TierL1NativeOS, WorkspaceDir: workspace, EgressPolicy: EgressBlocked}
}

func newTestWarmPool(t *testing.T, cfg WarmPoolConfig) (*WarmPool, *warmFakeProvisioner, *warmFakeClock) {
	t.Helper()
	fp := &warmFakeProvisioner{}
	clk := &warmFakeClock{t: time.Unix(1000, 0)}
	if cfg.Template.Tier == "" {
		cfg.Template = warmTestSpec("/ws/a")
	}
	cfg.Provision = fp.provision
	cfg.Now = clk.now
	p, err := NewWarmPool(cfg)
	if err != nil {
		t.Fatalf("NewWarmPool: %v", err)
	}
	return p, fp, clk
}

func mustReconcile(t *testing.T, p *WarmPool) WarmPoolReconcileResult {
	t.Helper()
	res, err := p.Reconcile(context.Background())
	if err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	return res
}

func TestWarmPoolFillsToDesired(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 3})
	res := mustReconcile(t, p)
	if res.Created != 3 || fp.calls != 3 {
		t.Fatalf("created=%d calls=%d, want 3/3", res.Created, fp.calls)
	}
	if st := p.Status(); st.Ready != 3 || st.Stale != 0 || st.Desired != 3 {
		t.Fatalf("status=%+v", st)
	}
	if res := mustReconcile(t, p); res != (WarmPoolReconcileResult{}) || fp.calls != 3 {
		t.Fatalf("steady-state reconcile did work: %+v calls=%d", res, fp.calls)
	}
}

func TestWarmPoolClaimRemovesAndReconcileRefills(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 2})
	mustReconcile(t, p)

	got, err := p.Claim()
	if err != nil {
		t.Fatalf("Claim: %v", err)
	}
	if got != Instance(fp.made[0]) {
		t.Fatalf("claim did not return the oldest instance")
	}
	if st := p.Status(); st.Ready != 1 {
		t.Fatalf("ready after claim=%d, want 1", st.Ready)
	}
	if _, err := p.Claim(); err != nil {
		t.Fatalf("second Claim: %v", err)
	}
	if _, err := p.Claim(); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("empty pool claim err=%v, want %s", err, ErrSandboxUnavailable)
	}

	res := mustReconcile(t, p)
	if res.Created != 2 || p.Status().Ready != 2 {
		t.Fatalf("refill created=%d ready=%d", res.Created, p.Status().Ready)
	}
	if fp.made[0].closeCount() != 0 {
		t.Fatalf("claimed instance was closed by the pool")
	}
}

func TestWarmPoolTemplateHashChangeRecreatesStale(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 3})
	mustReconcile(t, p)
	oldHash := p.Status().TemplateHash
	old := append([]*warmFakeInstance(nil), fp.made...)

	if err := p.SetTemplate(warmTestSpec("/ws/b")); err != nil {
		t.Fatalf("SetTemplate: %v", err)
	}
	st := p.Status()
	if st.TemplateHash == oldHash || st.Stale != 3 || st.Ready != 0 {
		t.Fatalf("after template change status=%+v", st)
	}
	if _, err := p.Claim(); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("claim handed out a stale instance: err=%v", err)
	}

	res := mustReconcile(t, p)
	if res.StaleRemoved != 3 || res.Created != 3 {
		t.Fatalf("recreate result=%+v", res)
	}
	for _, inst := range old {
		if inst.closeCount() != 1 {
			t.Fatalf("stale instance %d closed %d times, want 1", inst.id, inst.closeCount())
		}
	}
	for _, inst := range fp.made[3:] {
		if inst.spec.WorkspaceDir != "/ws/b" {
			t.Fatalf("recreated instance built from old template: %+v", inst.spec)
		}
	}
	if st := p.Status(); st.Ready != 3 || st.Stale != 0 {
		t.Fatalf("after recreate status=%+v", st)
	}

	same, err := TemplateHash(warmTestSpec("/ws/b"))
	if err != nil || same != p.Status().TemplateHash {
		t.Fatalf("TemplateHash not stable: %q vs %q (%v)", same, p.Status().TemplateHash, err)
	}
}

func TestWarmPoolReplenishRateLimitRecoversWithClock(t *testing.T) {
	p, fp, clk := newTestWarmPool(t, WarmPoolConfig{Replicas: 5, MaxRefillRate: 2})

	res := mustReconcile(t, p)
	if res.Created != 2 || res.Paced != 3 || res.RequeueAfter != 500*time.Millisecond {
		t.Fatalf("first pass=%+v, want 2 created, 3 paced, 500ms requeue", res)
	}
	res = mustReconcile(t, p)
	if res.Created != 0 || res.Paced != 3 || res.RequeueAfter != 500*time.Millisecond {
		t.Fatalf("same-instant pass=%+v, want nothing created", res)
	}

	clk.advance(500 * time.Millisecond)
	if res = mustReconcile(t, p); res.Created != 1 {
		t.Fatalf("after one token interval created=%d, want 1", res.Created)
	}

	// A long idle period still banks at most one second of tokens.
	clk.advance(time.Hour)
	if res = mustReconcile(t, p); res.Created != 2 || res.Paced != 0 || res.RequeueAfter != 0 {
		t.Fatalf("after idle pass=%+v, want 2 created and none paced", res)
	}
	if fp.calls != 5 || p.Status().Ready != 5 {
		t.Fatalf("calls=%d ready=%d, want 5/5", fp.calls, p.Status().Ready)
	}

	if _, err := p.Claim(); err != nil {
		t.Fatalf("Claim: %v", err)
	}
	if _, err := p.Claim(); err != nil {
		t.Fatalf("Claim: %v", err)
	}
	if _, err := p.Claim(); err != nil {
		t.Fatalf("Claim: %v", err)
	}
	clk.advance(time.Minute)
	if res = mustReconcile(t, p); res.Created != 2 || res.Paced != 1 {
		t.Fatalf("post-claim burst=%+v, want capped at bucket capacity 2", res)
	}
}

func TestWarmPoolMaxBatchCapsCreates(t *testing.T) {
	p, _, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 5, MaxBatch: 2})
	if res := mustReconcile(t, p); res.Created != 2 {
		t.Fatalf("created=%d, want batch cap 2", res.Created)
	}
	mustReconcile(t, p)
	mustReconcile(t, p)
	if st := p.Status(); st.Ready != 5 {
		t.Fatalf("ready=%d, want 5", st.Ready)
	}
}

func TestWarmPoolProvisionFailureDoesNotLeakOrDoubleCount(t *testing.T) {
	p, fp, clk := newTestWarmPool(t, WarmPoolConfig{Replicas: 4, MaxRefillRate: 4})
	fp.fail = func(call int) (bool, bool) {
		switch call {
		case 2:
			return false, true
		case 3:
			return true, true
		}
		return false, false
	}

	res, err := p.Reconcile(context.Background())
	if err == nil {
		t.Fatalf("Reconcile hid provision failures")
	}
	if res.Created != 2 || res.Failed != 2 {
		t.Fatalf("result=%+v, want 2 created 2 failed", res)
	}
	if st := p.Status(); st.Ready != 2 {
		t.Fatalf("ready=%d, want only successful creates counted", st.Ready)
	}
	for _, inst := range fp.made {
		if inst.id == 3 && inst.closeCount() != 1 {
			t.Fatalf("half-failed instance closed %d times, want 1", inst.closeCount())
		}
	}

	// Failed attempts spent their tokens, so an immediate retry is paced.
	if res = mustReconcile(t, p); res.Created != 0 || res.Paced != 2 {
		t.Fatalf("retry at same instant=%+v, want fully paced", res)
	}
	clk.advance(time.Second)
	if res = mustReconcile(t, p); res.Created != 2 {
		t.Fatalf("recovery created=%d, want 2", res.Created)
	}
	if st := p.Status(); st.Ready != 4 {
		t.Fatalf("ready=%d, want 4", st.Ready)
	}
}

func TestWarmPoolScaleDownDeletesNewestSurplus(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 5, MaxBatch: 2})
	for range 3 {
		mustReconcile(t, p)
	}
	if st := p.Status(); st.Ready != 5 {
		t.Fatalf("fill ready=%d, want 5", st.Ready)
	}
	if err := p.SetReplicas(1); err != nil {
		t.Fatalf("SetReplicas: %v", err)
	}

	if res := mustReconcile(t, p); res.Deleted != 2 || res.Created != 0 {
		t.Fatalf("first scale-down pass=%+v, want 2 deleted", res)
	}
	if fp.made[4].closeCount() != 1 || fp.made[3].closeCount() != 1 || fp.made[2].closeCount() != 0 {
		t.Fatalf("scale-down did not delete newest first")
	}
	if res := mustReconcile(t, p); res.Deleted != 2 {
		t.Fatalf("second scale-down pass=%+v, want 2 deleted", res)
	}
	if st := p.Status(); st.Ready != 1 {
		t.Fatalf("ready=%d, want 1", st.Ready)
	}
	got, err := p.Claim()
	if err != nil || got != Instance(fp.made[0]) {
		t.Fatalf("survivor is not the oldest instance: %v", err)
	}
	for _, inst := range fp.made {
		if inst.closeCount() > 1 {
			t.Fatalf("instance %d closed %d times", inst.id, inst.closeCount())
		}
	}
}

func TestWarmPoolTemplateChangeDuringProvisionDiscardsInstance(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 2})
	fp.fail = func(call int) (bool, bool) {
		if call == 1 {
			if err := p.SetTemplate(warmTestSpec("/ws/c")); err != nil {
				t.Errorf("SetTemplate: %v", err)
			}
		}
		return false, false
	}
	if _, err := p.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if fp.made[0].closeCount() != 1 || fp.made[1].closeCount() != 1 {
		t.Fatalf("instances built from a superseded template were kept")
	}
	if st := p.Status(); st.Ready != 0 || st.Stale != 0 {
		t.Fatalf("status=%+v, want empty", st)
	}
	fp.fail = nil
	mustReconcile(t, p)
	if st := p.Status(); st.Ready != 2 {
		t.Fatalf("ready=%d, want 2", st.Ready)
	}
}

func TestWarmPoolRegistryProvisionerResolvesTier(t *testing.T) {
	reg := NewRegistry()
	reg.RegisterProvider(&mockProvider{tier: TierL1NativeOS, available: true})
	p, err := NewWarmPool(WarmPoolConfig{
		Template:  warmTestSpec("/ws/r"),
		Replicas:  2,
		Provision: RegistryProvisioner(reg),
	})
	if err != nil {
		t.Fatalf("NewWarmPool: %v", err)
	}
	if res, err := p.Reconcile(context.Background()); err != nil || res.Created != 2 {
		t.Fatalf("Reconcile res=%+v err=%v", res, err)
	}
	inst, err := p.Claim()
	if err != nil || inst.Spec().WorkspaceDir != "/ws/r" {
		t.Fatalf("Claim inst=%v err=%v", inst, err)
	}

	if err := p.SetTemplate(Spec{Tier: TierL2Virtual, WorkspaceDir: "/ws/r", EgressPolicy: EgressBlocked}); err != nil {
		t.Fatalf("SetTemplate: %v", err)
	}
	res, err := p.Reconcile(context.Background())
	if !IsSandboxError(err, ErrSandboxUnavailable) || res.Failed != 2 || res.StaleRemoved != 1 {
		t.Fatalf("unresolvable tier res=%+v err=%v", res, err)
	}
	if st := p.Status(); st.Ready != 0 || st.Stale != 0 {
		t.Fatalf("status=%+v, want empty", st)
	}
}

func TestWarmPoolConcurrentClaimsAreDistinct(t *testing.T) {
	const n = 16
	p, _, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: n})
	mustReconcile(t, p)

	got := make([]Instance, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			inst, err := p.Claim()
			if err != nil {
				t.Errorf("Claim: %v", err)
				return
			}
			got[i] = inst
		}(i)
	}
	wg.Wait()

	seen := map[Instance]bool{}
	for _, inst := range got {
		if inst == nil || seen[inst] {
			t.Fatalf("claims were not distinct")
		}
		seen[inst] = true
	}
	if st := p.Status(); st.Ready != 0 {
		t.Fatalf("ready=%d, want 0", st.Ready)
	}
}

func TestWarmPoolCloseClosesMembersAndRefuses(t *testing.T) {
	p, fp, _ := newTestWarmPool(t, WarmPoolConfig{Replicas: 2})
	mustReconcile(t, p)
	if err := p.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	for _, inst := range fp.made {
		if inst.closeCount() != 1 {
			t.Fatalf("instance %d closed %d times", inst.id, inst.closeCount())
		}
	}
	if _, err := p.Claim(); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("Claim after Close err=%v", err)
	}
	if _, err := p.Reconcile(context.Background()); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("Reconcile after Close err=%v", err)
	}
	if err := p.SetReplicas(3); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("SetReplicas after Close err=%v", err)
	}
	if err := p.SetTemplate(warmTestSpec("/ws/b")); !IsSandboxError(err, ErrSandboxUnavailable) {
		t.Fatalf("SetTemplate after Close err=%v", err)
	}
}

func TestWarmPoolRejectsInvalidConfig(t *testing.T) {
	ok := func(context.Context, Spec) (Instance, error) { return nil, nil }
	cases := []WarmPoolConfig{
		{Template: warmTestSpec("/ws"), Replicas: 1},
		{Template: warmTestSpec("/ws"), Replicas: -1, Provision: ok},
		{Template: warmTestSpec("/ws"), Replicas: 1, MaxRefillRate: -1, Provision: ok},
		{Template: Spec{Tier: "bogus"}, Replicas: 1, Provision: ok},
	}
	for i, cfg := range cases {
		if _, err := NewWarmPool(cfg); err == nil {
			t.Fatalf("case %d: invalid config accepted", i)
		}
	}
}
