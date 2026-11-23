package sandbox

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

type lifecycleFakeRuntime struct {
	mu    sync.Mutex
	calls []string
	fail  map[LifecycleOp]error
	spec  Spec
	prov  string
	// hook runs outside the manager lock before the call is recorded.
	hook map[LifecycleOp]func(id string)
}

func (f *lifecycleFakeRuntime) record(op LifecycleOp, id string) error {
	if h := f.hook[op]; h != nil {
		h(id)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, string(op)+":"+id)
	return f.fail[op]
}

func (f *lifecycleFakeRuntime) Create(_ context.Context, id string, p Provider, spec Spec) error {
	f.mu.Lock()
	f.spec, f.prov = spec, p.Name()
	f.mu.Unlock()
	return f.record(LifecycleOpCreate, id)
}
func (f *lifecycleFakeRuntime) Pause(_ context.Context, id string) error {
	return f.record(LifecycleOpPause, id)
}
func (f *lifecycleFakeRuntime) Resume(_ context.Context, id string) error {
	return f.record(LifecycleOpResume, id)
}
func (f *lifecycleFakeRuntime) Snapshot(_ context.Context, id, _ string) error {
	return f.record(LifecycleOpSnapshot, id)
}
func (f *lifecycleFakeRuntime) Terminate(_ context.Context, id string) error {
	return f.record(LifecycleOpDelete, id)
}

func (f *lifecycleFakeRuntime) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

type lifecycleFakeClock struct{ now time.Time }

func (c *lifecycleFakeClock) Now() time.Time { return c.now }

var lifecycleEpoch = time.Date(2030, 1, 2, 3, 4, 5, 0, time.UTC)

type lifecycleHarness struct {
	m     *LifecycleManager
	rt    *lifecycleFakeRuntime
	clock *lifecycleFakeClock
	reg   *Registry
}

func newLifecycleHarness(t *testing.T, mutate func(*LifecycleConfig)) *lifecycleHarness {
	t.Helper()
	reg := NewRegistry()
	reg.RegisterProvider(&mockProvider{tier: TierL1NativeOS, available: true})
	reg.RegisterProvider(&mockProvider{tier: TierL2Virtual, available: true})
	h := &lifecycleHarness{rt: &lifecycleFakeRuntime{fail: map[LifecycleOp]error{}, hook: map[LifecycleOp]func(string){}}, clock: &lifecycleFakeClock{now: lifecycleEpoch}, reg: reg}
	cfg := LifecycleConfig{Registry: reg, Runtime: h.rt, Now: h.clock.Now}
	if mutate != nil {
		mutate(&cfg)
	}
	m, err := NewLifecycleManager(cfg)
	if err != nil {
		t.Fatalf("NewLifecycleManager: %v", err)
	}
	h.m = m
	return h
}

func lifecycleSpec(tier Tier) Spec {
	return Spec{Tier: tier, WorkspaceDir: "/workspace", EgressPolicy: EgressBlocked}
}

func (h *lifecycleHarness) create(t *testing.T, timeout time.Duration, md map[string]string) LifecycleSandbox {
	t.Helper()
	sb, err := h.m.Create(context.Background(), LifecycleCreateRequest{Spec: lifecycleSpec(TierL1NativeOS), Timeout: timeout, Metadata: md})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	return sb
}

func (h *lifecycleHarness) force(id string, s LifecycleState) {
	h.m.mu.Lock()
	h.m.records[id].view.State = s
	h.m.mu.Unlock()
}

func (h *lifecycleHarness) state(t *testing.T, id string) LifecycleState {
	t.Helper()
	sb, err := h.m.Get(id)
	if err != nil {
		t.Fatalf("Get(%s): %v", id, err)
	}
	return sb.State
}

func TestSandboxLifecycleTransitionTableMatchesSpec(t *testing.T) {
	want := map[[2]LifecycleState]bool{
		{LifecyclePending, LifecycleRunning}:     true,
		{LifecycleRunning, LifecyclePausing}:     true,
		{LifecyclePausing, LifecyclePaused}:      true,
		{LifecyclePaused, LifecycleResuming}:     true,
		{LifecycleResuming, LifecycleRunning}:    true,
		{LifecycleRunning, LifecycleStopping}:    true,
		{LifecyclePaused, LifecycleStopping}:     true,
		{LifecycleStopping, LifecycleTerminated}: true,
		{LifecyclePending, LifecycleFailed}:      true,
		{LifecycleRunning, LifecycleFailed}:      true,
		{LifecyclePaused, LifecycleFailed}:       true,
		{LifecycleResuming, LifecycleFailed}:     true,
	}
	for _, from := range LifecycleStates {
		for _, to := range LifecycleStates {
			if got := LifecycleCanTransition(from, to); got != want[[2]LifecycleState{from, to}] {
				t.Errorf("LifecycleCanTransition(%s, %s) = %v, want %v", from, to, got, !got)
			}
		}
	}
}

func TestSandboxLifecycleOperationMatrix(t *testing.T) {
	type opFn func(h *lifecycleHarness, id string) error
	ops := map[LifecycleOp]opFn{
		LifecycleOpPause: func(h *lifecycleHarness, id string) error {
			_, err := h.m.Pause(context.Background(), id)
			return err
		},
		LifecycleOpResume: func(h *lifecycleHarness, id string) error {
			_, err := h.m.Resume(context.Background(), id)
			return err
		},
		LifecycleOpSnapshot: func(h *lifecycleHarness, id string) error {
			_, err := h.m.Snapshot(context.Background(), id, "s")
			return err
		},
		LifecycleOpRenew: func(h *lifecycleHarness, id string) error {
			_, err := h.m.Renew(id, lifecycleEpoch.Add(time.Hour))
			return err
		},
		LifecycleOpDelete: func(h *lifecycleHarness, id string) error {
			return h.m.Delete(context.Background(), id)
		},
	}
	legal := map[LifecycleOp][]LifecycleState{
		LifecycleOpPause:    {LifecycleRunning},
		LifecycleOpResume:   {LifecyclePaused},
		LifecycleOpSnapshot: {LifecycleRunning},
		LifecycleOpRenew:    {LifecyclePending, LifecycleRunning, LifecyclePausing, LifecyclePaused, LifecycleResuming},
		LifecycleOpDelete:   {LifecycleRunning, LifecyclePaused, LifecycleTerminated, LifecycleFailed},
	}
	for op, fn := range ops {
		for _, st := range LifecycleStates {
			wantOK := false
			for _, s := range legal[op] {
				wantOK = wantOK || s == st
			}
			h := newLifecycleHarness(t, nil)
			id := h.create(t, 0, nil).ID
			h.force(id, st)
			before := h.rt.callCount()
			err := fn(h, id)
			if wantOK && err != nil {
				t.Errorf("%s from %s: unexpected error %v", op, st, err)
			}
			if !wantOK {
				var te *LifecycleTransitionError
				if !errors.Is(err, ErrLifecycleIllegalTransition) || !errors.As(err, &te) || te.From != st || te.Op != op {
					t.Errorf("%s from %s: err = %v, want illegal transition from %s", op, st, err, st)
				}
				if got := h.rt.callCount(); got != before {
					t.Errorf("%s from %s: refused op reached the runtime", op, st)
				}
				if got := h.state(t, id); got != st {
					t.Errorf("%s from %s: refused op changed state to %s", op, st, got)
				}
			}
		}
	}
}

func TestSandboxLifecyclePauseResumeRoundTrip(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	id := h.create(t, 0, nil).ID
	sb, err := h.m.Pause(context.Background(), id)
	if err != nil || sb.State != LifecyclePaused {
		t.Fatalf("Pause = %v, %v; want Paused", sb.State, err)
	}
	_, err = h.m.Pause(context.Background(), id)
	var te *LifecycleTransitionError
	if !errors.As(err, &te) || te.From != LifecyclePaused || te.To != LifecyclePausing {
		t.Fatalf("second Pause err = %v, want Paused -> Pausing refusal", err)
	}
	sb, err = h.m.Resume(context.Background(), id)
	if err != nil || sb.State != LifecycleRunning {
		t.Fatalf("Resume = %v, %v; want Running", sb.State, err)
	}
	want := []string{"create:" + id, "pause:" + id, "resume:" + id}
	if !reflect.DeepEqual(h.rt.calls, want) {
		t.Fatalf("runtime calls = %v, want %v", h.rt.calls, want)
	}
}

func TestSandboxLifecycleSnapshotRecordsDescriptor(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	id := h.create(t, 0, nil).ID
	snap, err := h.m.Snapshot(context.Background(), id, "before-import")
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if snap.SandboxID != id || snap.Name != "before-import" || snap.State != LifecycleSnapshotReady || !snap.CreatedAt.Equal(lifecycleEpoch) {
		t.Fatalf("snapshot = %+v", snap)
	}
	if got := h.state(t, id); got != LifecycleRunning {
		t.Fatalf("source state after snapshot = %s, want Running", got)
	}
	if _, err := h.m.Pause(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if _, err := h.m.Snapshot(context.Background(), id, "paused"); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Fatalf("snapshot of Paused err = %v, want illegal", err)
	}
	if err := h.m.Delete(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if got := h.m.Snapshots(id); len(got) != 1 || got[0].ID != snap.ID {
		t.Fatalf("snapshots after source delete = %+v, want the one recorded snapshot", got)
	}
}

func TestSandboxLifecycleRenewExtendsAndSurvivesSweep(t *testing.T) {
	h := newLifecycleHarness(t, func(c *LifecycleConfig) { c.MaxTimeout = 2 * time.Minute })
	keep := h.create(t, time.Minute, nil).ID
	lose := h.create(t, time.Minute, nil).ID

	far := lifecycleEpoch.Add(time.Hour)
	sb, err := h.m.Renew(keep, far)
	if err != nil || !sb.ExpiresAt.Equal(far) {
		t.Fatalf("Renew = %v, %v; want expiry %v beyond MaxTimeout", sb.ExpiresAt, err, far)
	}
	if _, err := h.m.Renew(keep, lifecycleEpoch); !errors.Is(err, ErrLifecycleInvalidRequest) {
		t.Fatalf("Renew to now err = %v, want invalid request", err)
	}
	if _, err := h.m.Renew(keep, lifecycleEpoch.Add(-time.Second)); !errors.Is(err, ErrLifecycleInvalidRequest) {
		t.Fatalf("Renew to past err = %v, want invalid request", err)
	}
	if got, _ := h.m.Get(keep); !got.ExpiresAt.Equal(far) {
		t.Fatalf("rejected renew changed expiry to %v", got.ExpiresAt)
	}

	expired, err := h.m.Sweep(context.Background(), lifecycleEpoch.Add(10*time.Minute))
	if err != nil || !reflect.DeepEqual(expired, []string{lose}) {
		t.Fatalf("Sweep = %v, %v; want only %s", expired, err, lose)
	}
	if got := h.state(t, keep); got != LifecycleRunning {
		t.Fatalf("renewed sandbox state = %s, want Running", got)
	}

	shorter := lifecycleEpoch.Add(90 * time.Second)
	if sb, err := h.m.Renew(keep, shorter); err != nil || !sb.ExpiresAt.Equal(shorter) {
		t.Fatalf("shortening Renew = %v, %v", sb.ExpiresAt, err)
	}
}

func TestSandboxLifecycleSweepBoundaryInstant(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	running := h.create(t, time.Minute, nil).ID
	paused := h.create(t, time.Minute, nil).ID
	forever := h.create(t, 0, nil).ID
	if _, err := h.m.Pause(context.Background(), paused); err != nil {
		t.Fatal(err)
	}
	deadline := lifecycleEpoch.Add(time.Minute)

	if got, err := h.m.Sweep(context.Background(), deadline.Add(-time.Nanosecond)); err != nil || len(got) != 0 {
		t.Fatalf("sweep before deadline = %v, %v; want none", got, err)
	}
	got, err := h.m.Sweep(context.Background(), deadline)
	if err != nil || !reflect.DeepEqual(got, []string{running, paused}) {
		t.Fatalf("sweep at deadline = %v, %v; want [%s %s]", got, err, running, paused)
	}
	for _, id := range []string{running, paused} {
		if st := h.state(t, id); st != LifecycleTerminated {
			t.Fatalf("%s state = %s, want Terminated", id, st)
		}
	}
	if st := h.state(t, forever); st != LifecycleRunning {
		t.Fatalf("no-timeout sandbox state = %s, want Running", st)
	}
	if again, err := h.m.Sweep(context.Background(), deadline.Add(time.Hour)); err != nil || len(again) != 0 {
		t.Fatalf("second sweep = %v, %v; want none", again, err)
	}
}

func TestSandboxLifecycleExpiredAndDeletedRefuse(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	expired := h.create(t, time.Minute, nil).ID
	deleted := h.create(t, 0, nil).ID
	if _, err := h.m.Sweep(context.Background(), lifecycleEpoch.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := h.m.Delete(context.Background(), deleted); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	if _, err := h.m.Resume(ctx, expired); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("resume expired err = %v", err)
	}
	if _, err := h.m.Renew(expired, lifecycleEpoch.Add(time.Hour)); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("renew expired err = %v", err)
	}
	if _, err := h.m.Snapshot(ctx, expired, ""); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("snapshot expired err = %v", err)
	}

	if _, err := h.m.Get(deleted); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("get deleted err = %v", err)
	}
	if _, err := h.m.Pause(ctx, deleted); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("pause deleted err = %v", err)
	}
	if _, err := h.m.Renew(deleted, lifecycleEpoch.Add(time.Hour)); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("renew deleted err = %v", err)
	}
	if err := h.m.Delete(ctx, deleted); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("delete deleted err = %v", err)
	}
	if _, err := h.m.Snapshot(ctx, "sbx-missing", ""); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("snapshot unknown err = %v", err)
	}

	if err := h.m.Delete(ctx, expired); err != nil {
		t.Fatalf("delete of Terminated sandbox: %v", err)
	}
	if _, err := h.m.Get(expired); !errors.Is(err, ErrLifecycleNotFound) {
		t.Errorf("get after deleting expired err = %v", err)
	}
}

func TestSandboxLifecycleTierResolution(t *testing.T) {
	t.Run("requested tier resolves through registry", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		sb := h.create(t, 0, nil)
		if sb.Tier != TierL1NativeOS || sb.Provider != "mock-l1_native_os" || h.rt.prov != sb.Provider || h.rt.spec.Tier != TierL1NativeOS {
			t.Fatalf("sandbox = %+v, runtime saw %s/%s", sb, h.rt.prov, h.rt.spec.Tier)
		}
	})
	t.Run("secure tier raises a weaker request", func(t *testing.T) {
		h := newLifecycleHarness(t, func(c *LifecycleConfig) { c.SecureTier = TierL2Virtual })
		sb := h.create(t, 0, nil)
		if sb.Tier != TierL2Virtual || h.rt.spec.Tier != TierL2Virtual || sb.Provider != "mock-l2_virtual" {
			t.Fatalf("secure tier not applied: %+v", sb)
		}
	})
	t.Run("secure tier does not lower a stronger request", func(t *testing.T) {
		h := newLifecycleHarness(t, func(c *LifecycleConfig) { c.SecureTier = TierL1NativeOS })
		sb, err := h.m.Create(context.Background(), LifecycleCreateRequest{Spec: lifecycleSpec(TierL2Virtual)})
		if err != nil || sb.Tier != TierL2Virtual {
			t.Fatalf("Create = %+v, %v; want l2", sb, err)
		}
	})
	t.Run("unresolvable secure tier refuses without a record", func(t *testing.T) {
		h := newLifecycleHarness(t, func(c *LifecycleConfig) { c.SecureTier = TierL2Virtual })
		h.reg.RegisterProvider(&mockProvider{tier: TierL2Virtual, available: false})
		_, err := h.m.Create(context.Background(), LifecycleCreateRequest{Spec: lifecycleSpec(TierL1NativeOS)})
		if !IsSandboxError(err, ErrSandboxUnavailable) {
			t.Fatalf("Create err = %v, want %s", err, ErrSandboxUnavailable)
		}
		if n := len(h.m.List(LifecycleListFilter{})); n != 0 || h.rt.callCount() != 0 {
			t.Fatalf("refused create left %d records and %d runtime calls", n, h.rt.callCount())
		}
	})
	t.Run("unregistered tier refuses", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		_, err := h.m.Create(context.Background(), LifecycleCreateRequest{Spec: lifecycleSpec(TierL0Wasm)})
		if !IsSandboxError(err, ErrSandboxUnavailable) {
			t.Fatalf("Create err = %v, want %s", err, ErrSandboxUnavailable)
		}
	})
	t.Run("timeout bounds", func(t *testing.T) {
		h := newLifecycleHarness(t, func(c *LifecycleConfig) { c.MaxTimeout = time.Hour })
		for _, d := range []time.Duration{-time.Second, 59 * time.Second, 2 * time.Hour} {
			if _, err := h.m.Create(context.Background(), LifecycleCreateRequest{Spec: lifecycleSpec(TierL1NativeOS), Timeout: d}); !errors.Is(err, ErrLifecycleInvalidRequest) {
				t.Errorf("timeout %s err = %v, want invalid request", d, err)
			}
		}
		sb := h.create(t, LifecycleMinTimeout, nil)
		if !sb.ExpiresAt.Equal(lifecycleEpoch.Add(LifecycleMinTimeout)) {
			t.Fatalf("ExpiresAt = %v", sb.ExpiresAt)
		}
	})
}

func TestSandboxLifecycleListFilter(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	a := h.create(t, 0, map[string]string{"project": "apollo", "team": "x"}).ID
	b := h.create(t, 0, map[string]string{"project": "apollo"}).ID
	c := h.create(t, 0, map[string]string{"project": "gemini"}).ID
	if _, err := h.m.Pause(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	ids := func(f LifecycleListFilter) []string {
		var out []string
		for _, s := range h.m.List(f) {
			out = append(out, s.ID)
		}
		return out
	}
	cases := []struct {
		name string
		f    LifecycleListFilter
		want []string
	}{
		{"all", LifecycleListFilter{}, []string{a, b, c}},
		{"running", LifecycleListFilter{States: []LifecycleState{LifecycleRunning}}, []string{a, c}},
		{"running or paused", LifecycleListFilter{States: []LifecycleState{LifecycleRunning, LifecyclePaused}}, []string{a, b, c}},
		{"metadata", LifecycleListFilter{Metadata: map[string]string{"project": "apollo"}}, []string{a, b}},
		{"metadata and state", LifecycleListFilter{States: []LifecycleState{LifecyclePaused}, Metadata: map[string]string{"project": "apollo"}}, []string{b}},
		{"metadata all pairs", LifecycleListFilter{Metadata: map[string]string{"project": "apollo", "team": "x"}}, []string{a}},
		{"no match", LifecycleListFilter{States: []LifecycleState{LifecycleFailed}}, nil},
	}
	for _, tc := range cases {
		if got := ids(tc.f); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
	got := h.m.List(LifecycleListFilter{})
	got[0].Metadata["project"] = "mutated"
	if again, _ := h.m.Get(a); again.Metadata["project"] != "apollo" {
		t.Fatal("List exposed internal metadata map")
	}
}

func TestSandboxLifecycleRuntimeFailure(t *testing.T) {
	boom := errors.New("runtime boom")
	ctx := context.Background()

	t.Run("create failure lands in Failed", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		h.rt.fail[LifecycleOpCreate] = boom
		sb, err := h.m.Create(ctx, LifecycleCreateRequest{Spec: lifecycleSpec(TierL1NativeOS)})
		if !errors.Is(err, boom) || sb.State != LifecycleFailed || h.state(t, sb.ID) != LifecycleFailed {
			t.Fatalf("Create = %+v, %v; want Failed", sb, err)
		}
		if err := h.m.Delete(ctx, sb.ID); err != nil {
			t.Fatalf("delete Failed sandbox: %v", err)
		}
	})
	t.Run("resume failure lands in Failed", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		id := h.create(t, 0, nil).ID
		if _, err := h.m.Pause(ctx, id); err != nil {
			t.Fatal(err)
		}
		h.rt.fail[LifecycleOpResume] = boom
		if _, err := h.m.Resume(ctx, id); !errors.Is(err, boom) {
			t.Fatalf("Resume err = %v", err)
		}
		if st := h.state(t, id); st != LifecycleFailed {
			t.Fatalf("state = %s, want Failed", st)
		}
	})
	t.Run("pause failure restores Running", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		id := h.create(t, 0, nil).ID
		h.rt.fail[LifecycleOpPause] = boom
		if _, err := h.m.Pause(ctx, id); !errors.Is(err, boom) {
			t.Fatalf("Pause err = %v", err)
		}
		if st := h.state(t, id); st != LifecycleRunning {
			t.Fatalf("state = %s, want Running", st)
		}
	})
	t.Run("delete failure restores state and keeps record", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		id := h.create(t, 0, nil).ID
		if _, err := h.m.Pause(ctx, id); err != nil {
			t.Fatal(err)
		}
		h.rt.fail[LifecycleOpDelete] = boom
		if err := h.m.Delete(ctx, id); !errors.Is(err, boom) {
			t.Fatalf("Delete err = %v", err)
		}
		if st := h.state(t, id); st != LifecyclePaused {
			t.Fatalf("state = %s, want Paused", st)
		}
		delete(h.rt.fail, LifecycleOpDelete)
		if err := h.m.Delete(ctx, id); err != nil {
			t.Fatalf("retry Delete: %v", err)
		}
	})
	t.Run("sweep failure keeps sandbox for next sweep", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		id := h.create(t, time.Minute, nil).ID
		h.rt.fail[LifecycleOpDelete] = boom
		got, err := h.m.Sweep(ctx, lifecycleEpoch.Add(time.Minute))
		if !errors.Is(err, boom) || len(got) != 0 || h.state(t, id) != LifecycleRunning {
			t.Fatalf("Sweep = %v, %v, state %s", got, err, h.state(t, id))
		}
		delete(h.rt.fail, LifecycleOpDelete)
		if got, err := h.m.Sweep(ctx, lifecycleEpoch.Add(time.Minute)); err != nil || len(got) != 1 {
			t.Fatalf("retry Sweep = %v, %v", got, err)
		}
	})
	t.Run("snapshot failure records Failed descriptor", func(t *testing.T) {
		h := newLifecycleHarness(t, nil)
		id := h.create(t, 0, nil).ID
		h.rt.fail[LifecycleOpSnapshot] = boom
		snap, err := h.m.Snapshot(ctx, id, "x")
		if !errors.Is(err, boom) || snap.State != LifecycleSnapshotFailed || h.state(t, id) != LifecycleRunning {
			t.Fatalf("Snapshot = %+v, %v", snap, err)
		}
	})
}

func TestSandboxLifecycleConfigValidation(t *testing.T) {
	rt := &lifecycleFakeRuntime{}
	bad := []LifecycleConfig{
		{Runtime: rt},
		{Registry: NewRegistry()},
		{Registry: NewRegistry(), Runtime: rt, SecureTier: Tier("l9")},
		{Registry: NewRegistry(), Runtime: rt, MaxTimeout: -time.Second},
	}
	for i, cfg := range bad {
		if _, err := NewLifecycleManager(cfg); !errors.Is(err, ErrLifecycleInvalidRequest) {
			t.Errorf("config %d err = %v, want invalid request", i, err)
		}
	}
}

// lifecycleBlock installs a hook on op that signals entry and waits for release.
func lifecycleBlock(h *lifecycleHarness, op LifecycleOp) (entered <-chan struct{}, release chan<- struct{}) {
	in, out := make(chan struct{}), make(chan struct{})
	h.rt.hook[op] = func(string) {
		close(in)
		<-out
	}
	return in, out
}

func TestSandboxLifecycleConcurrentOpsSerialize(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	id := h.create(t, 0, nil).ID
	entered, release := lifecycleBlock(h, LifecycleOpPause)
	ctx := context.Background()

	var wg sync.WaitGroup
	var pauseErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, pauseErr = h.m.Pause(ctx, id)
	}()
	<-entered
	if st := h.state(t, id); st != LifecyclePausing {
		t.Errorf("state during pause = %s, want Pausing", st)
	}
	before := h.rt.callCount()
	refused := map[string]error{
		"delete": h.m.Delete(ctx, id),
	}
	_, refused["resume"] = h.m.Resume(ctx, id)
	_, refused["pause"] = h.m.Pause(ctx, id)
	for name, err := range refused {
		var te *LifecycleTransitionError
		if !errors.As(err, &te) || te.From != LifecyclePausing {
			t.Errorf("%s during pause err = %v, want illegal transition from Pausing", name, err)
		}
	}
	if got := h.rt.callCount(); got != before {
		t.Errorf("refused ops reached the runtime: %d calls, want %d", got, before)
	}
	close(release)
	wg.Wait()
	if pauseErr != nil {
		t.Fatalf("Pause: %v", pauseErr)
	}
	if st := h.state(t, id); st != LifecyclePaused {
		t.Fatalf("final state = %s, want Paused", st)
	}
}

func TestSandboxLifecycleStateChangedDuringRuntimeCall(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	id := h.create(t, 0, nil).ID
	h.rt.hook[LifecycleOpPause] = func(id string) { h.force(id, LifecycleFailed) }
	_, err := h.m.Pause(context.Background(), id)
	var te *LifecycleTransitionError
	if !errors.Is(err, ErrLifecycleIllegalTransition) || !errors.As(err, &te) || te.From != LifecycleFailed {
		t.Fatalf("Pause err = %v, want illegal transition from Failed", err)
	}
	if st := h.state(t, id); st != LifecycleFailed {
		t.Fatalf("state = %s, want Failed", st)
	}
}

func TestSandboxLifecycleSnapshotHoldsSourceRunning(t *testing.T) {
	h := newLifecycleHarness(t, nil)
	id := h.create(t, time.Minute, nil).ID
	entered, release := lifecycleBlock(h, LifecycleOpSnapshot)
	ctx := context.Background()

	var wg sync.WaitGroup
	var snap LifecycleSnapshot
	var snapErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		snap, snapErr = h.m.Snapshot(ctx, id, "held")
	}()
	<-entered
	before := h.rt.callCount()
	if err := h.m.Delete(ctx, id); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("Delete during snapshot err = %v, want illegal", err)
	}
	if _, err := h.m.Pause(ctx, id); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("Pause during snapshot err = %v, want illegal", err)
	}
	if _, err := h.m.Snapshot(ctx, id, "second"); !errors.Is(err, ErrLifecycleIllegalTransition) {
		t.Errorf("second Snapshot during snapshot err = %v, want illegal", err)
	}
	if got, err := h.m.Sweep(ctx, lifecycleEpoch.Add(time.Minute)); !errors.Is(err, ErrLifecycleIllegalTransition) || len(got) != 0 {
		t.Errorf("Sweep during snapshot = %v, %v; want refusal", got, err)
	}
	if got := h.rt.callCount(); got != before {
		t.Errorf("refused ops reached the runtime: %d calls, want %d", got, before)
	}
	if st := h.state(t, id); st != LifecycleRunning {
		t.Errorf("state during snapshot = %s, want Running", st)
	}
	close(release)
	wg.Wait()
	if snapErr != nil || snap.State != LifecycleSnapshotReady {
		t.Fatalf("Snapshot = %+v, %v; want Ready", snap, snapErr)
	}
	if err := h.m.Delete(ctx, id); err != nil {
		t.Fatalf("Delete after snapshot: %v", err)
	}
}
