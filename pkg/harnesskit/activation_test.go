package harnesskit

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestActivateRefusesPendingProviderAndUnwindsOwnerScope(t *testing.T) {
	lock := activationLock(t, "audit-sink", "search-provider")
	closed := []string{}
	audit := &activationTestRuntime{id: "audit-sink", status: RuntimeReadiness{State: StateRunning}, closed: &closed}
	search := &activationTestRuntime{id: "search-provider", status: RuntimeReadiness{State: StateStarting, MissingServices: []string{"search.rpc"}}, closed: &closed}

	active, err := Activate(context.Background(), lock, activationTestServices{}, map[string]Factory{
		"audit-sink":      activationTestFactory{id: "audit-sink", runtime: audit},
		"search-provider": activationTestFactory{id: "search-provider", runtime: search},
	})
	if active != nil {
		t.Fatal("pending extension returned a ready activation")
	}
	var activationErr *ActivationError
	if !errors.As(err, &activationErr) {
		t.Fatalf("error type = %T, want *ActivationError: %v", err, err)
	}
	if got := err.Error(); !strings.Contains(got, "search-provider") || !strings.Contains(got, "search.rpc") || !strings.Contains(got, "pending") {
		t.Fatalf("error lost pending dependency evidence: %v", err)
	}
	rows := activationErr.Snapshot()
	if len(rows) != 2 || rows[0].ExtensionID != "audit-sink" || rows[0].State != StateRunning || rows[1].ExtensionID != "search-provider" || rows[1].State != StateStarting {
		t.Fatalf("snapshot = %+v", rows)
	}
	if audit.closeCalls != 1 || search.closeCalls != 1 || strings.Join(closed, ",") != "search-provider,audit-sink" {
		t.Fatalf("owner scope was not unwound in reverse: closed=%v audit=%d search=%d", closed, audit.closeCalls, search.closeCalls)
	}
}

func TestActivatePreservesOriginalFailureAndRefusesAbsentFactory(t *testing.T) {
	t.Run("failed runtime", func(t *testing.T) {
		lock := activationLock(t, "broken-provider", "sibling")
		original := errors.New("provider init exploded")
		broken := &activationTestRuntime{id: "broken-provider", status: RuntimeReadiness{State: StateFailed, Err: original}}
		sibling := &activationTestRuntime{id: "sibling", status: RuntimeReadiness{State: StateRunning}}
		_, err := Activate(context.Background(), lock, activationTestServices{}, map[string]Factory{
			"broken-provider": activationTestFactory{id: "broken-provider", runtime: broken},
			"sibling":         activationTestFactory{id: "sibling", runtime: sibling},
		})
		if !errors.Is(err, original) {
			t.Fatalf("original failure was not preserved: %v", err)
		}
		if got := err.Error(); !strings.Contains(got, "broken-provider") || !strings.Contains(got, original.Error()) {
			t.Fatalf("failure evidence missing: %v", err)
		}
		if broken.closeCalls != 1 || sibling.closeCalls != 1 {
			t.Fatalf("failed activation leaked runtimes: broken=%d sibling=%d", broken.closeCalls, sibling.closeCalls)
		}
	})

	t.Run("absent factory", func(t *testing.T) {
		lock := activationLock(t, "missing-provider", "sibling")
		sibling := &activationTestRuntime{id: "sibling", status: RuntimeReadiness{State: StateRunning}}
		_, err := Activate(context.Background(), lock, activationTestServices{}, map[string]Factory{
			"sibling": activationTestFactory{id: "sibling", runtime: sibling},
		})
		if got := err.Error(); !strings.Contains(got, "missing-provider") || !strings.Contains(got, "factory not mounted") {
			t.Fatalf("absent extension evidence missing: %v", err)
		}
		if sibling.closeCalls != 1 {
			t.Fatalf("absent extension left sibling mounted: close calls=%d", sibling.closeCalls)
		}
	})
}

func TestActivationOwnsSuccessfulRuntimeLifecycle(t *testing.T) {
	lock := activationLock(t, "provider", "sibling")
	order := []string{}
	provider := &activationTestRuntime{id: "provider", status: RuntimeReadiness{State: StateRunning}, drained: &order, closed: &order}
	sibling := &activationTestRuntime{id: "sibling", status: RuntimeReadiness{State: StateRunning}, drained: &order, closed: &order}
	active, err := Activate(context.Background(), lock, activationTestServices{}, map[string]Factory{
		"provider": activationTestFactory{id: "provider", runtime: provider},
		"sibling":  activationTestFactory{id: "sibling", runtime: sibling},
	})
	if err != nil {
		t.Fatal(err)
	}
	rows := active.Registry().Snapshot()
	if len(rows) != 2 || rows[0].State != StateRunning || rows[1].State != StateRunning {
		t.Fatalf("ready snapshot = %+v", rows)
	}
	if err := active.Drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := active.Close(); err != nil {
		t.Fatal(err)
	}
	if err := active.Close(); err != nil {
		t.Fatal(err)
	}
	if provider.drainCalls != 1 || sibling.drainCalls != 1 || provider.closeCalls != 1 || sibling.closeCalls != 1 {
		t.Fatalf("owner lifecycle calls provider=%d/%d sibling=%d/%d", provider.drainCalls, provider.closeCalls, sibling.drainCalls, sibling.closeCalls)
	}
	if got := strings.Join(order, ","); got != "drain:sibling,drain:provider,close:sibling,close:provider" {
		t.Fatalf("owner lifecycle order = %s", got)
	}
}

func TestActivateCompatibleRefusesBeforeFactoryStart(t *testing.T) {
	lock := activationLock(t, "provider")
	starts := 0
	active, report, err := ActivateCompatible(context.Background(), lock, activationTestServices{}, map[string]Factory{
		"provider": activationTestFactory{id: "provider", runtime: &activationTestRuntime{status: RuntimeReadiness{State: StateRunning}}, starts: &starts},
	}, contractWith(CapabilityRequirement{Name: "tools.invoke", MinRevision: 1, MaxRevision: 2}), hostWith(
		CapabilityOffer{Name: "tools.invoke", Revision: 3, Status: StatusStable},
	))
	if active != nil || err == nil || report.Compatible {
		t.Fatalf("incompatible activation = active:%v report:%+v err:%v", active, report, err)
	}
	var compatibilityErr *CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Report.Outcomes[0].Reason != ReasonRevisionAboveMax {
		t.Fatalf("activation lost machine refusal: %T %v", err, err)
	}
	if starts != 0 {
		t.Fatalf("factory started %d times before compatibility refusal", starts)
	}
}

func activationLock(t *testing.T, ids ...string) ProductLock {
	t.Helper()
	lock := ProductLock{Schema: ProductLockSchema, Environment: LockEnvironment{OS: "linux", Arch: "amd64", Contract: ContractVersion}}
	for _, id := range ids {
		lock.Components = append(lock.Components, LockedComponent{ID: id, Version: "1.0.0", Digest: "sha256:" + id, Source: "fixture/" + id})
	}
	lock.ID = lockDigest(t, lock)
	return lock
}

type activationTestServices struct{}

func (activationTestServices) Invoke(context.Context, Invocation) (Result, error) {
	return Result{}, nil
}

type activationTestFactory struct {
	id      string
	runtime Runtime
	err     error
	starts  *int
}

func (f activationTestFactory) Manifest() Extension {
	return Extension{ID: f.id, Plane: PlaneTools, Compatibility: ContractVersion, Provenance: Provenance{Source: "fixture/" + f.id, Version: "1.0.0"}}
}

func (f activationTestFactory) Start(context.Context, Services) (Runtime, error) {
	if f.starts != nil {
		*f.starts++
	}
	return f.runtime, f.err
}

type activationTestRuntime struct {
	id         string
	status     RuntimeReadiness
	drained    *[]string
	closed     *[]string
	drainCalls int
	closeCalls int
}

func (r *activationTestRuntime) Readiness() RuntimeReadiness { return r.status }

func (r *activationTestRuntime) Drain(context.Context) error {
	r.drainCalls++
	if r.drained != nil {
		*r.drained = append(*r.drained, "drain:"+r.id)
	}
	return nil
}

func (r *activationTestRuntime) Close() error {
	r.closeCalls++
	if r.closed != nil {
		prefix := ""
		if r.drained == r.closed {
			prefix = "close:"
		}
		*r.closed = append(*r.closed, prefix+r.id)
	}
	return nil
}
