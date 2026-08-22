package microagent_test

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type capabilityCaptureAgent struct {
	captured chan microagent.CapabilityEnvelope
}

type capabilityRootAgent struct {
	release <-chan struct{}
}

func (a *capabilityRootAgent) Step(ctx context.Context, _ microagent.Gateway) (bool, error) {
	select {
	case <-a.release:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func (a *capabilityCaptureAgent) Step(ctx context.Context, _ microagent.Gateway) (bool, error) {
	envelope, ok := microagent.ChildEnvelopeFromStep(ctx)
	if !ok {
		return false, errors.New("child capability envelope missing from execution context")
	}
	a.captured <- envelope
	return true, nil
}

func TestRequestChildIntersectsParentCapabilityEnvelope(t *testing.T) {
	parent := microagent.CapabilityEnvelope{
		Tools:               []string{"repo.read", "repo.write"},
		Paths:               []string{"internal/microagent", "docs/public.md"},
		NetworkDestinations: []string{"api.example.test:443"},
		Effects:             []string{"observe", "write"},
	}
	host, err := microagent.NewHost(stubPlanner{}, microagent.Config{
		Workers:          1,
		Queue:            2,
		SpawnBudget:      &microagent.SpawnBudget{MaxDepth: 2, MaxChildren: 1, MaxDescendants: 1},
		RootCapabilities: parent,
	})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer host.Close()

	releaseRoot := make(chan struct{})
	if err := host.Spawn("root", &capabilityRootAgent{release: releaseRoot}); err != nil {
		t.Fatalf("Spawn(root): %v", err)
	}
	captured := make(chan microagent.CapabilityEnvelope, 1)
	err = host.RequestChild(microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "child",
		Depth:    1,
		Capabilities: microagent.CapabilityEnvelope{
			Tools:               []string{"repo.read", "shell.exec"},
			Paths:               []string{"internal/microagent/contract.go", "private/credentials"},
			NetworkDestinations: []string{"api.example.test:443", "metadata.internal:80"},
			Effects:             []string{"observe", "delete"},
		},
	}, &capabilityCaptureAgent{captured: captured})
	if err != nil {
		t.Fatalf("RequestChild: %v", err)
	}
	close(releaseRoot)
	if err := host.Drain(context.Background()); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	got := <-captured
	want := microagent.CapabilityEnvelope{
		Tools:               []string{"repo.read"},
		Paths:               []string{filepath.Join("internal", "microagent", "contract.go")},
		NetworkDestinations: []string{"api.example.test:443"},
		Effects:             []string{"observe"},
	}
	if !got.Equal(want) {
		t.Fatalf("child capabilities = %#v, want strict intersection %#v", got, want)
	}
}

func TestRequestChildRefusesCapabilitiesWithoutParentEnvelope(t *testing.T) {
	sink := &countingSink{}
	host, err := microagent.NewHost(stubPlanner{}, microagent.Config{
		Workers:     1,
		Audit:       sink,
		SpawnBudget: &microagent.SpawnBudget{MaxDepth: 2, MaxChildren: 1, MaxDescendants: 1},
	})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	defer host.Close()

	err = host.RequestChild(microagent.SpawnRequest{
		ParentID:     "missing-parent",
		ChildID:      "child",
		Depth:        1,
		Capabilities: microagent.CapabilityEnvelope{Tools: []string{"repo.read"}},
	}, &capabilityCaptureAgent{captured: make(chan microagent.CapabilityEnvelope, 1)})
	if !errors.Is(err, microagent.ErrChildAuthority) {
		t.Fatalf("RequestChild without parent envelope = %v, want ErrChildAuthority", err)
	}
	if got := sink.kind(microagent.EventSpawn); got != 0 {
		t.Fatalf("spawn audit events = %d, want zero after fail-closed refusal", got)
	}
}
