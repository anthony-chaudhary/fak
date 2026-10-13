package model

import (
	"errors"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder stops at the selected fused operation. It witnesses routing
// and failure handling, not fused arithmetic or physical device execution.
type v41RoutedFusedSelectionBackend struct {
	*v41RoutedIDTestBackend
	fusedCalls int
	failure    error
}

var _ compute.V41RoutedFusedBackend = (*v41RoutedFusedSelectionBackend)(nil)

func (b *v41RoutedFusedSelectionBackend) V41RoutedFusedAvailable() bool { return true }
func (b *v41RoutedFusedSelectionBackend) SupportsV41RoutedFusedGateUp(gate, up compute.Dtype) bool {
	return gate == compute.Q2_K && up == compute.Q2_K
}
func (b *v41RoutedFusedSelectionBackend) SupportsV41RoutedFusedDown(down compute.Dtype) bool {
	return down == compute.Q3_K
}
func (b *v41RoutedFusedSelectionBackend) V41RoutedGateUpSwiGLUDeferred(gates, ups []compute.Tensor, ids, x compute.Tensor, picks int, limit float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	b.fusedCalls++
	return compute.Tensor{}, compute.V41DeferredStatus{}, b.failure
}
func (b *v41RoutedFusedSelectionBackend) V41RoutedDownSumDeferred(downs []compute.Tensor, mid, ids compute.Tensor, picks int, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("fused down reached after selected gate/up failure")
}

// The default fused route needs ID capability without opting into the legacy
// ID route. A selected fused error must remain terminal rather than decline.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41RoutedFusedDefaultSelectedFailure(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	for _, name := range []string{v41RoutedIDEnv, v41RoutedFusedEnv} {
		t.Setenv(name, "")
		if err := os.Unsetenv(name); err != nil {
			t.Fatal(err)
		}
	}
	m := v41IncrementalExpertFixture(t, true, false)
	sentinel := errors.New("selected fused gate/up failure")
	b := &v41RoutedFusedSelectionBackend{v41RoutedIDTestBackend: newV41RoutedIDTestBackend(), failure: sentinel}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	routed := s.v41RoutedLayerFunc()
	if routed == nil {
		t.Fatal("batched routed layer is unbound")
	}
	picks, x := v41RoutedLayerFaultPicks(m)
	sum, outcome, err := routed(0, picks, x)
	if sum != nil || outcome != v41RoutedExpertError || !errors.Is(err, sentinel) {
		t.Fatalf("selected fused failure: sum=%v outcome=%v err=%v", sum, outcome, err)
	}
	var closed *BackendForwardOperationError
	if !errors.As(err, &closed) || closed.Path != "v41-routed-expert" || closed.Stage != "gate/up" || closed.Layer != 0 || !s.BackendSessionClosed() || s.halFailure != err {
		t.Fatalf("selected fused error=%v session closed=%t saved error=%v", err, s.BackendSessionClosed(), s.halFailure)
	}
	if b.fusedCalls != 1 || b.idCalls["ids"] != 1 || b.idCalls["matvec"] != 0 || b.calls != 0 || b.publishes != 0 || b.deferredReads != 0 || s.HostFallbackObserved() {
		t.Fatalf("unexpected fallback: fused=%d id=%v activations=%d publishes=%d reads=%d host=%t", b.fusedCalls, b.idCalls, b.calls, b.publishes, b.deferredReads, s.HostFallbackObserved())
	}
}
