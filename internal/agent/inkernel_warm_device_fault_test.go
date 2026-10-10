package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// faultingGDNBackend fails every Qwen3.5 GDN device step, the way a device
// tensor preflight refusal does on real hardware.
type faultingGDNBackend struct{ *countingBackend }

func (*faultingGDNBackend) Qwen35GDNDecode(
	_,
	_, _, _, _,
	_, _, _, _, _,
	_, _ compute.Tensor,
	_, _, _, _, _ int,
	_ float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, errors.New("injected GDN device fault")
}

// A startup warm whose device forward fails closed must report a cold receipt
// with a closed reason; it must never crash the serving process.
// fak-test:runtime fast est=1s
func TestWarmPrefixDeviceFaultReportsColdInsteadOfPanicking(t *testing.T) {
	cfg := tinyHybridCfg()
	cfg.AttnOutputGate = false
	cfg.VocabSize = warmCfg().VocabSize
	backend := &faultingGDNBackend{countingBackend: &countingBackend{Backend: compute.Default(), deviceMemory: true}}
	p := NewInKernelPlannerWithConfig(model.NewSynthetic(cfg), loadProbeTok(t), "qwen35-device-fault", false, backend, false, InKernelPlannerConfig{
		RequireDeviceExecution: true,
		DenseGPULayers:         cfg.NumLayers,
	})
	p.quant = false
	p.SetWarmDiskConfig(WarmDiskConfig{Disabled: true})
	in := warmFixtureInputs()
	spec, err := p.DeriveWarmPrefix("tenant-fault", "agent-1", in)
	if err != nil {
		t.Fatalf("derive warm prefix: %v", err)
	}
	p.SetWarmPrefixInputs(in)

	receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(context.Background(), spec.Scope.Tenant, spec.Scope.Agent), spec)
	var operation *model.BackendForwardOperationError
	if !errors.As(err, &operation) {
		t.Fatalf("WarmPrefix err = %v, want a recovered *model.BackendForwardOperationError", err)
	}
	if receipt.Ready || receipt.Usable() || receipt.Status != WarmStatusCold || receipt.Reason != "device_fault" {
		t.Fatalf("receipt ready=%t status=%q reason=%q, want cold/device_fault", receipt.Ready, receipt.Status, receipt.Reason)
	}
}
