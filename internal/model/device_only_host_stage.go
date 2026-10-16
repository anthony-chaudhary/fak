package model

import (
	"errors"
	"fmt"
	"slices"
)

// HostStage names one host-side operation a Session can run during a forward.
type HostStage string

const (
	// HostStageEmbeddingRowUpload gathers one token's embedding row on the host and
	// uploads it as the forward's input activation (one H2D per token).
	HostStageEmbeddingRowUpload HostStage = "embedding-row-upload"
	// HostStageLogitsReadback reads the vocabulary logits back for host sampling
	// (one D2H per token).
	HostStageLogitsReadback HostStage = "logits-readback-sample"

	// HostStageHostMatKernel selects a host matmul kernel for a decode layer.
	HostStageHostMatKernel HostStage = "host-mat-kernel"
	// HostStageHostMatmul runs one projection GEMV on the host.
	HostStageHostMatmul HostStage = "host-matmul"
)

// ErrDeviceOnlyHostStage is the sentinel under every device-only host refusal.
var ErrDeviceOnlyHostStage = errors.New("model: device-only execution refuses an unsanctioned host stage")

// deviceOnlyAllowedHostStages is the measured per-token host work of the device
// decode path (strix1 Qwen Vulkan profile): everything else refuses.
var deviceOnlyAllowedHostStages = []HostStage{
	HostStageEmbeddingRowUpload,
	HostStageLogitsReadback,
}

// DeviceOnlyAllowedHostStages returns the sanctioned host stages in order.
func DeviceOnlyAllowedHostStages() []HostStage { return slices.Clone(deviceOnlyAllowedHostStages) }

// DeviceOnlyHostStageAllowed reports whether stage is sanctioned under device-only.
func DeviceOnlyHostStageAllowed(stage HostStage) bool {
	return slices.Contains(deviceOnlyAllowedHostStages, stage)
}

// observeUnsanctionedHostStage is the counting seam for an unsanctioned host
// stage that either refused (device-only) or ran under a host-path bypass.
// TICKET-02's fak_compute_host_fallback_total{stage} increments here.
var observeUnsanctionedHostStage = func(HostStage) {}

// SetHostPathBypass marks a session that runs host stages under an admitted
// bring-up bypass; token names the bypass and release runs once on Close.
func (s *Session) SetHostPathBypass(token string, release func()) {
	if s == nil {
		return
	}
	s.hostPathBypass, s.hostPathBypassRelease = token, release
}

// HostPathBypass is the admitted bypass token this session runs under, or "".
func (s *Session) HostPathBypass() string {
	if s == nil {
		return ""
	}
	return s.hostPathBypass
}

func (s *Session) releaseHostPathBypass() {
	if s == nil || s.hostPathBypassRelease == nil {
		return
	}
	release := s.hostPathBypassRelease
	s.hostPathBypassRelease = nil
	release()
}

// admitHostStage is the device-only allow-list: under ExecutionPolicyDeviceOnly a
// stage outside deviceOnlyAllowedHostStages closes the session and panics with a
// BackendForwardOperationError wrapping ErrDeviceOnlyHostStage.
func (s *Session) admitHostStage(stage HostStage) {
	if s == nil || DeviceOnlyHostStageAllowed(stage) {
		return
	}
	if s.executionPolicy != ExecutionPolicyDeviceOnly {
		if s.hostPathBypass != "" {
			observeUnsanctionedHostStage(stage)
		}
		return
	}
	observeUnsanctionedHostStage(stage)
	s.refuseHostFallbackCause(string(stage), fmt.Errorf("%w: %s", ErrDeviceOnlyHostStage, stage))
}
