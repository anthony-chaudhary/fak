package gateway

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default

// mhcPreCollapseError builds the exact error the V4.1 mHC pre-collapse readback raises on a
// fail-closed device-stage fault: a *model.V41MHCOperationError wrapping a
// *model.BackendForwardOperationError (Path "v41-mhc-pre-collapse", the named layer, the
// readback stage, and the errV41MHCResult cause). This is the shape recoverDevicePanic
// returns to the gateway, so it exercises the real error path end to end without a GPU.
func mhcPreCollapseError(layer int) error {
	bfo := &model.BackendForwardOperationError{
		Backend: "vulkan",
		Forward: "deepseek41",
		Path:    "v41-mhc-pre-collapse",
		Layer:   layer,
		Stage:   "readback",
		Cause:   errors.New("model: invalid selected V4.1 mHC result"),
	}
	return &model.V41MHCOperationError{Layer: layer, Op: "pre-collapse", Cause: bfo}
}

// A fail-closed device-stage failure (the V4.1 mHC pre-collapse readback at a specific layer)
// must reach the client as a 500 with a distinct code that names the op and layer, NOT the
// opaque 502 "upstream model error" fallthrough that hides the real cause (#3518). The cause
// (which can carry backend error text) must never cross the trust boundary.
func TestUpstreamErrorStatus_DeviceStageIsNotMaskedAs502(t *testing.T) {
	err := mhcPreCollapseError(22)
	status, code, msg := upstreamErrorStatus(err)
	if status != http.StatusInternalServerError {
		t.Fatalf("device-stage failure should be 500, got %d", status)
	}
	if code != "in_kernel_device_stage_failed" {
		t.Fatalf("device-stage failure should carry code in_kernel_device_stage_failed, got %q", code)
	}
	// The op (Path) and layer must be named so the client/ledger can branch on WHICH stage.
	if !strings.Contains(msg, "v41-mhc-pre-collapse") {
		t.Errorf("message should name the op, got %q", msg)
	}
	if !strings.Contains(msg, "layer 22") {
		t.Errorf("message should name the layer, got %q", msg)
	}
	// The cause text must NOT leak to the client.
	if strings.Contains(msg, "invalid selected V4.1 mHC result") {
		t.Errorf("cause text leaked into client message: %q", msg)
	}
}

// The device-stage failure gets its OWN ledger kind, distinct from the coarse "other" bucket
// and from "oom", and the kind stays in lockstep with the status-ladder code (#3518).
func TestUpstreamErrorKinds_DeviceStageOwnKind(t *testing.T) {
	err := mhcPreCollapseError(22)
	kind := upstreamErrorKind(err)
	if kind != "device_stage" {
		t.Fatalf("device-stage failure should carry kind device_stage, got %q", kind)
	}
	// Cross-ladder agreement: the status-ladder code and the ledger kind must pair.
	_, code, _ := upstreamErrorStatus(err)
	if code != "in_kernel_device_stage_failed" {
		t.Fatalf("status ladder code = %q, want in_kernel_device_stage_failed", code)
	}
	// The %w-wrapped form must classify identically (the planner may wrap the typed error).
	wrapped := fmt.Errorf("prefill turn: %w", err)
	if k := upstreamErrorKind(wrapped); k != "device_stage" {
		t.Fatalf("wrapped device-stage failure kind = %q, want device_stage", k)
	}
	if s, c, _ := upstreamErrorStatus(wrapped); s != http.StatusInternalServerError || c != "in_kernel_device_stage_failed" {
		t.Fatalf("wrapped device-stage failure = (%d, %q), want (500, in_kernel_device_stage_failed)", s, c)
	}
}

// The device-stage failure carries its op and layer as typed fields so a client or the perf
// ledger can branch on WHICH stage failed without parsing the message (#3518). The fields are
// gated on the device-stage code so they never ride on an unrelated error.
func TestUpstreamErrorFields_DeviceStageOpAndLayer(t *testing.T) {
	s := &Server{}
	err := mhcPreCollapseError(22)
	fields := s.upstreamErrorFields(err, "in_kernel_device_stage_failed")
	if fields == nil {
		t.Fatal("device-stage failure should carry typed fields")
	}
	if op, ok := fields["op"]; !ok || op != "v41-mhc-pre-collapse" {
		t.Errorf("op field = %v (present=%v), want v41-mhc-pre-collapse", op, ok)
	}
	if layer, ok := fields["layer"]; !ok || layer != 22 {
		t.Errorf("layer field = %v (present=%v), want 22", layer, ok)
	}
	// Gated on the code: a different code must not emit the device-stage fields.
	if f := s.upstreamErrorFields(err, "some_other_code"); f != nil {
		t.Errorf("device-stage fields must be gated on the code, got %v for a different code", f)
	}
}

// A device-stage failure with a negative layer (an admission refusal, not a per-layer readback)
// still names the op but omits the layer field, so the client never sees a bogus negative layer.
func TestUpstreamErrorStatus_DeviceStageNegativeLayer(t *testing.T) {
	s := &Server{}
	bfo := &model.BackendForwardOperationError{
		Backend: "vulkan",
		Forward: "deepseek41",
		Path:    "v41-mhc-reduced-geometry",
		Layer:   -1,
		Stage:   "admission",
		Cause:   errors.New("model: V4.1 mHC stream operations have no device dispatch on this backend"),
	}
	err := &model.V41MHCOperationError{Layer: -1, Op: "reduced-geometry", Cause: bfo}
	status, code, msg := upstreamErrorStatus(err)
	if status != http.StatusInternalServerError || code != "in_kernel_device_stage_failed" {
		t.Fatalf("negative-layer device-stage failure = (%d, %q), want (500, in_kernel_device_stage_failed)", status, code)
	}
	if !strings.Contains(msg, "v41-mhc-reduced-geometry") {
		t.Errorf("message should name the op, got %q", msg)
	}
	fields := s.upstreamErrorFields(err, "in_kernel_device_stage_failed")
	if fields == nil {
		t.Fatal("negative-layer device-stage failure should still carry the op field")
	}
	if _, present := fields["layer"]; present {
		t.Errorf("negative layer must not emit a layer field, got %v", fields)
	}
	if op, ok := fields["op"]; !ok || op != "v41-mhc-reduced-geometry" {
		t.Errorf("op field = %v (present=%v), want v41-mhc-reduced-geometry", op, ok)
	}
}
