//go:build vulkan && (windows || linux) && cgo && !v41_logit_scale_witness

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41LogitPostprocessDeviceOnlyVulkan$' -count=1 -timeout=10m
// Synthetic three-layer full-geometry fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The ordinary Vulkan build links no logit seam. With the V4.1 Flash
// postprocess (the identity) device-only admission names no logit op, and the
// device-only head (final norm and projection on the device; its BF16 boundary
// is the separate bf16-boundary op) returns the projection row unchanged,
// bit-identical to a host-policy Vulkan session's head. A configured logit
// scale is refused typed, at admission and, past it, at the head before any
// host arithmetic.
// Whole-forward greedy parity is TestV41DeviceForwardGreedyParityVulkan.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
func TestV41LogitPostprocessDeviceOnlyVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical device-only head witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	t.Logf("Vulkan device: %s", be.Tier())
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" {
		snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || snap.Identity.Device != expected {
			t.Fatalf("physical device=%q want=%q (available=%t err=%v)", snap.Identity.Device, expected, available, err)
		}
	}
	logitOps := func(ops []v41HostOp) []string {
		var names []string
		for _, op := range ops {
			if slices.Contains(v41LogitPostprocessOps, op.name) {
				names = append(names, op.name)
			}
		}
		return names
	}
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("layers3_engram_compressed_%t", compressed), func(t *testing.T) {
			fixture := func() *Model {
				m, _ := v41IncrementalEngramFixture(t, compressed)
				return m
			}
			ref, hostModel, only, scaled := fixture(), fixture(), fixture(), fixture()
			var sessions []*Session
			defer func() {
				for _, s := range sessions {
					s.Close()
				}
				for _, m := range []*Model{ref, hostModel, only, scaled} {
					if err := m.CloseWeights(); err != nil {
						t.Error(err)
					}
				}
			}()
			if v41UsesLogitScale(only.Cfg) || v41UsesLogitSoftcap(only.Cfg) || v41UsesLogitSuppression(only.Cfg) {
				t.Fatal("fixture head postprocess is not the identity")
			}
			session := func(m *Model, deviceOnly bool) (*Session, *v41ForwardState) {
				s, err := m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				sessions = append(sessions, s)
				if deviceOnly {
					s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
				}
				st := s.v41State()
				if st.denseProjection() == nil || st.finalNorm == nil {
					t.Fatal("Vulkan session left the head projection or final norm unbound; the head would compare host with host")
				}
				t.Logf("device_only=%t head seams: projection=%t final_norm=%t bf16_boundary=%t", deviceOnly, st.denseProjection() != nil, st.finalNorm != nil, st.bf16Boundary != nil)
				return s, st
			}
			H := only.Cfg.HiddenSize
			x := make([]float32, H)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
			}

			s, st := session(only, true)
			if s.v41LogitScaleFunc() != nil {
				t.Fatal("ordinary Vulkan build linked a logit scale seam")
			}
			if names := logitOps(s.v41DeviceOnlyUnboundOps()); len(names) != 0 {
				t.Fatalf("identity postprocess left logit ops unbound: %v", names)
			}
			var typed *v41DeviceOnlyHostOpError
			if err := s.v41DeviceOnlyAdmissionRefusal(); errors.As(err, &typed) {
				if names := logitOps(typed.ops); len(names) != 0 {
					t.Fatalf("admission names logit ops %v", names)
				}
				t.Logf("device-only admission refusal (remaining ops): %v", err)
			} else if err != nil {
				t.Fatalf("admission = %v, want nil or the typed host-op refusal", err)
			}
			refuse := st.logitHostRefusal()
			if refuse == nil {
				t.Fatal("device-only state permits the host postprocess")
			}

			_, hst := session(hostModel, false)
			if hst.logitHostRefusal() != nil {
				t.Fatal("host-policy state refuses the host postprocess")
			}
			want, err := hostModel.v41HeadWithLogitScale(x, hst.denseProjection(), hst.finalNorm, hst.bf16Boundary, hst.logitScale, nil)
			if err != nil {
				t.Fatal(err)
			}
			before, observed, _ := compute.CaptureBackendExecutionSnapshot(be)
			got, err := only.v41HeadWithLogitScale(x, st.denseProjection(), st.finalNorm, st.bf16Boundary, st.logitScale, refuse)
			if err != nil || s.BackendSessionClosed() || s.HostFallbackObserved() {
				t.Fatalf("device-only identity head = %v (closed=%t)", err, s.BackendSessionClosed())
			}
			if after, ok, _ := compute.CaptureBackendExecutionSnapshot(be); observed && ok {
				delta, err := compute.BackendExecutionDelta(before, after)
				if err != nil {
					t.Fatal(err)
				}
				if delta.Counters.ComputeDispatches == 0 {
					t.Fatal("device-only head issued no compute dispatches")
				}
				t.Logf("device-only head compute_dispatches=%d h2d_bytes=%d d2h_bytes=%d", delta.Counters.ComputeDispatches, delta.Counters.H2DBytes, delta.Counters.D2HBytes)
			}
			v41LogitScaleCheckBits(t, got, want)
			host, err := ref.v41HeadWithLogitScale(x, nil, nil, nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			var scale, delta float64
			for i, w := range host {
				scale = max(scale, math.Abs(float64(w)))
				delta = max(delta, math.Abs(float64(got[i]-w)))
			}
			if rel := delta / max(1, scale); rel > v41DeviceParityLogitTol {
				t.Fatalf("device-only head max_abs=%g rel=%g > %g against the host reference", delta, rel, v41DeviceParityLogitTol)
			} else {
				t.Logf("device-only identity head: %d logits bit-identical to the host-policy Vulkan head; max_abs=%.4g rel=%.4g against the host reference; argmax=%d/%d", len(got), delta, rel, argmaxF32(got), argmaxF32(host))
			}

			scaled.Cfg.LogitScale = 0.0625
			gated, gst := session(scaled, true)
			if names := logitOps(gated.v41DeviceOnlyUnboundOps()); !slices.Equal(names, []string{"logit-scale"}) {
				t.Fatalf("configured scale unbound logit ops = %v, want [logit-scale]", names)
			}
			if err := gated.v41DeviceOnlyAdmissionRefusal(); !errors.As(err, &typed) || !slices.Contains(v41HostOpNames(typed.ops), "logit-scale") {
				t.Fatalf("configured scale admission = %v, want it to name logit-scale", err)
			}
			_, err = scaled.v41HeadWithLogitScale(x, gst.denseProjection(), gst.finalNorm, gst.bf16Boundary, gst.logitScale, gst.logitHostRefusal())
			var closed *BackendForwardOperationError
			var host2 *v41DeviceOnlyHostOpError
			if !errors.As(err, &closed) || closed.Path != "device-only" || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &host2) || !slices.Equal(v41HostOpNames(host2.ops), []string{"logit-scale"}) {
				t.Fatalf("configured scale head = %v, want the typed logit-scale refusal", err)
			}
			if !gated.BackendSessionClosed() || !gated.HostFallbackObserved() {
				t.Fatal("head refusal left the session open")
			}
			t.Logf("configured scale refused at the head: %v", err)
		})
	}
}
