//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled V14/74 shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41EngramRowDecodeDeviceOnlyVulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// On the ordinary Vulkan build the Engram row decode binds, so device-only
// admits Engram models. On the role fixture with an Engram layer, in the packed
// and the Q2_K dialect, a device-only session makes no host row decode and no
// host BF16 round, its logits are bit-identical to a session that decodes the
// rows on the host, and it decodes the reference's greedy tokens. The default
// policy Engram session makes no host BF16 round either.
// fak-test:justify why=integration when=changed:internal/model/v41_engram_row_decode*
// fak-test:runtime slow est=30s lane=optin
func TestV41EngramRowDecodeDeviceOnlyVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	if op, ok := be.(compute.V41EngramRowDecodeBackend); !ok || !op.SupportsV41EngramRowDecode() {
		t.Fatal("ordinary Vulkan build did not admit the Engram row decode module on this device")
	}
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("binding_compressed_%t", compressed), func(t *testing.T) {
			m, _ := v41IncrementalEngramFixture(t, compressed)
			defer m.CloseWeights()
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			if unbound := v41HostOpNames(s.v41DeviceOnlyUnboundOps()); slices.Contains(unbound, "engram-row-decode") || slices.Contains(unbound, "engram-stream-injection") {
				t.Fatalf("device-only unbound ops = %v, want the Engram decode and injection bound", unbound)
			}
		})
	}
	t.Run("default-session", func(t *testing.T) {
		v41EngramRowDecodeVulkanSession(t, be, "engram-fixture", func() *Model { m, _ := v41IncrementalEngramFixture(t, false); return m }, false)
	})
	for _, encoding := range []compute.V41EngramRowEncoding{compute.V41EngramRowPacked, compute.V41EngramRowQ2K} {
		name := map[compute.V41EngramRowEncoding]string{compute.V41EngramRowPacked: "packed", compute.V41EngramRowQ2K: "q2k"}[encoding]
		t.Run("device-only-role-fixture-"+name, func(t *testing.T) {
			v41EngramRowDecodeVulkanSession(t, be, "role-fixture-"+name, func() *Model { return v41EngramRoleFixture(t, encoding) }, true)
		})
	}
}

// v41EngramRowDecodeVulkanSession decodes the reference prompt on the device
// (under device-only when deviceOnly) and requires: every Engram injection on
// the device seam, zero host row decodes and zero host BF16 rounds, logits
// bit-identical to a default-policy session whose gate decodes on the host,
// and, under device-only, the reference's greedy tokens.
func v41EngramRowDecodeVulkanSession(t *testing.T, be compute.Backend, label string, fixture func() *Model, deviceOnly bool) {
	t.Helper()
	prompt := []int{1, 2, 3}
	ref, devModel, hostArmModel := fixture(), fixture(), fixture()
	defer func() {
		for _, m := range []*Model{ref, devModel, hostArmModel} {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		}
	}()
	want := v41DeviceParityReference(t, ref, prompt, v41DeviceParitySteps)

	dev, err := devModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	if deviceOnly {
		dev.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		if len(v41DeviceOnlyHostOps) != 0 {
			t.Fatalf("static host inventory is not drained: %v", v41HostOpNames(v41DeviceOnlyHostOps))
		}
		if unbound := dev.v41DeviceOnlyUnboundOps(); len(unbound) != 0 {
			t.Fatalf("%s leaves device-only seams unbound: %v", label, v41HostOpNames(unbound))
		}
		if refusal := dev.v41DeviceOnlyAdmissionRefusal(); refusal != nil {
			t.Fatalf("%s device-only admission refusal = %v, want the empty list", label, refusal)
		}
	}
	ops := dev.v41State().gatherOps
	if ops == nil || ops.engramGate == nil || dev.v41EngramRowDecoder() == nil {
		t.Fatal("Engram gate or row decode seam was not bound")
	}
	gateCalls := 0
	bound := ops.engramGate
	ops.engramGate = func(l int, streams *v41DeviceStreams, rows *v41EngramGatheredRows, positions int, g v41EngramGeometry, eps float32) (bool, error) {
		gateCalls++
		return bound(l, streams, rows, positions, g, eps)
	}
	stop := enableV41BF16Witness()
	v41HostEngramRowDecodes.Store(0)
	var got v41DeviceParityRun
	var dispatches uint64
	refusal := recoverError(func() {
		dispatches = v41SeamDispatches(t, be, func() { got = v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps) })
	})
	rounds, decodes := v41HostBF16Rounds.Load(), v41HostEngramRowDecodes.Load()
	stop()
	if refusal != nil {
		t.Fatalf("%s device session refused (device-only=%t): %v", label, deviceOnly, refusal)
	}
	if dev.HostFallbackObserved() || dev.BackendSessionClosed() {
		t.Fatalf("%s device session observed a host fallback or closed", label)
	}
	if gateCalls == 0 || decodes != 0 || rounds != 0 {
		t.Fatalf("%s gate calls=%d host row decodes=%d host BF16 rounds=%d, want device injections and no host decode or round", label, gateCalls, decodes, rounds)
	}

	hostArm, err := hostArmModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer hostArm.Close()
	hostArm.v41State().gatherOps.engramGate = hostArm.v41EngramGateFuncWith(nil)
	stop = enableV41BF16Witness()
	v41HostEngramRowDecodes.Store(0)
	alt := v41DeviceParitySession(t, hostArm, prompt, v41DeviceParitySteps)
	hostDecodes := v41HostEngramRowDecodes.Load()
	stop()
	if hostDecodes == 0 {
		t.Fatalf("%s host-decode arm decoded no row on the host; the comparison would be vacuous", label)
	}
	for i := range alt.logits {
		for j := range alt.logits[i] {
			if math.Float32bits(got.logits[i][j]) != math.Float32bits(alt.logits[i][j]) {
				t.Fatalf("%s step %d logit %d device decode %08x host decode %08x", label, i, j, math.Float32bits(got.logits[i][j]), math.Float32bits(alt.logits[i][j]))
			}
		}
	}
	refWorst := 0.0
	for i := range want.logits {
		var scale, delta float64
		for j, w := range want.logits[i] {
			scale = max(scale, math.Abs(float64(w)))
			delta = max(delta, math.Abs(float64(got.logits[i][j]-w)))
		}
		refWorst = max(refWorst, delta/max(1, scale))
	}
	if deviceOnly && !slices.Equal(got.tokens, want.tokens) {
		t.Errorf("%s greedy tokens device=%v reference=%v", label, got.tokens, want.tokens)
	}
	t.Logf("V41_ENGRAM_ROW_DECODE_SESSION fixture=%s device_only=%t admission_refusal=none gate_calls=%d host_row_decodes=%d host_bf16_rounds=%d host_arm_row_decodes=%d dispatches=%d tokens device=%v reference=%v logits_bit_identical_to_host_decode_arm=true reference_worst_rel=%.4g",
		label, deviceOnly, gateCalls, decodes, rounds, hostDecodes, dispatches, got.tokens, want.tokens, refWorst)
}
