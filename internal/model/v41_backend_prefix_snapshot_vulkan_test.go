//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41BackendPrefixSnapshotVulkan is the physical Vulkan qualification
// witness for V4.1 prefix reuse (fak-private#3432, fak#12640). A full-geometry
// session on the actual Vulkan device captures a prefix through the serving
// cache path (PrefixSnapshot, PrepareForCache, CloneForRestore), restores it
// into a fresh session, and the restored branch must continue bit-identically
// to the unsnapshotted source: suffix prefill and fixed-token decode. Odd prefix
// lengths leave an incomplete compressor group and a partly filled window ring;
// the device KV mirror and every per-session binding must rebuild from the
// restored host state rather than carry the source's.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41BackendPrefixSnapshotVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical V4.1 prefix-snapshot witness requires FAK_VULKAN_REQUIRE_DEVICE=1")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" {
		snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || snap.Identity.Device != expected {
			t.Fatalf("physical device=%q want=%q (available=%t err=%v)", snap.Identity.Device, expected, available, err)
		}
	}
	for _, compressed := range []bool{false, true} {
		for _, prefix := range []int{5, 9} {
			t.Run(fmt.Sprintf("compressed_%t_prefix_%d", compressed, prefix), func(t *testing.T) {
				v41VulkanPrefixRestoreWitness(t, be, compressed, prefix)
			})
		}
	}
}

func v41VulkanPrefixRestoreWitness(t *testing.T, be compute.Backend, compressed bool, prefix int) {
	m := v41DecodeSubmitsLayeredFixture(t, 2, compressed)
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	if !m.Cfg.InKernelBackendPrefixReuseSupportedFor(be) {
		t.Fatal("Vulkan is not a qualified V4.1 prefix-reuse backend")
	}
	const suffix = 4
	prompt := v41DecodeSubmitsPrompt(prefix+suffix, m.Cfg.VocabSize)
	decode := []int{2, 4, 6}

	source, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if source.v41State().sharedAttentionFunc() == nil {
		t.Fatal("Vulkan session bound no shared attention")
	}
	if got := source.Prefill(prompt[:prefix]); len(got) == 0 {
		t.Fatal("source prefix prefill returned no logits")
	}
	snap, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	defer snap.Close()
	if !snap.v41BackendSnapshotSupported() || snap.v41Tokens != prefix {
		t.Fatalf("snapshot qualified=%t tokens=%d, want qualified %d", snap.v41BackendSnapshotSupported(), snap.v41Tokens, prefix)
	}
	snap.PrepareForCache()
	fork, err := snap.CloneForRestore()
	if err != nil {
		t.Fatalf("CloneForRestore: %v", err)
	}
	defer fork.Close()

	continueRun := func(s *Session) [][]float32 {
		out := [][]float32{append([]float32(nil), s.Prefill(prompt[prefix:])...)}
		for _, id := range decode {
			out = append(out, append([]float32(nil), s.Step(id)...))
		}
		return out
	}
	want := continueRun(source)

	target, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	if err := fork.Restore(target); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	got := continueRun(target)
	for i := range want {
		if len(want[i]) == 0 || len(got[i]) != len(want[i]) {
			t.Fatalf("window %d logits len got=%d want=%d", i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				t.Fatalf("window %d logit %d: restored %v != unsnapshotted %v", i, j, got[i][j], want[i][j])
			}
		}
	}
	if source.HostFallbackObserved() || target.HostFallbackObserved() {
		t.Error("a session observed a host fallback")
	}

	// A cold session that prefills the whole prompt in one call is the
	// no-reuse serving arm. Chunking may reorder accumulation, so it is logged
	// rather than required to be bitwise.
	cold, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer cold.Close()
	full := cold.Prefill(prompt)
	maxDiff := 0.0
	for j := range full {
		maxDiff = math.Max(maxDiff, math.Abs(float64(full[j]-got[0][j])))
	}
	t.Logf("v41_vulkan_prefix_restore compressed=%t prefix=%d suffix=%d decode=%d bitwise=true cold_full_prefill_max_abs_diff=%.3g argmax_restored=%d argmax_cold=%d",
		compressed, prefix, suffix, len(decode), maxDiff, v41ArgmaxF32(got[0]), v41ArgmaxF32(full))
}

func v41ArgmaxF32(v []float32) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}
