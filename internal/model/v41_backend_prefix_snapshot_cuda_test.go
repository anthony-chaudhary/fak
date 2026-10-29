//go:build cuda

package model

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41BackendPrefixSnapshotCUDA is the physical CUDA qualification witness
// for V4.1 prefix reuse, the CUDA twin of TestV41BackendPrefixSnapshotVulkan.
// A full-geometry session on the actual CUDA device captures a prefix through
// the serving cache path (PrefixSnapshot, PrepareForCache, CloneForRestore),
// restores it into a fresh session, and the restored branch must continue
// bit-identically to the unsnapshotted source: suffix prefill and fixed-token
// decode. Odd prefix lengths leave an incomplete compressor group and a partly
// filled window ring, so the compressed-attention and compressor ring state is
// covered; every per-session device binding must rebuild from the restored host
// state rather than carry the source's.
//
// Unlike Vulkan, most V4.1 seams have no CUDA device arm yet and run on the
// host, so a host fallback is logged rather than refused: the contract under
// test is continuation equality on the CUDA backend identity, whatever mix of
// device and host arms that backend executes today.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41BackendPrefixSnapshotCUDA(t *testing.T) {
	if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") != "1" {
		t.Skip("physical V4.1 prefix-snapshot witness requires FAK_CUDA_REQUIRE_DEVICE=1")
	}
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" || !be.Caps().DeviceMemory {
		t.Fatal("required actual CUDA device unavailable")
	}
	if expected := os.Getenv("FAK_CUDA_EXPECT_DEVICE"); expected != "" {
		snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || snap.Identity.Device != expected {
			t.Fatalf("physical device=%q want=%q (available=%t err=%v)", snap.Identity.Device, expected, available, err)
		}
	}
	for _, compressed := range []bool{false, true} {
		for _, prefix := range []int{5, 9} {
			t.Run(fmt.Sprintf("compressed_%t_prefix_%d", compressed, prefix), func(t *testing.T) {
				v41CUDAPrefixRestoreWitness(t, be, compressed, prefix)
			})
		}
	}
}

func v41CUDAPrefixRestoreWitness(t *testing.T, be compute.Backend, compressed bool, prefix int) {
	m := v41DecodeSubmitsLayeredFixture(t, 2, compressed)
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	if !m.Cfg.InKernelBackendPrefixReuseSupportedFor(be) {
		t.Fatal("CUDA is not a qualified V4.1 prefix-reuse backend")
	}
	const suffix = 4
	prompt := v41DecodeSubmitsPrompt(prefix+suffix, m.Cfg.VocabSize)
	decode := []int{2, 4, 6}

	source, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
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
	if source.halClosed || target.halClosed {
		t.Fatalf("a CUDA session closed its HAL (source=%v target=%v)", source.halFailure, target.halFailure)
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
	t.Logf("v41_cuda_prefix_restore compressed=%t prefix=%d suffix=%d decode=%d bitwise=true shared_attention_bound=%t host_fallback_source=%t host_fallback_target=%t cold_full_prefill_max_abs_diff=%.3g argmax_restored=%d argmax_cold=%d",
		compressed, prefix, suffix, len(decode), target.v41State().sharedAttentionFunc() != nil,
		source.HostFallbackObserved(), target.HostFallbackObserved(), maxDiff, v41CUDAArgmax(got[0]), v41CUDAArgmax(full))
}

func v41CUDAArgmax(v []float32) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}
