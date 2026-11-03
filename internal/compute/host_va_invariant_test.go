package compute

import (
	"errors"
	"os"
	"testing"
)

// TestHostVAEquivalenceInvariant pins the host-VA/device-VA equivalence
// contract for pinned host slabs (root-cause discipline adapted from SGLang
// PR #38967): which allocation primitive must back any host memory whose
// pointer is stored in a device-visible table and dereferenced by device
// code. On non-HIP targets it asserts the closed contract trivially; the
// ROCm-gated subtest runs on a live ROCm device and is quarantine-skippable
// elsewhere (rocmRequired-style skip policy, FAK_ROCM_REQUIRE_DEVICE to
// force). The final leg — a GPU kernel loading through a device-side pointer
// table built from a hipHostMalloc slab — stays open until the frocm ABI
// grows a pinned-host entry plus a deref kernel (ticket follow-up gap);
// this subtest covers the contract and the transfer path those tables rely
// on.
//
// fak-test:runtime fast est=1ms lane=default
func TestHostVAEquivalenceInvariant(t *testing.T) {
	t.Run("closed-contract", func(t *testing.T) {
		for _, tc := range []struct {
			platform, primitive string
			want                HostVAClass
		}{
			{"cuda", "cudaHostAlloc", HostVAEquivalent},
			{"CUDA", "cudaHostRegister", HostVAEquivalent}, // case-insensitive spelling
			{"hip", "hipHostMalloc", HostVAEquivalent},
			{"rocm", "hipHostMalloc", HostVAEquivalent},
			{"rocm", "hipHostRegister", HostVADivergent},
		} {
			got, err := HostVAEquivalence(tc.platform, tc.primitive)
			if err != nil || got != tc.want {
				t.Fatalf("HostVAEquivalence(%q,%q)=(%v,%v), want (%v,nil)", tc.platform, tc.primitive, got, err, tc.want)
			}
		}
		for _, pair := range [][2]string{
			{"metal", "cudaHostAlloc"},
			{"", "cudaHostAlloc"},
			{"cuda", "hipHostMalloc"},
			{"rocm", "cudaHostRegister"},
			{"cuda", "malloc"},
		} {
			if _, err := HostVAEquivalence(pair[0], pair[1]); !errors.Is(err, ErrHostVAContractUnknown) {
				t.Fatalf("HostVAEquivalence(%q,%q)=err %v, want ErrHostVAContractUnknown", pair[0], pair[1], err)
			}
		}
		for platform, want := range map[string]string{
			"cuda": "cudaHostAlloc",
			"HIP":  "hipHostMalloc",
			"rocm": "hipHostMalloc",
		} {
			got, err := DeviceDerefHostPrimitive(platform)
			if err != nil || got != want {
				t.Fatalf("DeviceDerefHostPrimitive(%q)=(%q,%v), want (%q,nil)", platform, got, err, want)
			}
		}
		if _, err := DeviceDerefHostPrimitive("metal"); !errors.Is(err, ErrHostVAContractUnknown) {
			t.Fatalf("DeviceDerefHostPrimitive(metal)=%v, want ErrHostVAContractUnknown", err)
		}
		if err := CheckDeviceVisibleHostBacking("cuda", "cudaHostAlloc", "cudaHostRegister"); err != nil {
			t.Fatalf("CheckDeviceVisibleHostBacking(cuda,...)=%v, want nil", err)
		}
		if err := CheckDeviceVisibleHostBacking("rocm", "hipHostMalloc"); err != nil {
			t.Fatalf("CheckDeviceVisibleHostBacking(rocm,hipHostMalloc)=%v, want nil", err)
		}
		if err := CheckDeviceVisibleHostBacking("rocm"); err != nil {
			t.Fatalf("empty backing list: %v, want nil", err)
		}
		if err := CheckDeviceVisibleHostBacking("rocm", "hipHostMalloc", "hipHostRegister"); !errors.Is(err, ErrHostVADivergentBacking) {
			t.Fatalf("divergent backing: %v, want ErrHostVADivergentBacking", err)
		}
		if err := CheckDeviceVisibleHostBacking("tpu", "cudaHostAlloc"); !errors.Is(err, ErrHostVAContractUnknown) {
			t.Fatalf("unknown platform: %v, want ErrHostVAContractUnknown", err)
		}
	})

	t.Run("roc-device-visible-table", func(t *testing.T) {
		b, ok := Lookup("rocm")
		if !ok {
			if os.Getenv("FAK_ROCM_REQUIRE_DEVICE") == "1" {
				t.Fatal("FAK_ROCM_REQUIRE_DEVICE=1: native ROCm backend unavailable")
			}
			t.Skip("quarantine-skippable: native ROCm backend unavailable; the closed contract is asserted trivially above")
		}
		// The registry's own platform spelling must be covered by the closed
		// contract and select the same-VA primitive.
		want := "hipHostMalloc"
		got, err := DeviceDerefHostPrimitive(b.Name())
		if err != nil || got != want {
			t.Fatalf("DeviceDerefHostPrimitive(%q)=(%q,%v), want (%q,nil)", b.Name(), got, err, want)
		}
		if err := CheckDeviceVisibleHostBacking(b.Name(), got); err != nil {
			t.Fatalf("CheckDeviceVisibleHostBacking(%q,%q)=%v, want nil", b.Name(), got, err)
		}
		if err := CheckDeviceVisibleHostBacking(b.Name(), "hipHostRegister"); !errors.Is(err, ErrHostVADivergentBacking) {
			t.Fatalf("hipHostRegister backing on %s: %v, want ErrHostVADivergentBacking", b.Name(), err)
		}
		// Publish host slab contents into resident storage through the
		// transfer path device-visible pointer tables rely on, then read them
		// back: values must survive the round trip exactly.
		src := []float32{1, -2, .5, .25}
		resident := b.Upload(NewF32(b, []int{4}, src), F32)
		defer b.Free(resident)
		out := b.Read(resident)
		if len(out) != len(src) {
			t.Fatalf("round trip length=%d, want %d", len(out), len(src))
		}
		for i := range src {
			if out[i] != src[i] {
				t.Fatalf("round trip[%d]=%v, want %v", i, out[i], src[i])
			}
		}
	})
}
