package compute

import "testing"

// rocm_arch_test.go — host-tractable witnesses for the ROCm device-arch taxonomy (#266).
// Every assertion here is hardware-independent: it checks the gfx→family mapping, the
// CDNA/RDNA split, and the offload-arch normalization that a HIP build keys on, none of
// which needs an AMD GPU. The device run (a HIP kernel actually executing on AMD silicon)
// is the deferred half — see ROCM-C002-NOTES.md.

func TestROCmArchLookupKnown(t *testing.T) {
	cases := []struct {
		gfx        string
		family     ROCmFamily
		wavefront  int
		datacenter bool
	}{
		{"gfx906", ROCmGCN5, 64, true},
		{"gfx908", ROCmCDNA1, 64, true},
		{"gfx90a", ROCmCDNA2, 64, true},
		{"gfx942", ROCmCDNA3, 64, true},
		{"gfx1010", ROCmRDNA1, 32, false},
		{"gfx1030", ROCmRDNA2, 32, false},
		{"gfx1032", ROCmRDNA2, 32, false},
		{"gfx1100", ROCmRDNA3, 32, false},
		{"gfx1102", ROCmRDNA3, 32, false},
		{"gfx1151", ROCmRDNA3_5, 32, false},
	}
	for _, c := range cases {
		a, ok := LookupROCmArch(c.gfx)
		if !ok {
			t.Fatalf("LookupROCmArch(%q): not found, want supported", c.gfx)
		}
		if a.Family != c.family {
			t.Errorf("%s: family = %v, want %v", c.gfx, a.Family, c.family)
		}
		if a.Wavefront != c.wavefront {
			t.Errorf("%s: wavefront = %d, want %d", c.gfx, a.Wavefront, c.wavefront)
		}
		if a.Family.Datacenter() != c.datacenter {
			t.Errorf("%s: Datacenter() = %v, want %v", c.gfx, a.Family.Datacenter(), c.datacenter)
		}
	}
}

// TestROCmCDNARDNAInvariant pins the load-bearing split: every supported target is exactly
// one of CDNA, RDNA, or the GCN5 datacenter part — never both, never neither — and the
// wavefront width is the one its family mandates (64 for GCN/CDNA, 32 for RDNA). A new row
// added to the table with a mistyped family or wavefront fails here.
func TestROCmCDNARDNAInvariant(t *testing.T) {
	for _, a := range KnownROCmArches() {
		if a.Family.IsCDNA() && a.Family.IsRDNA() {
			t.Errorf("%s: family %v claims both CDNA and RDNA", a.GFX, a.Family)
		}
		isGCN5 := a.Family == ROCmGCN5
		if !a.Family.IsCDNA() && !a.Family.IsRDNA() && !isGCN5 {
			t.Errorf("%s: family %v is neither CDNA, RDNA, nor GCN5", a.GFX, a.Family)
		}
		if a.Family.IsRDNA() {
			if a.Wavefront != 32 {
				t.Errorf("%s: RDNA wavefront = %d, want 32", a.GFX, a.Wavefront)
			}
			if a.Family.Datacenter() {
				t.Errorf("%s: RDNA classified as datacenter", a.GFX)
			}
		} else { // GCN5 or CDNA — the 64-lane Instinct datacenter line
			if a.Wavefront != 64 {
				t.Errorf("%s: %v wavefront = %d, want 64", a.GFX, a.Family, a.Wavefront)
			}
			if !a.Family.Datacenter() {
				t.Errorf("%s: %v not classified as datacenter", a.GFX, a.Family)
			}
		}
	}
}

// TestROCmOffloadArchNormalization checks that the offload-arch selector accepts the noisy
// forms ROCm actually reports (case, whitespace, a target-feature suffix) and canonicalizes
// them to the bare gfx token hipcc wants — while still rejecting an unsupported part.
func TestROCmOffloadArchNormalization(t *testing.T) {
	for _, in := range []string{"gfx90a", "GFX90A", "  gfx90a ", "gfx90a:sramecc+:xnack-"} {
		got, ok := ROCmOffloadArch(in)
		if !ok || got != "gfx90a" {
			t.Errorf("ROCmOffloadArch(%q) = (%q,%v), want (gfx90a,true)", in, got, ok)
		}
	}
	for _, in := range []string{"gfx700", "sm_90", "", "gfx", "rdna3"} {
		if got, ok := ROCmOffloadArch(in); ok {
			t.Errorf("ROCmOffloadArch(%q) = (%q,true), want unsupported", in, got)
		}
	}
}

// TestROCmRX7600IsRDNA3 ties the taxonomy to a real, already-witnessed AMD card: the RX 7600
// (gfx1102) that the Vulkan backend runs on with numerical parity (VULKAN-AMD-RESULTS.md) is
// the consumer-RDNA3 target the ROCm backend will compile for via --offload-arch=gfx1102.
func TestROCmRX7600IsRDNA3(t *testing.T) {
	a, ok := LookupROCmArch("gfx1102")
	if !ok {
		t.Fatal("gfx1102 (RX 7600) must be a supported target")
	}
	if a.Family != ROCmRDNA3 || a.Family.Datacenter() || a.Wavefront != 32 {
		t.Errorf("gfx1102 = %+v, want RDNA3 consumer wave32", a)
	}
	if off, ok := ROCmOffloadArch("gfx1102"); !ok || off != "gfx1102" {
		t.Errorf("ROCmOffloadArch(gfx1102) = (%q,%v), want (gfx1102,true)", off, ok)
	}
}

// TestROCmGfx1151IsRDNA3_5 witnesses AMD Strix Halo (Ryzen AI Max+ 395, 40 CUs RDNA 3.5)
// recognition in the ROCm architecture taxonomy.
func TestROCmGfx1151IsRDNA3_5(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 (Strix Halo APU) must be a supported target")
	}
	if a.Family != ROCmRDNA3_5 {
		t.Errorf("gfx1151 family = %v, want ROCmRDNA3_5", a.Family)
	}
	if a.Family.String() != "RDNA3.5" {
		t.Errorf("gfx1151 family string = %q, want RDNA3.5", a.Family.String())
	}
	if a.Wavefront != 32 {
		t.Errorf("gfx1151 wavefront = %d, want 32", a.Wavefront)
	}
	if a.Family.Datacenter() {
		t.Errorf("gfx1151 classified as datacenter, want false")
	}
	if !a.Family.IsRDNA() {
		t.Errorf("gfx1151 IsRDNA() = false, want true")
	}
	if a.Family.IsCDNA() {
		t.Errorf("gfx1151 IsCDNA() = true, want false")
	}
	if off, ok := ROCmOffloadArch("gfx1151"); !ok || off != "gfx1151" {
		t.Errorf("ROCmOffloadArch(gfx1151) = (%q,%v), want (gfx1151,true)", off, ok)
	}
}

// TestROCmGfx1151CompilerFlagsAndLDS verifies compiler flags and LDS tuning for 4-token speculative tree masks.
func TestROCmGfx1151CompilerFlagsAndLDS(t *testing.T) {
	a, ok := LookupROCmArch("gfx1151")
	if !ok {
		t.Fatal("gfx1151 not found")
	}
	flags := a.CompilerFlags()
	hasWave32 := false
	hasOffload := false
	for _, f := range flags {
		if f == "-mwavefrontsize32" {
			hasWave32 = true
		}
		if f == "--offload-arch=gfx1151" {
			hasOffload = true
		}
	}
	if !hasWave32 {
		t.Errorf("flags %+v missing -mwavefrontsize32", flags)
	}
	if !hasOffload {
		t.Errorf("flags %+v missing --offload-arch=gfx1151", flags)
	}

	lds := a.TuneMTPTreeMaskLDS(4, 128)
	if lds.DraftDepth != 4 {
		t.Errorf("lds.DraftDepth = %d, want 4", lds.DraftDepth)
	}
	if lds.TotalLDSBytes <= 0 || lds.TotalLDSBytes > 65536 {
		t.Errorf("lds.TotalLDSBytes = %d, must fit in 64KB CU LDS", lds.TotalLDSBytes)
	}
	if lds.MaxWavesPerCU != 32 {
		t.Errorf("lds.MaxWavesPerCU = %d, want 32", lds.MaxWavesPerCU)
	}
}
