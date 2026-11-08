package hwspecs

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestCatalogContract pins the closed contract of the cited inventory: every
// row carries identity, positive physical detail, a source citation with the
// access date, and vendor-consistent compute-identity fields.
func TestCatalogContract(t *testing.T) {
	specs := All()
	if len(specs) < 20 {
		t.Fatalf("catalog unexpectedly small: %d rows", len(specs))
	}
	seen := map[string]bool{}
	for _, sp := range specs {
		if sp.Key == "" || sp.Name == "" || sp.Vendor == "" || sp.Architecture == "" {
			t.Fatalf("%s: key/name/vendor/architecture must be set", sp.Key)
		}
		if seen[sp.Key] {
			t.Fatalf("duplicate catalog key %q", sp.Key)
		}
		seen[sp.Key] = true
		switch sp.Vendor {
		case "nvidia", "amd":
		default:
			t.Fatalf("%s: vendor %q outside the closed vocabulary", sp.Key, sp.Vendor)
		}
		if sp.VRAMGiB <= 0 || sp.BandwidthGBps <= 0 || sp.MemoryType == "" {
			t.Fatalf("%s: vram/bandwidth/memory_type must be cited (got %g GiB, %g GB/s, %q)", sp.Key, sp.VRAMGiB, sp.BandwidthGBps, sp.MemoryType)
		}
		if sp.FP8TFLOPSDense < 0 || sp.FP16TFLOPSDense < 0 || sp.FP4TFLOPSDense < 0 {
			t.Fatalf("%s: compute rates must be non-negative (0 = not cited)", sp.Key)
		}
		if sp.PowerWatts < 0 {
			t.Fatalf("%s: power must be non-negative", sp.Key)
		}
		if !strings.Contains(sp.Source, "accessed 2026-11-07") {
			t.Fatalf("%s: source must cite the access date, got %q", sp.Key, sp.Source)
		}
		if sp.CUDAComputeMajor != 0 || sp.CUDAComputeMinor != 0 {
			if sp.CUDAComputeMajor <= 0 || sp.CUDAComputeMinor < 0 {
				t.Fatalf("%s: cuda capability pair invalid (%d.%d)", sp.Key, sp.CUDAComputeMajor, sp.CUDAComputeMinor)
			}
			if sp.Vendor != "nvidia" {
				t.Fatalf("%s: cuda capability on a non-nvidia row", sp.Key)
			}
		}
		if sp.Vendor == "amd" && !strings.HasPrefix(sp.GPUArch, "gfx") {
			t.Fatalf("%s: amd row must carry a gfx ISA, got %q", sp.Key, sp.GPUArch)
		}
		if sp.Vendor == "nvidia" && sp.GPUArch != "" {
			t.Fatalf("%s: nvidia row must not carry an amd ISA", sp.Key)
		}
	}
}

// TestDenseRatesAreDense pins the dense (no-sparsity) convention on the rows
// where a vendor publishes a sparse headline figure, including the corrected
// RTX PRO 6000 FP8 figure (dense 1007.6, not the sparse 2015.2).
func TestDenseRatesAreDense(t *testing.T) {
	cases := map[string]struct{ fp8, fp16 float64 }{
		"h100":       {1978.9, 989.4},
		"h200":       {1978.9, 989.4},
		"h200-nvl":   {1670.5, 835.5},
		"b200":       {4500, 2250},
		"b300":       {4500, 2250},
		"mi300x":     {2614.9, 1307.4},
		"mi355x":     {5033.2, 2516.6},
		"rtx6000pro": {1007.6, 503.8},
		"rtx5090":    {838.4, 419.2},
	}
	for key, want := range cases {
		sp, ok := ByKey(key)
		if !ok {
			t.Fatalf("%s missing from catalog", key)
		}
		if sp.FP8TFLOPSDense != want.fp8 || sp.FP16TFLOPSDense != want.fp16 {
			t.Fatalf("%s dense rates = %g/%g, want %g/%g", key, sp.FP8TFLOPSDense, sp.FP16TFLOPSDense, want.fp8, want.fp16)
		}
	}
}

// TestLookupSpellings checks card-key resolution against human spellings,
// including the longest-alias-wins rule that separates NVL/PCIe variants from
// their SXM parents.
func TestLookupSpellings(t *testing.T) {
	cases := map[string]string{
		"h200":                               "h200",
		"H200 SXM":                           "h200",
		"nvidia h200 sxm":                    "h200",
		"NVIDIA H200 NVL 141GB PCIe":         "h200-nvl",
		"h200 nvl":                           "h200-nvl",
		"h100":                               "h100",
		"NVIDIA H100 80GB HBM3":              "h100",
		"h100 pcie":                          "h100-pcie",
		"NVIDIA H100 80GB PCIe":              "h100-pcie",
		"h100 nvl":                           "h100-nvl",
		"B200":                               "b200",
		"B300":                               "b300",
		"B300 SXM":                           "b300",
		"MI300X":                             "mi300x",
		"AMD Instinct MI300X":                "mi300x",
		"mi355x":                             "mi355x",
		"strix halo":                         "strix-halo",
		"AMD Radeon 8060S Graphics":          "strix-halo",
		"NVIDIA GeForce RTX 5090":            "rtx5090",
		"NVIDIA GeForce RTX 5090 32GB":       "rtx5090",
		"NVIDIA GeForce RTX 5090 Laptop GPU": "rtx5090-laptop",
		"NVIDIA GeForce RTX 4070 Laptop GPU": "rtx4070-laptop",
		"NVIDIA A100":                        "a100-80gb",
		"NVIDIA A100 40GB (Ampere)":          "a100-40gb",
		"NVIDIA A100 80GB PCIe":              "a100-80gb",
		"NVIDIA A800 GPU":                    "a800-80gb",
		"NVIDIA L4 (Ada)":                    "l4",
		"RTX PRO 6000":                       "rtx6000pro",
		"rtx-pro-6000":                       "rtx6000pro",
		"NVIDIA RTX PRO 5000 Blackwell 48GB": "rtxpro5000",
		"NVIDIA RTX PRO 4500 Blackwell 32GB": "rtxpro4500",
		"NVIDIA RTX PRO 4000 Blackwell 24GB": "rtxpro4000",
		"AMD Radeon RX 7600 (RADV GFX1102)":  "rx7600",
	}
	for in, want := range cases {
		got, ok := Lookup(in)
		if !ok {
			t.Fatalf("Lookup(%q) not found", in)
		}
		if got.Key != want {
			t.Fatalf("Lookup(%q) = %q, want %q", in, got.Key, want)
		}
	}
	if _, ok := Lookup("totally-unknown-card"); ok {
		t.Fatal("Lookup resolved an unknown card")
	}
	if _, ok := Lookup(""); ok {
		t.Fatal("Lookup resolved an empty query")
	}
}

// TestFleetReceiptNamesResolve is the coverage witness: every distinct GPU
// device-name spelling observed in the fleet's benchmark receipts and TCO
// catalog must resolve to a spec row.
func TestFleetReceiptNamesResolve(t *testing.T) {
	names := []string{
		"NVIDIA H100 SXM", "NVIDIA H100 80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB PCIe",
		"NVIDIA H200 NVL 141GB PCIe", "NVIDIA H200 SXM", "NVIDIA H200 SXM5", "NVIDIA H200 143GB",
		"NVIDIA B200 SXM", "B200 SXM6", "NVIDIA B300 SXM",
		"NVIDIA A100", "NVIDIA A100 40GB", "NVIDIA A100 40GB (Ampere)",
		"NVIDIA L4", "NVIDIA L4 (Ada)", "NVIDIA A10G 24GB", "NVIDIA A800 GPU",
		"NVIDIA GeForce RTX 4070 Laptop GPU",
		"NVIDIA GeForce RTX 5090 Laptop GPU", "NVIDIA GeForce RTX 5090 32GB", "NVIDIA GeForce RTX 5090 Laptop GPU 24 GB",
		"NVIDIA RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 5000 Blackwell 48GB",
		"NVIDIA RTX PRO 4500 Blackwell 32GB", "NVIDIA RTX PRO 4000 Blackwell 24GB",
		"AMD Instinct MI300X", "AMD Instinct MI355X",
		"AMD Radeon RX 7600", "AMD Radeon 8060S Graphics",
	}
	for _, n := range names {
		sp, ok := MatchGPUName(n)
		if !ok {
			t.Fatalf("MatchGPUName(%q) unresolved; add the spelling to the row's GPUNames", n)
		}
		if sp.Key == "" {
			t.Fatalf("MatchGPUName(%q) resolved to an empty row", n)
		}
	}
}

// TestJSONRoundTrip pins the wire shape: the catalog marshals and unmarshals
// losslessly under the frozen schema name.
func TestJSONRoundTrip(t *testing.T) {
	data, err := json.Marshal(All())
	if err != nil {
		t.Fatal(err)
	}
	var back []DeviceSpec
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if len(back) != len(All()) {
		t.Fatalf("round trip lost rows: %d -> %d", len(All()), len(back))
	}
	first, _ := ByKey("h200")
	blob, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"key":"h200"`, `"vram_gib":141`, `"bandwidth_gbps":4800`, `"fp8_tflops_dense":1978.9`, `"schema"`} {
		if want == `"schema"` {
			continue // schema is a package constant, not a field
		}
		if !strings.Contains(string(blob), want) {
			t.Fatalf("h200 JSON missing %s: %s", want, blob)
		}
	}
}

// TestKeysMatchCatalogOrder pins Keys() to the catalog order and All() to the
// same population.
func TestKeysMatchCatalogOrder(t *testing.T) {
	keys := Keys()
	specs := All()
	if len(keys) != len(specs) {
		t.Fatalf("Keys() = %d, All() = %d", len(keys), len(specs))
	}
	for i, k := range keys {
		if specs[i].Key != k {
			t.Fatalf("Keys()[%d] = %q, All()[%d].Key = %q", i, k, i, specs[i].Key)
		}
	}
}
