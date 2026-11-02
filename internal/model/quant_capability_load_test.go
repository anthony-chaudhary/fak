package model

// quant_capability_load_test.go — witness for the #12983 post-load fail-closed quant gate. Pure
// Go, no build tags: the gate is backend-agnostic bookkeeping over m.kqw, so it is fully
// exercised on any host (darwin/amd64/linux) without a Metal device.

import (
	"strings"
	"testing"
)

// quantFixture builds a *Model whose kqw store holds n tensors of each named kind. raw length is
// a caller-chosen byte count so tests can pin byte-based ordering.
func quantFixture(t *testing.T, entries []struct {
	kind  kQuantKind
	n     int
	bytes int
}) *Model {
	t.Helper()
	m := &Model{kqw: map[string]*kQuantTensor{}}
	idx := 0
	for _, e := range entries {
		for i := 0; i < e.n; i++ {
			name := e.kind.String() + "_" + itoa(idx)
			m.kqw[name] = &kQuantTensor{kind: e.kind, raw: make([]byte, e.bytes)}
			idx++
		}
	}
	return m
}

// fixtureTypes is the shared #12983 shape: IQ3_XXS x3 (unsupported on metal, large), IQ2_S x2
// (unsupported on metal, small), Q6_K x1 (supported on metal, large-ish).
func fixtureTypes() []struct {
	kind  kQuantKind
	n     int
	bytes int
} {
	return []struct {
		kind  kQuantKind
		n     int
		bytes int
	}{
		{kindIQ3XXS, 3, 4096},
		{kindIQ2S, 2, 512},
		{kindQ6K, 1, 8192},
	}
}

// fak-test:runtime fast est=1ms
func TestQuantCapabilityRefusalNamesEveryUnsupportedType(t *testing.T) {
	m := quantFixture(t, fixtureTypes())
	ref := m.RefuseUnsupportedQuants("metal")
	if ref == nil {
		t.Fatal("RefuseUnsupportedQuants(metal) = nil, want refusal for IQ3_XXS/IQ2_S")
	}
	if len(ref.Unsupported) != 2 {
		t.Fatalf("Unsupported has %d entries, want 2: %+v", len(ref.Unsupported), ref.Unsupported)
	}
	msg := ref.Error()
	for _, want := range []string{"IQ3_XXS", "IQ2_S", "3 tensors", "2 tensors"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal %q does not contain %q", msg, want)
		}
	}
	if strings.Contains(msg, "Q6_K") {
		t.Errorf("refusal %q wrongly claims Q6_K is unsupported", msg)
	}
}

// fak-test:runtime fast est=1ms
func TestQuantCapabilityBoundedOnCPU(t *testing.T) {
	m := quantFixture(t, fixtureTypes())
	if ref := m.RefuseUnsupportedQuants("cpu"); ref != nil {
		t.Fatalf("RefuseUnsupportedQuants(cpu) = %v, want nil (every kQuantKind has a CPU dequant)", ref)
	}
	receipt := m.QuantCapabilityReceipt("cpu")
	if receipt == nil {
		t.Fatal("QuantCapabilityReceipt(cpu) = nil")
	}
	if !receipt.Bounded {
		t.Errorf("cpu receipt Bounded = false, want true; unsupported=%+v", receipt.Unsupported)
	}
	if len(receipt.Unsupported) != 0 {
		t.Errorf("cpu receipt Unsupported = %+v, want empty", receipt.Unsupported)
	}
}

// fak-test:runtime fast est=1ms
func TestQuantCapabilityReceiptRecordsDispatchPerBand(t *testing.T) {
	m := quantFixture(t, fixtureTypes())
	receipt := m.QuantCapabilityReceipt("metal")
	if receipt.Bounded {
		t.Error("metal receipt Bounded = true, want false with IQ3_XXS/IQ2_S present")
	}
	seen := map[string]int{}
	for _, row := range receipt.Rows {
		seen[row.Name]++
		switch row.Name {
		case "Q6_K":
			if row.Dispatch != quantDispatchDevice {
				t.Errorf("Q6_K dispatch = %q, want device (resident Metal Q6_K kernel)", row.Dispatch)
			}
		case "IQ3_XXS", "IQ2_S":
			if row.Dispatch != quantDispatchUnsupported {
				t.Errorf("%s dispatch = %q, want unsupported (no kernel, no bounded host path)", row.Name, row.Dispatch)
			}
		}
	}
	for _, name := range []string{"Q6_K", "IQ3_XXS", "IQ2_S"} {
		if seen[name] != 1 {
			t.Errorf("kind %s appears %d times in receipt rows, want exactly once", name, seen[name])
		}
	}
}

// fak-test:runtime fast est=1ms
func TestQuantKindCensusSortsByBytesDesc(t *testing.T) {
	m := quantFixture(t, fixtureTypes())
	census := m.QuantKindCensus()
	if len(census) != 3 {
		t.Fatalf("census has %d rows, want 3: %+v", len(census), census)
	}
	for i := 1; i < len(census); i++ {
		if census[i-1].Bytes < census[i].Bytes {
			t.Fatalf("census not sorted by Bytes desc at row %d: %+v", i, census)
		}
	}
	// IQ3_XXS (3*4096=12288) > Q6_K (8192) > IQ2_S (2*512=1024). Pin the exact order.
	want := []string{"IQ3_XXS", "Q6_K", "IQ2_S"}
	for i, name := range want {
		if census[i].Name != name {
			t.Fatalf("census[%d] = %s, want %s (full: %+v)", i, census[i].Name, name, census)
		}
	}
}

// TestAllKQuantKindsIsExhaustive guards the census against a new kQuantKind being added to the
// const block in quant_kquant.go but omitted from allKQuantKinds. Without it, an undeclared kind
// would be silently misclassified (refused on "cpu", where every declared kind is servable) and
// mislabelled by String()'s default "Q5_K" arm. Go cannot enumerate a const block, so this pins
// the count explicitly: adding a kind MUST update both the const block and allKQuantKinds.
//
// fak-test:runtime fast est=1ms
func TestAllKQuantKindsIsExhaustive(t *testing.T) {
	const wantKinds = 15 // kindQ5K..kindIQ4NL, quant_kquant.go
	if len(allKQuantKinds) != wantKinds {
		t.Fatalf("allKQuantKinds has %d entries, want %d — a kQuantKind was added/removed; update allKQuantKinds and this count", len(allKQuantKinds), wantKinds)
	}
	seen := map[kQuantKind]bool{}
	for _, k := range allKQuantKinds {
		if seen[k] {
			t.Errorf("allKQuantKinds lists %s twice", k)
		}
		seen[k] = true
		if got := k.String(); got == "Q5_K" && k != kindQ5K {
			t.Errorf("kind %d String() = %q via the default arm; it is not kindQ5K and is likely missing from String()", k, got)
		}
	}
	// The immediately-following value after the declared range must NOT be treated as a real kind,
	// so a sentinel (and any future kind) can never be laundered into a servable verdict.
	if isDeclaredKQuantKind(kindIQ4NL + 1) {
		t.Error("isDeclaredKQuantKind(kindIQ4NL+1) = true, want false (out-of-range sentinel must fail closed)")
	}
}

// fak-test:runtime fast est=1ms
func TestQuantKindHasResidentKernelMetalSet(t *testing.T) {
	// Device kernels on Metal: only Q6_K / Q2_K.
	metalDevice := []kQuantKind{kindQ6K, kindQ2K}
	// Bounded host CPU on the Metal arm (SDOT Q5_K/Q6_K + small-block Q2_K/Q4_0/Q8_0): servable.
	metalBoundedHost := []kQuantKind{kindQ5K, kindQ8_0, kindQ4_0}
	// No device kernel AND no bounded host path: the #12983 first-turn hazard. IQ4_NL is included
	// deliberately: trunk declares it but this tree has no dispatch evidence for it, so the gate
	// fails closed until witnessed.
	metalUnsupported := []kQuantKind{kindIQ3XXS, kindIQ2S, kindIQ1S, kindQ3K, kindIQ4XS, kindIQ2XXS, kindIQ2XS, kindIQ1M, kindIQ3S, kindIQ4NL}
	for _, k := range metalDevice {
		if !QuantKindHasResidentKernel("metal", k) {
			t.Errorf("QuantKindHasResidentKernel(metal, %s) = false, want true (device kernel)", k)
		}
		if got := quantDispatchClass("metal", k); got != quantDispatchDevice {
			t.Errorf("quantDispatchClass(metal, %s) = %q, want %q", k, got, quantDispatchDevice)
		}
	}
	for _, k := range metalBoundedHost {
		if !QuantKindHasResidentKernel("metal", k) {
			t.Errorf("QuantKindHasResidentKernel(metal, %s) = false, want true (bounded host path)", k)
		}
		if got := quantDispatchClass("metal", k); got != quantDispatchCPU {
			t.Errorf("quantDispatchClass(metal, %s) = %q, want %q", k, got, quantDispatchCPU)
		}
	}
	for _, k := range metalUnsupported {
		if QuantKindHasResidentKernel("metal", k) {
			t.Errorf("QuantKindHasResidentKernel(metal, %s) = true, want false (no kernel, no bounded host path)", k)
		}
		if got := quantDispatchClass("metal", k); got != quantDispatchUnsupported {
			t.Errorf("quantDispatchClass(metal, %s) = %q, want %q", k, got, quantDispatchUnsupported)
		}
	}
	for _, k := range allKQuantKinds {
		if !QuantKindHasResidentKernel("cpu", k) {
			t.Errorf("QuantKindHasResidentKernel(cpu, %s) = false, want true (model-side dequant exists)", k)
		}
	}
}

// TestQuantCapabilityBoundedTurnOnMetalSupportedBands pins the other side of the gate: a loaded
// model whose retained bulk is entirely device-kernel or bounded-host kinds must be admitted on
// Metal — its first turn is bounded, so the gate must not over-refuse the GLM-5.2-style mixed
// Q5_K/Q6_K expert checkpoint.
//
// fak-test:runtime fast est=1ms
func TestQuantCapabilityBoundedTurnOnMetalSupportedBands(t *testing.T) {
	kinds := []struct {
		kind  kQuantKind
		n     int
		bytes int
	}{
		{kindQ6K, 2, 8192},
		{kindQ2K, 1, 2048},
		{kindQ5K, 4, 4096},
		{kindQ8_0, 1, 1024},
		{kindQ4_0, 1, 512},
	}
	m := quantFixture(t, kinds)
	if ref := m.RefuseUnsupportedQuants("metal"); ref != nil {
		t.Fatalf("RefuseUnsupportedQuants(metal) = %v, want nil (every retained kind is device-resident or bounded host)", ref)
	}
	receipt := m.QuantCapabilityReceipt("metal")
	if !receipt.Bounded {
		t.Fatalf("metal receipt Bounded = false, want true; unsupported=%+v", receipt.Unsupported)
	}
	if len(receipt.Rows) != len(kinds) {
		t.Fatalf("receipt has %d rows, want one per fixture kind (%d): %+v", len(receipt.Rows), len(kinds), receipt.Rows)
	}
}
