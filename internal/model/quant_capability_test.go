package model

import (
	"strings"
	"testing"
)

// quant_capability_test.go — witness for the #12983 fail-closed quant load gate. Pure Go, no
// build tags: the gate is backend-agnostic bookkeeping over m.kqw, so it is fully exercised on
// any host (darwin/amd64/linux) without a Metal device.

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
			if row.Dispatch != "device" {
				t.Errorf("Q6_K dispatch = %q, want device (resident Metal Q6_K kernel)", row.Dispatch)
			}
		case "IQ3_XXS", "IQ2_S":
			if row.Dispatch != "unsupported" {
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

func TestQuantCapabilityCensusSortsByBytesDesc(t *testing.T) {
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
	// Q6_K (8192) > IQ3_XXS (3*4096=12288)? No: 12288 > 8192 > 512. Pin the exact order.
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
func TestAllKQuantKindsIsExhaustive(t *testing.T) {
	const wantKinds = 14 // kindQ5K..kindIQ3S, quant_kquant.go
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
	if isDeclaredKQuantKind(kindIQ3S + 1) {
		t.Error("isDeclaredKQuantKind(kindIQ3S+1) = true, want false (out-of-range sentinel must fail closed)")
	}
}

func TestQuantKindHasResidentKernelMetalSet(t *testing.T) {
	// Device kernels on Metal: only Q6_K / Q2_K.
	metalDevice := []kQuantKind{kindQ6K, kindQ2K}
	// Bounded host CPU on the Metal arm (SDOT Q5_K/Q6_K + small-block Q2_K/Q4_0/Q8_0): servable.
	metalBoundedHost := []kQuantKind{kindQ5K, kindQ8_0, kindQ4_0}
	// No device kernel AND no bounded host path: the #12983 first-turn hazard.
	metalUnsupported := []kQuantKind{kindIQ3XXS, kindIQ2S, kindIQ1S, kindQ3K, kindIQ4XS, kindIQ2XXS, kindIQ2XS, kindIQ1M, kindIQ3S}
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

// TestQuantCapabilityBoundedTurnOnMetalIQBand is the #12983 regression witness named by the issue's
// witness command (`-run 'QuantCapabilityRefusal|BoundedTurn'`). It pins the aggregate-hazard shape
// from the issue's own type census: a UD-mix artifact carrying the IQ family (the compute-dominant
// bands with no kernel and no bounded host path) MUST be refused on the Metal arm, while the
// ordinary Q4_K-mix minority (Q5_K/Q6_K) must NOT be — that minority is intentionally served by the
// bounded host SDOT path (metal_q4k_on.go:448).
func TestQuantCapabilityBoundedTurnOnMetalIQBand(t *testing.T) {
	// Shape mirrors the issue census: IQ bands dominate, Q6_K/Q5_K minority are kernel/bounded-host.
	entries := []struct {
		kind  kQuantKind
		n     int
		bytes int
	}{
		{kindIQ3XXS, 113, 98 * 1024},
		{kindIQ2S, 68, 82 * 1024},
		{kindIQ3S, 58, 98 * 1024},
		{kindIQ2XXS, 49, 66 * 1024},
		{kindIQ2XS, 35, 74 * 1024},
		{kindIQ1S, 21, 50 * 1024},
		{kindIQ4XS, 20, 136 * 1024},
		{kindQ3K, 6, 110 * 1024},
		{kindQ6K, 1, 210 * 1024},
		{kindQ5K, 1, 176 * 1024},
	}
	m := quantFixture(t, entries)
	ref := m.RefuseUnsupportedQuants("metal")
	if ref == nil {
		t.Fatal("RefuseUnsupportedQuants(metal) = nil, want refusal for the IQ/Q3_K band")
	}
	// Exactly the 8 IQ/Q3_K kinds are refused; Q6_K and Q5_K are not.
	if len(ref.Unsupported) != 8 {
		t.Fatalf("Unsupported has %d entries, want 8 (the IQ/Q3_K band): %+v", len(ref.Unsupported), ref.Unsupported)
	}
	msg := ref.Error()
	for _, want := range []string{"IQ3_XXS", "IQ2_S", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ1_S", "IQ4_XS", "Q3_K", "113 tensors", "68 tensors"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal %q does not name %q", msg, want)
		}
	}
	for _, banned := range []string{"Q6_K", "Q5_K"} {
		if strings.Contains(msg, banned) {
			t.Errorf("refusal %q wrongly names bounded-host %s as unsupported", msg, banned)
		}
	}
	// The same model on a pure-CPU serve is bounded (every kind has a host dequant).
	if r := m.RefuseUnsupportedQuants("cpu"); r != nil {
		t.Errorf("RefuseUnsupportedQuants(cpu) = %v, want nil", r)
	}
}
