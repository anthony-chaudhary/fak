package fp4meta

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDecodeE2M1BitPatterns(t *testing.T) {
	want := []float64{0, 0.5, 1, 1.5, 2, 3, 4, 6, math.Copysign(0, -1), -0.5, -1, -1.5, -2, -3, -4, -6}
	for bits, expected := range want {
		got, err := DecodeE2M1(byte(bits))
		if err != nil {
			t.Fatalf("DecodeE2M1(0x%x): %v", bits, err)
		}
		if math.Float64bits(got) != math.Float64bits(expected) {
			t.Errorf("DecodeE2M1(0x%x) = %v (%x), want %v (%x)", bits, got, math.Float64bits(got), expected, math.Float64bits(expected))
		}
	}
	if _, err := DecodeE2M1(0x10); err == nil {
		t.Fatal("DecodeE2M1 accepted a value wider than four bits")
	}
}

func TestJSONGoldenRecords(t *testing.T) {
	capabilities := Capabilities{
		Variants:     []Variant{VariantE2M1, VariantNVFP4, VariantMXFP4},
		ScaleFormats: []ScaleEncoding{ScaleBinary32, ScaleE4M3, ScaleUE8M0},
		Accumulators: []Accumulator{AccumulatorFP16, AccumulatorFP32},
		Hardware:     []string{"nvidia/sm100"},
		Runtime:      true,
	}
	tests := []struct {
		name      string
		variant   Variant
		blockSize int
		scale     ScaleEncoding
	}{
		{name: "e2m1", variant: VariantE2M1, scale: ScaleBinary32},
		{name: "nvfp4", variant: VariantNVFP4, blockSize: 16, scale: ScaleE4M3},
		{name: "mxfp4", variant: VariantMXFP4, blockSize: 32, scale: ScaleUE8M0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", test.name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			descriptor, decision, err := Parse(raw, capabilities)
			if err != nil {
				t.Fatal(err)
			}
			if decision.Outcome != OutcomeAccept || decision.Reason != ReasonSupported {
				t.Fatalf("decision = %#v", decision)
			}
			if descriptor.Variant != test.variant || descriptor.Scale.BlockSize != test.blockSize || descriptor.Scale.Encoding != test.scale {
				t.Fatalf("descriptor conflated variant semantics: %#v", descriptor)
			}
			canonical, err := MarshalCanonical(descriptor)
			if err != nil {
				t.Fatal(err)
			}
			var got, want any
			if err := json.Unmarshal(canonical, &got); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("canonical JSON changed golden record\ngot:  %s\nwant: %s", canonical, raw)
			}
		})
	}
}

func TestUnknownVariantsAndSchemasAbstain(t *testing.T) {
	base := goldenDescriptor(t, "e2m1")

	unknownVariant := base
	unknownVariant.Variant = Variant("vendor-fp4-v9")
	decision := Adjudicate(unknownVariant, Capabilities{})
	if decision.Outcome != OutcomeAbstain || decision.Reason != ReasonUnknownVariant {
		t.Fatalf("unknown variant decision = %#v", decision)
	}

	unknownSchema := base
	unknownSchema.Schema = "fak.fp4meta/v2"
	decision = Adjudicate(unknownSchema, Capabilities{})
	if decision.Outcome != OutcomeAbstain || decision.Reason != ReasonUnknownSchema {
		t.Fatalf("unknown schema decision = %#v", decision)
	}
}

func TestInvalidAndUnsupportedCombinationsAreTyped(t *testing.T) {
	mxfp4 := goldenDescriptor(t, "mxfp4")
	mxfp4.Scale.BlockSize = 16
	decision := Adjudicate(mxfp4, allCapabilities())
	if decision.Outcome != OutcomeRefuse || decision.Reason != ReasonInvalidDescriptor {
		t.Fatalf("invalid microscaling decision = %#v", decision)
	}

	nvfp4 := goldenDescriptor(t, "nvfp4")
	decision = Adjudicate(nvfp4, Capabilities{
		Variants:     []Variant{VariantNVFP4},
		ScaleFormats: []ScaleEncoding{ScaleE4M3},
		Accumulators: []Accumulator{AccumulatorFP16},
		Hardware:     []string{"nvidia/sm100"},
	})
	if decision.Outcome != OutcomeDelegate || decision.Reason != ReasonRuntimeDelegation {
		t.Fatalf("runtime decision = %#v", decision)
	}

	decision = Adjudicate(nvfp4, Capabilities{
		Variants:     []Variant{VariantNVFP4},
		ScaleFormats: []ScaleEncoding{ScaleE4M3},
		Accumulators: []Accumulator{AccumulatorFP16},
		Runtime:      true,
	})
	if decision.Outcome != OutcomeAbstain || decision.Reason != ReasonHardwareUnverified {
		t.Fatalf("hardware decision = %#v", decision)
	}
}

func TestParseRejectsUnknownFieldsAndTrailingValues(t *testing.T) {
	for _, raw := range [][]byte{
		[]byte(`{"schema":"fak.fp4meta/v1","surprise":true}`),
		[]byte(`{} {}`),
	} {
		_, decision, err := Parse(raw, Capabilities{})
		if err == nil || decision.Outcome != OutcomeRefuse || decision.Reason != ReasonInvalidDescriptor {
			t.Fatalf("Parse(%s) = %#v, %v", raw, decision, err)
		}
	}
}

func goldenDescriptor(t *testing.T, name string) Descriptor {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var descriptor Descriptor
	if err := json.Unmarshal(raw, &descriptor); err != nil {
		t.Fatal(err)
	}
	return descriptor
}

func allCapabilities() Capabilities {
	return Capabilities{
		Variants:     []Variant{VariantE2M1, VariantNVFP4, VariantMXFP4},
		ScaleFormats: []ScaleEncoding{ScaleBinary32, ScaleE4M3, ScaleUE8M0},
		Accumulators: []Accumulator{AccumulatorFP16, AccumulatorFP32},
		Hardware:     []string{"nvidia/sm100"},
		Runtime:      true,
	}
}

// fak-test:runtime fast est=1ms
func TestParseNVFP4RejectsAmbiguousScaleFields(t *testing.T) {
	t.Parallel()
	raw, err := json.Marshal(goldenDescriptor(t, "nvfp4"))
	if err != nil {
		t.Fatal(err)
	}
	good := string(raw)
	// Repeated names in separate objects are not duplicates. In this fixture,
	// encoding and scale both declare exponent_bits, while artifact, recipe,
	// and runtime each declare version.
	for _, valid := range []string{
		good,
		strings.Replace(good, `"block_size":16`, `"BLOCK_SIZE":16`, 1),
	} {
		if _, decision, err := Parse([]byte(valid), allCapabilities()); err != nil || decision.Outcome != OutcomeAccept {
			t.Fatalf("valid descriptor = %#v, %v", decision, err)
		}
	}
	for _, tc := range []struct {
		name, old, replacement string
	}{
		{"nested scale", `"block_size":16`, `"block_size":32,"block_size":16`},
		{"escaped scale", `"block_size":16`, `"\u0062lock_size":32,"block_size":16`},
		{"casefold scale", `"block_size":16`, `"BLOCK_SIZE":32,"block_size":16`},
		{"repeated scale object", `"scale":{`, `"scale":{"block_size":32},"scale":{`},
		{"top-level variant", `"variant":"nvfp4"`, `"variant":"mxfp4","variant":"nvfp4"`},
		{"identical scale", `"block_size":16`, `"block_size":16,"block_size":16`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if strings.Count(good, tc.old) != 1 {
				t.Fatalf("fixture must contain %q exactly once", tc.old)
			}
			ambiguous := strings.Replace(good, tc.old, tc.replacement, 1)
			descriptor, decision, err := Parse([]byte(ambiguous), allCapabilities())
			if err == nil || decision.Outcome != OutcomeRefuse || decision.Reason != ReasonInvalidDescriptor {
				t.Fatalf("ambiguous descriptor = %#v, %v; want invalid-descriptor refusal", decision, err)
			}
			if !reflect.DeepEqual(descriptor, Descriptor{}) {
				t.Fatalf("refusal exposed partial descriptor: %#v", descriptor)
			}
		})
	}
}
