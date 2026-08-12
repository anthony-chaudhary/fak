package model

// fp4meta_test.go pins the four dispositions the FP4 metadata contract keeps apart. The split
// that matters is abstain vs refuse: an artifact this build cannot READ (a newer schema, a
// vocabulary word it has no meaning for) must not be reported as a broken artifact, and a
// document that contradicts the published definition of the format it names must not be
// accepted just because every field parsed.

import (
	"encoding/json"
	"strings"
	"testing"
)

// fp4NVFP4 is a well-formed NVFP4 document on a target that runs it natively: E2M1 elements in
// 16-element blocks, an E4M3 per-block scale under a per-tensor scale (2 levels), fp32
// accumulate. Each case below mutates exactly one thing, so the mutation is the cause.
func fp4NVFP4() FP4Metadata {
	return FP4Metadata{
		Schema:      FP4MetadataSchema,
		Format:      FP4FormatNVFP4,
		Encoding:    "e2m1",
		BlockScale:  FP4BlockScale{Elements: 16, Encoding: FP4ScaleE4M3, Levels: 2},
		Exponent:    FP4Exponent{Bits: 2, Bias: 1},
		Accumulator: FP4AccumulatorFP32,
		Hardware:    FP4HardwareCapability{Runtime: "fak-native", Accelerator: "sm_100", NativeDecode: true, NativeGEMM: true},
		ClaimScope:  FP4ClaimArtifact,
	}
}

// fp4MXFP4 is the OCP microscaling shape: E2M1 elements in 32-element blocks under a single
// E8M0 power-of-two scale. It is here so a per-format rule that merely hard-coded NVFP4's
// geometry would fail.
func fp4MXFP4() FP4Metadata {
	m := fp4NVFP4()
	m.Format = FP4FormatMXFP4
	m.BlockScale = FP4BlockScale{Elements: 32, Encoding: FP4ScaleE8M0, Levels: 1}
	return m
}

func TestAdjudicateFP4Metadata(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FP4Metadata)
		want   FP4Disposition
		reason FP4Reason
		detail string // a substring an operator can act on
	}{
		{name: "nvfp4/native", mutate: func(*FP4Metadata) {}, want: FP4Accept, reason: FP4ReasonSupported},
		{name: "mxfp4/native", mutate: func(m *FP4Metadata) { *m = fp4MXFP4() }, want: FP4Accept, reason: FP4ReasonSupported},
		{
			name:   "schema/newer-build-abstains",
			mutate: func(m *FP4Metadata) { m.Schema = "fp4meta/v2" },
			want:   FP4Abstain, reason: FP4ReasonUnknownSchema, detail: "fp4meta/v2",
		},
		{
			name:   "format/unknown-abstains",
			mutate: func(m *FP4Metadata) { m.Format = "nvfp3" },
			want:   FP4Abstain, reason: FP4ReasonUnknownFormat, detail: "nvfp3",
		},
		{
			name:   "scale-encoding/unknown-abstains",
			mutate: func(m *FP4Metadata) { m.BlockScale.Encoding = "ue5m2" },
			want:   FP4Abstain, reason: FP4ReasonUnknownFormat, detail: "ue5m2",
		},
		{
			name:   "accumulator/unknown-abstains",
			mutate: func(m *FP4Metadata) { m.Accumulator = "fp24" },
			want:   FP4Abstain, reason: FP4ReasonUnknownFormat, detail: "fp24",
		},
		{
			name:   "block/non-power-of-two-refuses",
			mutate: func(m *FP4Metadata) { m.Format = FP4FormatE2M1; m.BlockScale.Elements = 24 },
			want:   FP4Refuse, reason: FP4ReasonMalformed, detail: "power of two",
		},
		{
			name:   "block/scale-with-no-levels-refuses",
			mutate: func(m *FP4Metadata) { m.Format = FP4FormatE2M1; m.BlockScale.Levels = 0 },
			want:   FP4Refuse, reason: FP4ReasonMalformed, detail: "0 scale levels",
		},
		{
			name:   "block/no-scale-but-a-block-refuses",
			mutate: func(m *FP4Metadata) { m.Format = FP4FormatE2M1; m.BlockScale.Encoding = FP4ScaleNone },
			want:   FP4Refuse, reason: FP4ReasonMalformed, detail: "declares elements=16",
		},
		{
			name:   "exponent/too-wide-for-4-bits-refuses",
			mutate: func(m *FP4Metadata) { m.Exponent.Bits = 4 },
			want:   FP4Refuse, reason: FP4ReasonMalformed, detail: "does not fit a 4-bit element",
		},
		{
			name:   "mxfp4/nvfp4-block-size-refuses",
			mutate: func(m *FP4Metadata) { *m = fp4MXFP4(); m.BlockScale.Elements = 16 },
			want:   FP4Refuse, reason: FP4ReasonUnsupportedCombination, detail: "32-element blocks",
		},
		{
			name:   "nvfp4/mxfp4-scale-encoding-refuses",
			mutate: func(m *FP4Metadata) { m.BlockScale.Encoding = FP4ScaleE8M0 },
			want:   FP4Refuse, reason: FP4ReasonUnsupportedCombination, detail: "e4m3 scale over 16-element blocks",
		},
		{
			name:   "e2m1/wrong-bias-refuses",
			mutate: func(m *FP4Metadata) { m.Exponent.Bias = 7 },
			want:   FP4Refuse, reason: FP4ReasonUnsupportedCombination, detail: "bias 1",
		},
		{
			name:   "accumulator/fp16-refuses",
			mutate: func(m *FP4Metadata) { m.Accumulator = FP4AccumulatorFP16 },
			want:   FP4Refuse, reason: FP4ReasonUnsupportedCombination, detail: "accumulate in fp32",
		},
		{
			name:   "claim/measured-hardware-without-a-device-refuses",
			mutate: func(m *FP4Metadata) { m.ClaimScope = FP4ClaimMeasuredHardware; m.Hardware.Accelerator = "" },
			want:   FP4Refuse, reason: FP4ReasonMalformed, detail: "measured_hardware requires",
		},
		{
			name:   "claim/measured-hardware-with-a-device-accepts",
			mutate: func(m *FP4Metadata) { m.ClaimScope = FP4ClaimMeasuredHardware },
			want:   FP4Accept, reason: FP4ReasonSupported,
		},
		{
			name: "claim/producer-delegates",
			mutate: func(m *FP4Metadata) {
				m.ClaimScope = FP4ClaimRuntimeDelegated
				m.Hardware.Runtime = "tensorrt-llm-pytorch"
			},
			want: FP4Delegate, reason: FP4ReasonRuntimeDelegation, detail: "tensorrt-llm-pytorch",
		},
		{
			name:   "hardware/no-native-gemm-delegates",
			mutate: func(m *FP4Metadata) { m.Hardware.NativeGEMM = false },
			want:   FP4Delegate, reason: FP4ReasonRuntimeDelegation, detail: "native_gemm=false",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := fp4NVFP4()
			tc.mutate(&m)
			got := AdjudicateFP4Metadata(m)
			if got.Disposition != tc.want || got.Reason != tc.reason {
				t.Fatalf("AdjudicateFP4Metadata = %s/%s, want %s/%s (detail %q)", got.Disposition, got.Reason, tc.want, tc.reason, got.Detail)
			}
			if tc.detail != "" && !strings.Contains(got.Detail, tc.detail) {
				t.Fatalf("detail = %q, want it to name %q so an operator can act on the refusal without reading this package", got.Detail, tc.detail)
			}
			// Only a readable, coherent document hands its metadata on; an abstain or a refusal
			// must not pass a document the caller could mistake for a vetted one.
			if wantEcho := tc.want == FP4Accept || tc.want == FP4Delegate; wantEcho != (got.Metadata != nil) {
				t.Fatalf("metadata echoed = %v, want %v for disposition %s", got.Metadata != nil, wantEcho, got.Disposition)
			}
		})
	}
}

// TestParseFP4MetadataIsStrict pins the wire half: a document this build would have to read
// PARTIALLY is refused, because the field it dropped is exactly the one that changes how the
// payload must be decoded.
func TestParseFP4MetadataIsStrict(t *testing.T) {
	good, err := json.Marshal(fp4NVFP4())
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if got := ParseFP4Metadata(good); got.Disposition != FP4Accept {
		t.Fatalf("round-tripped fixture = %s/%s (%s), want accept", got.Disposition, got.Reason, got.Detail)
	}

	cases := []struct {
		name string
		data string
		want FP4Disposition
	}{
		{name: "unknown-field", data: `{"schema":"fp4meta/v1","format":"nvfp4","tile_shape":[128,128]}`, want: FP4Refuse},
		{name: "trailing-json", data: string(good) + `{"schema":"fp4meta/v1"}`, want: FP4Refuse},
		{name: "truncated", data: `{"schema":"fp4meta/v1","format":`, want: FP4Refuse},
		{name: "not-an-object", data: `["fp4meta/v1"]`, want: FP4Refuse},
		{name: "unknown-schema", data: `{"schema":"fp4meta/v9"}`, want: FP4Abstain},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseFP4Metadata([]byte(tc.data))
			if got.Disposition != tc.want {
				t.Fatalf("ParseFP4Metadata(%s) = %s/%s (%s), want %s", tc.name, got.Disposition, got.Reason, got.Detail, tc.want)
			}
			if got.Metadata != nil {
				t.Fatalf("a %s parse echoed metadata back; nothing here is trustworthy enough to hand on", got.Disposition)
			}
		})
	}
}
