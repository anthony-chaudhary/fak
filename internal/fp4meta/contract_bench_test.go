package fp4meta

import (
	"testing"
)

func BenchmarkValidateContract(b *testing.B) {
	descriptor := Descriptor{
		Schema:  SchemaV1,
		Variant: VariantNVFP4,
		Encoding: FloatEncoding{
			Bits:         4,
			SignBits:     1,
			ExponentBits: 2,
			MantissaBits: 1,
			ExponentBias: 1,
			FiniteOnly:   true,
		},
		Scale: BlockScale{
			Scope:        ScalePerBlock,
			BlockSize:    16,
			Encoding:     ScaleE4M3,
			ExponentBits: 4,
			ExponentBias: 7,
			ExponentOnly: false,
		},
		Accumulator: AccumulatorFP16,
		Artifact: Artifact{
			Format:  "safetensors",
			Version: "1.0",
		},
		Recipe: Recipe{
			ID:      "nvfp4-default",
			Version: "1.0",
		},
		Hardware: HardwareEnvelope{
			Vendor:       "nvidia",
			Architecture: "sm100",
			Measured:     true,
			Witness:      "sha256:benchmark-witness",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := validate(descriptor); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkValidateContractParallel(b *testing.B) {
	descriptor := Descriptor{
		Schema:  SchemaV1,
		Variant: VariantNVFP4,
		Encoding: FloatEncoding{
			Bits:         4,
			SignBits:     1,
			ExponentBits: 2,
			MantissaBits: 1,
			ExponentBias: 1,
			FiniteOnly:   true,
		},
		Scale: BlockScale{
			Scope:        ScalePerBlock,
			BlockSize:    16,
			Encoding:     ScaleE4M3,
			ExponentBits: 4,
			ExponentBias: 7,
			ExponentOnly: false,
		},
		Accumulator: AccumulatorFP16,
		Artifact: Artifact{
			Format:  "safetensors",
			Version: "1.0",
		},
		Recipe: Recipe{
			ID:      "nvfp4-default",
			Version: "1.0",
		},
		Hardware: HardwareEnvelope{
			Vendor:       "nvidia",
			Architecture: "sm100",
			Measured:     true,
			Witness:      "sha256:benchmark-witness",
		},
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := validate(descriptor); err != nil {
				b.Fatal(err)
			}
		}
	})
}
