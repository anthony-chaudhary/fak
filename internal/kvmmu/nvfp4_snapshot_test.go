package kvmmu_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func makeDeterministicBytes(n int, seed byte) []byte {
	out := make([]byte, n)
	x := seed
	for i := range out {
		x = x*131 + 17
		out[i] = x
	}
	return out
}

// fak-test:runtime fast est=100ms
func TestPackedKVSnapshotRoundTripAndCompression(t *testing.T) {
	t.Parallel()

	const (
		layers  = 4
		seqLen  = 128
		kvHeads = 8
		headDim = 128
	)
	epoch := model.WeightEpochStamp{
		Generation: 7,
		Digest:     "sha256:qwen3.8-epoch7-deadbeef",
	}

	fp32Bytes := int64(layers) * int64(seqLen) * int64(kvHeads) * int64(headDim) * 2 * 4
	fp16Bytes := int64(layers) * int64(seqLen) * int64(kvHeads) * int64(headDim) * 2 * 2

	t.Run("nvfp4_roundtrip_and_compression", func(t *testing.T) {
		kDataLen, kScalesLen, vDataLen, vScalesLen, err := kvmmu.ExpectedPayloadSizes(
			kvmmu.PackedKVDTypeNVFP4, layers, seqLen, kvHeads, headDim,
		)
		if err != nil {
			t.Fatalf("ExpectedPayloadSizes: %v", err)
		}

		orig := &kvmmu.PackedKVSnapshot{
			ModelArch:    "qwen3.8",
			DType:        kvmmu.PackedKVDTypeNVFP4,
			WeightEpoch:  epoch.Generation,
			WeightDigest: epoch.Digest,
			NumLayers:    layers,
			NumKVHeads:   kvHeads,
			HeadDim:      headDim,
			SeqLen:       seqLen,
			KGlobalScale: 1.25,
			VGlobalScale: 0.75,
			KData:        makeDeterministicBytes(kDataLen, 0x11),
			KScales:      makeDeterministicBytes(kScalesLen, 0x22),
			VData:        makeDeterministicBytes(vDataLen, 0x33),
			VScales:      makeDeterministicBytes(vScalesLen, 0x44),
		}

		encoded, err := kvmmu.EncodePackedKVSnapshot(orig)
		if err != nil {
			t.Fatalf("EncodePackedKVSnapshot: %v", err)
		}

		ratioFP32 := float64(fp32Bytes) / float64(len(encoded))
		ratioFP16 := float64(fp16Bytes) / float64(len(encoded))
		if ratioFP32 < 6.5 {
			t.Fatalf("nvfp4 compression vs FP32 = %.3fx, want >= 6.5x (encoded=%d fp32=%d)", ratioFP32, len(encoded), fp32Bytes)
		}
		if ratioFP16 < 3.2 {
			t.Fatalf("nvfp4 compression vs FP16 = %.3fx, want >= 3.2x (encoded=%d fp16=%d)", ratioFP16, len(encoded), fp16Bytes)
		}

		got, err := kvmmu.DecodePackedKVSnapshot(encoded, "qwen3.8", kvmmu.PackedKVDTypeNVFP4, epoch)
		if err != nil {
			t.Fatalf("DecodePackedKVSnapshot: %v", err)
		}
		if got.ModelArch != orig.ModelArch || got.DType != orig.DType ||
			got.WeightEpoch != orig.WeightEpoch || got.WeightDigest != orig.WeightDigest ||
			got.NumLayers != orig.NumLayers || got.NumKVHeads != orig.NumKVHeads ||
			got.HeadDim != orig.HeadDim || got.SeqLen != orig.SeqLen ||
			got.KGlobalScale != orig.KGlobalScale || got.VGlobalScale != orig.VGlobalScale {
			t.Fatalf("decoded metadata mismatch: got %+v, want %+v", got, orig)
		}
		if !bytes.Equal(got.KData, orig.KData) || !bytes.Equal(got.KScales, orig.KScales) ||
			!bytes.Equal(got.VData, orig.VData) || !bytes.Equal(got.VScales, orig.VScales) {
			t.Fatal("decoded payload bytes are not bit-exact")
		}
	})

	t.Run("fp8k_nvfp4v_roundtrip_and_compression", func(t *testing.T) {
		kDataLen, kScalesLen, vDataLen, vScalesLen, err := kvmmu.ExpectedPayloadSizes(
			kvmmu.PackedKVDTypeFP8KNVFP4V, layers, seqLen, kvHeads, headDim,
		)
		if err != nil {
			t.Fatalf("ExpectedPayloadSizes: %v", err)
		}

		orig := &kvmmu.PackedKVSnapshot{
			ModelArch:    "deepseek-v4.1-flash",
			DType:        kvmmu.PackedKVDTypeFP8KNVFP4V,
			WeightEpoch:  epoch.Generation,
			WeightDigest: epoch.Digest,
			NumLayers:    layers,
			NumKVHeads:   kvHeads,
			HeadDim:      headDim,
			SeqLen:       seqLen,
			KGlobalScale: 1.0,
			VGlobalScale: 0.5,
			KData:        makeDeterministicBytes(kDataLen, 0x55),
			KScales:      makeDeterministicBytes(kScalesLen, 0x66),
			VData:        makeDeterministicBytes(vDataLen, 0x77),
			VScales:      makeDeterministicBytes(vScalesLen, 0x88),
		}

		encoded, err := kvmmu.EncodePackedKVSnapshot(orig)
		if err != nil {
			t.Fatalf("EncodePackedKVSnapshot: %v", err)
		}

		ratioFP32 := float64(fp32Bytes) / float64(len(encoded))
		if ratioFP32 < 4.5 {
			t.Fatalf("fp8k_nvfp4v compression vs FP32 = %.3fx, want >= 4.5x (encoded=%d fp32=%d)", ratioFP32, len(encoded), fp32Bytes)
		}

		got, err := kvmmu.DecodePackedKVSnapshot(encoded, "deepseek-v4.1-flash", kvmmu.PackedKVDTypeFP8KNVFP4V, epoch)
		if err != nil {
			t.Fatalf("DecodePackedKVSnapshot: %v", err)
		}
		if !bytes.Equal(got.KData, orig.KData) || len(got.KScales) != 0 ||
			!bytes.Equal(got.VData, orig.VData) || !bytes.Equal(got.VScales, orig.VScales) {
			t.Fatal("decoded fp8k_nvfp4v payload bytes are not bit-exact")
		}
	})

	t.Run("tamper_and_binding_refusal", func(t *testing.T) {
		kDataLen, kScalesLen, vDataLen, vScalesLen, err := kvmmu.ExpectedPayloadSizes(
			kvmmu.PackedKVDTypeNVFP4, layers, 16, kvHeads, headDim,
		)
		if err != nil {
			t.Fatalf("ExpectedPayloadSizes: %v", err)
		}
		orig := &kvmmu.PackedKVSnapshot{
			ModelArch:    "qwen3.8",
			DType:        kvmmu.PackedKVDTypeNVFP4,
			WeightEpoch:  epoch.Generation,
			WeightDigest: epoch.Digest,
			NumLayers:    layers,
			NumKVHeads:   kvHeads,
			HeadDim:      headDim,
			SeqLen:       16,
			KGlobalScale: 1.0,
			VGlobalScale: 1.0,
			KData:        makeDeterministicBytes(kDataLen, 0x10),
			KScales:      makeDeterministicBytes(kScalesLen, 0x20),
			VData:        makeDeterministicBytes(vDataLen, 0x30),
			VScales:      makeDeterministicBytes(vScalesLen, 0x40),
		}
		encoded, err := kvmmu.EncodePackedKVSnapshot(orig)
		if err != nil {
			t.Fatalf("EncodePackedKVSnapshot: %v", err)
		}

		// Corrupt payload byte -> ErrSnapshotCorrupt
		tamperedPayload := append([]byte(nil), encoded...)
		tamperedPayload[len(tamperedPayload)-40] ^= 0xff
		if _, err := kvmmu.DecodePackedKVSnapshot(tamperedPayload, "qwen3.8", kvmmu.PackedKVDTypeNVFP4, epoch); !errors.Is(err, kvmmu.ErrSnapshotCorrupt) {
			t.Fatalf("tampered payload: got err=%v, want ErrSnapshotCorrupt", err)
		}

		// Corrupt checksum byte -> ErrSnapshotCorrupt
		tamperedSum := append([]byte(nil), encoded...)
		tamperedSum[len(tamperedSum)-1] ^= 0x01
		if _, err := kvmmu.DecodePackedKVSnapshot(tamperedSum, "qwen3.8", kvmmu.PackedKVDTypeNVFP4, epoch); !errors.Is(err, kvmmu.ErrSnapshotCorrupt) {
			t.Fatalf("tampered checksum: got err=%v, want ErrSnapshotCorrupt", err)
		}

		// Truncated frame -> ErrSnapshotCorrupt
		if _, err := kvmmu.DecodePackedKVSnapshot(encoded[:20], "qwen3.8", kvmmu.PackedKVDTypeNVFP4, epoch); !errors.Is(err, kvmmu.ErrSnapshotCorrupt) {
			t.Fatalf("truncated frame: got err=%v, want ErrSnapshotCorrupt", err)
		}

		// Mismatched WeightEpoch generation -> ErrWeightEpochMismatch
		wrongGen := model.WeightEpochStamp{Generation: epoch.Generation + 1, Digest: epoch.Digest}
		if _, err := kvmmu.DecodePackedKVSnapshot(encoded, "qwen3.8", kvmmu.PackedKVDTypeNVFP4, wrongGen); !errors.Is(err, kvmmu.ErrWeightEpochMismatch) {
			t.Fatalf("mismatched generation: got err=%v, want ErrWeightEpochMismatch", err)
		}

		// Mismatched WeightDigest -> ErrWeightEpochMismatch
		wrongDigest := model.WeightEpochStamp{Generation: epoch.Generation, Digest: "sha256:other"}
		if _, err := kvmmu.DecodePackedKVSnapshot(encoded, "qwen3.8", kvmmu.PackedKVDTypeNVFP4, wrongDigest); !errors.Is(err, kvmmu.ErrWeightEpochMismatch) {
			t.Fatalf("mismatched digest: got err=%v, want ErrWeightEpochMismatch", err)
		}

		// Mismatched ModelArch -> ErrSnapshotArchMismatch
		if _, err := kvmmu.DecodePackedKVSnapshot(encoded, "inkling2", kvmmu.PackedKVDTypeNVFP4, epoch); !errors.Is(err, kvmmu.ErrSnapshotArchMismatch) {
			t.Fatalf("mismatched arch: got err=%v, want ErrSnapshotArchMismatch", err)
		}

		// Mismatched DType -> ErrSnapshotDTypeMismatch
		if _, err := kvmmu.DecodePackedKVSnapshot(encoded, "qwen3.8", kvmmu.PackedKVDTypeFP8KNVFP4V, epoch); !errors.Is(err, kvmmu.ErrSnapshotDTypeMismatch) {
			t.Fatalf("mismatched dtype: got err=%v, want ErrSnapshotDTypeMismatch", err)
		}
	})
}
