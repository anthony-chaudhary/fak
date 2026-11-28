package kvmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// PackedKVBlockSize is the element group size for block-16 NVFP4 scales.
	PackedKVBlockSize = 16

	// PackedKVSnapshotVersion is the wire format version emitted by EncodePackedKVSnapshot.
	PackedKVSnapshotVersion uint16 = 1

	packedKVSnapshotMagic = "FAKNVKV1"
)

// PackedKVDType identifies the KV quantization layout inside a packed snapshot.
type PackedKVDType string

const (
	// PackedKVDTypeNVFP4 stores both K and V as packed block-16 E2M1 codes with E4M3FN scales.
	PackedKVDTypeNVFP4 PackedKVDType = "nvfp4"
	// PackedKVDTypeFP8KNVFP4V stores K as FP8 (1 byte/elem) and V as packed block-16 NVFP4.
	PackedKVDTypeFP8KNVFP4V PackedKVDType = "fp8k_nvfp4v"
)

var (
	// ErrSnapshotCorrupt is returned when snapshot bytes fail magic, framing, or SHA-256 verification.
	ErrSnapshotCorrupt = errors.New("kvmmu: packed KV snapshot corrupt or checksum mismatch")
	// ErrSnapshotInvalid is returned when snapshot metadata or payload sizes violate layout invariants.
	ErrSnapshotInvalid = errors.New("kvmmu: invalid packed KV snapshot metadata or payload")
	// ErrSnapshotArchMismatch is returned when a decoded snapshot's model arch does not match expected.
	ErrSnapshotArchMismatch = errors.New("kvmmu: packed KV snapshot model arch mismatch")
	// ErrSnapshotDTypeMismatch is returned when a decoded snapshot's KV dtype does not match expected.
	ErrSnapshotDTypeMismatch = errors.New("kvmmu: packed KV snapshot dtype mismatch")
)

// PackedKVSnapshot holds a serialized block-16 NVFP4 (or asymmetric FP8-K / NVFP4-V) KV cache
// bound to a specific model architecture, KV dtype, and weight epoch state.
type PackedKVSnapshot struct {
	Version      uint16
	ModelArch    string
	DType        PackedKVDType
	WeightEpoch  uint64
	WeightDigest string
	NumLayers    int
	NumKVHeads   int
	HeadDim      int
	SeqLen       int
	KGlobalScale float32
	VGlobalScale float32
	KData        []byte
	KScales      []byte
	VData        []byte
	VScales      []byte
}

// ExpectedPayloadSizes returns the exact byte lengths required for KData, KScales, VData, and VScales
// given the tensor geometry and KV dtype.
func ExpectedPayloadSizes(dtype PackedKVDType, layers, seqLen, kvHeads, headDim int) (kData, kScales, vData, vScales int, err error) {
	if layers <= 0 || kvHeads <= 0 || headDim <= 0 || seqLen < 0 {
		return 0, 0, 0, 0, fmt.Errorf("%w: invalid geometry layers=%d seq=%d heads=%d dim=%d",
			ErrSnapshotInvalid, layers, seqLen, kvHeads, headDim)
	}
	if headDim%PackedKVBlockSize != 0 {
		return 0, 0, 0, 0, fmt.Errorf("%w: headDim=%d must be divisible by %d",
			ErrSnapshotInvalid, headDim, PackedKVBlockSize)
	}
	elems := layers * seqLen * kvHeads * headDim
	nvfp4Data := elems / 2
	nvfp4Scales := elems / PackedKVBlockSize

	switch dtype {
	case PackedKVDTypeNVFP4:
		return nvfp4Data, nvfp4Scales, nvfp4Data, nvfp4Scales, nil
	case PackedKVDTypeFP8KNVFP4V:
		return elems, 0, nvfp4Data, nvfp4Scales, nil
	default:
		return 0, 0, 0, 0, fmt.Errorf("%w: unsupported KV dtype %q", ErrSnapshotInvalid, dtype)
	}
}

func dtypeToTag(dtype PackedKVDType) (uint8, error) {
	switch dtype {
	case PackedKVDTypeNVFP4:
		return 1, nil
	case PackedKVDTypeFP8KNVFP4V:
		return 2, nil
	default:
		return 0, fmt.Errorf("%w: unsupported KV dtype %q", ErrSnapshotInvalid, dtype)
	}
}

func tagToDType(tag uint8) (PackedKVDType, error) {
	switch tag {
	case 1:
		return PackedKVDTypeNVFP4, nil
	case 2:
		return PackedKVDTypeFP8KNVFP4V, nil
	default:
		return "", fmt.Errorf("%w: unknown dtype tag %d", ErrSnapshotCorrupt, tag)
	}
}

// EncodePackedKVSnapshot serializes a PackedKVSnapshot into a versioned, SHA-256 checksummed
// binary frame without dequantizing the packed payloads.
func EncodePackedKVSnapshot(snap *PackedKVSnapshot) ([]byte, error) {
	if snap == nil {
		return nil, fmt.Errorf("%w: nil snapshot", ErrSnapshotInvalid)
	}
	if snap.ModelArch == "" {
		return nil, fmt.Errorf("%w: empty ModelArch", ErrSnapshotInvalid)
	}
	if len(snap.ModelArch) > math.MaxUint16 || len(snap.WeightDigest) > math.MaxUint16 {
		return nil, fmt.Errorf("%w: metadata string exceeds uint16 length", ErrSnapshotInvalid)
	}
	dtypeTag, err := dtypeToTag(snap.DType)
	if err != nil {
		return nil, err
	}
	wantKData, wantKScales, wantVData, wantVScales, err := ExpectedPayloadSizes(
		snap.DType, snap.NumLayers, snap.SeqLen, snap.NumKVHeads, snap.HeadDim,
	)
	if err != nil {
		return nil, err
	}
	if len(snap.KData) != wantKData || len(snap.KScales) != wantKScales ||
		len(snap.VData) != wantVData || len(snap.VScales) != wantVScales {
		return nil, fmt.Errorf("%w: payload sizes got (%d,%d,%d,%d), want (%d,%d,%d,%d)",
			ErrSnapshotInvalid,
			len(snap.KData), len(snap.KScales), len(snap.VData), len(snap.VScales),
			wantKData, wantKScales, wantVData, wantVScales)
	}

	version := snap.Version
	if version == 0 {
		version = PackedKVSnapshotVersion
	} else if version != PackedKVSnapshotVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrSnapshotInvalid, version)
	}

	headerLen := 8 + 2 + 1 + 1 + 8 + 2 + len(snap.WeightDigest) + 2 + len(snap.ModelArch) + 4*4 + 4*2 + 8*4
	totalLen := headerLen + len(snap.KData) + len(snap.KScales) + len(snap.VData) + len(snap.VScales) + sha256.Size
	buf := make([]byte, 0, totalLen)

	buf = append(buf, packedKVSnapshotMagic...)
	buf = binary.LittleEndian.AppendUint16(buf, version)
	buf = append(buf, dtypeTag, 0)
	buf = binary.LittleEndian.AppendUint64(buf, snap.WeightEpoch)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(snap.WeightDigest)))
	buf = append(buf, snap.WeightDigest...)
	buf = binary.LittleEndian.AppendUint16(buf, uint16(len(snap.ModelArch)))
	buf = append(buf, snap.ModelArch...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(snap.NumLayers))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(snap.NumKVHeads))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(snap.HeadDim))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(snap.SeqLen))
	buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(snap.KGlobalScale))
	buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(snap.VGlobalScale))

	for _, payload := range [][]byte{snap.KData, snap.KScales, snap.VData, snap.VScales} {
		buf = binary.LittleEndian.AppendUint64(buf, uint64(len(payload)))
		buf = append(buf, payload...)
	}

	sum := sha256.Sum256(buf)
	buf = append(buf, sum[:]...)
	return buf, nil
}

// DecodePackedKVSnapshot verifies the SHA-256 checksum and framing of raw snapshot bytes,
// then enforces model arch, KV dtype, and WeightEpoch/WeightDigest binding before returning
// the decoded PackedKVSnapshot.
func DecodePackedKVSnapshot(raw []byte, expectedArch string, expectedDType PackedKVDType, expectedEpoch model.WeightEpochStamp) (*PackedKVSnapshot, error) {
	// Minimum size: magic(8) + version(2) + dtype(1) + flags(1) + epoch(8) + digestLen(2) +
	// archLen(2) + 4*uint32(16) + 2*float32(8) + 4*uint64(32) + sha256(32) = 112 bytes.
	const minFrameBytes = 8 + 2 + 1 + 1 + 8 + 2 + 2 + 16 + 8 + 32 + sha256.Size
	if len(raw) < minFrameBytes {
		return nil, fmt.Errorf("%w: truncated frame (%d bytes)", ErrSnapshotCorrupt, len(raw))
	}
	if string(raw[:8]) != packedKVSnapshotMagic {
		return nil, fmt.Errorf("%w: invalid magic", ErrSnapshotCorrupt)
	}

	bodyEnd := len(raw) - sha256.Size
	wantSum := sha256.Sum256(raw[:bodyEnd])
	if !bytes.Equal(raw[bodyEnd:], wantSum[:]) {
		return nil, fmt.Errorf("%w: sha256 mismatch", ErrSnapshotCorrupt)
	}

	r := raw[8:bodyEnd]
	readU16 := func() (uint16, bool) {
		if len(r) < 2 {
			return 0, false
		}
		v := binary.LittleEndian.Uint16(r[:2])
		r = r[2:]
		return v, true
	}
	readU32 := func() (uint32, bool) {
		if len(r) < 4 {
			return 0, false
		}
		v := binary.LittleEndian.Uint32(r[:4])
		r = r[4:]
		return v, true
	}
	readU64 := func() (uint64, bool) {
		if len(r) < 8 {
			return 0, false
		}
		v := binary.LittleEndian.Uint64(r[:8])
		r = r[8:]
		return v, true
	}
	readBytes := func(n uint64) ([]byte, bool) {
		if n > uint64(len(r)) {
			return nil, false
		}
		out := make([]byte, n)
		copy(out, r[:n])
		r = r[n:]
		return out, true
	}

	version, ok := readU16()
	if !ok || version != PackedKVSnapshotVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrSnapshotCorrupt, version)
	}
	if len(r) < 2 {
		return nil, fmt.Errorf("%w: truncated dtype header", ErrSnapshotCorrupt)
	}
	dtype, err := tagToDType(r[0])
	if err != nil {
		return nil, err
	}
	r = r[2:] // dtypeTag + flags

	weightEpoch, ok := readU64()
	if !ok {
		return nil, fmt.Errorf("%w: truncated weight epoch", ErrSnapshotCorrupt)
	}
	digestLen, ok := readU16()
	if !ok {
		return nil, fmt.Errorf("%w: truncated digest len", ErrSnapshotCorrupt)
	}
	digestBytes, ok := readBytes(uint64(digestLen))
	if !ok {
		return nil, fmt.Errorf("%w: truncated digest", ErrSnapshotCorrupt)
	}
	archLen, ok := readU16()
	if !ok {
		return nil, fmt.Errorf("%w: truncated arch len", ErrSnapshotCorrupt)
	}
	archBytes, ok := readBytes(uint64(archLen))
	if !ok {
		return nil, fmt.Errorf("%w: truncated arch", ErrSnapshotCorrupt)
	}

	numLayers, ok1 := readU32()
	numKVHeads, ok2 := readU32()
	headDim, ok3 := readU32()
	seqLen, ok4 := readU32()
	kGlobalBits, ok5 := readU32()
	vGlobalBits, ok6 := readU32()
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 {
		return nil, fmt.Errorf("%w: truncated geometry header", ErrSnapshotCorrupt)
	}

	readPayload := func() ([]byte, bool) {
		n, ok := readU64()
		if !ok {
			return nil, false
		}
		return readBytes(n)
	}
	kData, ok1 := readPayload()
	kScales, ok2 := readPayload()
	vData, ok3 := readPayload()
	vScales, ok4 := readPayload()
	if !ok1 || !ok2 || !ok3 || !ok4 || len(r) != 0 {
		return nil, fmt.Errorf("%w: malformed payload framing", ErrSnapshotCorrupt)
	}

	snap := &PackedKVSnapshot{
		Version:      version,
		ModelArch:    string(archBytes),
		DType:        dtype,
		WeightEpoch:  weightEpoch,
		WeightDigest: string(digestBytes),
		NumLayers:    int(numLayers),
		NumKVHeads:   int(numKVHeads),
		HeadDim:      int(headDim),
		SeqLen:       int(seqLen),
		KGlobalScale: math.Float32frombits(kGlobalBits),
		VGlobalScale: math.Float32frombits(vGlobalBits),
		KData:        kData,
		KScales:      kScales,
		VData:        vData,
		VScales:      vScales,
	}

	wantKData, wantKScales, wantVData, wantVScales, err := ExpectedPayloadSizes(
		snap.DType, snap.NumLayers, snap.SeqLen, snap.NumKVHeads, snap.HeadDim,
	)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrSnapshotCorrupt, err)
	}
	if len(snap.KData) != wantKData || len(snap.KScales) != wantKScales ||
		len(snap.VData) != wantVData || len(snap.VScales) != wantVScales {
		return nil, fmt.Errorf("%w: payload length mismatch", ErrSnapshotCorrupt)
	}

	if expectedArch != "" && snap.ModelArch != expectedArch {
		return nil, fmt.Errorf("%w: got %q, want %q", ErrSnapshotArchMismatch, snap.ModelArch, expectedArch)
	}
	if expectedDType != "" && snap.DType != expectedDType {
		return nil, fmt.Errorf("%w: got %q, want %q", ErrSnapshotDTypeMismatch, snap.DType, expectedDType)
	}
	if snap.WeightEpoch != expectedEpoch.Generation || snap.WeightDigest != expectedEpoch.Digest {
		return nil, fmt.Errorf("%w: snapshot (epoch=%d digest=%q) != live (epoch=%d digest=%q)",
			ErrWeightEpochMismatch, snap.WeightEpoch, snap.WeightDigest, expectedEpoch.Generation, expectedEpoch.Digest)
	}

	return snap, nil
}
