package kvcache

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	// SnapshotMagic identifies a versioned, checksummed cross-model KV snapshot.
	SnapshotMagic = "FAKKVSNP"
	// SnapshotVersion is the current binary wire/disk format version.
	SnapshotVersion = uint32(1)
	// snapshotHeaderSize = magic(8) + version(4) + flags(4) + sha256(32).
	snapshotHeaderSize = 8 + 4 + 4 + sha256.Size
)

var (
	// ErrSnapshotCorrupt indicates malformed framing, truncated payload, or checksum mismatch.
	ErrSnapshotCorrupt = errors.New("kvcache: corrupt snapshot")
	// ErrSnapshotIncompatible indicates the decoded snapshot does not match the expected layout/arch.
	ErrSnapshotIncompatible = errors.New("kvcache: incompatible snapshot layout")
)

// SnapshotLayer holds one layer's (or loop slot's) packed KV prefix state.
// For PrecisionNVFP4, KCodes/VCodes hold packed E2M1 nibbles and KScales/VScales
// hold OCP E4M3FN block-16 scales directly without dequantization.
type SnapshotLayer struct {
	Index   int
	KCodes  []byte
	VCodes  []byte
	KScales []byte
	VScales []byte
	Extra   []byte
}

// Snapshot is a model-agnostic, versioned KV prefix snapshot.
type Snapshot struct {
	Layout    LayoutDesc
	Tokens    []int
	Positions []int
	Layers    []SnapshotLayer
}

// EncodeSnapshot serializes s into a versioned, SHA-256-checksummed binary blob.
func EncodeSnapshot(s *Snapshot) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("%w: nil snapshot", ErrSnapshotCorrupt)
	}
	if err := s.Layout.Validate(); err != nil {
		return nil, err
	}
	if len(s.Layers) != s.Layout.Layers {
		return nil, fmt.Errorf("%w: layer count %d != layout.Layers %d", ErrSnapshotCorrupt, len(s.Layers), s.Layout.Layers)
	}
	if len(s.Positions) != 0 && len(s.Positions) != len(s.Tokens) {
		return nil, fmt.Errorf("%w: positions len %d != tokens len %d", ErrSnapshotCorrupt, len(s.Positions), len(s.Tokens))
	}

	archBytes := []byte(s.Layout.Arch)
	precBytes := []byte(s.Layout.Precision)
	mode := s.Layout.Layout
	if mode == "" {
		mode = LayoutContiguous
	}
	modeBytes := []byte(mode)

	payloadLen := 2 + len(archBytes) +
		2 + len(precBytes) +
		2 + len(modeBytes) +
		4*6 + // layers, kvHeads, headDim, capacity, pageTokens, extraStateFloatsPerSlot
		4 + 8*len(s.Tokens) +
		4 + 8*len(s.Positions) +
		4 // layer count

	for _, l := range s.Layers {
		payloadLen += 4 * 6 // index + 5 byte-slice lengths
		payloadLen += len(l.KCodes) + len(l.VCodes) + len(l.KScales) + len(l.VScales) + len(l.Extra)
	}

	buf := make([]byte, snapshotHeaderSize+payloadLen)
	off := snapshotHeaderSize

	writeBytes := func(b []byte) {
		binary.LittleEndian.PutUint16(buf[off:], uint16(len(b)))
		off += 2
		copy(buf[off:], b)
		off += len(b)
	}
	writeU32 := func(v uint32) {
		binary.LittleEndian.PutUint32(buf[off:], v)
		off += 4
	}
	writeBlob := func(b []byte) {
		writeU32(uint32(len(b)))
		copy(buf[off:], b)
		off += len(b)
	}

	writeBytes(archBytes)
	writeBytes(precBytes)
	writeBytes(modeBytes)
	writeU32(uint32(s.Layout.Layers))
	writeU32(uint32(s.Layout.KVHeads))
	writeU32(uint32(s.Layout.HeadDim))
	writeU32(uint32(s.Layout.Capacity))
	writeU32(uint32(s.Layout.PageTokens))
	writeU32(uint32(s.Layout.ExtraStateFloatsPerSlot))

	writeU32(uint32(len(s.Tokens)))
	for _, tok := range s.Tokens {
		binary.LittleEndian.PutUint64(buf[off:], uint64(int64(tok)))
		off += 8
	}
	writeU32(uint32(len(s.Positions)))
	for _, pos := range s.Positions {
		binary.LittleEndian.PutUint64(buf[off:], uint64(int64(pos)))
		off += 8
	}

	writeU32(uint32(len(s.Layers)))
	for _, l := range s.Layers {
		writeU32(uint32(l.Index))
		writeBlob(l.KCodes)
		writeBlob(l.VCodes)
		writeBlob(l.KScales)
		writeBlob(l.VScales)
		writeBlob(l.Extra)
	}

	copy(buf[0:8], SnapshotMagic)
	binary.LittleEndian.PutUint32(buf[8:12], SnapshotVersion)
	binary.LittleEndian.PutUint32(buf[12:16], 0)
	sum := sha256.Sum256(buf[snapshotHeaderSize:])
	copy(buf[16:48], sum[:])
	return buf, nil
}

// DecodeSnapshot verifies the header, version, and SHA-256 checksum, then decodes
// the snapshot. If expected is non-nil, it verifies Arch (when non-empty),
// Precision, Layers, KVHeads, and HeadDim compatibility.
func DecodeSnapshot(data []byte, expected *LayoutDesc) (*Snapshot, error) {
	if len(data) < snapshotHeaderSize {
		return nil, ErrSnapshotCorrupt
	}
	if string(data[0:8]) != SnapshotMagic {
		return nil, ErrSnapshotCorrupt
	}
	if binary.LittleEndian.Uint32(data[8:12]) != SnapshotVersion {
		return nil, ErrSnapshotCorrupt
	}
	if binary.LittleEndian.Uint32(data[12:16]) != 0 {
		return nil, ErrSnapshotCorrupt
	}
	payload := data[snapshotHeaderSize:]
	sum := sha256.Sum256(payload)
	var want [sha256.Size]byte
	copy(want[:], data[16:48])
	if sum != want {
		return nil, ErrSnapshotCorrupt
	}

	r := snapshotReader{buf: payload}
	arch, err := r.readShortString()
	if err != nil {
		return nil, err
	}
	precStr, err := r.readShortString()
	if err != nil {
		return nil, err
	}
	modeStr, err := r.readShortString()
	if err != nil {
		return nil, err
	}
	layers, err := r.readInt32()
	if err != nil {
		return nil, err
	}
	kvHeads, err := r.readInt32()
	if err != nil {
		return nil, err
	}
	headDim, err := r.readInt32()
	if err != nil {
		return nil, err
	}
	capacity, err := r.readInt32()
	if err != nil {
		return nil, err
	}
	pageTokens, err := r.readInt32()
	if err != nil {
		return nil, err
	}
	extraFloats, err := r.readInt32()
	if err != nil {
		return nil, err
	}

	layout := LayoutDesc{
		Arch:                    arch,
		Precision:               Precision(precStr),
		Layers:                  layers,
		KVHeads:                 kvHeads,
		HeadDim:                 headDim,
		Capacity:                capacity,
		Layout:                  LayoutMode(modeStr),
		PageTokens:              pageTokens,
		ExtraStateFloatsPerSlot: extraFloats,
	}
	if err := layout.Validate(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrSnapshotCorrupt, err)
	}
	if expected != nil {
		if err := checkLayoutCompatible(layout, *expected); err != nil {
			return nil, err
		}
	}

	tokens, err := r.readInt64Slice()
	if err != nil {
		return nil, err
	}
	positions, err := r.readInt64Slice()
	if err != nil {
		return nil, err
	}
	if len(positions) != 0 && len(positions) != len(tokens) {
		return nil, ErrSnapshotCorrupt
	}

	layerCount, err := r.readInt32()
	if err != nil || layerCount != layout.Layers {
		return nil, ErrSnapshotCorrupt
	}
	layerSlices := make([]SnapshotLayer, layerCount)
	for i := range layerSlices {
		idx, err := r.readInt32()
		if err != nil {
			return nil, err
		}
		kCodes, err := r.readBlob()
		if err != nil {
			return nil, err
		}
		vCodes, err := r.readBlob()
		if err != nil {
			return nil, err
		}
		kScales, err := r.readBlob()
		if err != nil {
			return nil, err
		}
		vScales, err := r.readBlob()
		if err != nil {
			return nil, err
		}
		extra, err := r.readBlob()
		if err != nil {
			return nil, err
		}
		layerSlices[i] = SnapshotLayer{
			Index:   idx,
			KCodes:  kCodes,
			VCodes:  vCodes,
			KScales: kScales,
			VScales: vScales,
			Extra:   extra,
		}
	}
	if r.off != len(r.buf) {
		return nil, ErrSnapshotCorrupt
	}
	return &Snapshot{
		Layout:    layout,
		Tokens:    tokens,
		Positions: positions,
		Layers:    layerSlices,
	}, nil
}

func checkLayoutCompatible(got, want LayoutDesc) error {
	if want.Arch != "" && got.Arch != want.Arch {
		return fmt.Errorf("%w: arch %q != %q", ErrSnapshotIncompatible, got.Arch, want.Arch)
	}
	if want.Precision != "" && got.Precision != want.Precision {
		return fmt.Errorf("%w: precision %q != %q", ErrSnapshotIncompatible, got.Precision, want.Precision)
	}
	if want.Layers > 0 && got.Layers != want.Layers {
		return fmt.Errorf("%w: layers %d != %d", ErrSnapshotIncompatible, got.Layers, want.Layers)
	}
	if want.KVHeads > 0 && got.KVHeads != want.KVHeads {
		return fmt.Errorf("%w: kv_heads %d != %d", ErrSnapshotIncompatible, got.KVHeads, want.KVHeads)
	}
	if want.HeadDim > 0 && got.HeadDim != want.HeadDim {
		return fmt.Errorf("%w: head_dim %d != %d", ErrSnapshotIncompatible, got.HeadDim, want.HeadDim)
	}
	return nil
}

// Float32sToBytes encodes a float32 slice into little-endian bytes.
func Float32sToBytes(src []float32) []byte {
	if len(src) == 0 {
		return nil
	}
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// BytesToFloat32s decodes little-endian bytes into a float32 slice.
func BytesToFloat32s(src []byte) ([]float32, error) {
	if len(src)%4 != 0 {
		return nil, fmt.Errorf("%w: float32 byte length %d not divisible by 4", ErrSnapshotCorrupt, len(src))
	}
	if len(src) == 0 {
		return nil, nil
	}
	out := make([]float32, len(src)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out, nil
}

type snapshotReader struct {
	buf []byte
	off int
}

func (r *snapshotReader) readShortString() (string, error) {
	if len(r.buf)-r.off < 2 {
		return "", ErrSnapshotCorrupt
	}
	n := int(binary.LittleEndian.Uint16(r.buf[r.off:]))
	r.off += 2
	if len(r.buf)-r.off < n {
		return "", ErrSnapshotCorrupt
	}
	s := string(r.buf[r.off : r.off+n])
	r.off += n
	return s, nil
}

func (r *snapshotReader) readInt32() (int, error) {
	if len(r.buf)-r.off < 4 {
		return 0, ErrSnapshotCorrupt
	}
	v := binary.LittleEndian.Uint32(r.buf[r.off:])
	r.off += 4
	return int(int32(v)), nil
}

func (r *snapshotReader) readInt64Slice() ([]int, error) {
	n, err := r.readInt32()
	if err != nil || n < 0 {
		return nil, ErrSnapshotCorrupt
	}
	bytesNeeded := int64(n) * 8
	if int64(len(r.buf)-r.off) < bytesNeeded {
		return nil, ErrSnapshotCorrupt
	}
	if n == 0 {
		return nil, nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = int(int64(binary.LittleEndian.Uint64(r.buf[r.off:])))
		r.off += 8
	}
	return out, nil
}

func (r *snapshotReader) readBlob() ([]byte, error) {
	n, err := r.readInt32()
	if err != nil || n < 0 {
		return nil, ErrSnapshotCorrupt
	}
	if len(r.buf)-r.off < n {
		return nil, ErrSnapshotCorrupt
	}
	if n == 0 {
		return nil, nil
	}
	out := make([]byte, n)
	copy(out, r.buf[r.off:r.off+n])
	r.off += n
	return out, nil
}
