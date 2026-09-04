// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
)

// KPACK constants defining container signature, version, and parsing boundaries.
// Borrowed from ROCm/hrx-system libhrx/kpack.c.
const (
	// KPACKMagic is the 4-byte ASCII signature "KPAK" (0x4B50414B).
	KPACKMagic uint32 = 0x4B50414B

	// KPACKVersion1 defines version 1 of the container format.
	KPACKVersion1 uint32 = 1

	// KPACKHeaderSize is the fixed 32-byte binary header size.
	KPACKHeaderSize = 32

	// MaxMsgpackDepth limits MessagePack container recursion depth to 32
	// to prevent stack overflow from deeply nested or malicious TOC structures.
	MaxMsgpackDepth = 32
)

// CompressionType represents the compression codec applied to kernel code objects.
type CompressionType uint8

const (
	// CompressionNone indicates uncompressed raw kernel binary payload.
	CompressionNone CompressionType = 0

	// CompressionZSTD indicates Zstandard-compressed kernel binary payload.
	CompressionZSTD CompressionType = 1
)

// Common KPACK errors.
var (
	ErrKPACKInvalidMagic   = errors.New("kpack: invalid magic number (expected KPAK)")
	ErrKPACKInvalidVersion = errors.New("kpack: unsupported version")
	ErrKPACKTruncated      = errors.New("kpack: archive truncated or payload out of bounds")
	ErrKPACKNestingTooDeep = errors.New("kpack: msgpack nesting depth exceeded limit (max 32)")
	ErrKPACKInvalidTOC     = errors.New("kpack: invalid or malformed table of contents")
)

// KPACKHeader is the fixed 32-byte header at the start of every KPACK container file.
type KPACKHeader struct {
	Magic         uint32 // 0x4B50414B ("KPAK")
	Version       uint32 // Version number (e.g. 1)
	Flags         uint32 // Reserved flags
	EntryCount    uint32 // Number of target entries in the archive
	TOCSize       uint32 // Size of the MessagePack Table of Contents in bytes
	TOCOffset     uint32 // Byte offset of the MessagePack TOC (typically 32)
	PayloadOffset uint64 // Byte offset where kernel payload data begins
}

// KPACKEntry represents a single target architecture code object in the container.
type KPACKEntry struct {
	Target           string          // Architecture ID e.g. "gfx1151", "gfx942:sramecc+:xnack-"
	Compression      CompressionType // Compression algorithm (None or ZSTD)
	Offset           uint64          // Absolute byte offset in the archive
	CompressedSize   uint64          // Stored payload size in bytes
	UncompressedSize uint64          // Uncompressed payload size in bytes
	Data             []byte          // Code object payload bytes
}

// KPACKArchive represents a parsed multi-target kernel container.
type KPACKArchive struct {
	Header  KPACKHeader
	Entries []KPACKEntry
}

// Targets returns a slice of all target architecture IDs present in the archive.
func (a *KPACKArchive) Targets() []string {
	targets := make([]string, len(a.Entries))
	for i, e := range a.Entries {
		targets[i] = e.Target
	}
	return targets
}

// Get returns the entry matching the exact target string, or nil if not found.
func (a *KPACKArchive) Get(target string) (*KPACKEntry, bool) {
	for i := range a.Entries {
		if strings.EqualFold(a.Entries[i].Target, strings.TrimSpace(target)) {
			return &a.Entries[i], true
		}
	}
	return nil, false
}

// Resolve resolves the requested architecture against available targets in the archive
// using the dynamic fallback ladder and returns the best matching entry.
func (a *KPACKArchive) Resolve(arch string) (*KPACKEntry, string, bool) {
	matched, ok := ResolveTarget(arch, a.Targets())
	if !ok {
		return nil, "", false
	}
	entry, found := a.Get(matched)
	return entry, matched, found
}

// WriteKPACK serializes a slice of KPACKEntry items into the KPACK binary format.
func WriteKPACK(w io.Writer, entries []KPACKEntry) error {
	if w == nil {
		return errors.New("kpack: nil writer")
	}

	prepared := make([]KPACKEntry, len(entries))
	var curRelOffset uint64
	for i, e := range entries {
		prepared[i] = e
		if prepared[i].CompressedSize == 0 {
			prepared[i].CompressedSize = uint64(len(e.Data))
		}
		if prepared[i].UncompressedSize == 0 {
			prepared[i].UncompressedSize = prepared[i].CompressedSize
		}
		prepared[i].Offset = curRelOffset
		curRelOffset += prepared[i].CompressedSize
	}

	tocBytes, err := encodeTOC(prepared)
	if err != nil {
		return fmt.Errorf("kpack: failed to encode TOC: %w", err)
	}

	tocSize := uint32(len(tocBytes))
	tocOffset := uint32(KPACKHeaderSize)
	payloadOffset := uint64(KPACKHeaderSize + tocSize)

	var hdr [KPACKHeaderSize]byte
	binary.BigEndian.PutUint32(hdr[0:4], KPACKMagic)
	binary.BigEndian.PutUint32(hdr[4:8], KPACKVersion1)
	binary.BigEndian.PutUint32(hdr[8:12], 0) // Flags
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(entries)))
	binary.BigEndian.PutUint32(hdr[16:20], tocSize)
	binary.BigEndian.PutUint32(hdr[20:24], tocOffset)
	binary.BigEndian.PutUint64(hdr[24:32], payloadOffset)

	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}

	if _, err := w.Write(tocBytes); err != nil {
		return err
	}

	for _, e := range prepared {
		if len(e.Data) > 0 {
			if _, err := w.Write(e.Data); err != nil {
				return err
			}
		}
	}

	return nil
}

// ReadKPACK parses a KPACK archive from an io.ReaderAt given the total archive size.
func ReadKPACK(r io.ReaderAt, size int64) (*KPACKArchive, error) {
	if r == nil {
		return nil, errors.New("kpack: nil reader")
	}
	if size < KPACKHeaderSize {
		return nil, ErrKPACKTruncated
	}

	var hdrBuf [KPACKHeaderSize]byte
	if _, err := r.ReadAt(hdrBuf[:], 0); err != nil {
		return nil, ErrKPACKTruncated
	}

	magic := binary.BigEndian.Uint32(hdrBuf[0:4])
	if magic != KPACKMagic {
		return nil, ErrKPACKInvalidMagic
	}
	version := binary.BigEndian.Uint32(hdrBuf[4:8])
	if version != KPACKVersion1 {
		return nil, ErrKPACKInvalidVersion
	}
	flags := binary.BigEndian.Uint32(hdrBuf[8:12])
	entryCount := binary.BigEndian.Uint32(hdrBuf[12:16])
	tocSize := binary.BigEndian.Uint32(hdrBuf[16:20])
	tocOffset := binary.BigEndian.Uint32(hdrBuf[20:24])
	payloadOffset := binary.BigEndian.Uint64(hdrBuf[24:32])

	if int64(tocOffset)+int64(tocSize) > size || int64(tocOffset) < KPACKHeaderSize {
		return nil, ErrKPACKTruncated
	}
	if int64(payloadOffset) > size {
		return nil, ErrKPACKTruncated
	}

	header := KPACKHeader{
		Magic:         magic,
		Version:       version,
		Flags:         flags,
		EntryCount:    entryCount,
		TOCSize:       tocSize,
		TOCOffset:     tocOffset,
		PayloadOffset: payloadOffset,
	}

	tocBytes := make([]byte, tocSize)
	if _, err := r.ReadAt(tocBytes, int64(tocOffset)); err != nil {
		return nil, ErrKPACKTruncated
	}

	entries, err := parseTOC(tocBytes)
	if err != nil {
		return nil, err
	}

	for i := range entries {
		if entries[i].Offset < payloadOffset {
			entries[i].Offset += payloadOffset
		}
		if int64(entries[i].Offset)+int64(entries[i].CompressedSize) > size {
			return nil, ErrKPACKTruncated
		}
		if entries[i].CompressedSize > 0 {
			entries[i].Data = make([]byte, entries[i].CompressedSize)
			if _, err := r.ReadAt(entries[i].Data, int64(entries[i].Offset)); err != nil {
				return nil, ErrKPACKTruncated
			}
		}
	}

	return &KPACKArchive{
		Header:  header,
		Entries: entries,
	}, nil
}

// TargetFallbackChain expands an AMDGPU target architecture into a ranked candidate
// fallback chain from most specific to generic.
// Covering: gfx1151, gfx1150, gfx1100, gfx1102, gfx1200, gfx1201, gfx942, gfx90a, gfx908.
func TargetFallbackChain(arch string) []string {
	norm := strings.ToLower(strings.TrimSpace(arch))
	if norm == "" {
		return nil
	}

	var base, features string
	if idx := strings.IndexByte(norm, ':'); idx != -1 {
		base = norm[:idx]
		features = norm[idx:]
	} else {
		base = norm
		features = ""
	}

	chain := make([]string, 0, 4)
	if features != "" {
		chain = append(chain, norm)
	}
	if base != "" && base != "generic" {
		chain = append(chain, base)
	}

	switch {
	case base == "gfx1151" || base == "gfx1150" || base == "gfx1100" || base == "gfx1102" || strings.HasPrefix(base, "gfx11"):
		chain = append(chain, "gfx11-generic", "generic")
	case base == "gfx1200" || base == "gfx1201" || strings.HasPrefix(base, "gfx12"):
		chain = append(chain, "gfx12-generic", "generic")
	case base == "gfx942" || base == "gfx90a" || base == "gfx908" || strings.HasPrefix(base, "gfx9"):
		chain = append(chain, "gfx9-generic")
	case base == "gfx11-generic" || base == "gfx12-generic":
		chain = append(chain, "generic")
	case base == "gfx9-generic":
		// gfx9-generic has no further generic fallback in CDNA
	case base == "generic":
		if len(chain) == 0 {
			chain = append(chain, "generic")
		}
	default:
		chain = append(chain, "generic")
	}

	seen := make(map[string]struct{}, len(chain))
	deduped := make([]string, 0, len(chain))
	for _, c := range chain {
		if _, ok := seen[c]; !ok {
			seen[c] = struct{}{}
			deduped = append(deduped, c)
		}
	}
	return deduped
}

// ResolveTarget matches a requested target architecture against available targets,
// resolving exact matches, feature-variant subsets, and ranked fallback chains.
func ResolveTarget(arch string, availableTargets []string) (string, bool) {
	if len(availableTargets) == 0 {
		return "", false
	}
	normArch := strings.ToLower(strings.TrimSpace(arch))
	if normArch == "" {
		return "", false
	}

	// 1. Exact match on target string
	for _, avail := range availableTargets {
		if strings.EqualFold(normArch, strings.TrimSpace(avail)) {
			return avail, true
		}
	}

	archBase, archFeats := parseArchFeatures(normArch)

	// 2. Feature-variant match: if arch has features, check if available targets
	// contain a compatible feature subset of the same base architecture.
	if len(archFeats) > 0 {
		var bestMatch string
		bestScore := -1
		for _, avail := range availableTargets {
			availBase, availFeats := parseArchFeatures(strings.ToLower(strings.TrimSpace(avail)))
			if strings.EqualFold(availBase, archBase) && len(availFeats) > 0 {
				compat := true
				matches := 0
				for k, v := range availFeats {
					archVal, ok := archFeats[k]
					if !ok || (archVal != v && archVal != "any" && v != "any") {
						compat = false
						break
					}
					if archVal == v {
						matches++
					}
				}
				if compat && matches > bestScore {
					bestMatch = avail
					bestScore = matches
				}
			}
		}
		if bestMatch != "" {
			return bestMatch, true
		}
	}

	// 3. Fallback chain resolution (base arch -> family generic -> generic)
	chain := TargetFallbackChain(normArch)
	for _, cand := range chain {
		for _, avail := range availableTargets {
			if strings.EqualFold(cand, strings.TrimSpace(avail)) {
				return avail, true
			}
		}
	}

	return "", false
}

// parseArchFeatures splits an architecture string into base processor and feature map.
// e.g. "gfx942:sramecc+:xnack-" -> base: "gfx942", features: {"sramecc": "+", "xnack": "-"}
func parseArchFeatures(s string) (string, map[string]string) {
	features := make(map[string]string)
	parts := strings.Split(s, ":")
	if len(parts) == 0 {
		return "", features
	}
	base := strings.TrimSpace(parts[0])
	for _, p := range parts[1:] {
		p = strings.TrimSpace(p)
		if len(p) == 0 {
			continue
		}
		lastChar := p[len(p)-1]
		if lastChar == '+' || lastChar == '-' {
			featName := p[:len(p)-1]
			features[featName] = string(lastChar)
		} else if idx := strings.IndexByte(p, ':'); idx != -1 {
			features[p[:idx]] = p[idx+1:]
		} else {
			features[p] = "any"
		}
	}
	return base, features
}

// --- Stack-Bounded MessagePack Encoder and Decoder ---

type msgpackEncoder struct {
	buf bytes.Buffer
}

func (e *msgpackEncoder) writeMapHeader(count int) {
	if count <= 15 {
		e.buf.WriteByte(byte(0x80 | count))
	} else if count <= 0xffff {
		e.buf.WriteByte(0xde)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(count))
		e.buf.Write(b[:])
	} else {
		e.buf.WriteByte(0xdf)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(count))
		e.buf.Write(b[:])
	}
}

func (e *msgpackEncoder) writeArrayHeader(count int) {
	if count <= 15 {
		e.buf.WriteByte(byte(0x90 | count))
	} else if count <= 0xffff {
		e.buf.WriteByte(0xdc)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(count))
		e.buf.Write(b[:])
	} else {
		e.buf.WriteByte(0xdd)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(count))
		e.buf.Write(b[:])
	}
}

func (e *msgpackEncoder) writeString(s string) {
	l := len(s)
	if l <= 31 {
		e.buf.WriteByte(byte(0xa0 | l))
	} else if l <= 0xff {
		e.buf.WriteByte(0xd9)
		e.buf.WriteByte(byte(l))
	} else if l <= 0xffff {
		e.buf.WriteByte(0xda)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(l))
		e.buf.Write(b[:])
	} else {
		e.buf.WriteByte(0xdb)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(l))
		e.buf.Write(b[:])
	}
	e.buf.WriteString(s)
}

func (e *msgpackEncoder) writeUint(v uint64) {
	if v <= 0x7f {
		e.buf.WriteByte(byte(v))
	} else if v <= 0xff {
		e.buf.WriteByte(0xcc)
		e.buf.WriteByte(byte(v))
	} else if v <= 0xffff {
		e.buf.WriteByte(0xcd)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(v))
		e.buf.Write(b[:])
	} else if v <= 0xffffffff {
		e.buf.WriteByte(0xce)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(v))
		e.buf.Write(b[:])
	} else {
		e.buf.WriteByte(0xcf)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], v)
		e.buf.Write(b[:])
	}
}

func encodeTOC(entries []KPACKEntry) ([]byte, error) {
	var enc msgpackEncoder
	enc.writeMapHeader(2)

	enc.writeString("version")
	enc.writeUint(uint64(KPACKVersion1))

	enc.writeString("entries")
	enc.writeArrayHeader(len(entries))

	for _, e := range entries {
		enc.writeMapHeader(5)

		enc.writeString("target")
		enc.writeString(e.Target)

		enc.writeString("compression")
		enc.writeUint(uint64(e.Compression))

		enc.writeString("offset")
		enc.writeUint(e.Offset)

		enc.writeString("compressed_size")
		enc.writeUint(e.CompressedSize)

		enc.writeString("uncompressed_size")
		enc.writeUint(e.UncompressedSize)
	}

	return enc.buf.Bytes(), nil
}

type msgpackDecoder struct {
	data  []byte
	pos   int
	depth int
}

func (d *msgpackDecoder) readByte() (byte, error) {
	if d.pos >= len(d.data) {
		return 0, ErrKPACKTruncated
	}
	b := d.data[d.pos]
	d.pos++
	return b, nil
}

func (d *msgpackDecoder) readBytes(n int) ([]byte, error) {
	if n < 0 || d.pos+n > len(d.data) {
		return nil, ErrKPACKTruncated
	}
	b := d.data[d.pos : d.pos+n]
	d.pos += n
	return b, nil
}

func (d *msgpackDecoder) enterContainer() error {
	d.depth++
	if d.depth > MaxMsgpackDepth {
		return ErrKPACKNestingTooDeep
	}
	return nil
}

func (d *msgpackDecoder) leaveContainer() {
	if d.depth > 0 {
		d.depth--
	}
}

func (d *msgpackDecoder) readString() (string, error) {
	b, err := d.readByte()
	if err != nil {
		return "", err
	}
	var strLen int
	if b >= 0xa0 && b <= 0xbf {
		strLen = int(b & 0x1f)
	} else if b == 0xd9 {
		l, err := d.readByte()
		if err != nil {
			return "", err
		}
		strLen = int(l)
	} else if b == 0xda {
		raw, err := d.readBytes(2)
		if err != nil {
			return "", err
		}
		strLen = int(binary.BigEndian.Uint16(raw))
	} else if b == 0xdb {
		raw, err := d.readBytes(4)
		if err != nil {
			return "", err
		}
		strLen = int(binary.BigEndian.Uint32(raw))
	} else {
		return "", ErrKPACKInvalidTOC
	}
	bytes, err := d.readBytes(strLen)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func (d *msgpackDecoder) readUint() (uint64, error) {
	b, err := d.readByte()
	if err != nil {
		return 0, err
	}
	if b <= 0x7f {
		return uint64(b), nil
	}
	switch b {
	case 0xcc:
		v, err := d.readByte()
		return uint64(v), err
	case 0xcd:
		raw, err := d.readBytes(2)
		if err != nil {
			return 0, err
		}
		return uint64(binary.BigEndian.Uint16(raw)), nil
	case 0xce:
		raw, err := d.readBytes(4)
		if err != nil {
			return 0, err
		}
		return uint64(binary.BigEndian.Uint32(raw)), nil
	case 0xcf:
		raw, err := d.readBytes(8)
		if err != nil {
			return 0, err
		}
		return binary.BigEndian.Uint64(raw), nil
	case 0xd0:
		v, err := d.readByte()
		if err != nil {
			return 0, err
		}
		if int8(v) < 0 {
			return 0, ErrKPACKInvalidTOC
		}
		return uint64(v), nil
	case 0xd1:
		raw, err := d.readBytes(2)
		if err != nil {
			return 0, err
		}
		v := int16(binary.BigEndian.Uint16(raw))
		if v < 0 {
			return 0, ErrKPACKInvalidTOC
		}
		return uint64(v), nil
	case 0xd2:
		raw, err := d.readBytes(4)
		if err != nil {
			return 0, err
		}
		v := int32(binary.BigEndian.Uint32(raw))
		if v < 0 {
			return 0, ErrKPACKInvalidTOC
		}
		return uint64(v), nil
	case 0xd3:
		raw, err := d.readBytes(8)
		if err != nil {
			return 0, err
		}
		v := int64(binary.BigEndian.Uint64(raw))
		if v < 0 {
			return 0, ErrKPACKInvalidTOC
		}
		return uint64(v), nil
	default:
		return 0, ErrKPACKInvalidTOC
	}
}

func (d *msgpackDecoder) readMapHeader() (int, error) {
	b, err := d.readByte()
	if err != nil {
		return 0, err
	}
	var count int
	if b >= 0x80 && b <= 0x8f {
		count = int(b & 0x0f)
	} else if b == 0xde {
		raw, err := d.readBytes(2)
		if err != nil {
			return 0, err
		}
		count = int(binary.BigEndian.Uint16(raw))
	} else if b == 0xdf {
		raw, err := d.readBytes(4)
		if err != nil {
			return 0, err
		}
		count = int(binary.BigEndian.Uint32(raw))
	} else {
		return 0, ErrKPACKInvalidTOC
	}
	if err := d.enterContainer(); err != nil {
		return 0, err
	}
	return count, nil
}

func (d *msgpackDecoder) readArrayHeader() (int, error) {
	b, err := d.readByte()
	if err != nil {
		return 0, err
	}
	var count int
	if b >= 0x90 && b <= 0x9f {
		count = int(b & 0x0f)
	} else if b == 0xdc {
		raw, err := d.readBytes(2)
		if err != nil {
			return 0, err
		}
		count = int(binary.BigEndian.Uint16(raw))
	} else if b == 0xdd {
		raw, err := d.readBytes(4)
		if err != nil {
			return 0, err
		}
		count = int(binary.BigEndian.Uint32(raw))
	} else {
		return 0, ErrKPACKInvalidTOC
	}
	if err := d.enterContainer(); err != nil {
		return 0, err
	}
	return count, nil
}

func (d *msgpackDecoder) skipValue() error {
	b, err := d.readByte()
	if err != nil {
		return err
	}
	if b <= 0x7f || b >= 0xe0 {
		return nil
	}
	if b >= 0xa0 && b <= 0xbf {
		l := int(b & 0x1f)
		_, err := d.readBytes(l)
		return err
	}
	if b >= 0x80 && b <= 0x8f {
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(b & 0x0f)
		for i := 0; i < count*2; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	}
	if b >= 0x90 && b <= 0x9f {
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(b & 0x0f)
		for i := 0; i < count; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	}

	switch b {
	case 0xc0, 0xc2, 0xc3:
		return nil
	case 0xc4, 0xd9:
		l, err := d.readByte()
		if err != nil {
			return err
		}
		_, err = d.readBytes(int(l))
		return err
	case 0xc5, 0xda:
		raw, err := d.readBytes(2)
		if err != nil {
			return err
		}
		_, err = d.readBytes(int(binary.BigEndian.Uint16(raw)))
		return err
	case 0xc6, 0xdb:
		raw, err := d.readBytes(4)
		if err != nil {
			return err
		}
		_, err = d.readBytes(int(binary.BigEndian.Uint32(raw)))
		return err
	case 0xcc, 0xd0:
		_, err := d.readByte()
		return err
	case 0xcd, 0xd1:
		_, err := d.readBytes(2)
		return err
	case 0xce, 0xd2, 0xca:
		_, err := d.readBytes(4)
		return err
	case 0xcf, 0xd3, 0xcb:
		_, err := d.readBytes(8)
		return err
	case 0xd4:
		_, err := d.readBytes(2)
		return err
	case 0xd5:
		_, err := d.readBytes(3)
		return err
	case 0xd6:
		_, err := d.readBytes(5)
		return err
	case 0xd7:
		_, err := d.readBytes(9)
		return err
	case 0xd8:
		_, err := d.readBytes(17)
		return err
	case 0xc7:
		l, err := d.readByte()
		if err != nil {
			return err
		}
		_, err = d.readBytes(1 + int(l))
		return err
	case 0xc8:
		raw, err := d.readBytes(2)
		if err != nil {
			return err
		}
		_, err = d.readBytes(1 + int(binary.BigEndian.Uint16(raw)))
		return err
	case 0xc9:
		raw, err := d.readBytes(4)
		if err != nil {
			return err
		}
		_, err = d.readBytes(1 + int(binary.BigEndian.Uint32(raw)))
		return err
	case 0xdc:
		raw, err := d.readBytes(2)
		if err != nil {
			return err
		}
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(binary.BigEndian.Uint16(raw))
		for i := 0; i < count; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	case 0xdd:
		raw, err := d.readBytes(4)
		if err != nil {
			return err
		}
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(binary.BigEndian.Uint32(raw))
		for i := 0; i < count; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	case 0xde:
		raw, err := d.readBytes(2)
		if err != nil {
			return err
		}
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(binary.BigEndian.Uint16(raw))
		for i := 0; i < count*2; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	case 0xdf:
		raw, err := d.readBytes(4)
		if err != nil {
			return err
		}
		if err := d.enterContainer(); err != nil {
			return err
		}
		count := int(binary.BigEndian.Uint32(raw))
		for i := 0; i < count*2; i++ {
			if err := d.skipValue(); err != nil {
				return err
			}
		}
		d.leaveContainer()
		return nil
	default:
		return ErrKPACKInvalidTOC
	}
}

func parseTOC(tocBytes []byte) ([]KPACKEntry, error) {
	if len(tocBytes) == 0 {
		return nil, ErrKPACKInvalidTOC
	}
	d := &msgpackDecoder{data: tocBytes}

	b := tocBytes[0]
	isMap := (b >= 0x80 && b <= 0x8f) || b == 0xde || b == 0xdf
	isArray := (b >= 0x90 && b <= 0x9f) || b == 0xdc || b == 0xdd

	if !isMap && !isArray {
		return nil, ErrKPACKInvalidTOC
	}

	var entries []KPACKEntry

	if isArray {
		arrLen, err := d.readArrayHeader()
		if err != nil {
			return nil, err
		}
		entries = make([]KPACKEntry, 0, arrLen)
		for i := 0; i < arrLen; i++ {
			entry, err := parseEntryMap(d)
			if err != nil {
				return nil, err
			}
			entries = append(entries, entry)
		}
		d.leaveContainer()
	} else {
		mapLen, err := d.readMapHeader()
		if err != nil {
			return nil, err
		}
		for i := 0; i < mapLen; i++ {
			key, err := d.readString()
			if err != nil {
				return nil, err
			}
			switch strings.ToLower(key) {
			case "entries", "targets":
				arrLen, err := d.readArrayHeader()
				if err != nil {
					return nil, err
				}
				for j := 0; j < arrLen; j++ {
					entry, err := parseEntryMap(d)
					if err != nil {
						return nil, err
					}
					entries = append(entries, entry)
				}
				d.leaveContainer()
			case "version":
				if err := d.skipValue(); err != nil {
					return nil, err
				}
			default:
				if d.pos < len(d.data) {
					valB := d.data[d.pos]
					valIsMap := (valB >= 0x80 && valB <= 0x8f) || valB == 0xde || valB == 0xdf
					if valIsMap {
						entry, err := parseEntryMap(d)
						if err != nil {
							return nil, err
						}
						if entry.Target == "" {
							entry.Target = key
						}
						entries = append(entries, entry)
					} else {
						if err := d.skipValue(); err != nil {
							return nil, err
						}
					}
				}
			}
		}
		d.leaveContainer()
	}
	return entries, nil
}

func parseEntryMap(d *msgpackDecoder) (KPACKEntry, error) {
	var entry KPACKEntry
	mapLen, err := d.readMapHeader()
	if err != nil {
		return entry, err
	}
	for j := 0; j < mapLen; j++ {
		fieldName, err := d.readString()
		if err != nil {
			return entry, err
		}
		switch strings.ToLower(fieldName) {
		case "target", "arch":
			s, err := d.readString()
			if err != nil {
				return entry, err
			}
			entry.Target = s
		case "compression", "comp", "format":
			if d.pos < len(d.data) {
				vb := d.data[d.pos]
				if (vb >= 0xa0 && vb <= 0xbf) || vb == 0xd9 || vb == 0xda || vb == 0xdb {
					strVal, err := d.readString()
					if err != nil {
						return entry, err
					}
					if strings.EqualFold(strVal, "zstd") {
						entry.Compression = CompressionZSTD
					} else {
						entry.Compression = CompressionNone
					}
				} else {
					v, err := d.readUint()
					if err != nil {
						return entry, err
					}
					entry.Compression = CompressionType(v)
				}
			}
		case "offset":
			v, err := d.readUint()
			if err != nil {
				return entry, err
			}
			entry.Offset = v
		case "compressed_size", "size", "csize":
			v, err := d.readUint()
			if err != nil {
				return entry, err
			}
			entry.CompressedSize = v
		case "uncompressed_size", "raw_size", "usize", "orig_size":
			v, err := d.readUint()
			if err != nil {
				return entry, err
			}
			entry.UncompressedSize = v
		default:
			if err := d.skipValue(); err != nil {
				return entry, err
			}
		}
	}
	d.leaveContainer()
	if entry.UncompressedSize == 0 && entry.CompressedSize > 0 {
		entry.UncompressedSize = entry.CompressedSize
	}
	return entry, nil
}
