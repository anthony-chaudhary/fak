package l3kv

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
)

// SpanCompressionMagic is the 4-byte magic header [0xFA, 0x4C, 0x5A, 0x34] (FA-LZ4)
// placed at the start of compressed L3 KV spans (#11040, parent #10964).
var SpanCompressionMagic = [4]byte{0xFA, 0x4C, 0x5A, 0x34}

const (
	spanHeaderLen = 12      // 4-byte magic + 4-byte uncompressed length + 4-byte CRC32
	maxSpanBytes  = 1 << 30 // 1 GiB sanity safety cap on span payloads
)

// IsCompressedSpan reports whether payload begins with the FA-LZ4 magic header.
func IsCompressedSpan(payload []byte) bool {
	return len(payload) >= len(SpanCompressionMagic) &&
		bytes.Equal(payload[:len(SpanCompressionMagic)], SpanCompressionMagic[:])
}

// CompressSpan compresses raw span bytes using fast LZ4 block encoding framed
// with the 4-byte FA-LZ4 magic header, a 4-byte little-endian uncompressed
// length prefix, and a 4-byte CRC32 IEEE checksum.
//
// Transparent fail-safe handling: if raw payload is smaller than or equal to
// the compressed framing (e.g. incompressible random data or tiny payloads),
// raw is returned uncompressed so storage never inflates.
func CompressSpan(raw []byte) ([]byte, error) {
	if len(raw) == 0 {
		return raw, nil
	}
	if len(raw) > maxSpanBytes {
		return nil, fmt.Errorf("l3kv: raw span size %d exceeds maximum %d", len(raw), maxSpanBytes)
	}

	lz4Payload := compressLZ4Block(raw)
	framedLen := spanHeaderLen + len(lz4Payload)

	// If compression does not reduce size, leave uncompressed (fail-safe).
	// Guard against raw payload accidentally starting with the magic bytes.
	if framedLen >= len(raw) && !IsCompressedSpan(raw) {
		return raw, nil
	}

	out := make([]byte, framedLen)
	copy(out[0:4], SpanCompressionMagic[:])
	binary.LittleEndian.PutUint32(out[4:8], uint32(len(raw)))
	binary.LittleEndian.PutUint32(out[8:12], crc32.ChecksumIEEE(raw))
	copy(out[spanHeaderLen:], lz4Payload)
	return out, nil
}

// DecompressSpan verifies the 4-byte FA-LZ4 magic header, validates the CRC32
// checksum, and decodes the LZ4 payload back to bit-identical bytes.
//
// Transparent fail-safe handling: if payload is uncompressed (does not start
// with FA-LZ4 magic), it is returned cleanly as-is.
func DecompressSpan(payload []byte) ([]byte, error) {
	if len(payload) == 0 {
		return payload, nil
	}
	if !IsCompressedSpan(payload) {
		// Uncompressed legacy or fail-safe payload: return as-is.
		return payload, nil
	}
	if len(payload) < spanHeaderLen {
		return nil, errors.New("l3kv: truncated compressed span header")
	}

	uncompressedLen := int(binary.LittleEndian.Uint32(payload[4:8]))
	if uncompressedLen < 0 || uncompressedLen > maxSpanBytes {
		return nil, fmt.Errorf("l3kv: invalid uncompressed span length: %d", uncompressedLen)
	}
	expectedCRC := binary.LittleEndian.Uint32(payload[8:12])

	decompressed, err := decompressLZ4Block(payload[spanHeaderLen:], uncompressedLen)
	if err != nil {
		return nil, fmt.Errorf("l3kv: decompress span lz4: %w", err)
	}

	actualCRC := crc32.ChecksumIEEE(decompressed)
	if actualCRC != expectedCRC {
		return nil, fmt.Errorf("l3kv: span CRC32 checksum mismatch: got 0x%08x, want 0x%08x", actualCRC, expectedCRC)
	}

	return decompressed, nil
}

// compressLZ4Block encodes src using standard LZ4 block format.
func compressLZ4Block(src []byte) []byte {
	n := len(src)
	if n == 0 {
		return nil
	}

	// LZ4 requires at least 13 bytes for match search (5 trailing literals + 4 match + 4 literals).
	if n < 13 {
		dst := make([]byte, 0, 1+n)
		tokenLit := n
		dst = append(dst, byte(tokenLit<<4))
		dst = append(dst, src...)
		return dst
	}

	const (
		hashLog   = 14
		tableSize = 1 << hashLog
	)
	table := make([]int32, tableSize)

	dst := make([]byte, 0, n)
	anchor := 0
	pos := 0
	matchLimit := n - 5

	for pos <= matchLimit-4 {
		v := binary.LittleEndian.Uint32(src[pos : pos+4])
		h := (v * 2654435761) >> (32 - hashLog)
		ref := int(table[h]) - 1
		table[h] = int32(pos + 1)

		if ref >= 0 && pos-ref <= 65535 && binary.LittleEndian.Uint32(src[ref:ref+4]) == v {
			matchLen := 4
			for pos+matchLen < matchLimit && src[pos+matchLen] == src[ref+matchLen] {
				matchLen++
			}

			// Emit literal token and match token.
			litLen := pos - anchor
			tokenLit := litLen
			if tokenLit > 15 {
				tokenLit = 15
			}
			tokenMatch := matchLen - 4
			if tokenMatch > 15 {
				tokenMatch = 15
			}
			dst = append(dst, byte((tokenLit<<4)|tokenMatch))

			// Emit extended literal length if needed.
			if litLen >= 15 {
				rem := litLen - 15
				for rem >= 255 {
					dst = append(dst, 255)
					rem -= 255
				}
				dst = append(dst, byte(rem))
			}
			dst = append(dst, src[anchor:pos]...)

			// Emit 2-byte match offset (little-endian).
			offset := uint16(pos - ref)
			dst = append(dst, byte(offset), byte(offset>>8))

			// Emit extended match length if needed.
			if matchLen-4 >= 15 {
				rem := (matchLen - 4) - 15
				for rem >= 255 {
					dst = append(dst, 255)
					rem -= 255
				}
				dst = append(dst, byte(rem))
			}

			pos += matchLen
			anchor = pos

			if pos-2 >= 0 && pos-2 <= matchLimit-4 {
				v2 := binary.LittleEndian.Uint32(src[pos-2 : pos+2])
				h2 := (v2 * 2654435761) >> (32 - hashLog)
				table[h2] = int32(pos - 2 + 1)
			}
			continue
		}
		pos++
	}

	// Emit trailing literals (at least 5 bytes).
	litLen := n - anchor
	tokenLit := litLen
	if tokenLit > 15 {
		tokenLit = 15
	}
	dst = append(dst, byte(tokenLit<<4))
	if litLen >= 15 {
		rem := litLen - 15
		for rem >= 255 {
			dst = append(dst, 255)
			rem -= 255
		}
		dst = append(dst, byte(rem))
	}
	dst = append(dst, src[anchor:n]...)
	return dst
}

// decompressLZ4Block decodes an LZ4 block of expected length dstLen.
func decompressLZ4Block(src []byte, dstLen int) ([]byte, error) {
	if dstLen < 0 || dstLen > maxSpanBytes {
		return nil, fmt.Errorf("l3kv: invalid decompressed span length: %d", dstLen)
	}
	if dstLen == 0 {
		return []byte{}, nil
	}

	dst := make([]byte, dstLen)
	sp := 0
	dp := 0

	for sp < len(src) {
		token := src[sp]
		sp++

		// 1. Literal length
		litLen := int(token >> 4)
		if litLen == 15 {
			for {
				if sp >= len(src) {
					return nil, errors.New("l3kv: truncated lz4 literal length")
				}
				b := int(src[sp])
				sp++
				litLen += b
				if b != 255 {
					break
				}
			}
		}

		if litLen < 0 || dp+litLen < 0 || dp+litLen > dstLen || sp+litLen > len(src) {
			return nil, errors.New("l3kv: lz4 literal bounds overflow")
		}
		copy(dst[dp:dp+litLen], src[sp:sp+litLen])
		dp += litLen
		sp += litLen

		if sp == len(src) {
			break
		}

		// 2. Match offset
		if sp+2 > len(src) {
			return nil, errors.New("l3kv: truncated lz4 match offset")
		}
		offset := int(binary.LittleEndian.Uint16(src[sp : sp+2]))
		sp += 2
		if offset == 0 || offset > dp {
			return nil, fmt.Errorf("l3kv: invalid lz4 match offset %d at position %d", offset, dp)
		}

		// 3. Match length
		matchLen := int(token&0x0F) + 4
		if (token & 0x0F) == 15 {
			for {
				if sp >= len(src) {
					return nil, errors.New("l3kv: truncated lz4 match length")
				}
				b := int(src[sp])
				sp++
				matchLen += b
				if b != 255 {
					break
				}
			}
		}

		if matchLen < 4 || dp+matchLen < 0 || dp+matchLen > dstLen {
			return nil, errors.New("l3kv: lz4 match bounds overflow")
		}

		matchStart := dp - offset
		for i := 0; i < matchLen; i++ {
			dst[dp+i] = dst[matchStart+i]
		}
		dp += matchLen
	}

	if dp != dstLen {
		return nil, fmt.Errorf("l3kv: lz4 decompressed length mismatch: got %d, want %d", dp, dstLen)
	}
	return dst, nil
}
