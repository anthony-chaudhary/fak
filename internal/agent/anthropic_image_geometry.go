package agent

// anthropic_image_geometry.go — geometry-aware per-image token cost (#5165), the precision
// follow-up to the flat imageTokenCost ceiling shipped in f992aa051.
//
// Anthropic bills an image at ~(width*height)/750 tokens, capped at ~imageTokenCost. The flat
// constant charges EVERY image that ceiling. That is correct and safe — it can never under-count,
// which is the failure mode that lets a real overflow slip past the budget — but it prices a 64x64
// favicon and a 1512x982 full-page screenshot identically at ~1600 tokens, when the favicon is
// really ~5. In an image-heavy session that over-count is what the compaction budget compares
// against, so fak sheds turns it never needed to shed.
//
// This file recovers the pixel geometry only where it is CHEAP: an explicit width/height on the
// image source, or the fixed-offset header of the base64 payload fak is already holding in memory.
// It never decodes pixel data and never pulls in an image codec — just the container headers.
//
// The flat cost stays the floor in the sense the issue means it: whenever geometry is NOT
// recoverable — a URL source with no dimensions, an unknown or truncated container, a JPEG whose
// frame header sits past the scan bound — the image falls back to imageTokenCost exactly as today.
// A geometry-priced image is therefore the only thing that moves, and it moves toward the truth.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"strings"
)

// imageTokenPixelsPerToken is Anthropic's documented per-image cost divisor: an image costs
// ~(width*height)/750 tokens. imageTokenCost remains the cap AND the unknown-geometry default.
const imageTokenPixelsPerToken = 750

// imageHeaderScanBytes bounds how much of an image's base64 payload is decoded to find the
// dimensions. PNG, GIF and WebP carry theirs at fixed offsets inside the first ~30 bytes; only
// JPEG needs a walk down the segment chain. 8 KiB reaches the frame header of the overwhelming
// majority of real JPEGs while keeping the decode cheap enough to sit inside the compaction walk,
// which re-estimates every element on every pass. A JPEG whose frame header sits past this bound
// (a large embedded EXIF thumbnail pushes it out) simply takes the flat cost.
const imageHeaderScanBytes = 8 << 10

// maxPlausibleImageDim rejects a dimension no real attachment carries. It is a guard against a
// corrupt or hostile header yielding an absurd pixel count, and it keeps width*height inside int64
// on every GOARCH (including 32-bit, where plain int would overflow).
const maxPlausibleImageDim = 1 << 20

// imageBlockTokenCost prices ONE image content block in the same ~token currency as the budget.
// It returns the geometry-derived ~(w*h)/750 capped at imageTokenCost when the dimensions are
// recoverable, and the flat imageTokenCost otherwise — so an image fak cannot measure is never
// charged less than it is today.
func imageBlockTokenCost(blk json.RawMessage) int {
	w, h, ok := imageBlockDimensions(blk)
	if !ok {
		return imageTokenCost
	}
	cost := (int64(w) * int64(h)) / imageTokenPixelsPerToken
	if cost >= imageTokenCost {
		return imageTokenCost
	}
	if cost < 1 {
		// A sub-750-pixel image still occupies a content block on the wire; never charge a real
		// image zero, or a wall of thumbnails would estimate to nothing at all.
		return 1
	}
	return int(cost)
}

// imageBlockDimensions recovers an image block's pixel geometry without decoding the image. An
// explicit width/height wins where present: it is exact, costs no decode, and is the only geometry
// a non-base64 (URL) source can offer. Otherwise the base64 payload's container header is sniffed.
// ok is false whenever the geometry cannot be established, which routes the caller to the flat cost.
func imageBlockDimensions(blk json.RawMessage) (w, h int, ok bool) {
	var b struct {
		Width  int `json:"width"`
		Height int `json:"height"`
		Source struct {
			Data   string `json:"data"`
			Width  int    `json:"width"`
			Height int    `json:"height"`
		} `json:"source"`
	}
	if json.Unmarshal(blk, &b) != nil {
		return 0, 0, false
	}
	if w, h, ok := plausibleDims(int64(b.Source.Width), int64(b.Source.Height)); ok {
		return w, h, true
	}
	if w, h, ok := plausibleDims(int64(b.Width), int64(b.Height)); ok {
		return w, h, true
	}
	if b.Source.Data == "" {
		return 0, 0, false
	}
	return sniffImageDimensions(decodeBase64Prefix(b.Source.Data, imageHeaderScanBytes))
}

// decodeBase64Prefix decodes at most maxBytes of a base64 payload. Cutting on a 4-character quantum
// boundary keeps the prefix independently decodable, so a multi-megabyte screenshot costs a few KiB
// of work rather than a full decode.
func decodeBase64Prefix(data string, maxBytes int) []byte {
	// Whitespace is legal inside a JSON string but not inside base64.StdEncoding. Real payloads
	// carry none, so only pay for the strip when one is actually present.
	if strings.ContainsAny(data, " \t\r\n") {
		data = strings.Map(func(r rune) rune {
			switch r {
			case ' ', '\t', '\r', '\n':
				return -1
			}
			return r
		}, data)
	}
	chars := (maxBytes/3 + 1) * 4
	if chars > len(data) {
		chars = len(data)
	}
	chars -= chars % 4
	if chars == 0 {
		return nil
	}
	out, err := base64.StdEncoding.DecodeString(data[:chars])
	if err != nil {
		return nil
	}
	return out
}

// sniffImageDimensions dispatches on the container's magic bytes rather than the block's declared
// media_type: a mislabelled source then falls back to the flat cost instead of mis-parsing some
// other format's header as geometry. These four are the container types the API accepts.
func sniffImageDimensions(b []byte) (w, h int, ok bool) {
	switch {
	case len(b) >= 8 && string(b[:8]) == "\x89PNG\r\n\x1a\n":
		return pngDimensions(b)
	case len(b) >= 6 && (string(b[:6]) == "GIF87a" || string(b[:6]) == "GIF89a"):
		return gifDimensions(b)
	case len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WEBP":
		return webpDimensions(b)
	case len(b) >= 2 && b[0] == 0xFF && b[1] == 0xD8:
		return jpegDimensions(b)
	}
	return 0, 0, false
}

// pngDimensions reads the IHDR chunk: an 8-byte signature, a 4-byte chunk length, the "IHDR" type,
// then width and height as big-endian uint32.
func pngDimensions(b []byte) (w, h int, ok bool) {
	if len(b) < 24 || string(b[12:16]) != "IHDR" {
		return 0, 0, false
	}
	return plausibleDims(int64(binary.BigEndian.Uint32(b[16:20])), int64(binary.BigEndian.Uint32(b[20:24])))
}

// gifDimensions reads the logical screen descriptor: little-endian uint16 width and height directly
// after the 6-byte header.
func gifDimensions(b []byte) (w, h int, ok bool) {
	if len(b) < 10 {
		return 0, 0, false
	}
	return plausibleDims(int64(binary.LittleEndian.Uint16(b[6:8])), int64(binary.LittleEndian.Uint16(b[8:10])))
}

// webpDimensions handles the three RIFF payload flavours. The chunk FourCC sits at offset 12 and
// its body at offset 20.
func webpDimensions(b []byte) (w, h int, ok bool) {
	if len(b) < 16 {
		return 0, 0, false
	}
	switch string(b[12:16]) {
	case "VP8X":
		// Extended format: 24-bit little-endian (width-1) and (height-1) at offset 24.
		if len(b) < 30 {
			return 0, 0, false
		}
		cw := int64(b[24]) | int64(b[25])<<8 | int64(b[26])<<16
		ch := int64(b[27]) | int64(b[28])<<8 | int64(b[29])<<16
		return plausibleDims(cw+1, ch+1)
	case "VP8 ":
		// Lossy: a 3-byte frame tag, the 3-byte start code 0x9D012A, then 14-bit little-endian
		// width and height.
		if len(b) < 30 || b[23] != 0x9D || b[24] != 0x01 || b[25] != 0x2A {
			return 0, 0, false
		}
		return plausibleDims(
			int64(binary.LittleEndian.Uint16(b[26:28])&0x3FFF),
			int64(binary.LittleEndian.Uint16(b[28:30])&0x3FFF),
		)
	case "VP8L":
		// Lossless: a 0x2F signature byte, then 14-bit (width-1) and 14-bit (height-1) packed
		// little-endian.
		if len(b) < 25 || b[20] != 0x2F {
			return 0, 0, false
		}
		bits := uint32(b[21]) | uint32(b[22])<<8 | uint32(b[23])<<16 | uint32(b[24])<<24
		return plausibleDims(int64(bits&0x3FFF)+1, int64((bits>>14)&0x3FFF)+1)
	}
	return 0, 0, false
}

// jpegDimensions walks the segment chain from just past the SOI. Each segment is 0xFF, a marker
// byte, then a big-endian uint16 length that counts itself. A start-of-frame segment carries a
// 1-byte sample precision, then height and width as big-endian uint16.
func jpegDimensions(b []byte) (w, h int, ok bool) {
	for i := 2; i+4 <= len(b); {
		if b[i] != 0xFF {
			return 0, 0, false
		}
		marker := b[i+1]
		switch {
		case marker == 0xFF:
			// Fill byte before the real marker.
			i++
			continue
		case marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD9):
			// Standalone markers (restart, SOI, EOI) carry no length payload.
			i += 2
			continue
		case marker == 0xDA:
			// Start of scan: entropy-coded data follows and no frame header was found ahead of it.
			return 0, 0, false
		}
		if isJPEGStartOfFrame(marker) {
			if i+9 > len(b) {
				return 0, 0, false
			}
			return plausibleDims(
				int64(binary.BigEndian.Uint16(b[i+7:i+9])),
				int64(binary.BigEndian.Uint16(b[i+5:i+7])),
			)
		}
		size := int(binary.BigEndian.Uint16(b[i+2 : i+4]))
		if size < 2 {
			return 0, 0, false
		}
		i += 2 + size
	}
	return 0, 0, false
}

// isJPEGStartOfFrame reports whether a marker is one of the SOF0..SOF15 frame headers. 0xC4, 0xC8
// and 0xCC sit in the same numeric range but are Huffman-table / JPEG-extension / arithmetic-table
// segments, not frame headers, and carry no dimensions.
func isJPEGStartOfFrame(marker byte) bool {
	return marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC
}

// plausibleDims validates a candidate geometry and narrows it to int. It rejects zero, negative and
// absurd dimensions so a corrupt header can never produce a bogus pixel count.
func plausibleDims(w, h int64) (int, int, bool) {
	if w <= 0 || h <= 0 || w > maxPlausibleImageDim || h > maxPlausibleImageDim {
		return 0, 0, false
	}
	return int(w), int(h), true
}
