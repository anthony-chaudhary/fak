package tokenizer

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// DefaultInlineImageMinBytes is the minimum data-URL length (in bytes) that
// ExtractInlineImageRefs lifts out of a raw JSON request before parsing.
const DefaultInlineImageMinBytes = 4096

// InlineImageRef is a zero-copy reference to an inline data URL inside a raw
// request buffer, paired with the compact marker that replaces it for JSON parsing.
type InlineImageRef struct {
	Index  int
	Start  int    // byte offset of the data URL payload (after opening '"')
	End    int    // byte offset of the closing '"'
	Data   []byte // zero-copy slice raw[Start:End]
	Marker string
}

// ExtractInlineImageRefs scans raw for quoted "data:..." URLs whose length is at
// least minBytes (defaults to DefaultInlineImageMinBytes when <= 0) and replaces
// them with compact "data:x-fak-image-ref,<idx>" markers. When no data URL meets
// the threshold, raw is returned unchanged with nil refs and zero allocations.
func ExtractInlineImageRefs(raw []byte, minBytes int) (rewritten []byte, refs []InlineImageRef) {
	if minBytes <= 0 {
		minBytes = DefaultInlineImageMinBytes
	}
	needle := []byte(`"data:`)
	pos := 0
	lastCopied := 0
	var out bytes.Buffer

	for pos < len(raw) {
		idx := bytes.Index(raw[pos:], needle)
		if idx < 0 {
			break
		}
		quoteStart := pos + idx
		// Skip escaped quotes inside JSON string literals.
		if quoteStart > 0 && raw[quoteStart-1] == '\\' {
			pos = quoteStart + len(needle)
			continue
		}
		urlStart := quoteStart + 1
		// Find the unescaped closing quote.
		urlEnd := -1
		for i := urlStart + len("data:"); i < len(raw); i++ {
			if raw[i] == '"' && raw[i-1] != '\\' {
				urlEnd = i
				break
			}
		}
		if urlEnd < 0 {
			break
		}
		if urlEnd-urlStart >= minBytes {
			refIdx := len(refs)
			marker := fmt.Sprintf("data:x-fak-image-ref,%d", refIdx)
			refs = append(refs, InlineImageRef{
				Index:  refIdx,
				Start:  urlStart,
				End:    urlEnd,
				Data:   raw[urlStart:urlEnd],
				Marker: marker,
			})
			if out.Len() == 0 {
				// Pre-size conservatively: rewritten JSON is smaller than raw.
				out.Grow(len(raw) - (urlEnd - urlStart) + len(marker))
			}
			out.Write(raw[lastCopied:urlStart])
			out.WriteString(marker)
			lastCopied = urlEnd
		}
		pos = urlEnd + 1
	}

	if len(refs) == 0 {
		return raw, nil
	}
	out.Write(raw[lastCopied:])
	return out.Bytes(), refs
}

// ImageFormat identifies the container format of an inline image header.
type ImageFormat string

const (
	ImageFormatPNG  ImageFormat = "png"
	ImageFormatGIF  ImageFormat = "gif"
	ImageFormatWebP ImageFormat = "webp"
	ImageFormatJPEG ImageFormat = "jpeg"
)

// ImageDimensions holds the oriented pixel dimensions read from an image header.
type ImageDimensions struct {
	Format      ImageFormat
	Width       int
	Height      int
	Orientation int // EXIF orientation (1-8) when present, 0 otherwise
}

var (
	errInvalidDataURL    = errors.New("tokenizer: invalid base64 image data URL")
	errUnsupportedImage  = errors.New("tokenizer: unsupported or truncated image header")
	errInvalidDimensions = errors.New("tokenizer: invalid image dimensions")
)

// ImageDimensionsFromDataURL parses a "data:<mime>;base64,<payload>" URL and reads
// the image dimensions directly from the container header without decoding pixels.
func ImageDimensionsFromDataURL(dataURL []byte) (ImageDimensions, error) {
	if !bytes.HasPrefix(dataURL, []byte("data:")) {
		return ImageDimensions{}, errInvalidDataURL
	}
	comma := bytes.IndexByte(dataURL, ',')
	if comma < 0 {
		return ImageDimensions{}, errInvalidDataURL
	}
	meta := string(dataURL[len("data:"):comma])
	if !strings.Contains(strings.ToLower(meta), ";base64") {
		return ImageDimensions{}, errInvalidDataURL
	}
	return ImageDimensionsFromBase64(dataURL[comma+1:])
}

// ImageDimensionsFromBase64 reads PNG, GIF, WebP, or JPEG dimensions directly from
// base64-encoded bytes using a bounded streaming decoder (no full-image decode).
func ImageDimensionsFromBase64(b64 []byte) (ImageDimensions, error) {
	// Trim optional whitespace/padding? Standard and raw base64 both work via NewDecoder
	// after stripping ASCII whitespace if present; fast path avoids copying when clean.
	r := newCleanBase64Reader(b64)
	dec := base64.NewDecoder(base64.StdEncoding, r)
	return ImageDimensionsFromReader(dec)
}

// ImageDimensionsFromHeader reads image dimensions from raw encoded container bytes.
func ImageDimensionsFromHeader(header []byte) (ImageDimensions, error) {
	return ImageDimensionsFromReader(bytes.NewReader(header))
}

// ImageDimensionsFromReader reads PNG, GIF, WebP, or JPEG dimensions from r,
// consuming only the container header bytes needed to determine width and height.
func ImageDimensionsFromReader(r io.Reader) (ImageDimensions, error) {
	var magic [32]byte
	n, err := io.ReadFull(r, magic[:12])
	if err != nil && err != io.ErrUnexpectedEOF {
		return ImageDimensions{}, fmt.Errorf("%w: %v", errUnsupportedImage, err)
	}
	if n < 10 {
		return ImageDimensions{}, errUnsupportedImage
	}

	switch {
	case bytes.HasPrefix(magic[:n], []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}):
		// Need bytes 12..23 (rest of IHDR: 4B length, 4B "IHDR", 4B width, 4B height).
		if _, err := io.ReadFull(r, magic[12:24]); err != nil {
			return ImageDimensions{}, errUnsupportedImage
		}
		if string(magic[12:16]) != "IHDR" {
			return ImageDimensions{}, errUnsupportedImage
		}
		w := int(binary.BigEndian.Uint32(magic[16:20]))
		h := int(binary.BigEndian.Uint32(magic[20:24]))
		if w <= 0 || h <= 0 {
			return ImageDimensions{}, errInvalidDimensions
		}
		return ImageDimensions{Format: ImageFormatPNG, Width: w, Height: h}, nil

	case bytes.HasPrefix(magic[:n], []byte("GIF87a")) || bytes.HasPrefix(magic[:n], []byte("GIF89a")):
		w := int(binary.LittleEndian.Uint16(magic[6:8]))
		h := int(binary.LittleEndian.Uint16(magic[8:10]))
		if w <= 0 || h <= 0 {
			return ImageDimensions{}, errInvalidDimensions
		}
		return ImageDimensions{Format: ImageFormatGIF, Width: w, Height: h}, nil

	case bytes.HasPrefix(magic[:n], []byte("RIFF")) && n >= 12 && bytes.Equal(magic[8:12], []byte("WEBP")):
		// Read chunk FourCC (4B) + size (4B) + up to 10B of payload (bytes 12..30).
		if _, err := io.ReadFull(r, magic[12:30]); err != nil {
			return ImageDimensions{}, errUnsupportedImage
		}
		return parseWebPHeader(magic[:30])

	case magic[0] == 0xFF && magic[1] == 0xD8:
		return parseJPEGHeader(io.MultiReader(bytes.NewReader(magic[2:n]), r))

	default:
		return ImageDimensions{}, errUnsupportedImage
	}
}

func parseWebPHeader(h []byte) (ImageDimensions, error) {
	if len(h) < 30 {
		return ImageDimensions{}, errUnsupportedImage
	}
	chunk := string(h[12:16])
	switch chunk {
	case "VP8 ":
		// Bytes 20..22: frame tag, 23..25: 0x9d 0x01 0x2a, 26..27: width, 28..29: height.
		if h[23] != 0x9d || h[24] != 0x01 || h[25] != 0x2a {
			return ImageDimensions{}, errUnsupportedImage
		}
		w := int(binary.LittleEndian.Uint16(h[26:28]) & 0x3FFF)
		hgt := int(binary.LittleEndian.Uint16(h[28:30]) & 0x3FFF)
		if w <= 0 || hgt <= 0 {
			return ImageDimensions{}, errInvalidDimensions
		}
		return ImageDimensions{Format: ImageFormatWebP, Width: w, Height: hgt}, nil

	case "VP8L":
		if h[20] != 0x2f {
			return ImageDimensions{}, errUnsupportedImage
		}
		bits := binary.LittleEndian.Uint32(h[21:25])
		w := int(bits&0x3FFF) + 1
		hgt := int((bits>>14)&0x3FFF) + 1
		return ImageDimensions{Format: ImageFormatWebP, Width: w, Height: hgt}, nil

	case "VP8X":
		w := int(h[24]) | int(h[25])<<8 | int(h[26])<<16 + 1
		hgt := int(h[27]) | int(h[28])<<8 | int(h[29])<<16 + 1
		return ImageDimensions{Format: ImageFormatWebP, Width: w, Height: hgt}, nil

	default:
		return ImageDimensions{}, errUnsupportedImage
	}
}

func parseJPEGHeader(r io.Reader) (ImageDimensions, error) {
	var buf [8]byte
	orientation := 0

	for {
		// Find next marker (0xFF followed by non-0xFF, non-0x00).
		if _, err := io.ReadFull(r, buf[:1]); err != nil {
			return ImageDimensions{}, errUnsupportedImage
		}
		if buf[0] != 0xFF {
			return ImageDimensions{}, errUnsupportedImage
		}
		for buf[0] == 0xFF {
			if _, err := io.ReadFull(r, buf[:1]); err != nil {
				return ImageDimensions{}, errUnsupportedImage
			}
		}
		marker := buf[0]
		if marker == 0x00 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0xD8 {
			continue
		}
		if marker == 0xD9 || marker == 0xDA { // EOI or SOS (image data begins)
			return ImageDimensions{}, errUnsupportedImage
		}

		if _, err := io.ReadFull(r, buf[:2]); err != nil {
			return ImageDimensions{}, errUnsupportedImage
		}
		segLen := int(binary.BigEndian.Uint16(buf[:2]))
		if segLen < 2 {
			return ImageDimensions{}, errUnsupportedImage
		}
		payloadLen := segLen - 2

		if isJPEGSOF(marker) {
			if payloadLen < 5 {
				return ImageDimensions{}, errUnsupportedImage
			}
			if _, err := io.ReadFull(r, buf[:5]); err != nil {
				return ImageDimensions{}, errUnsupportedImage
			}
			h := int(binary.BigEndian.Uint16(buf[1:3]))
			w := int(binary.BigEndian.Uint16(buf[3:5]))
			if w <= 0 || h <= 0 {
				return ImageDimensions{}, errInvalidDimensions
			}
			if orientation >= 5 && orientation <= 8 {
				w, h = h, w
			}
			return ImageDimensions{
				Format:      ImageFormatJPEG,
				Width:       w,
				Height:      h,
				Orientation: orientation,
			}, nil
		}

		if marker == 0xE1 && payloadLen >= 14 && orientation == 0 {
			// Cap APP1 read to 64 KiB (max JPEG segment size anyway).
			app1 := make([]byte, payloadLen)
			if _, err := io.ReadFull(r, app1); err != nil {
				return ImageDimensions{}, errUnsupportedImage
			}
			if orient, ok := parseEXIFOrientation(app1); ok {
				orientation = orient
			}
			continue
		}

		if _, err := io.CopyN(io.Discard, r, int64(payloadLen)); err != nil {
			return ImageDimensions{}, errUnsupportedImage
		}
	}
}

func isJPEGSOF(marker byte) bool {
	switch marker {
	case 0xC0, 0xC1, 0xC2, 0xC3,
		0xC5, 0xC6, 0xC7,
		0xC9, 0xCA, 0xCB,
		0xCD, 0xCE, 0xCF:
		return true
	default:
		return false
	}
}

func parseEXIFOrientation(app1 []byte) (int, bool) {
	if len(app1) < 14 || string(app1[:6]) != "Exif\x00\x00" {
		return 0, false
	}
	tiff := app1[6:]
	if len(tiff) < 8 {
		return 0, false
	}
	var order binary.ByteOrder
	switch string(tiff[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 0, false
	}
	if order.Uint16(tiff[2:4]) != 0x002A {
		return 0, false
	}
	ifdOff := int(order.Uint32(tiff[4:8]))
	if ifdOff < 8 || ifdOff+2 > len(tiff) {
		return 0, false
	}
	count := int(order.Uint16(tiff[ifdOff : ifdOff+2]))
	for i := 0; i < count; i++ {
		entryOff := ifdOff + 2 + i*12
		if entryOff+12 > len(tiff) {
			break
		}
		tag := order.Uint16(tiff[entryOff : entryOff+2])
		if tag == 0x0112 { // Orientation
			typ := order.Uint16(tiff[entryOff+2 : entryOff+4])
			cnt := order.Uint32(tiff[entryOff+4 : entryOff+8])
			if typ == 3 && cnt == 1 {
				val := int(order.Uint16(tiff[entryOff+8 : entryOff+10]))
				if val >= 1 && val <= 8 {
					return val, true
				}
			}
			return 0, false
		}
	}
	return 0, false
}

// newCleanBase64Reader returns an io.Reader over b64 that skips ASCII whitespace
// without copying when none is present.
func newCleanBase64Reader(b64 []byte) io.Reader {
	if bytes.IndexAny(b64, " \r\n\t") < 0 {
		return bytes.NewReader(b64)
	}
	return &whitespaceStripReader{src: b64}
}

type whitespaceStripReader struct {
	src []byte
	pos int
}

func (r *whitespaceStripReader) Read(p []byte) (int, error) {
	n := 0
	for r.pos < len(r.src) && n < len(p) {
		b := r.src[r.pos]
		r.pos++
		if b == ' ' || b == '\r' || b == '\n' || b == '\t' {
			continue
		}
		p[n] = b
		n++
	}
	if n == 0 && r.pos >= len(r.src) {
		return 0, io.EOF
	}
	return n, nil
}

// VisionResizeRule parameterizes model-specific image resizing and token counting.
// DefaultQwenVLResizeRule matches Qwen2-VL / Qwen2.5-VL (factor=28, min=3136, max=12845056).
type VisionResizeRule struct {
	PatchSize      int // ViT patch size (e.g. 14 for Qwen2-VL, 16 for Qwen3-VL)
	MergeSize      int // Spatial merge factor (e.g. 2)
	MinPixels      int // Minimum resized pixel area
	MaxPixels      int // Maximum resized pixel area
	MaxAspectRatio int // Maximum max(w,h)/min(w,h); defaults to 200 when <= 0
}

// DefaultQwenVLResizeRule is the standard Qwen2-VL / Qwen2.5-VL smart_resize rule.
var DefaultQwenVLResizeRule = VisionResizeRule{
	PatchSize:      14,
	MergeSize:      2,
	MinPixels:      4 * 28 * 28,
	MaxPixels:      16384 * 28 * 28,
	MaxAspectRatio: 200,
}

// VisionGrid is the resized pixel dimensions and resulting image-token count.
type VisionGrid struct {
	ResizedWidth  int
	ResizedHeight int
	GridRows      int
	GridCols      int
	TokenCount    int
}

// VisionTokenCount applies rule to dims and returns the exact post-resize grid and
// image-token count without decoding pixels.
func VisionTokenCount(dims ImageDimensions, rule VisionResizeRule) (VisionGrid, error) {
	if dims.Width <= 0 || dims.Height <= 0 {
		return VisionGrid{}, errInvalidDimensions
	}
	if rule.PatchSize <= 0 || rule.MergeSize <= 0 {
		return VisionGrid{}, errors.New("tokenizer: invalid vision resize patch/merge size")
	}
	factor := rule.PatchSize * rule.MergeSize
	maxAspect := rule.MaxAspectRatio
	if maxAspect <= 0 {
		maxAspect = 200
	}
	w, h := dims.Width, dims.Height
	longSide, shortSide := w, h
	if h > w {
		longSide, shortSide = h, w
	}
	if longSide > shortSide*maxAspect {
		return VisionGrid{}, fmt.Errorf("tokenizer: image aspect ratio %d:%d exceeds %d", longSide, shortSide, maxAspect)
	}

	hBar := max(factor, int(math.Round(float64(h)/float64(factor)))*factor)
	wBar := max(factor, int(math.Round(float64(w)/float64(factor)))*factor)

	area := hBar * wBar
	if rule.MaxPixels > 0 && area > rule.MaxPixels {
		beta := math.Sqrt(float64(h*w) / float64(rule.MaxPixels))
		hBar = max(factor, int(math.Floor(float64(h)/beta/float64(factor)))*factor)
		wBar = max(factor, int(math.Floor(float64(w)/beta/float64(factor)))*factor)
	} else if rule.MinPixels > 0 && area < rule.MinPixels {
		beta := math.Sqrt(float64(rule.MinPixels) / float64(h*w))
		hBar = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}

	rows := hBar / factor
	cols := wBar / factor
	return VisionGrid{
		ResizedWidth:  wBar,
		ResizedHeight: hBar,
		GridRows:      rows,
		GridCols:      cols,
		TokenCount:    rows * cols,
	}, nil
}

// CountInlineImageTokens computes the exact image-token count directly from a base64
// data URL using header-only dimension parsing and the model's resize rule.
func CountInlineImageTokens(dataURL []byte, rule VisionResizeRule) (VisionGrid, error) {
	dims, err := ImageDimensionsFromDataURL(dataURL)
	if err != nil {
		return VisionGrid{}, err
	}
	return VisionTokenCount(dims, rule)
}
