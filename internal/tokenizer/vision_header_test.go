package tokenizer

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"
)

func buildPNGHeader(w, h uint32) []byte {
	var b [24]byte
	copy(b[:8], []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	binary.BigEndian.PutUint32(b[8:12], 13)
	copy(b[12:16], "IHDR")
	binary.BigEndian.PutUint32(b[16:20], w)
	binary.BigEndian.PutUint32(b[20:24], h)
	return b[:]
}

func buildGIFHeader(magic string, w, h uint16) []byte {
	var b [10]byte
	copy(b[:6], magic)
	binary.LittleEndian.PutUint16(b[6:8], w)
	binary.LittleEndian.PutUint16(b[8:10], h)
	return b[:]
}

func buildWebPVP8Header(w, h uint16) []byte {
	var b [30]byte
	copy(b[0:4], "RIFF")
	copy(b[8:12], "WEBP")
	copy(b[12:16], "VP8 ")
	b[23], b[24], b[25] = 0x9d, 0x01, 0x2a
	binary.LittleEndian.PutUint16(b[26:28], w&0x3FFF)
	binary.LittleEndian.PutUint16(b[28:30], h&0x3FFF)
	return b[:]
}

func buildWebPVP8LHeader(w, h uint32) []byte {
	var b [30]byte
	copy(b[0:4], "RIFF")
	copy(b[8:12], "WEBP")
	copy(b[12:16], "VP8L")
	b[20] = 0x2f
	bits := ((w - 1) & 0x3FFF) | (((h - 1) & 0x3FFF) << 14)
	binary.LittleEndian.PutUint32(b[21:25], bits)
	return b[:]
}

func buildWebPVP8XHeader(w, h uint32) []byte {
	var b [30]byte
	copy(b[0:4], "RIFF")
	copy(b[8:12], "WEBP")
	copy(b[12:16], "VP8X")
	wm := w - 1
	hm := h - 1
	b[24], b[25], b[26] = byte(wm), byte(wm>>8), byte(wm>>16)
	b[27], b[28], b[29] = byte(hm), byte(hm>>8), byte(hm>>16)
	return b[:]
}

func buildJPEGHeader(w, h uint16, exifOrientation uint16) []byte {
	var out bytes.Buffer
	out.Write([]byte{0xFF, 0xD8}) // SOI
	if exifOrientation > 0 {
		// APP1 with little-endian TIFF + 1 IFD entry (Orientation tag 0x0112).
		var app1 [28]byte
		copy(app1[0:6], "Exif\x00\x00")
		copy(app1[6:8], "II")
		binary.LittleEndian.PutUint16(app1[8:10], 0x002A)
		binary.LittleEndian.PutUint32(app1[10:14], 8) // IFD0 offset
		binary.LittleEndian.PutUint16(app1[14:16], 1) // 1 entry
		binary.LittleEndian.PutUint16(app1[16:18], 0x0112)
		binary.LittleEndian.PutUint16(app1[18:20], 3) // SHORT
		binary.LittleEndian.PutUint32(app1[20:24], 1) // count 1
		binary.LittleEndian.PutUint16(app1[24:26], exifOrientation)
		out.Write([]byte{0xFF, 0xE1})
		_ = binary.Write(&out, binary.BigEndian, uint16(len(app1)+2))
		out.Write(app1[:])
	}
	// Dummy APP0 segment to exercise marker skipping.
	out.Write([]byte{0xFF, 0xE0, 0x00, 0x06, 'J', 'F', 'I', 'F'})
	// SOF0 segment.
	out.Write([]byte{0xFF, 0xC0, 0x00, 0x08, 8})
	_ = binary.Write(&out, binary.BigEndian, h)
	_ = binary.Write(&out, binary.BigEndian, w)
	out.WriteByte(3)
	return out.Bytes()
}

// fak-test:runtime fast est=50ms lane=default
func TestVisionExtractInlineImageRefsZeroCopy(t *testing.T) {
	imgPayload := "data:image/png;base64," + base64.StdEncoding.EncodeToString(append(buildPNGHeader(560, 280), make([]byte, 5000)...))
	smallPayload := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buildPNGHeader(28, 28))
	raw := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"` + imgPayload + `"}},{"type":"image_url","image_url":{"url":"` + smallPayload + `"}},{"type":"text","text":"hi"}]}]}`)

	rewritten, refs := ExtractInlineImageRefs(raw, DefaultInlineImageMinBytes)
	if len(refs) != 1 {
		t.Fatalf("len(refs) = %d, want 1", len(refs))
	}
	if !bytes.Equal(refs[0].Data, []byte(imgPayload)) {
		t.Fatalf("ref[0].Data mismatch")
	}
	// Verify zero-copy: refs[0].Data aliases raw's backing array.
	if &refs[0].Data[0] != &raw[refs[0].Start] {
		t.Fatalf("ref[0].Data is not zero-copy")
	}
	if !bytes.Contains(rewritten, []byte(refs[0].Marker)) {
		t.Fatalf("rewritten JSON missing marker %q", refs[0].Marker)
	}
	if !json.Valid(rewritten) {
		t.Fatalf("rewritten JSON is invalid: %s", rewritten)
	}

	// Below threshold: returns raw unchanged with nil refs.
	unchanged, none := ExtractInlineImageRefs(raw, 100000)
	if none != nil || &unchanged[0] != &raw[0] {
		t.Fatalf("below-threshold scan should return raw unchanged with nil refs")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestVisionHeaderDimensionsFromBase64(t *testing.T) {
	cases := []struct {
		name    string
		mime    string
		header  []byte
		wantFmt ImageFormat
		wantW   int
		wantH   int
	}{
		{"png", "image/png", buildPNGHeader(1280, 720), ImageFormatPNG, 1280, 720},
		{"gif87a", "image/gif", buildGIFHeader("GIF87a", 320, 240), ImageFormatGIF, 320, 240},
		{"gif89a", "image/gif", buildGIFHeader("GIF89a", 640, 480), ImageFormatGIF, 640, 480},
		{"webp_vp8", "image/webp", buildWebPVP8Header(800, 600), ImageFormatWebP, 800, 600},
		{"webp_vp8l", "image/webp", buildWebPVP8LHeader(1024, 768), ImageFormatWebP, 1024, 768},
		{"webp_vp8x", "image/webp", buildWebPVP8XHeader(1920, 1080), ImageFormatWebP, 1920, 1080},
		{"jpeg_normal", "image/jpeg", buildJPEGHeader(1600, 900, 1), ImageFormatJPEG, 1600, 900},
		{"jpeg_exif_rot90", "image/jpeg", buildJPEGHeader(1600, 900, 6), ImageFormatJPEG, 900, 1600},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Append truncated garbage after header to prove no full-image decode occurs.
			withTail := append(append([]byte(nil), tc.header...), []byte("not-a-valid-pixel-stream")...)
			b64 := base64.StdEncoding.EncodeToString(withTail)
			// Inject whitespace to exercise streaming reader.
			dataURL := []byte("data:" + tc.mime + ";base64," + strings.ReplaceAll(b64, "A", "A\n"))

			got, err := ImageDimensionsFromDataURL(dataURL)
			if err != nil {
				t.Fatalf("ImageDimensionsFromDataURL: %v", err)
			}
			if got.Format != tc.wantFmt || got.Width != tc.wantW || got.Height != tc.wantH {
				t.Fatalf("got (%s, %dx%d), want (%s, %dx%d)", got.Format, got.Width, got.Height, tc.wantFmt, tc.wantW, tc.wantH)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestVisionTokenCountResizeRules(t *testing.T) {
	// Exact multiple of factor=28: 560x280 -> 20x10 = 200 tokens.
	g, err := VisionTokenCount(ImageDimensions{Width: 560, Height: 280}, DefaultQwenVLResizeRule)
	if err != nil {
		t.Fatalf("VisionTokenCount exact: %v", err)
	}
	if g.ResizedWidth != 560 || g.ResizedHeight != 280 || g.TokenCount != 200 {
		t.Fatalf("exact = %+v, want 560x280 (200 tokens)", g)
	}

	// Small image scaled up to minPixels (3136 = 56x56 = 2x2 tokens).
	gMin, err := VisionTokenCount(ImageDimensions{Width: 28, Height: 28}, DefaultQwenVLResizeRule)
	if err != nil {
		t.Fatalf("VisionTokenCount min: %v", err)
	}
	if gMin.ResizedWidth*gMin.ResizedHeight < DefaultQwenVLResizeRule.MinPixels || gMin.TokenCount != 4 {
		t.Fatalf("min = %+v, want >= minPixels and 4 tokens", gMin)
	}

	// Large image scaled down to <= maxPixels.
	gMax, err := VisionTokenCount(ImageDimensions{Width: 8000, Height: 4000}, DefaultQwenVLResizeRule)
	if err != nil {
		t.Fatalf("VisionTokenCount max: %v", err)
	}
	if gMax.ResizedWidth*gMax.ResizedHeight > DefaultQwenVLResizeRule.MaxPixels {
		t.Fatalf("max = %+v, exceeded MaxPixels %d", gMax, DefaultQwenVLResizeRule.MaxPixels)
	}

	// End-to-end from raw JSON data URL -> header dims -> exact token count.
	pngURL := []byte("data:image/png;base64," + base64.StdEncoding.EncodeToString(buildPNGHeader(560, 280)))
	e2e, err := CountInlineImageTokens(pngURL, DefaultQwenVLResizeRule)
	if err != nil || e2e.TokenCount != 200 {
		t.Fatalf("CountInlineImageTokens = %+v, %v; want 200 tokens", e2e, err)
	}
}
