package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestV41Q3KTokenPanelPackedLoads binds the sole shader transformation to a
// packed-byte/address witness. The scalar visit order and all floating-point
// expressions remain unchanged. This does not establish native device parity
// or that a driver preserves the source-level load reduction.
// fak-test:runtime fast est=20ms
func TestV41Q3KTokenPanelPackedLoads(t *testing.T) {
	source, err := os.ReadFile("shaders/v41_q3k_token_panel.comp")
	if err != nil {
		t.Fatal(err)
	}
	if err := v41Q3PackedLoadSource(string(source)); err != nil {
		t.Fatal(err)
	}
	// A wrong straddling shift must fail the source-bound contract, even when
	// aligned blocks would still decode correctly.
	broken := strings.Replace(string(source), "32u - shift", "8u", 1)
	if broken == string(source) || v41Q3PackedLoadSource(broken) == nil {
		t.Fatal("negative control accepted a corrupt cross-word byte extraction")
	}
	brokenActivation := strings.Replace(string(source), "sum1 + decoded2 * xv.z", "sum0 + decoded2 * xv.z", 1)
	if brokenActivation == string(source) || v41Q3PackedLoadSource(brokenActivation) == nil {
		t.Fatal("negative control accepted a missing prior scalar addition")
	}
	brokenMetadata := strings.Replace(string(source), "metadata3 >> metadataShift", "metadata2 >> metadataShift", 1)
	if brokenMetadata == string(source) || v41Q3PackedLoadSource(brokenMetadata) == nil {
		t.Fatal("negative control accepted the wrong half-scale word")
	}
	brokenScale := strings.Replace(string(source), "(index >> 2u) * 2u", "(index >> 2u)", 1)
	if brokenScale == string(source) || v41Q3PackedLoadSource(brokenScale) == nil {
		t.Fatal("negative control accepted a corrupt scale high-bit selector")
	}
	// Vary each of the twelve encoded bytes through every value, against
	// zero and all-one backgrounds. The old word-array decoder is retained
	// independently as the oracle; every decoded value is in [0, 63].
	decodeScale := func(low0, low1, high uint32, index uint) int {
		byteShift := (index & 3) * 8
		low := low0
		if index&4 != 0 {
			low = low1
		}
		lowShift := byteShift
		if index >= 8 {
			lowShift += 4
		}
		highShift := byteShift + (index>>2)*2
		return int((low>>lowShift)&15 | ((high>>highShift)&3)<<4)
	}
	legacyScales := func(a, b, high uint32) [16]int {
		words := [4]uint32{
			(a & 0x0f0f0f0f) | ((high & 0x03030303) << 4),
			(b & 0x0f0f0f0f) | (((high >> 2) & 0x03030303) << 4),
			((a >> 4) & 0x0f0f0f0f) | (((high >> 4) & 0x03030303) << 4),
			((b >> 4) & 0x0f0f0f0f) | (((high >> 6) & 0x03030303) << 4),
		}
		var scales [16]int
		for i, word := range words {
			for j := 0; j < 4; j++ {
				scales[i*4+j] = int((word >> uint(8*j)) & 255)
			}
		}
		return scales
	}
	for _, background := range []byte{0, 255} {
		for variedByte := 0; variedByte < 12; variedByte++ {
			for value := 0; value < 256; value++ {
				var encoded [12]byte
				for i := range encoded {
					encoded[i] = background
				}
				encoded[variedByte] = byte(value)
				a, b, high := binary.LittleEndian.Uint32(encoded[:4]), binary.LittleEndian.Uint32(encoded[4:8]), binary.LittleEndian.Uint32(encoded[8:])
				want := legacyScales(a, b, high)
				for i := uint(0); i < 16; i++ {
					if got := decodeScale(a, b, high, i); got != want[i] || got < 0 || got > 63 {
						t.Fatalf("scale index=%d byte=%d value=%d background=%d got=%d want=%d", i, variedByte, value, background, got, want[i])
					}
				}
			}
		}
	}
	// Four scalar visits and one vector visit cover the same contiguous input
	// bytes. Keep every rounded scalar product and running addition in order.
	for _, tokens := range []int{2, 4, 8} {
		const width = 3 * 256
		input := make([]float32, tokens*width)
		weights := make([]float32, width)
		for i := range input {
			input[i] = float32((i*37)%251-125) / 31
		}
		for i := range weights {
			weights[i] = float32((i*13)%127-63) / 19
		}
		scalar, quartet := make([]float32, tokens), make([]float32, tokens)
		for col := 0; col < width; col++ {
			for token := 0; token < tokens; token++ {
				product := float32(weights[col] * input[token*width+col])
				scalar[token] = float32(scalar[token] + product)
			}
		}
		loads, stores := 0, 0
		for col := 0; col < width; col += 4 {
			for token := 0; token < tokens; token++ {
				vectorIndex := token*(width>>2) + (col >> 2)
				start := vectorIndex * 4
				if start != token*width+col || start+4 > len(input) {
					t.Fatal("activation quartet changed token stride or crossed storage")
				}
				xv := input[start : start+4]
				loads++
				sum0 := float32(quartet[token] + float32(weights[col]*xv[0]))
				sum1 := float32(sum0 + float32(weights[col+1]*xv[1]))
				sum2 := float32(sum1 + float32(weights[col+2]*xv[2]))
				sum3 := float32(sum2 + float32(weights[col+3]*xv[3]))
				quartet[token] = sum3
				stores++
			}
		}
		for token := 0; token < tokens; token++ {
			if quartet[token] != scalar[token] {
				t.Fatalf("tokens=%d token=%d quartet=%g scalar=%g", tokens, token, quartet[token], scalar[token])
			}
		}
		if loads != tokens*width/4 || stores != tokens*width/4 {
			t.Fatalf("tokens=%d loads=%d stores=%d", tokens, loads, stores)
		}
	}
	const blocks = 3 // Both alignments and an odd, minimally padded tail.
	raw := make([]byte, blocks*110+2)
	for seed := 0; seed < 256; seed++ {
		for i := range raw {
			raw[i] = byte(i*73 + i/7 + seed)
		}
		for sb := 0; sb < blocks; sb++ {
			base, visits, loads := sb*110, 0, 0
			packed := func(offset int) uint32 {
				word, shift := offset/4, uint(offset&3)*8
				read := func(index int) uint32 {
					if index < 0 || index*4+4 > len(raw) {
						t.Fatalf("raw load out of bounds: %d", index)
					}
					loads++
					return binary.LittleEndian.Uint32(raw[index*4:])
				}
				lo := read(word)
				if shift == 0 {
					return lo
				}
				return lo>>shift | read(word+1)<<(32-shift)
			}
			word, metadataShift := (base+96)/4, uint(base&3)*8
			var metadata [4]uint32
			for i := range metadata {
				if (word+i)*4+4 > len(raw) {
					t.Fatal("metadata word load crossed the padded extent")
				}
				metadata[i] = binary.LittleEndian.Uint32(raw[(word+i)*4:])
				loads++
			}
			low0, low1, high := metadata[0], metadata[1], metadata[2]
			if metadataShift != 0 {
				low0 = metadata[0]>>metadataShift | metadata[1]<<(32-metadataShift)
				low1 = metadata[1]>>metadataShift | metadata[2]<<(32-metadataShift)
				high = metadata[2]>>metadataShift | metadata[3]<<(32-metadataShift)
			}
			if loads != 4 || low0 != binary.LittleEndian.Uint32(raw[base+96:]) ||
				low1 != binary.LittleEndian.Uint32(raw[base+100:]) || high != binary.LittleEndian.Uint32(raw[base+104:]) ||
				uint16(metadata[3]>>metadataShift) != binary.LittleEndian.Uint16(raw[base+108:]) {
				t.Fatalf("block=%d metadata extraction changed or used %d word loads", sb, loads)
			}
			wantScales := legacyScales(binary.LittleEndian.Uint32(raw[base+96:]), binary.LittleEndian.Uint32(raw[base+100:]), binary.LittleEndian.Uint32(raw[base+104:]))
			for i := uint(0); i < 16; i++ {
				if got := decodeScale(low0, low1, high, i); got != wantScales[i] {
					t.Fatalf("seed=%d block=%d scale=%d got=%d want=%d", seed, sb, i, got, wantScales[i])
				}
			}
			loads = 0
			qi, mask, scale := 0, uint32(1), 0
			for half := 0; half < 256; half += 128 {
				for j := 0; j < 4; j++ {
					for sub := 0; sub < 2; sub++ {
						for l := 0; l < 16; l += 4 {
							laneBase := sub*16 + l
							quant, high := packed(base+32+qi+laneBase), packed(base+laneBase)
							for component := 0; component < 4; component++ {
								lane := laneBase + component
								code := int((quant >> uint(component*8+j*2)) & 3)
								if (high>>uint(component*8))&mask == 0 {
									code -= 4
								}
								col := half + j*32 + lane
								// Independent direct-byte addressing derives its
								// role from the next scalar column, not packed loops.
								wantHalf, wantJ, wantLane := visits/128, (visits%128)/32, visits%32
								want := int((raw[base+32+wantHalf*32+wantLane] >> uint(wantJ*2)) & 3)
								if raw[base+wantLane]&byte(1<<uint(wantHalf*4+wantJ)) == 0 {
									want -= 4
								}
								if col != visits || scale != visits/16 || code != want {
									t.Fatalf("seed=%d block=%d visit=%d col=%d scale=%d code=%d want=%d", seed, sb, visits, col, scale, code, want)
								}
								visits++
							}
						}
						scale++
					}
					mask <<= 1
				}
				qi += 32
			}
			wantLoads := 128
			if base&3 != 0 {
				wantLoads = 256
			}
			if visits != 256 || loads != wantLoads {
				t.Fatalf("block=%d visits=%d packed loads=%d want=%d (scalar loads=512)", sb, visits, loads, wantLoads)
			}
		}
	}
}

func v41Q3PackedLoadSource(source string) error {
	normalize := func(s string) string {
		comments := regexp.MustCompile(`(?s)//[^\n]*|/\*.*?\*/`)
		return strings.Join(strings.Fields(comments.ReplaceAllString(s, "")), "")
	}
	source = normalize(source)
	metadataAfter, metadataBefore := normalize(v41Q3MetadataWordsAfter), normalize(v41Q3MetadataWordsBefore)
	if strings.Count(source, metadataAfter) != 1 {
		return fmt.Errorf("metadata word sharing or half-scale extraction changed")
	}
	source = strings.Replace(source, metadataAfter, metadataBefore, 1)
	activationHelper := normalize(v41Q3ActivationQuartetHelper)
	if strings.Count(source, activationHelper) != 1 {
		return fmt.Errorf("quartet weight decoder changed")
	}
	source = strings.Replace(source, activationHelper, "", 1)
	for _, replacement := range [][2]string{
		{"readonly buffer X { vec4 x4[]; };", "readonly buffer X { float x[]; };"},
		{v41Q3ActivationQuartetAfter, v41Q3PackedLoadAfter},
	} {
		after, before := normalize(replacement[0]), normalize(replacement[1])
		if strings.Count(source, after) != 1 {
			return fmt.Errorf("aligned activation quartet or ordered accumulation changed")
		}
		source = strings.Replace(source, after, before, 1)
	}
	for _, replacement := range [][2]string{
		{v41Q3DirectScaleAfter, v41Q3DirectScaleBefore},
		{v41Q3DirectScalePreludeAfter, v41Q3DirectScalePreludeBefore},
		{"float dl = d * float(unpackScale(scaleLow0, scaleLow1, scaleHigh, uint(si)) - 32);", "float dl = d * float(scales[si] - 32);"},
	} {
		after, before := normalize(replacement[0]), normalize(replacement[1])
		if strings.Count(source, after) != 1 {
			return fmt.Errorf("direct packed-scale decoder or ordered scale use changed")
		}
		source = strings.Replace(source, after, before, 1)
	}
	helper, before, after := normalize(v41Q3PackedLoadHelper), normalize(v41Q3PackedLoadBefore), normalize(v41Q3PackedLoadAfter)
	if strings.Count(source, helper) != 1 || strings.Count(source, after) != 1 {
		return fmt.Errorf("packed helper or exact scalar-ordered quartet loop changed")
	}
	restored := strings.Replace(source, helper, "", 1)
	restored = strings.Replace(restored, after, before, 1)
	// Reversing just this transformation must recover the complete prior
	// active source: decoder, geometry, ABI, guards and publication included.
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(restored))); got != "d4b48808d7b613a45933c84c867c37b5f5d4d130c6368ae93c699bd494cdc86d" {
		return fmt.Errorf("unrelated Q3 shader contract changed: %s", got)
	}
	return nil
}

const v41Q3PackedLoadHelper = `// Four consecutive bytes share their raw word loads. Every caller uses a
// four-byte lane group inside the block's quant or high-mask array. Q3 blocks
// are 110 bytes, so their starting alignment alternates between zero and two;
// the straddling read still stays inside those arrays' enclosing raw words.
uint packedFourBytes(uint offset) {
    uint word = offset >> 2u;
    uint shift = (offset & 3u) * 8u;
    uint lo = raw[word];
    if (shift == 0u) return lo;
    return (lo >> shift) | (raw[word + 1u] << (32u - shift));
}

`

const v41Q3PackedLoadBefore = `                    for (uint l = 0u; l < 16u; ++l) {
                        uint lane = sub * 16u + l;
                        int code = int((byteAt(base + 32u + qi + lane) >> shift) & 3u);
                        if ((byteAt(base + lane) & mask) == 0u) code -= 4;
                        float decoded = dl * float(code);
                        uint col = sb * 256u + halfBase + j * 32u + lane;
                        for (uint token = 0u; token < tokens; ++token) {
                            sums[token] += decoded * x[token * inDim + col];
                        }
                    }`

const v41Q3PackedLoadAfter = `                    for (uint l = 0u; l < 16u; l += 4u) {
                        uint laneBase = sub * 16u + l;
                        uint quantBytes = packedFourBytes(base + 32u + qi + laneBase);
                        uint highBytes = packedFourBytes(base + laneBase);
                        for (uint component = 0u; component < 4u; ++component) {
                            uint lane = laneBase + component;
                            uint byteShift = component * 8u;
                            int code = int((quantBytes >> (byteShift + shift)) & 3u);
                            if (((highBytes >> byteShift) & mask) == 0u) code -= 4;
                            float decoded = dl * float(code);
                            uint col = sb * 256u + halfBase + j * 32u + lane;
                            for (uint token = 0u; token < tokens; ++token) {
                                sums[token] += decoded * x[token * inDim + col];
                            }
                        }
                    }`

const v41Q3DirectScaleBefore = `void unpackScales(uint sc, out int scales[16]) {
    uint aux0 = leU32(sc), aux1 = leU32(sc + 4u), tmp = leU32(sc + 8u);
    uint words[4];
    words[0] = (aux0 & 0x0f0f0f0fu) | ((tmp & 0x03030303u) << 4u);
    words[1] = (aux1 & 0x0f0f0f0fu) | (((tmp >> 2u) & 0x03030303u) << 4u);
    words[2] = ((aux0 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 4u) & 0x03030303u) << 4u);
    words[3] = ((aux1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint b = (words[i] >> uint(8 * j)) & 255u;
            scales[i * 4 + j] = b >= 128u ? int(b) - 256 : int(b);
        }
    }
}
`

const v41Q3DirectScaleAfter = `// Decode the next six-bit scale directly from the three packed words.
// The scalar consumer visits indices 0..15 once per superblock; keeping these
// words avoids the sixteen-element unpack/copy arrays without changing order.
int unpackScale(uint low0, uint low1, uint high, uint index) {
    uint byteShift = (index & 3u) * 8u;
    uint low = (index & 4u) == 0u ? low0 : low1;
    uint lowShift = byteShift + (index >= 8u ? 4u : 0u);
    uint highShift = byteShift + (index >> 2u) * 2u;
    return int(((low >> lowShift) & 15u) | (((high >> highShift) & 3u) << 4u));
}
`

const v41Q3DirectScalePreludeBefore = `        int scales[16];
        unpackScales(base + 96u, scales);`

const v41Q3DirectScalePreludeAfter = `        uint scaleLow0 = packedFourBytes(base + 96u);
        uint scaleLow1 = packedFourBytes(base + 100u);
        uint scaleHigh = packedFourBytes(base + 104u);`

const v41Q3ActivationQuartetHelper = `// Decode all four scalar weights before visiting tokens. Each token then
// consumes one aligned activation vec4 while retaining four ordered additions.
float decodedLane(uint quantBytes, uint highBytes, uint shift, uint mask, uint component, float dl) {
    uint byteShift = component * 8u;
    int code = int((quantBytes >> (byteShift + shift)) & 3u);
    if (((highBytes >> byteShift) & mask) == 0u) code -= 4;
    float decoded = dl * float(code);
    return decoded;
}

`

const v41Q3ActivationQuartetAfter = `                    for (uint l = 0u; l < 16u; l += 4u) {
                        uint laneBase = sub * 16u + l;
                        uint quantBytes = packedFourBytes(base + 32u + qi + laneBase);
                        uint highBytes = packedFourBytes(base + laneBase);
                        float decoded0 = decodedLane(quantBytes, highBytes, shift, mask, 0u, dl);
                        float decoded1 = decodedLane(quantBytes, highBytes, shift, mask, 1u, dl);
                        float decoded2 = decodedLane(quantBytes, highBytes, shift, mask, 2u, dl);
                        float decoded3 = decodedLane(quantBytes, highBytes, shift, mask, 3u, dl);
                        uint col4 = (sb * 256u + halfBase + j * 32u + laneBase) >> 2u;
                        for (uint token = 0u; token < tokens; ++token) {
                            vec4 xv = x4[token * (inDim >> 2u) + col4];
                            float sum0 = sums[token] + decoded0 * xv.x;
                            float sum1 = sum0 + decoded1 * xv.y;
                            float sum2 = sum1 + decoded2 * xv.z;
                            float sum3 = sum2 + decoded3 * xv.w;
                            sums[token] = sum3;
                        }
                    }`

const v41Q3MetadataWordsBefore = `        float d = unpackHalf2x16(byteAt(base + 108u) | (byteAt(base + 109u) << 8u)).x;
        uint scaleLow0 = packedFourBytes(base + 96u);
        uint scaleLow1 = packedFourBytes(base + 100u);
        uint scaleHigh = packedFourBytes(base + 104u);`

const v41Q3MetadataWordsAfter = `        // Four enclosing words cover all twelve scale bytes and the half scale.
        // Blocks start at byte alignment zero or two; no fifth word is needed.
        uint metadataWord = (base + 96u) >> 2u;
        uint metadataShift = (base & 3u) * 8u;
        uint metadata0 = raw[metadataWord];
        uint metadata1 = raw[metadataWord + 1u];
        uint metadata2 = raw[metadataWord + 2u];
        uint metadata3 = raw[metadataWord + 3u];
        uint scaleLow0 = metadata0, scaleLow1 = metadata1, scaleHigh = metadata2;
        if (metadataShift != 0u) {
            scaleLow0 = (metadata0 >> metadataShift) | (metadata1 << (32u - metadataShift));
            scaleLow1 = (metadata1 >> metadataShift) | (metadata2 << (32u - metadataShift));
            scaleHigh = (metadata2 >> metadataShift) | (metadata3 << (32u - metadataShift));
        }
        float d = unpackHalf2x16((metadata3 >> metadataShift) & 65535u).x;`
