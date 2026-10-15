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
