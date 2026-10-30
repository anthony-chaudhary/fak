package strix

import (
	"regexp"
	"strings"
	"testing"
)

// TestGFX1151ISA_Guard_NoPhantomWMMAOpcodes pins the gfx11 ISA floor (#13542).
//
// The complete gfx11 (RDNA 3 / 3.5) WMMA instruction set is:
//
//	v_wmma_{f32,f16,bf16}_16x16x16_{f16,bf16}
//	v_wmma_i32_16x16x16_{iu8,iu4}
//
// No FP4 form, no FP8 form, and no 16x16x32 tile exists on gfx11
// (v_wmma_i32_16x16x32_iu4 is gfx12/RDNA4 only). This guard scans the CODE
// lines (comments stripped) of both Strix Halo assembly surfaces — the embedded
// device source and the emitted on-the-fly-dequant kernel — and refuses any
// WMMA mnemonic outside that ISA. It additionally anchors the positive case:
// both surfaces must emit the real f16 WMMA that the FP4 path feeds after
// in-register E2M1 dequant.
//
// Red on the parent tree: asm_gfx1151.s carried the nonexistent
// v_wmma_f32_16x16x16_fp4 mnemonic and the emitted kernel did not carry the
// f16 anchor. Green once the FP4 path dequantizes E2M1 in VGPRs and issues
// v_wmma_f32_16x16x16_f16.
func TestGFX1151ISA_Guard_NoPhantomWMMAOpcodes(t *testing.T) {
	phantom := []*regexp.Regexp{
		regexp.MustCompile(`^v_wmma_\w*_fp4$`),          // no FP4 WMMA on gfx11
		regexp.MustCompile(`^v_wmma_\w*fp8\w*$`),        // no FP8 WMMA on gfx11
		regexp.MustCompile(`^v_wmma_\w*_16x16x32_\w*$`), // no 16x16x32 tile on gfx11
	}
	sources := map[string]string{
		"embedded GFX1151 assembly":  GFX1151AssemblySource,
		"emitted FP4 dequant kernel": EmitROCmFP4WMMASource(),
	}
	const anchor = "v_wmma_f32_16x16x16_f16"
	for name, src := range sources {
		code := stripAsmComments(src)
		for _, line := range strings.Split(code, "\n") {
			fields := strings.Fields(line)
			if len(fields) == 0 {
				continue
			}
			op := fields[0]
			if !strings.HasPrefix(op, "v_wmma_") {
				continue
			}
			for _, re := range phantom {
				if re.MatchString(op) {
					t.Errorf("%s: phantom WMMA opcode %q is not in the gfx11 ISA", name, op)
				}
			}
		}
		if !lineEmitsOpcode(code, anchor) {
			t.Errorf("%s: missing positive anchor %s (the FP4 path must feed the f16 WMMA)", name, anchor)
		}
	}
}
