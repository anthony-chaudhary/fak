package compute

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Source-only witness: literal bindings replace per-load switches without
// changing either quantized reduction order. Paired Q2 gate/up work also reuses
// the unchanged activation prelude. Compiler/device parity and fused
// performance still require the existing Vulkan witnesses on physical silicon.
func TestV41RoutedFusedSlotHoistSourceContract(t *testing.T) {
	t.Parallel()
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile("shaders/" + name + ".comp")
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(b), "\r\n", "\n")
	}
	section := func(s, start, end string) string {
		t.Helper()
		i := strings.Index(s, start)
		if i < 0 {
			t.Fatalf("missing section start %q", start)
		}
		s = s[i:]
		j := strings.Index(s, end)
		if j < 0 {
			t.Fatalf("missing section end %q", end)
		}
		return s[:j]
	}
	canonical := func(s string) string {
		return strings.Join(strings.Fields(strings.ReplaceAll(s, "\\\n", "\n")), " ")
	}
	down := read("v41_routed_down_sum")
	gate := read("v41_routed_gate_up_swiglu_id")
	q3 := read("v41_q3k_matvec_id")
	q2 := read("v41_q2k_matvec_id")
	for name, source := range map[string]string{"down": down, "gate/up": gate} {
		body := section(source, "#version", "void main()")
		for _, forbidden := range []string{"switch (slot)", "wload(", "gload(", "uload(", "\nuint slot;"} {
			if strings.Contains(body, forbidden) {
				t.Errorf("%s selects slot inside a load: %q", name, forbidden)
			}
		}
		if strings.Count(source, "switch (slot)") != 1 {
			t.Errorf("%s must have exactly one outer slot switch", name)
		}
	}
	// The down kernel hoists packed words; its independent packed-layout witness
	// binds that integer decoder. Keep the complete floating-point loop nest
	// identical to routed-id, not merely a few selected accumulation terms.
	refRow := section(q3, "        uint qi = 0u;", "Q3K_SLOT(q3kRow0")
	refRow = strings.NewReplacer("xPick", "midBase", " * x[", " * mid[").Replace(refRow)
	if canonical(section(down, "        uint qi = 0u;", "Q3K_SLOT(q3kRow0")) != canonical(refRow) {
		t.Error("down row no longer matches routed-id Q3_K floating-point loop order")
	}
	if canonical(section(down, "void q3kScales", "// Q3K_SLOT")) != canonical(section(q3, "void q3kScales", "// Q3K_SLOT")) {
		t.Error("down scale decoder differs from routed-id")
	}
	// Compare the complete row body to the unchanged routed-id bank. Only the
	// outer loop is paired: each accumulator keeps the same sb/j order.
	refBank := canonical(section(q2, "#define Q2K_BANK(RAW, ACC)", "void main()"))
	rowStart := strings.Index(refBank, "for (uint r = 0u; r < ROWS; ++r) {")
	if rowStart < 0 || !strings.HasSuffix(refBank, " }") {
		t.Fatal("routed-id Q2_K bank no longer has the expected row/outer-loop shape")
	}
	refRowQ2 := strings.TrimSuffix(refBank[rowStart:], " }")
	row := canonical(section(gate, "#define Q2K_ROW(RAW, ACC)", "#define Q2K_PAIR(GRAW, URAW)"))
	if row != "#define Q2K_ROW(RAW, ACC) "+refRowQ2 {
		t.Error("gate/up row differs from routed-id Q2_K arithmetic or addressing")
	}
	activation := canonical(section(q2, "        vec4 xv[4];", "        for (uint r = 0u; r < ROWS; ++r) {"))
	superblockLoop := "for (uint sb = tid >> 4u; sb < nsb; sb += LANES / 16u) { "
	if refBank != "#define Q2K_BANK(RAW, ACC) for (uint r = 0u; r < ROWS; ++r) ACC[r] = 0.0; "+superblockLoop+activation+" "+refRowQ2+" }" {
		t.Fatal("routed-id Q2_K bank has unchecked initialization or outer-loop work")
	}
	// Pin the entire paired wrapper, including both zero initializations and
	// exactly one original activation load/sum followed by G then U row work.
	paired := "#define Q2K_PAIR(GRAW, URAW) " +
		"for (uint r = 0u; r < ROWS; ++r) accG[r] = 0.0; " +
		"for (uint r = 0u; r < ROWS; ++r) accU[r] = 0.0; " +
		superblockLoop + activation + " Q2K_ROW(GRAW, accG) Q2K_ROW(URAW, accU) }"
	if canonical(section(gate, "#define Q2K_PAIR(GRAW, URAW)", "void main()")) != paired {
		t.Error("gate/up must share one activation prelude without changing accumulator order")
	}

	for slot := 0; slot < 8; slot++ {
		label := fmt.Sprintf("case %du", slot)
		if slot == 7 {
			label = "default"
		}
		for _, pair := range [][2]string{
			{down, fmt.Sprintf("Q3K_SLOT(q3kRow%d, byteAt%d, raw%d)", slot, slot, slot)},
			{down, fmt.Sprintf("%s: d = q3kRow%d(row, blocks, midBase); break;", label, slot)},
			{gate, fmt.Sprintf("%s: { Q2K_PAIR(g%d, u%d) break; }", label, slot, slot)},
		} {
			if !strings.Contains(pair[0], pair[1]) {
				t.Errorf("missing correct slot mapping %q", pair[1])
			}
		}
	}
	for _, term := range []string{
		"precise float acc = 0.0;",
		"for (uint j = 0u; j < picks; ++j) {\n        uint p = ids[2u * picks + j];\n        uint slot = ids[p];\n        uint midBase = p * uint(pc.midDim);\n        float d;\n        switch (slot)",
		"uint bits = floatBitsToUint(d);\n        uint rounded = roundBF16(bits);",
		"atomicMin(status[4u * p + 2u], row);",
		"acc = acc + uintBitsToFloat(rounded);",
	} {
		if !strings.Contains(down, term) {
			t.Errorf("down pick order/publication changed: missing %q", term)
		}
	}
	if !strings.Contains(gate, "uint pick = gl_WorkGroupID.y;\n    uint slot = ids[pick];") {
		t.Error("gate/up slot must remain workgroup-uniform")
	}
	// Every reduction barrier remains after the complete slot switch.
	dispatch := section(gate, "    switch (slot)", "    for (uint r = 0u; r < ROWS; ++r) {\n        red[")
	if strings.Contains(dispatch, "barrier()") {
		t.Error("barrier moved inside slot switch")
	}
	// Freeze the complete unchanged gate epilogue, including shared-memory
	// destinations, both barriers, reduction order, status records and every
	// publication/fault branch. Checking a few terms would miss an extra add,
	// a removed barrier or a changed status field elsewhere in this suffix.
	gateSuffix := `for (uint r = 0u; r < ROWS; ++r) {
        red[r * LANES + tid] = accG[r];
        red[(ROWS + r) * LANES + tid] = accU[r];
    }
    barrier();
    for (uint stride = LANES / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (uint r = 0u; r < ROWS; ++r) {
                red[r * LANES + tid] += red[r * LANES + tid + stride];
                red[(ROWS + r) * LANES + tid] += red[(ROWS + r) * LANES + tid + stride];
            }
        }
        barrier();
    }

    uint rec = 4u * pick;
    if (gl_GlobalInvocationID.x == 0u) {
        status[rec] = 0x52535731u;
        status[rec + 1u] = outDim;
    }
    if (tid >= ROWS) return;
    uint row = rowBase + tid;
    if (row >= outDim) return;
    float weight = uintBitsToFloat(ids[pc.picks + int(pick)]);
    uint gb = floatBitsToUint(red[tid * LANES]);
    uint ub = floatBitsToUint(red[(ROWS + tid) * LANES]);
    uint gr = roundBF16(gb);
    uint ur = roundBF16(ub);
    uint base = pick * outDim + row;
    if (nonfinite(gb) || nonfinite(ub) || nonfinite(gr) || nonfinite(ur)) {
        atomicMin(status[rec + 2u], row);
        mid[base] = uintBitsToFloat(0x7fc00000u);
        return;
    }
    precise float g = uintBitsToFloat(gr);
    precise float u = uintBitsToFloat(ur);
    if (pc.limit > 0.0) {
        g = min(g, pc.limit);
        u = max(-pc.limit, min(u, pc.limit));
    }
    precise float t = 1.0 + exp(-g);
    precise float f = (g / t) * u;
    uint fb = floatBitsToUint(f);
    if (nonfinite(fb)) {
        atomicMin(status[rec + 2u], row);
        mid[base] = uintBitsToFloat(0x7fc00000u);
        return;
    }
    precise float w = f * weight;
    uint wb = floatBitsToUint(w);
    uint wr = roundBF16(wb);
    if (nonfinite(wb) || nonfinite(wr)) {
        atomicMin(status[rec + 3u], row);
        wr = 0x7fc00000u;
    }
    mid[base] = uintBitsToFloat(wr);
}`
	start := strings.Index(gate, "    for (uint r = 0u; r < ROWS; ++r) {\n        red[")
	if start < 0 || canonical(gate[start:]) != canonical(gateSuffix) {
		t.Error("gate/up reduction, barriers or BF16/status publication suffix changed")
	}
	// Freeze both rounding helpers and the gate nonfinite classifier. The
	// down helper is outside the row macro and otherwise escapes comparison.
	bf16 := "uint roundBF16(uint bits) { return (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u; }"
	if canonical(section(down, "uint roundBF16", "void main()")) != canonical(bf16) {
		t.Error("down BF16 round-to-nearest-even helper changed")
	}
	gateHelpers := "const uint kExponent = 0x7f800000u;\n" + bf16 + "\nbool nonfinite(uint bits) { return (bits & kExponent) == kExponent; }"
	if canonical(section(gate, "const uint kExponent", "vec4 bytesOf")) != canonical(gateHelpers) {
		t.Error("gate BF16 rounding or nonfinite classification changed")
	}
}
