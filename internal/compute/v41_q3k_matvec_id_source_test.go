package compute

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Source witness only: the Q3_K routed-id GEMV selects its ring slot once per
// workgroup and each per-slot row body keeps q3k_matmul.comp's arithmetic and
// byte addressing term for term. A per-load slot switch measured 19.2 ms/call
// against 52 us for per-pick q3k_matmul on strix3. No compiler or GPU proof.
func TestV41Q3KMatVecIDSlotHoistSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(string(raw), "\r\n", "\n")
	}
	id := read("shaders/v41_q3k_matvec_id.comp")
	ref := read("shaders/q3k_matmul.comp")

	for _, forbidden := range []string{"\nuint slot;", "wload(", "switch (slot)"} {
		body := id[:strings.Index(id, "void main()")]
		if strings.Contains(body, forbidden) {
			t.Errorf("row body selects the slot per load: found %q before main", forbidden)
		}
	}
	main := id[strings.Index(id, "void main()"):]
	if strings.Count(main, "switch (slot)") != 1 || !strings.Contains(main, "uint slot = ids[pick];") {
		t.Error("main must read the pick's slot once and branch on it once")
	}
	for s := 0; s < 8; s++ {
		stamp := fmt.Sprintf("Q3K_SLOT(q3kRow%d, byteAt%d, raw%d)", s, s, s)
		if !strings.Contains(id, stamp) {
			t.Errorf("missing per-slot row body %q", stamp)
		}
		call := fmt.Sprintf("sum = q3kRow%d(row, blocks, xPick); break;", s)
		if !strings.Contains(main, call) {
			t.Errorf("main does not dispatch slot %d to its row body: missing %q", s, call)
		}
	}
	if !strings.Contains(id, "return (RAW[offset >> 2u] >> ((offset & 3u) * 8u)) & 255u;") {
		t.Error("per-slot byte load no longer matches q3k_matmul's byteAt")
	}
	for _, term := range []string{
		"uint base = (row * blocks + sb) * 110u;",
		"uint qBase  = base + 32u;",
		"uint scBase = base + 96u;",
		"float dl0 = d * float(scales[si] - 32); si++;",
		"if ((BYTE(hmBase + l) & m) == 0u) code -= 4;",
		"int code = int((BYTE(qBase + qi + 16u + l) >> shift) & 3u);",
		"sum += dl1 * float(code) * x[xBlock + n + j * 32u + 16u + l];",
		"words[3] = ((aux1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u);",
		"scales[i * 4 + j] = (b >= 128u) ? int(b) - 256 : int(b);",
	} {
		refTerm := strings.ReplaceAll(term, "BYTE(", "byteAt(")
		if !strings.Contains(id, term) || !strings.Contains(ref, refTerm) {
			t.Errorf("id row body and q3k_matmul.comp disagree on %q", term)
		}
	}
	if !strings.Contains(read("vulkan_shim.cpp"), "(uint64_t(out) + 1u) / 2u : (uint64_t(out) + 63u) / 64u;") {
		t.Error("Q3_K id grid is no longer one thread per output row")
	}
}
