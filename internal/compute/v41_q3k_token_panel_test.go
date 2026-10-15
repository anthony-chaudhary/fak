package compute

import (
	"bytes"
	"encoding/binary"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// fak-test:runtime fast est=10ms
func TestV41Q3KTokenPanelOptIn(t *testing.T) {
	for _, q2 := range []string{"", "0", "1", "true", " 1", "1 "} {
		for _, q3 := range []string{"", "0", "1", "true", " 1", "1 "} {
			t.Setenv("FAK_V41_Q2K_TOKEN_PANEL", q2)
			t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", q3)
			if got, want := v41Q3KTokenPanelOptedIn(), q2 == "1" || q3 == "1"; got != want {
				t.Fatalf("Q2=%q Q3=%q: got %v want %v", q2, q3, got, want)
			}
		}
	}
}

// fak-test:runtime fast est=10ms
// Bind the real tagged native guards without claiming Vulkan execution.
func TestV41Q3KTokenPanelOptInSource(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "vulkan_v41_q3k_token_panel.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	text := func(n ast.Node) string {
		var b bytes.Buffer
		if err := format.Node(&b, fset, n); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	guards := map[string]struct {
		index     int
		condition string
	}{
		"SupportsV41Q3KTokenPanel": {0, "v == nil || !v41Q3KTokenPanelOptedIn() || weight.Dtype != Q3_K || weight.Layout != RowMajor || len(weight.Shape) != 2"},
		"V41Q3KTokenPanel":         {3, "v == nil || !v41Q3KTokenPanelOptedIn()"},
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		want, ok := guards[fn.Name.Name]
		if !ok {
			continue
		}
		if fn.Body == nil || len(fn.Body.List) <= want.index+1 {
			t.Fatalf("missing guard in %s", fn.Name.Name)
		}
		guard, ok := fn.Body.List[want.index].(*ast.IfStmt)
		if !ok || text(guard.Cond) != want.condition || len(guard.Body.List) != 1 {
			t.Fatalf("%s lost its early shared policy guard", fn.Name.Name)
		}
		if _, ok := guard.Body.List[0].(*ast.ReturnStmt); !ok {
			t.Fatalf("%s policy guard no longer rejects", fn.Name.Name)
		}
		if fn.Name.Name == "V41Q3KTokenPanel" &&
			(text(fn.Body.List[1]) != "plan, invalid := validateV41Q3KTokenPanel(weight, input, rows)" || text(fn.Body.List[4]) != "vulkanMu.Lock()") {
			t.Fatal("selected validation/policy/lock order changed")
		}
		delete(guards, fn.Name.Name)
	}
	if len(guards) != 0 {
		t.Fatal("missing native Q3 methods")
	}
	q2, err := os.ReadFile("vulkan_v41_q2k_token_panel.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(q2), "FAK_V41_Q3K_TOKEN_PANEL") || strings.Contains(string(q2), "v41Q3KTokenPanelOptedIn") || strings.Count(string(q2), `os.Getenv("FAK_V41_Q2K_TOKEN_PANEL") != "1"`) != 2 {
		t.Fatal("Q2 must retain its exclusive legacy policy")
	}
}

// fak-test:runtime fast est=50ms
// CPU indexing, scalar-order and decode-sharing witness only. Native shader
// compilation, device parity, occupancy and bandwidth remain unqualified.
func TestV41Q3KTokenPanelContract(t *testing.T) {
	const out, in = 65, 768 // A partial workgroup and an odd number of Q3 blocks.
	raw := make([]byte, out*(in/256)*110)
	for i := range raw {
		raw[i] = byte(i*73 + i/7 + 19)
	}
	for off := 0; off < len(raw); off += 110 {
		binary.LittleEndian.PutUint16(raw[off+108:], 0x1800)
	}
	for _, rows := range []int{2, 4, 8} {
		plan, err := PlanV41Q3KTokenPanel(out, in, rows)
		if err != nil || plan.PayloadBytes != len(raw) || plan.WeightBytes != len(raw)+2 || plan.InputBytes != rows*in*4 || plan.OutputBytes != rows*out*4 || plan.WorkgroupsX != 2 {
			t.Fatalf("rows=%d plan=%+v err=%v", rows, plan, err)
		}
		x := make([]float32, rows*in)
		for token := 0; token < rows; token++ {
			for col := 0; col < in; col++ {
				x[token*in+col] = float32((col*3+token*11)%31-15) / 32
			}
		}
		got, decodes := v41Q3KTokenPanelScalar(raw, x, out, in, rows)
		if decodes != out*in {
			t.Fatalf("rows=%d decodes=%d, want token-independent %d", rows, decodes, out*in)
		}
		for token := 0; token < rows; token++ {
			for row := 0; row < out; row++ {
				weight := raw[row*(in/256)*110 : (row+1)*(in/256)*110]
				// Existing independent packed decoder and scalar dot retain the
				// same per-output accumulation order; no token shares an accumulator.
				want := q3kRowDot(weight, x[token*in:(token+1)*in], make([]float32, 256))
				if math.Float32bits(got[token*out+row]) != math.Float32bits(want) {
					t.Fatalf("rows=%d token=%d output=%d got=%g want=%g", rows, token, row, got[token*out+row], want)
				}
			}
		}
	}
	for _, rows := range []int{2, 4, 8} {
		if plan, err := PlanV41Q3KTokenPanel(5120, 2304, rows); err != nil || plan.WeightBytes != 5068800 {
			t.Fatalf("published down geometry rows=%d plan=%+v err=%v", rows, plan, err)
		}
	}
	for _, dims := range [][3]int{{65, 768, 1}, {65, 768, 3}, {65, 768, 9}, {0, 256, 2}, {65, 767, 2}, {1 << 30, 256, 8}, {65, 1 << 30, 8}} {
		if _, err := PlanV41Q3KTokenPanel(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid geometry %v admitted", dims)
		}
	}
	shader, err := os.ReadFile("shaders/v41_q3k_token_panel.comp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(shader)
	decoded := strings.Index(source, "float decoded = dl * float(code);")
	if decoded < 0 || !strings.Contains(source[decoded:], "for (uint token = 0u; token < tokens; ++token)") || strings.Count(source, "int code = int(") != 1 || !strings.Contains(source, "layout(local_size_x = 64)") {
		t.Fatal("shader must decode once before its per-token accumulator loop")
	}
}

func v41Q3KTokenPanelScalar(raw []byte, x []float32, out, in, rows int) ([]float32, int) {
	y := make([]float32, rows*out)
	decodes := 0
	for row := 0; row < out; row++ {
		for sb := 0; sb < in/256; sb++ {
			base := (row*(in/256) + sb) * 110
			d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+108:])))
			scales := unpackQ3KScales(raw[base+96 : base+108])
			qi, si, mask := 0, 0, byte(1)
			for half := 0; half < 256; half += 128 {
				shift := uint(0)
				for j := 0; j < 4; j++ {
					for sub := 0; sub < 2; sub++ {
						dl := d * float32(scales[si]-32)
						si++
						for l := 0; l < 16; l++ {
							lane := sub*16 + l
							code := int((raw[base+32+qi+lane] >> shift) & 3)
							if raw[base+lane]&mask == 0 {
								code -= 4
							}
							decoded := dl * float32(code)
							decodes++
							col := sb*256 + half + j*32 + lane
							for token := 0; token < rows; token++ {
								y[token*out+row] += decoded * x[token*in+col]
							}
						}
					}
					shift += 2
					mask <<= 1
				}
				qi += 32
			}
		}
	}
	return y, decodes
}
