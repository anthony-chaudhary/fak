package compute

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"math"
	"os"
	"regexp"
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

// fak-test:runtime fast est=500ms
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
	for _, rows := range []int{2, 4, 8, 9, 31, 64} {
		plan, err := PlanV41Q3KTokenPanel(out, in, rows)
		if err != nil || plan.PayloadBytes != len(raw) || plan.WeightBytes != len(raw)+2 || plan.InputBytes != rows*in*4 || plan.OutputBytes != rows*out*4 || plan.WorkgroupsX != 2 || plan.WorkgroupsY != uint32((rows+7)/8) {
			t.Fatalf("rows=%d plan=%+v err=%v", rows, plan, err)
		}
		x := make([]float32, rows*in)
		for token := 0; token < rows; token++ {
			for col := 0; col < in; col++ {
				x[token*in+col] = float32((col*3+token*11)%31-15) / 32
			}
		}
		got, decodes := v41Q3KTokenPanelScalar(raw, x, out, in, rows)
		if decodes != out*in*int(plan.WorkgroupsY) {
			t.Fatalf("rows=%d decodes=%d, want one decode per eight-token tile %d", rows, decodes, out*in*int(plan.WorkgroupsY))
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
	for _, rows := range []int{2, 4, 8, 9, 31, 64} {
		if plan, err := PlanV41Q3KTokenPanel(5120, 2304, rows); err != nil || plan.WeightBytes != 5068800 {
			t.Fatalf("published down geometry rows=%d plan=%+v err=%v", rows, plan, err)
		}
	}
	for _, dims := range [][3]int{{65, 768, 1}, {65, 768, 0}, {65, 768, 65}, {0, 256, 2}, {65, 767, 2}, {1 << 30, 256, 8}, {65, 1 << 30, 8}} {
		if _, err := PlanV41Q3KTokenPanel(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid geometry %v admitted", dims)
		}
	}
	// Every admitted total maps each token once, including 1..7-token tails.
	// No inactive lane may name storage beyond the full input/output extent.
	for rows := 2; rows <= 64; rows++ {
		plan, err := PlanV41Q3KTokenPanel(out, in, rows)
		if err != nil {
			t.Fatal(err)
		}
		visits := make([]int, rows)
		for group := uint32(0); group < plan.WorkgroupsY; group++ {
			base := int(group) * 8
			count := min(8, rows-base)
			if count < 1 || count > 8 {
				t.Fatal("invalid local token count")
			}
			for local := 0; local < count; local++ {
				token := base + local
				if token < 0 || token >= rows || (token+1)*in*4 > plan.InputBytes || (token+1)*out*4 > plan.OutputBytes {
					t.Fatal("tile crossed full-panel storage")
				}
				visits[token]++
			}
		}
		for _, count := range visits {
			if count != 1 {
				t.Fatal("token omitted or duplicated")
			}
		}
	}
	native, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	if err := v41Q3TokenGridNativeSource(string(native)); err != nil {
		t.Fatal(err)
	}
	for _, replacement := range [][2]string{
		{"!g_v41Q3TokenGridModule && tokens != 2", "false && tokens != 2"},
		{"g_v41Q3TokenGridModule = q3GridABI == 1", "g_v41Q3TokenGridModule = true"},
		{"if (q3GridABI < 0) return false;", ""},
	} {
		broken := strings.Replace(string(native), replacement[0], replacement[1], 1)
		if broken == string(native) || v41Q3TokenGridNativeSource(broken) == nil {
			t.Fatal("negative control accepted unsafe loaded-module grid admission")
		}
	}
	const q3Dispatch = "dispatch(g_kern[K_V41_Q3K_TOKEN_PANEL], bufs, &pc, sizeof(pc), uint32_t((uint64_t(out) + 63u) / 64u), uint32_t((uint64_t(tokens) + 7u) / 8u), true)"
	broken := strings.Replace(string(native), q3Dispatch, strings.Replace(q3Dispatch, "/ 8u)", "/ 4u)", 1), 1)
	if broken == string(native) || v41Q3TokenGridNativeSource(broken) == nil {
		t.Fatal("negative control accepted wrong native token grid")
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
	for tokenBase := 0; tokenBase < rows; tokenBase += 8 {
		tokenEnd := min(tokenBase+8, rows)
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
								for token := tokenBase; token < tokenEnd; token++ {
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
	}
	return y, decodes
}

func v41Q3TokenGridNativeSource(source string) error {
	comments := regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
	source = comments.ReplaceAllString(source, "")
	normalize := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	for _, text := range []string{
		"uint32_t g_maxComputeWorkGroupCountY = 0;",
		"bool g_v41Q3TokenGridModule = false;",
		"g_maxComputeWorkGroupCountY = props2.properties.limits.maxComputeWorkGroupCount[1];",
	} {
		if strings.Count(normalize(source), text) != 1 {
			return fmt.Errorf("missing actual device Y limit: %s", text)
		}
	}
	extract := func(name string) (string, error) {
		start := strings.Index(source, name)
		if start < 0 {
			return "", fmt.Errorf("missing native Q3 function %s", name)
		}
		brace := strings.Index(source[start:], "{")
		if brace < 0 {
			return "", fmt.Errorf("missing native Q3 body %s", name)
		}
		depth, end := 1, start+brace+1
		for ; end < len(source) && depth != 0; end++ {
			if source[end] == '{' {
				depth++
			}
			if source[end] == '}' {
				depth--
			}
		}
		if depth != 0 {
			return "", fmt.Errorf("unterminated native Q3 body %s", name)
		}
		return normalize(source[start:end]), nil
	}
	// Pin actual loaded-byte classification and its publication only after a
	// successful pipeline build. A filename or mere framing is insufficient.
	for name, want := range map[string]string{
		"static int v41TokenGridModuleABI(": "1de863aa99e27cc08a870eef8e21dde6eb65babdb907531e998bd0e5b9f6ef98",
		"bool buildKernel(":                 "708c28a14a899f178be4bf8f3ed16fda7bf66dd4bec1a9bf113267335d98a29d",
	} {
		body, err := extract(name)
		if err != nil {
			return err
		}
		if fmt.Sprintf("%x", sha256.Sum256([]byte(body))) != want {
			return fmt.Errorf("loaded-module ABI admission changed: %s", name)
		}
	}
	var native string
	for _, name := range []string{"fvk_have_v41_q3k_token_panel", "fvk_v41_q3k_token_panel_shape", "fvk_v41_q3k_token_panel_f32"} {
		body, err := extract("int " + name + "(")
		if err != nil {
			return err
		}
		native += body
	}
	for _, replacement := range [][2]string{
		{"if (!g_v41Q3TokenGridModule && tokens != 2 && tokens != 4 && tokens != 8) return 0; ", ""},
		{"(tokens < 2 || tokens > 64)", "(tokens != 2 && tokens != 4 && tokens != 8)"},
		{"const uint64_t tokenGroups = (uint64_t(tokens) + 7u) / 8u; ", ""},
		{" && g_maxComputeWorkGroupCountY != 0 && tokenGroups <= g_maxComputeWorkGroupCountY", ""},
		{"uint32_t((uint64_t(tokens) + 7u) / 8u), true)", "1, true)"},
	} {
		if strings.Count(native, replacement[0]) != 1 {
			return fmt.Errorf("native token bounds or dispatch grid changed")
		}
		native = strings.Replace(native, replacement[0], replacement[1], 1)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(native))); got != "2a20749e4a80d35419135a616cdd17375aa3fb4245ec6a44d10b4f16d43e291b" {
		return fmt.Errorf("native Q3 availability/storage/state contract changed: %s", got)
	}
	return nil
}
