package covmatrix

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// repoRoot resolves the repository root from this test's source path:
// internal/covmatrix/precision_test.go -> two dirs up from internal/.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Skip("cannot locate test source path")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
}

func readSource(t *testing.T, rel ...string) string {
	t.Helper()
	p := filepath.Join(append([]string{repoRoot(t)}, rel...)...)
	b, err := os.ReadFile(p)
	if err != nil {
		t.Skipf("source %s not readable (%v); cross-check skipped", filepath.Join(rel...), err)
	}
	return string(b)
}

// TestPrecisionsMatchDtypeSource is the precision-axis analogue of TestResolverTokensExistInSource:
// every token in the Precisions roster must be a real compute.Dtype.String() tag, so the axis
// cannot drift from the kernel's Dtype enum (internal/compute/compute.go).
func TestPrecisionsMatchDtypeSource(t *testing.T) {
	if len(Precisions) == 0 {
		t.Fatal("Precisions roster is empty")
	}
	src := readSource(t, "internal", "compute", "compute.go")
	for _, p := range Precisions {
		needle := "\"" + p + "\""
		if !strings.Contains(src, needle) {
			t.Errorf("precision %q pins a token with no matching %s in compute.go Dtype.String() — roster drifted from the Dtype enum",
				p, needle)
		}
	}
}

// dtypeIdent maps a precision token to the compute.Dtype Go identifier its dispatch case uses.
var dtypeIdent = map[string]string{"f32": "F32", "q8_0": "Q8_0", "q4_k": "Q4_K"}

// TestCPUReferencePrecisionsMatchSource grounds the canonical supported-dtype row against the
// cpu reference dispatch (cpuref.go MatMul switch): every precision backendPrecisions claims
// cpu supports must appear as its Dtype case in cpuref.go, so the table cannot over-claim the
// kernel's coverage. The cpu row is the anchor because it is the topology-aware reference every
// other backend is measured against.
func TestCPUReferencePrecisionsMatchSource(t *testing.T) {
	src := readSource(t, "internal", "compute", "cpuref.go")
	for p, ok := range backendPrecisions["cpu"] {
		if !ok {
			continue
		}
		ident := dtypeIdent[p]
		if ident == "" {
			t.Errorf("cpu precision %q has no known compute.Dtype identifier to cross-check", p)
			continue
		}
		if !strings.Contains(src, "case "+ident+":") {
			t.Errorf("backendPrecisions[cpu] claims %q but no `case %s:` appears in cpuref.go — table over-claims the cpu reference",
				p, ident)
		}
	}
}

// TestQuantFencesAreHonestInSource grounds the metal/vulkan quant fences: the table marks
// those quant cells unsupported, and the backend source must carry the honest "only F32"
// refusal marker that makes them FENCED (not a silent wrong result).
func TestQuantFencesAreHonestInSource(t *testing.T) {
	if backendPrecisions["metal"]["q8_0"] || backendPrecisions["metal"]["q4_k"] {
		t.Fatal("table claims metal serves a quant precision; update this cross-check if metal gained a quant path")
	}
	metal := readSource(t, "internal", "compute", "metal.go")
	if !strings.Contains(metal, "only F32") {
		t.Error("metal quant cells are marked FENCED but metal.go carries no 'only F32' refusal marker — fence is ungrounded")
	}
	if backendPrecisions["vulkan"]["q4_k"] {
		t.Fatal("table claims vulkan serves q4_k; update this cross-check if vulkan gained a q4_k path")
	}
	vulkan := readSource(t, "internal", "compute", "vulkan.go")
	if !strings.Contains(vulkan, "only F32") {
		t.Error("vulkan q4_k cell is marked FENCED but vulkan.go carries no 'only F32' refusal marker — fence is ungrounded")
	}
}

// TestGridXEnumeratesEveryCrossCell is the enumeration witness: the tensor contains exactly
// one cell per (family, backend, precision) triple, with no duplicates.
func TestGridXEnumeratesEveryCrossCell(t *testing.T) {
	cells := GridX()
	want := len(Families) * len(Backends) * len(Precisions)
	if len(cells) != want {
		t.Errorf("GridX returned %d cells, want %d (%d families × %d backends × %d precisions)",
			len(cells), want, len(Families), len(Backends), len(Precisions))
	}
	seen := make(map[string]bool, len(cells))
	for _, c := range cells {
		key := c.Family + "/" + c.Backend + "/" + c.Precision
		if seen[key] {
			t.Errorf("duplicate cross cell: %s", key)
		}
		seen[key] = true
	}
}

// TestGridXNoSilentUndefined is the issue's core witness: no cross cell is silently UNDEFINED.
// Every gap is an honest fence, so the tensor never carries a reachable-but-unfenced cell.
func TestGridXNoSilentUndefined(t *testing.T) {
	for _, c := range GridX() {
		if c.Support == Undefined {
			t.Errorf("cross cell %s × %s × %s is silently UNDEFINED — every gap must be an honest fence",
				c.Family, c.Backend, c.Precision)
		}
		if c.Support == "" {
			t.Errorf("cross cell %s × %s × %s has an empty support level", c.Family, c.Backend, c.Precision)
		}
	}
}

// TestGridXDeterministic is the snapshot witness: two builds at one commit are byte-identical.
func TestGridXDeterministic(t *testing.T) {
	a, err := json.Marshal(GridX())
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	b, err := json.Marshal(GridX())
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(a) != string(b) {
		t.Error("GridX is not deterministic across two calls")
	}
}

// TestClassifyX checks the meet-of-two-gates logic on representative cross cells, including the
// epic's headline question (q4_k on vulkan) and the precision-dominates-base fence.
func TestClassifyX(t *testing.T) {
	llama := Family{Name: "Llama", Topology: PreNorm, OracleInCI: true}
	qwen := Family{Name: "Qwen2/3.x", Topology: PreNorm, OracleInCI: false}
	gemma := Family{Name: "Gemma2/3", Topology: SandwichNorm, OracleInCI: false}

	tests := []struct {
		name      string
		family    Family
		backend   string
		precision string
		want      Support
	}{
		{"Llama f32 on cpu (CI oracle) is SUPPORTED", llama, "cpu", "f32", Supported},
		{"Llama q8_0 on cpu (reference serves quant) is SUPPORTED", llama, "cpu", "q8_0", Supported},
		{"Llama q4_k on cuda (PreNorm + cuda q4_k path) is SUPPORTED", llama, "cuda", "q4_k", Supported},
		{"Qwen q4_k on vulkan is FENCED (vulkan has no q4_k path)", qwen, "vulkan", "q4_k", Fenced},
		{"Qwen q8_0 on metal is FENCED (metal is f32-only)", qwen, "metal", "q8_0", Fenced},
		{"Qwen q8_0 on vulkan (vulkan q8 path) is SUPPORTED", qwen, "vulkan", "q8_0", Supported},
		{"Gemma f32 on cuda is FENCED (topology fence dominates)", gemma, "cuda", "f32", Fenced},
		{"Gemma q4_k on cuda is FENCED (topology fence, precision supported)", gemma, "cuda", "q4_k", Fenced},
		{"Gemma f32 on cpu (no CI oracle) is PROOF-PATH-ONLY", gemma, "cpu", "f32", ProofPathOnly},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyX(tt.family, tt.backend, tt.precision)
			if got != tt.want {
				t.Errorf("classifyX(%s, %s, %s) = %v, want %v", tt.family.Name, tt.backend, tt.precision, got, tt.want)
			}
		})
	}
}

// TestCoverageFace checks the complete-model-support coverage face sums consistently with the
// tensor and that per-family rows partition the cross cells.
func TestCoverageFace(t *testing.T) {
	face := Coverage()
	if face.Cells != len(Families)*len(Backends)*len(Precisions) {
		t.Errorf("Coverage.Cells = %d, want %d", face.Cells, len(Families)*len(Backends)*len(Precisions))
	}
	if got := face.Supported + face.Fenced + face.ProofPath + face.Undefined; got != face.Cells {
		t.Errorf("coverage levels sum to %d, want %d (every cell classified exactly once)", got, face.Cells)
	}
	if face.Undefined != 0 {
		t.Errorf("coverage face reports %d undefined cross cells; the tensor must carry none", face.Undefined)
	}
	if len(face.ByFamily) != len(Families) {
		t.Fatalf("coverage face has %d family rows, want %d", len(face.ByFamily), len(Families))
	}
	totalByFamily, supByFamily := 0, 0
	for _, fc := range face.ByFamily {
		if fc.Total != len(Backends)*len(Precisions) {
			t.Errorf("family %s has %d cross cells, want %d", fc.Family, fc.Total, len(Backends)*len(Precisions))
		}
		if fc.Supported > fc.Total {
			t.Errorf("family %s supported %d exceeds total %d", fc.Family, fc.Supported, fc.Total)
		}
		if fc.Complete != (fc.Total > 0 && fc.Supported == fc.Total) {
			t.Errorf("family %s Complete=%v inconsistent with %d/%d", fc.Family, fc.Complete, fc.Supported, fc.Total)
		}
		totalByFamily += fc.Total
		supByFamily += fc.Supported
	}
	if totalByFamily != face.Cells {
		t.Errorf("per-family totals sum to %d, want %d", totalByFamily, face.Cells)
	}
	if supByFamily != face.Supported {
		t.Errorf("per-family supported sum to %d, want %d", supByFamily, face.Supported)
	}
}

// TestBuildXDeterministicAndHonest is the control-pane witness: the cross-tensor fold is
// deterministic, carries the debt key, and reports zero silently-undefined debt (every gap is
// a fence — FENCED ≠ debt).
func TestBuildXDeterministicAndHonest(t *testing.T) {
	a, err := json.Marshal(BuildX())
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	b, err := json.Marshal(BuildX())
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(a) != string(b) {
		t.Error("BuildX is not deterministic across two calls")
	}

	payload := BuildX()
	if payload.Schema != SchemaX {
		t.Errorf("payload schema = %s, want %s", payload.Schema, SchemaX)
	}
	debt, ok := payload.Corpus[DebtKeyX]
	if !ok {
		t.Fatalf("payload corpus missing %q key", DebtKeyX)
	}
	if d, isInt := debt.(int); !isInt || d != 0 {
		t.Errorf("cross_support_debt = %v, want 0 (every gap is an honest fence, not debt)", debt)
	}
	foundCorrectness := false
	for _, kpi := range payload.KPIs {
		if kpi.Key == "no_undefined_cross_cells" {
			foundCorrectness = true
			if len(kpi.Defects) != 0 {
				t.Errorf("no_undefined_cross_cells carries %d defects; expected none", len(kpi.Defects))
			}
		}
	}
	if !foundCorrectness {
		t.Error("payload missing no_undefined_cross_cells KPI")
	}
}
