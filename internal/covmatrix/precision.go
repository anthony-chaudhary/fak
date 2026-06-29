// precision.go — E3 of the combinatorial-growth epic (#1248): widens the support tensor
// from covmatrix's 2-D (family × backend) grid to the 3-D cross-support tensor
// (family × backend × precision), and surfaces the complete-model-support coverage face
// (how much of the cross-product reaches its target rung). "Cross support" is the
// intersection question the epic names — does Q4_K work on Vulkan for Qwen? — which the
// 2-D grid could not express because it had no precision axis.
//
// The precision axis is the weight Dtype the kernel dispatches a matmul/upload path on:
// the same lowercase tags compute.Dtype.String() emits ("f32", "q8_0", "q4_k",
// internal/compute/compute.go). Following this package's foundation-leaf discipline it
// pins those tokens rather than importing internal/compute, and precision_test.go cross-
// checks every token against the Dtype.String() source so the roster cannot drift from the
// kernel's Dtype enum — the same source-cross-check the family roster runs against the
// resolver (TestResolverTokensExistInSource). The wider Dtype set (f16/bf16/i8/i4/fp8) and
// the orthogonal KVPrecision tier are not weight-matmul dispatch keys on the reference
// path, so they are out of this axis by design (noted, not silently dropped).
//
// Honesty is preserved end to end. Every backend ends its weight-dtype switch in an honest
// panic (cpuref.go MatMul default, metal.go "only F32 today", vulkan.go "only F32 today",
// cuda.go Upload), so a (backend, precision) pair the backend does not implement is FENCED,
// never a silently-reachable wrong result. A FENCE is honest — it is NOT debt — exactly as
// in the 2-D grid. By construction classifyX never returns Undefined; precision_test.go
// asserts that invariant, which is the issue's "no cell silently UNDEFINED" witness.

package covmatrix

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// SchemaX is the control-pane envelope id for the cross-support tensor. It is distinct from
// Schema so the 3-D fold does not clobber the 2-D matrix's growth_debt corpus key.
const SchemaX = "fak-cross-support-tensor/1"

// DebtKeyX is the corpus key the cross-tensor fold writes. The debt is the count of
// silently-UNDEFINED cross cells — structurally 0, because every unimplemented precision is
// an honest fence; the key exists so a future backend that adds a dtype path WITHOUT a panic
// guard would surface as nonzero debt instead of slipping in silently.
const DebtKeyX = "cross_support_debt"

// Precisions is the weight-precision axis: the compute.Dtype tiers the kernel actually
// dispatches a matmul/upload path for, as the Dtype.String() tags from
// internal/compute/compute.go. precision_test.go cross-checks each token against that source
// so the roster cannot drift from the kernel's Dtype enum. Order is stable for deterministic
// output (lightest correctness-currency first).
var Precisions = []string{"f32", "q8_0", "q4_k"}

// backendPrecisions maps each backend to the weight dtypes it implements a real
// upload/matmul path for. The facts are pinned from the backend dispatch sites — each of
// which ends its dtype switch in an honest panic, so an ABSENT (backend, precision) pair is
// a FENCE (an honest refusal), never a silent wrong result:
//
//   - cpu    cpuBackend.MatMul / BatchedMatMul switch (internal/compute/cpuref.go):
//     f32, q8_0, q4_k — the topology-aware scalar reference; default: panic.
//   - cuda   cudaBackend.Upload                        (internal/compute/cuda.go):
//     f32, q8_0 (resident + narrow), q4_k (raw super-block bytes); else panic.
//   - metal  metalBackend.MatMul / Upload              (internal/compute/metal.go):
//     f32 only — "quantized device GEMM is a tracked follow-up" panics.
//   - vulkan vulkanBackend.Upload                      (internal/compute/vulkan.go):
//     f32, q8_0 (int8 code + scale path); q4_k hits "only F32 today" → panic.
//
// precision_test.go cross-checks the cpu reference row (the canonical supported-dtype set)
// against cpuref.go so this table cannot silently over-claim the kernel's coverage.
var backendPrecisions = map[string]map[string]bool{
	"cpu":    {"f32": true, "q8_0": true, "q4_k": true},
	"cuda":   {"f32": true, "q8_0": true, "q4_k": true},
	"metal":  {"f32": true},
	"vulkan": {"f32": true, "q8_0": true},
}

// precisionSupported reports whether backend implements a weight path for precision. A false
// result means the backend's dtype switch panics on that precision — an honest fence.
func precisionSupported(backend, precision string) bool {
	return backendPrecisions[backend][precision]
}

// XCell is one (family, backend, precision) cross-support cell — the intersection the epic
// names ("does q4_k work on vulkan for qwen?"). It reuses the 2-D grid's Support vocabulary
// so the two faces speak the same honesty language.
type XCell struct {
	Family    string  `json:"family"`
	Backend   string  `json:"backend"`
	Precision string  `json:"precision"`
	Support   Support `json:"support"`
	Topology  string  `json:"topology"`
}

// classifyX classifies one cross cell as the meet of two gates: the base (family, backend)
// topology gate (classify) and the (backend, precision) dtype gate (precisionSupported). A
// precision the backend does not implement is an honest FENCE (the backend's dtype switch
// panics) and dominates; otherwise the cell inherits the base (family, backend) support.
// Because classify never returns Undefined and the precision gap maps to Fenced, classifyX
// never returns Undefined — every gap is a fence. That is the "no cell silently UNDEFINED"
// witness, asserted in precision_test.go.
func classifyX(f Family, backend, precision string) Support {
	if !precisionSupported(backend, precision) {
		return Fenced
	}
	return classify(f, backend)
}

// GridX computes every (family, backend, precision) cross cell, in deterministic
// (family, backend, precision) order — the byte-stable enumeration the snapshot witness
// requires.
func GridX() []XCell {
	cells := make([]XCell, 0, len(Families)*len(Backends)*len(Precisions))
	for _, f := range Families {
		for _, b := range Backends {
			for _, p := range Precisions {
				cells = append(cells, XCell{
					Family:    f.Name,
					Backend:   b,
					Precision: p,
					Support:   classifyX(f, b, p),
					Topology:  string(f.Topology),
				})
			}
		}
	}
	return cells
}

// FamilyCoverage is one architecture's row on the complete-model-support coverage face: how
// many of its cross cells reach the target rung (SUPPORTED) out of the whole precision ×
// backend ladder. Complete is true only when the family reaches SUPPORTED on every cross
// cell — no fenced, proof-path, or undefined gap anywhere in its ladder.
type FamilyCoverage struct {
	Family    string `json:"family"`
	Supported int    `json:"supported"`
	Total     int    `json:"total"`
	Complete  bool   `json:"complete"`
}

// CoverageFace is the complete-model-support coverage face over the cross tensor: the count
// of cross cells at each support level plus the per-family breakdown. The headline is
// Supported/Cells — the fraction of the (family × backend × precision) cross-product that
// reaches the target rung. Fenced cells are honest gaps (the backend refuses the precision),
// not debt; proof-path cells run on the scalar reference without a CI oracle.
type CoverageFace struct {
	Cells     int              `json:"cells"`
	Supported int              `json:"supported"`
	Fenced    int              `json:"fenced"`
	ProofPath int              `json:"proof_path_only"`
	Undefined int              `json:"undefined"`
	ByFamily  []FamilyCoverage `json:"by_family"`
}

// Coverage folds GridX into the complete-model-support coverage face. Output is
// deterministic: GridX is already in (family, backend, precision) order and the per-family
// rows follow the Families roster order.
func Coverage() CoverageFace {
	face := CoverageFace{Cells: len(Families) * len(Backends) * len(Precisions)}
	supByFamily := make(map[string]int, len(Families))
	totByFamily := make(map[string]int, len(Families))
	for _, c := range GridX() {
		totByFamily[c.Family]++
		switch c.Support {
		case Supported:
			face.Supported++
			supByFamily[c.Family]++
		case Fenced:
			face.Fenced++
		case ProofPathOnly:
			face.ProofPath++
		case Undefined:
			face.Undefined++
		}
	}
	for _, f := range Families {
		sup, tot := supByFamily[f.Name], totByFamily[f.Name]
		face.ByFamily = append(face.ByFamily, FamilyCoverage{
			Family:    f.Name,
			Supported: sup,
			Total:     tot,
			Complete:  tot > 0 && sup == tot,
		})
	}
	return face
}

// undefinedXCells returns the silently-undefined cross cells in deterministic order. It is
// structurally empty (every gap is a fence), but it is computed — not assumed — so the
// invariant is witnessed by the same machinery that would catch a regression.
func undefinedXCells(cells []XCell) []XCell {
	var out []XCell
	for _, c := range cells {
		if c.Support == Undefined {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Family != out[j].Family {
			return out[i].Family < out[j].Family
		}
		if out[i].Backend != out[j].Backend {
			return out[i].Backend < out[j].Backend
		}
		return out[i].Precision < out[j].Precision
	})
	return out
}

// BuildX folds the cross tensor into the control-pane Payload, surfacing the
// complete-model-support coverage face through the same scorecard pipe the 2-D matrix uses.
// cross_support_debt is the count of silently-UNDEFINED cross cells (the correctness gate);
// cross_support_coverage is the advisory SOFT fraction that reaches the target rung — a
// fenced or proof-path cell is honest, not a defect, so it never reds the gate.
func BuildX() scorecard.Payload {
	cells := GridX()
	face := Coverage()

	undefined := undefinedXCells(cells)
	kpiUndefined := scorecard.KPI{
		Key:    "no_undefined_cross_cells",
		Group:  "correctness",
		Detail: fmt.Sprintf("%d (family,backend,precision) cross cell(s) reachable with neither a fence nor a CI witness", len(undefined)),
		Score:  pct(len(cells)-len(undefined), len(cells)),
	}
	for _, c := range undefined {
		kpiUndefined.Defects = append(kpiUndefined.Defects,
			fmt.Sprintf("%s × %s × %s is reachable but neither fenced nor witnessed", c.Family, c.Backend, c.Precision))
	}

	// Cross-support coverage is advisory: a FENCED (backend refuses the precision) or
	// PROOF-PATH-ONLY cell is honest, not a defect. Surfacing the gaps as SOFT keeps the gate
	// from reding on honest fences while showing how much of the cross-product reaches the
	// target rung — the complete-model-support coverage face.
	var soft []string
	for _, c := range cells {
		if c.Support != Supported {
			soft = append(soft, fmt.Sprintf("%s × %s × %s: %s", c.Family, c.Backend, c.Precision, c.Support))
		}
	}
	kpiCoverage := scorecard.KPI{
		Key:    "cross_support_coverage",
		Group:  "coverage",
		Detail: fmt.Sprintf("%d/%d cross cells reach the target rung (rest fenced/proof-path — honest, not debt)", face.Supported, face.Cells),
		Score:  pct(face.Supported, face.Cells),
		Soft:   soft,
	}

	corpus := map[string]any{
		"families":        len(Families),
		"backends":        len(Backends),
		"precisions":      len(Precisions),
		"cross_cells":     face.Cells,
		"supported":       face.Supported,
		"fenced":          face.Fenced,
		"proof_path_only": face.ProofPath,
		"undefined":       face.Undefined,
	}

	return scorecard.Fold(SchemaX, []scorecard.KPI{kpiUndefined, kpiCoverage}, DebtKeyX, nil, scorecard.Messages{
		Finding: fmt.Sprintf("%d undefined cross cell(s) — a family×backend×precision path is reachable without a fence or a CI witness",
			len(undefined)),
		FindingClean: fmt.Sprintf("every one of %d family×backend×precision cross cells is supported, fenced, or proof-path — none silently undefined",
			len(cells)),
		NextAction:      "install an honest fence or a conformance witness for each undefined cross cell",
		NextActionClean: "hold the line: regenerate the cross tensor on every model/backend/precision change and re-check cross_support_debt",
		ExtraCorpus:     corpus,
	})
}
