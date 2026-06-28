// Package covmatrix is the C1 keystone of the combinatorial-growth epic (#1079/#1080):
// it derives fak's model × backend support grid from the kernel's own structural
// facts and folds the result into the shared scorecard control-pane as a growth_debt
// integer.
//
// The problem it replaces: the support cross-product was tracked by a hand-written
// prose table (docs/notes/model-arch-seam-status-487.md) that goes stale, and by
// oracle tests that SKIP in CI — so the 12 requirePreNorm panic cells and the wider
// honestly-unsupported set were invisible unless you read the source or hit them at
// runtime. This package makes the grid a generated, deterministic artifact: the same
// commit always yields the same matrix, and a new model/backend changes it as a diff.
//
// What "derived from the tree" means here. The atoms below — the family roster, the
// topology each family lowers to, the backend roster, and the accelerated-path fence
// rule — are the same facts the kernel encodes in code (internal/model/tensor_resolver.go
// resolveSpecFor, BlockTopology in arch.go, the requirePreNorm/requireGLMDsaSession
// call sites in kv.go, the --backend enum in cmd/fak/serve.go). They are pinned here as
// the single classification table the matrix is computed from; covmatrix_test.go cross-
// checks the family roster against the resolver source so the table cannot silently drift
// from the kernel it describes (that test is the C1 acceptance gate). The follow-on within
// #1080 is to read the topology + fence facts straight from go/ast so even the per-family
// topology cannot drift; the roster cross-check lands that guarantee for the family axis now.
package covmatrix

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane envelope id for the coverage matrix.
const Schema = "fak-coverage-matrix/1"

// DebtKey is the corpus key the control-pane fold reads (corpus.growth_debt).
const DebtKey = "growth_debt"

// Topology is the block topology a family lowers to. PreNorm is the only one the
// accelerated hot-path copies (HAL/Metal/quant-batch) implement today; the others run
// on the scalar f32 proof path and panic on the accelerated paths (requirePreNorm,
// internal/model/kv.go). This mirrors BlockTopology in internal/model/arch.go.
type Topology string

const (
	PreNorm          Topology = "PreNorm"
	PostNorm         Topology = "PostNorm"
	SandwichNorm     Topology = "SandwichNorm"
	ParallelResidual Topology = "ParallelResidual"
	// SparseAttn marks the MLA/DSA/MSA families whose sparse-attention index is
	// host-resident; their accelerated path is gated by a dedicated fence
	// (requireGLMDsaSession, internal/model/kv.go) rather than requirePreNorm.
	SparseAttn Topology = "SparseAttn"
)

// accelerated reports whether a backend uses one of the hot-path copies that still
// hardcode Llama PreNorm. The cpu reference path runs the topology-aware scalar
// blockStep / cacheless layer(), so it serves every topology.
func accelerated(backend string) bool { return backend != "cpu" }

// Family is one model architecture in the kernel's resolver, with the topology it
// lowers to and whether it carries a CI-runnable on-disk numeric oracle. The roster
// tracks internal/model/tensor_resolver.go resolveSpecFor (+ the identity Llama default).
type Family struct {
	// Name is the canonical family label shown in the grid.
	Name string
	// ResolverToken is the substring resolveSpecFor matches on (tensor_resolver.go),
	// or "" for the identity Llama default. covmatrix_test.go cross-checks every
	// non-empty token against the resolver source so the roster cannot drift.
	ResolverToken string
	// Topology is the block topology this family lowers to (topologyForFamily, weights.go).
	Topology Topology
	// OracleInCI is true only when a weight-free, checkpoint-independent numeric witness
	// runs in CI for this family. Today only Llama (the SmolLM2 anchor + Float32bits gate)
	// qualifies; every other family's HF oracle is the checkpoint-gated #474 set that
	// SKIPs under -short. This is the honest "asserted, not proven" boundary the epic names.
	OracleInCI bool
}

// Families is the kernel's architecture roster, derived from the resolveSpecFor switch
// (internal/model/tensor_resolver.go:121) plus the identity Llama default. Order is
// stable for deterministic output. Keep this in sync with the resolver — the cross-check
// test (covmatrix_test.go) reds the trunk on drift.
var Families = []Family{
	{Name: "Llama", ResolverToken: "", Topology: PreNorm, OracleInCI: true},
	{Name: "Qwen2/3.x", ResolverToken: "", Topology: PreNorm, OracleInCI: false},
	{Name: "GPT-NeoX", ResolverToken: "gptneox", Topology: ParallelResidual, OracleInCI: false},
	{Name: "Falcon", ResolverToken: "falcon", Topology: ParallelResidual, OracleInCI: false},
	{Name: "MPT", ResolverToken: "mpt", Topology: PreNorm, OracleInCI: false},
	{Name: "StableLM", ResolverToken: "stablelm", Topology: PreNorm, OracleInCI: false},
	{Name: "OLMo2", ResolverToken: "olmo2", Topology: PostNorm, OracleInCI: false},
	{Name: "Cohere", ResolverToken: "cohere", Topology: ParallelResidual, OracleInCI: false},
	{Name: "Gemma2/3", ResolverToken: "gemma", Topology: SandwichNorm, OracleInCI: false},
	{Name: "Mixtral-MoE", ResolverToken: "mixtral", Topology: PreNorm, OracleInCI: false},
	{Name: "gpt-oss-MoE", ResolverToken: "gptoss", Topology: PreNorm, OracleInCI: false},
	{Name: "DeepSeek-MLA", ResolverToken: "deepseek", Topology: SparseAttn, OracleInCI: false},
	{Name: "GLM-5.2-DSA", ResolverToken: "", Topology: SparseAttn, OracleInCI: false},
	{Name: "MiniMax-MSA", ResolverToken: "", Topology: SparseAttn, OracleInCI: false},
}

// Backends is the --backend roster (cmd/fak/serve.go). cpu is the topology-aware
// reference; the rest are accelerated hot-path copies today.
var Backends = []string{"cpu", "cuda", "metal", "vulkan"}

// Support is the classification of one (family, backend) cell.
type Support string

const (
	// Supported: the cell runs and (for the family axis) has a CI-runnable witness.
	Supported Support = "SUPPORTED"
	// ProofPathOnly: correct on the scalar cpu path but not on this accelerated backend.
	ProofPathOnly Support = "PROOF-PATH-ONLY"
	// Fenced: the accelerated path panics honestly (requirePreNorm / requireGLMDsaSession)
	// rather than silently diverging. A fence is honest — it is NOT debt.
	Fenced Support = "FENCED"
	// Undefined: the dispatch can reach this cell with neither a fence nor a passing
	// witness — a silently-reachable wrong-result path. THIS is growth_debt.
	Undefined Support = "UNDEFINED"
)

// Cell is one (family, backend) classification.
type Cell struct {
	Family   string  `json:"family"`
	Backend  string  `json:"backend"`
	Support  Support `json:"support"`
	Topology string  `json:"topology"`
}

// classify decides one cell's support level from the family's topology and the backend.
//
//   - cpu serves every topology (the scalar proof path). A family with a CI oracle is
//     SUPPORTED there; one without is PROOF-PATH-ONLY (it runs, but the numeric claim is
//     unproven in CI — the honest #474 boundary).
//   - an accelerated backend serves PreNorm families (the hot-path copies implement
//     PreNorm). For a non-PreNorm topology the kernel installs an honest fence
//     (requirePreNorm / requireGLMDsaSession) → FENCED. A non-PreNorm cell with NO fence
//     would be UNDEFINED — the silently-reachable wrong result the metric exists to catch.
func classify(f Family, backend string) Support {
	if !accelerated(backend) {
		if f.OracleInCI {
			return Supported
		}
		return ProofPathOnly
	}
	if f.Topology == PreNorm {
		return Supported
	}
	// Every non-PreNorm topology has an installed accelerated-path fence today
	// (requirePreNorm for PostNorm/SandwichNorm/ParallelResidual; requireGLMDsaSession
	// for the SparseAttn families). The fence is what keeps these cells out of debt.
	return Fenced
}

// Grid computes every (family, backend) cell, in deterministic (family, backend) order.
func Grid() []Cell {
	cells := make([]Cell, 0, len(Families)*len(Backends))
	for _, f := range Families {
		for _, b := range Backends {
			cells = append(cells, Cell{
				Family:   f.Name,
				Backend:  b,
				Support:  classify(f, b),
				Topology: string(f.Topology),
			})
		}
	}
	return cells
}

// StaleReason names why a cell is on the --stale honest-but-incomplete list.
// The matrix tracks no per-cell oracle DATE, so "stale" here is the structural
// residual: a cell that RUNS but whose family carries no CI-runnable numeric
// oracle (OracleInCI == false), so its correctness is asserted, not proven in
// CI. This is the union the C5 ticket (#1084) names — "oracle older than N days
// OR support level PROOF-PATH-ONLY past a grace window" — in the limit where a
// family with no CI oracle has an oracle age of effectively infinite (older than
// any N). The N-days refinement (discriminating by a real per-family oracle date)
// is the follow-on once an oracle-date ledger exists; the structural list is what
// ships today. A FENCED cell is honest-AND-complete (it refuses) and an UNDEFINED
// cell is growth_debt — neither is "stale".
type StaleReason string

const (
	// StaleProofPath: the cell is PROOF-PATH-ONLY — it runs on the scalar cpu
	// proof path but its family has no CI oracle, so it is past any grace window
	// (the #487/S4 residual carried forever).
	StaleProofPath StaleReason = "PROOF-PATH-ONLY: runs on the cpu proof path, no CI oracle (correctness asserted, not proven)"
	// StaleUnwitnessed: the cell is SUPPORTED by topology — a PreNorm family on an
	// accelerated backend — but its family has no CI oracle, so the accelerated
	// numeric claim is unwitnessed in CI (the "oracle older than N days" criterion,
	// at infinite age).
	StaleUnwitnessed StaleReason = "SUPPORTED but no CI oracle: accelerated path runs, numeric claim unwitnessed in CI"
)

// StaleCell is one honest-but-incomplete cell surfaced by the --stale lens.
type StaleCell struct {
	Cell
	Reason StaleReason `json:"reason"`
}

// StaleCells returns the honest-but-incomplete residual: every cell that RUNS
// (SUPPORTED or PROOF-PATH-ONLY) whose family carries no CI-runnable numeric
// oracle. Llama (the only OracleInCI family today) never appears; FENCED cells
// (honest refusals) and UNDEFINED cells (growth_debt) are excluded by design.
// Output is deterministic: Grid() is already in (family, backend) order.
func StaleCells() []StaleCell {
	oracle := make(map[string]bool, len(Families))
	for _, f := range Families {
		oracle[f.Name] = f.OracleInCI
	}
	var out []StaleCell
	for _, c := range Grid() {
		if oracle[c.Family] {
			continue // a CI oracle witnesses this family — not stale
		}
		switch c.Support {
		case ProofPathOnly:
			out = append(out, StaleCell{Cell: c, Reason: StaleProofPath})
		case Supported:
			out = append(out, StaleCell{Cell: c, Reason: StaleUnwitnessed})
		}
	}
	return out
}

// countBy tallies cells by support level.
func countBy(cells []Cell) map[Support]int {
	m := map[Support]int{}
	for _, c := range cells {
		m[c.Support]++
	}
	return m
}

// Build folds the grid into the control-pane Payload. growth_debt is the count of
// UNDEFINED cells (silently-reachable, unfenced, unwitnessed). The KPIs split the grid
// into the axes a maintainer acts on: undefined cells (the debt), and accelerated-path
// coverage (how much of the cross-product is still proof-path/fenced — advisory, since a
// fence is honest, so those are SOFT not debt).
func Build() scorecard.Payload {
	cells := Grid()
	counts := countBy(cells)

	undefined := undefinedCells(cells)
	kpiUndefined := scorecard.KPI{
		Key:    "no_undefined_cells",
		Group:  "correctness",
		Detail: fmt.Sprintf("%d (family,backend) cell(s) reachable with neither a fence nor a CI witness", len(undefined)),
		Score:  pct(len(cells)-len(undefined), len(cells)),
	}
	for _, c := range undefined {
		kpiUndefined.Defects = append(kpiUndefined.Defects,
			fmt.Sprintf("%s × %s is reachable but neither fenced nor witnessed", c.Family, c.Backend))
	}

	// Accelerated coverage is advisory: a FENCED or PROOF-PATH-ONLY cell is honest, not a
	// defect. Surfacing it as SOFT keeps the gate from reding on honest gaps while still
	// showing how much of the cross-product the hot path covers (the #487/S4 residual).
	accelProven := 0
	accelTotal := 0
	var soft []string
	for _, c := range cells {
		if c.Backend == "cpu" {
			continue
		}
		accelTotal++
		if c.Support == Supported {
			accelProven++
		} else {
			soft = append(soft, fmt.Sprintf("%s × %s: %s", c.Family, c.Backend, c.Support))
		}
	}
	kpiAccel := scorecard.KPI{
		Key:    "accelerated_coverage",
		Group:  "coverage",
		Detail: fmt.Sprintf("%d/%d accelerated cells run (rest fenced/proof-path — honest, not debt)", accelProven, accelTotal),
		Score:  pct(accelProven, accelTotal),
		Soft:   soft,
	}

	corpus := map[string]any{
		"families":        len(Families),
		"backends":        len(Backends),
		"cells":           len(cells),
		"supported":       counts[Supported],
		"proof_path_only": counts[ProofPathOnly],
		"fenced":          counts[Fenced],
		"undefined":       counts[Undefined],
	}

	return scorecard.Fold(Schema, []scorecard.KPI{kpiUndefined, kpiAccel}, DebtKey, nil, scorecard.Messages{
		Finding: fmt.Sprintf("%d undefined cell(s) — a model×backend path is reachable without a fence or a CI witness",
			len(undefined)),
		FindingClean: fmt.Sprintf("every one of %d model×backend cells is supported, fenced, or proof-path — none silently undefined",
			len(cells)),
		NextAction:      "install an honest fence (requirePreNorm-style) or a conformance witness (#1081) for each undefined cell",
		NextActionClean: "hold the line: regenerate the matrix on every model/backend change and re-check growth_debt (#1084)",
		ExtraCorpus:     corpus,
	})
}

// undefinedCells returns the debt cells in deterministic order.
func undefinedCells(cells []Cell) []Cell {
	var out []Cell
	for _, c := range cells {
		if c.Support == Undefined {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Family != out[j].Family {
			return out[i].Family < out[j].Family
		}
		return out[i].Backend < out[j].Backend
	})
	return out
}

// pct renders n/total as a 0-100 score (100 when total is 0, so an empty axis is clean).
func pct(n, total int) float64 {
	if total == 0 {
		return 100
	}
	return 100 * float64(n) / float64(total)
}
