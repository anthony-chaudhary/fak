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
	// runs in CI for this family. Llama qualifies via the SmolLM2 anchor + Float32bits
	// gate (refactor_test.go's legacy hand-copy web); OLMo2 and Qwen2/3.x via the
	// independent HF-semantics references in internal/model/family_cpu_oracle_test.go
	// (#1271 Lane 1); MPT via internal/model/family_mpt_cpu_oracle_test.go, whose
	// reference adds the ALiBi slope table (including the non-power-of-two head
	// reorder), the mean-subtracting bias-free LayerNorm, and the fused-Wqkv row cut;
	// Falcon via internal/model/family_falcon_cpu_oracle_test.go, whose reference adds
	// the parallel-residual dataflow (both branches read the same pre-block x through
	// the same norm) and the multi-query fused-qkv cut. That Falcon reference covers the
	// 7B multi_query variant — the one production's contiguous qkv split and single
	// aliased input_layernorm actually implement; new_decoder_architecture (40B/180B)
	// and Falcon-RW use different layouts and are NOT witnessed by it.
	// GPT-NeoX via internal/model/family_gptneox_cpu_oracle_test.go, whose reference adds
	// the parallel residual, the two learned-bias LayerNorms, the INTERLEAVED fused-QKV
	// split and the exact-erf GELU, and which runs at BOTH published rotary_pct lineages
	// (1.0 and 0.25); StableLM via internal/model/family_stablelm_cpu_oracle_test.go at
	// both use_qkv_bias lineages. Their partial-rotary halves were red until invFreqDenom
	// stopped denominating inv_freq by head_dim (rope.go) — the bit is set on the fix, not
	// on a widened tolerance.
	// Cohere via internal/model/family_cohere_cpu_oracle_test.go, whose reference adds the
	// half-split rotary pairing, plus internal/model/cohere_loader_routing_test.go, which
	// pins the three HF-source f32 loaders to the re-layout constructor the oracle proves —
	// the bit is set on that FIX, not on the pairing it replaced. gptq.go still routes
	// through newModel, so the quantized loader is outside the claim.
	// Gemma2/3 via internal/model/family_gemma_cpu_oracle_test.go, the first SandwichNorm
	// reference, covering both lineages: four distinct sandwich norms, the (1+w) gain, the
	// sqrt(hidden_size) embedding scale, the local/global cadence and Gemma3's two rope
	// bases, a query_pre_attn_scalar deliberately != head_dim, Gemma2's two soft-caps and
	// Gemma3's per-head QK-norm.
	// Mixtral-MoE via internal/model/family_mixtral_cpu_oracle_test.go, whose reference
	// reads the HF SOURCE expert names rather than production's canonical aliases, and
	// which fails if routing is degenerate; gpt-oss-MoE via
	// internal/model/family_gptoss_cpu_oracle_test.go — set on the sliding-window cadence
	// FIX (familySlidingWindowPattern). Neither covers the batched MoE kernel, and gpt-oss
	// additionally excludes YaRN and the MXFP4 expert dequant.
	// DeepSeek-MLA via internal/model/family_deepseek_cpu_oracle_test.go, which witnesses
	// the batched prefill and the separately written cached decode against the SAME
	// reference and pins the MLA head_dim derivation without which the rotary table is too
	// short to cover the pe slice. It builds the dense first_k_dense_replace lineage, so
	// the routed DeepseekMoE block and YaRN are outside it.
	// GLM-5.2-DSA via internal/model/family_glm_cpu_oracle_test.go, which compares the
	// discrete top-k SELECTION SETS as well as the logits. Its fixture omits
	// mlp.gate.weight, so glmMoeFFN and glmRoute are unwitnessed, and its two indexer scale
	// constants are provably unreachable by ANY end-to-end oracle: each multiplies every
	// score by the same positive constant ahead of a ranking.
	// MiniMax-MSA via internal/model/family_minimax_cpu_oracle_test.go, whose
	// block-selector reference is an independent RE-IMPLEMENTATION of the documented
	// algorithm rather than a transcription (no shipped transformers release carries
	// minimax_m3_vl), so it catches implementation defects but not a spec misreading it
	// shares with production. Every other semantic there IS transcribed from a cited file.
	// That completes the roster, so the honest boundary this bit records MOVED rather than
	// disappearing. It used to run between FAMILIES — a few proven, most merely asserted.
	// It now runs between PATHS: every oracle above is a CPU f32 HF-checkpoint-path
	// witness, and the quantized loaders (which hold their projections in their own decoded
	// stores rather than the f32 blob), the GGUF path (a different constructor) and the
	// accelerated backends (separate hot-path copies) are witnessed by NONE of them. The
	// checkpoint-gated #474 oracles that SKIP under -short remain the only cover there.
	OracleInCI bool
}

// Families is the kernel's architecture roster, derived from the resolveSpecFor switch
// (internal/model/tensor_resolver.go:121) plus the identity Llama default. Order is
// stable for deterministic output. Keep this in sync with the resolver — the cross-check
// test (covmatrix_test.go) reds the trunk on drift.
var Families = []Family{
	{Name: "Llama", ResolverToken: "", Topology: PreNorm, OracleInCI: true},
	{Name: "Qwen2/3.x", ResolverToken: "", Topology: PreNorm, OracleInCI: true},
	{Name: "GPT-NeoX", ResolverToken: "gptneox", Topology: ParallelResidual, OracleInCI: true},
	{Name: "Falcon", ResolverToken: "falcon", Topology: ParallelResidual, OracleInCI: true},
	{Name: "MPT", ResolverToken: "mpt", Topology: PreNorm, OracleInCI: true},
	{Name: "StableLM", ResolverToken: "stablelm", Topology: PreNorm, OracleInCI: true},
	{Name: "OLMo2", ResolverToken: "olmo2", Topology: PostNorm, OracleInCI: true},
	{Name: "Cohere", ResolverToken: "cohere", Topology: ParallelResidual, OracleInCI: true},
	{Name: "Gemma2/3", ResolverToken: "gemma", Topology: SandwichNorm, OracleInCI: true},
	{Name: "Mixtral-MoE", ResolverToken: "mixtral", Topology: PreNorm, OracleInCI: true},
	{Name: "gpt-oss-MoE", ResolverToken: "gptoss", Topology: PreNorm, OracleInCI: true},
	{Name: "DeepSeek-MLA", ResolverToken: "deepseek", Topology: SparseAttn, OracleInCI: true},
	{Name: "GLM-5.2-DSA", ResolverToken: "", Topology: SparseAttn, OracleInCI: true},
	{Name: "MiniMax-MSA", ResolverToken: "", Topology: SparseAttn, OracleInCI: true},
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
// residual: a cell that RUNS but that no CI-runnable numeric oracle executes, so
// its correctness is asserted, not proven in CI. That is either a family with no
// oracle at all (OracleInCI == false) or ANY accelerated cell, because every oracle
// in the roster witnesses the cpu path only. This is the union the C5 ticket (#1084) names — "oracle older than N days
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
	// accelerated backend — but no CI oracle executes that backend, so the accelerated
	// numeric claim is unwitnessed in CI (the "oracle older than N days" criterion,
	// at infinite age). A CPU oracle for the same family does NOT clear this: the
	// accelerated backend is a separate hot-path copy the cpu oracle never runs.
	StaleUnwitnessed StaleReason = "SUPPORTED but no CI oracle: accelerated path runs, numeric claim unwitnessed in CI"
)

// StaleCell is one honest-but-incomplete cell surfaced by the --stale lens.
type StaleCell struct {
	Cell
	Reason StaleReason `json:"reason"`
}

// StaleCells returns the honest-but-incomplete residual: every cell that RUNS
// (SUPPORTED or PROOF-PATH-ONLY) that no CI-runnable numeric oracle witnesses.
// FENCED cells (honest refusals) and UNDEFINED cells (growth_debt) are excluded by
// design. Output is deterministic: Grid() is already in (family, backend) order.
//
// An OracleInCI family clears its OWN cpu cell and nothing else. Every oracle in the
// roster is a CPU f32 HF-checkpoint-path witness (see Family.OracleInCI), and each
// accelerated backend is a separate hot-path copy that no cpu oracle executes — so an
// accelerated cell classified SUPPORTED from topology alone stays unwitnessed however
// many families carry an oracle. Exempting the whole family on a cpu-only witness is
// what this function used to do; it under-reported from the start, and once the roster
// completed it reported an empty residual for a grid with 18 unwitnessed accelerated
// cells. The per-family shortcut is gone: the doctrine (#1244) and FromSupport
// (internal/supportmaturity) both already describe this function as flagging "the
// accelerated-SUPPORTED cells whose witness is in fact absent", and it now does.
func StaleCells() []StaleCell {
	oracle := make(map[string]bool, len(Families))
	for _, f := range Families {
		oracle[f.Name] = f.OracleInCI
	}
	var out []StaleCell
	for _, c := range Grid() {
		if oracle[c.Family] && !accelerated(c.Backend) {
			continue // the family's CPU oracle witnesses its cpu cell — not stale
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
	defectLabels := make([]string, 0, len(undefined))
	for _, c := range undefined {
		defectLabels = append(defectLabels, fmt.Sprintf("%s × %s", c.Family, c.Backend))
	}
	kpiUndefined := undefinedCorrectnessKPI("no_undefined_cells", "(family,backend) cell(s)", len(cells), defectLabels)

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

// undefinedOf filters cells to those isUndefined reports true for (preserving
// order) and stable-sorts the result with less — the "filter to the debt cells,
// then sort deterministically" idiom both the family-axis matrix (undefinedCells)
// and the cross-axis matrix (undefinedXCells, precision.go) share.
func undefinedOf[T any](cells []T, isUndefined func(T) bool, less func(a, b T) bool) []T {
	var out []T
	for _, c := range cells {
		if isUndefined(c) {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return less(out[i], out[j]) })
	return out
}

// undefinedCells returns the debt cells in deterministic order.
func undefinedCells(cells []Cell) []Cell {
	return undefinedOf(cells,
		func(c Cell) bool { return c.Support == Undefined },
		func(a, b Cell) bool {
			if a.Family != b.Family {
				return a.Family < b.Family
			}
			return a.Backend < b.Backend
		})
}

// pct renders n/total as a 0-100 score (100 when total is 0, so an empty axis is clean).
func pct(n, total int) float64 {
	if total == 0 {
		return 100
	}
	return 100 * float64(n) / float64(total)
}

// undefinedCorrectnessKPI builds the shared "no undefined cells" correctness KPI that both
// the 2-D matrix (Build) and the 3-D cross tensor (BuildX) emit. key is the KPI id, cellNoun
// names the cell kind in the Detail line (e.g. "(family,backend) cell(s)"), total is the whole
// axis size, and defectLabels are the pre-formatted "A × B[ × C]" labels for each undefined
// cell (empty/nil when clean — the cell types differ, so each caller formats its own labels).
// The Group, Detail, Score, and per-defect suffix are byte-identical to the form both callers
// inlined before this extraction.
func undefinedCorrectnessKPI(key, cellNoun string, total int, defectLabels []string) scorecard.KPI {
	kpi := scorecard.KPI{
		Key:    key,
		Group:  "correctness",
		Detail: fmt.Sprintf("%d %s reachable with neither a fence nor a CI witness", len(defectLabels), cellNoun),
		Score:  pct(total-len(defectLabels), total),
	}
	for _, label := range defectLabels {
		kpi.Defects = append(kpi.Defects, label+" is reachable but neither fenced nor witnessed")
	}
	return kpi
}
