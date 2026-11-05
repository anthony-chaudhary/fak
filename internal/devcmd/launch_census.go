// launch-census — static census of the kernel-launch and parallel-for
// dispatch events a decoded token issues through the public engine's decode
// paths, classified per-token / per-layer / per-request with multiplicity
// factors (the active-experts factor for MoE expert dispatch).
//
// The ad-hoc method this replaces is the prose estimate in
// docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md lever 2 ("one parFor
// across all active experts per proj, not ~24 tiny parFors/layer", "~1.8x"):
// a hand-wave that cannot be recomputed after a refactor, cannot flag a NEW
// dispatch site added to the decode path, and cannot say how many launches a
// batching change actually removes per decoded token. This census converts it
// into a version-controlled, recomputable, trendable metric:
//
//   - a closed inventory of the decode-path launch sites (MoE expert GEMVs in
//     BOTH their looped and batched representations, the router GEMVs, the
//     GLM-DSA attention/indexer device ops) plus the universal dispatch
//     primitives (parFor / parForRange / the backend MatMul submit seam / the
//     CUDA H2D/D2H copy plumbing); each pinned to file + function + source
//     pattern;
//   - verification against the LIVE source (go/ast, reusing the glue-census
//     scanner): a vanished function or pattern is DRIFT and fails the run —
//     when a batching change lands, the looped rows it retires drift loudly
//     instead of the metric silently rotting;
//   - discovery of UNCOVERED launch markers (parFor( / parForRange( /
//     be.MatMul( / fcuda copies) in the decode-path files that no inventory
//     row pins — the new-regression signal;
//   - shape-parameterized math: per-class layer counters + the active-experts
//     (top-k) factor turn the inventory into launches/token for ANY model
//     shape (--model-shape JSON or named presets incl. glm-dsa), reported for
//     BOTH the looped and the batched expert-dispatch variant so the lever-2
//     headroom is a computed delta, not a prose multiplier;
//   - a cross-check mode that compares the computed looped expert-dispatch
//     count per MoE layer against the documented ~24 parFors/layer reference
//     (the delta is reported, never hardcoded).
//
// Scope note: CUDA <<<grid,block>>> sites live in the .cu/.c sources behind
// the cgo wrappers; the census pins the Go-side submit seams (one wrapper
// call = one device submission funnel), which is where the decode path's
// launch cadence is visible from Go.

package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
)

// LaunchClass is the cadence class of an inventoried launch site. Per-layer
// classes scale with the matching model-shape counter; per-active-expert
// scales with ActiveExperts x MoELayers; per-token / per-request are constant
// per decoded token / request; primitive rows are drift guards only (Factor 0
// — they name the dispatch plumbing and are NOT summed).
type LaunchClass string

const (
	LaunchPerIndexLayer   LaunchClass = "per-index-layer"   // glmDsaIndexStep projections + indexSelect
	LaunchPerKVLayer      LaunchClass = "per-kv-layer"      // glmDsaAppendAttentionKV projections
	LaunchPerAttendLayer  LaunchClass = "per-attend-layer"  // sparse attend + o_proj
	LaunchPerMoELayer     LaunchClass = "per-moe-layer"     // router GEMV, batched expert rows
	LaunchPerActiveExpert LaunchClass = "per-active-expert" // looped expert GEMVs: x ActiveExperts per MoE layer
	LaunchPerToken        LaunchClass = "per-token"
	LaunchPerRequest      LaunchClass = "per-request"
	LaunchPrimitive       LaunchClass = "primitive"
)

// LaunchVariant distinguishes mutually exclusive dispatch representations of
// the SAME logical work: "looped" (one dispatch per active expert x
// projection — today's documented regime) vs "batched" (lever 2: one dispatch
// per projection across all active experts). "" = shared by both variants.
const (
	variantPerExpert    = "looped"
	variantBatched      = "batched"
	launchVariantShared = ""
)

// LaunchCallSite is one version-controlled inventory row. Pattern must occur
// in the enclosing function's source text (or anywhere in the file when Func
// is empty) — verification re-checks this against the live tree on every run.
type LaunchCallSite struct {
	ID      string      `json:"id"`
	File    string      `json:"file"`
	Func    string      `json:"func"` // "" = file-level
	Pattern string      `json:"pattern"`
	Class   LaunchClass `json:"class"`
	Variant string      `json:"variant,omitempty"`
	Factor  int         `json:"factor"` // launches contributed per unit of Class
	Note    string      `json:"note"`
}

// launchCensusInventory is the closed set of decode-path launch sites for the
// small-batch DECODE regime documented in GLM52-DECODE-PATH-TO-10-TOKS
// (hybrid cpu-offload serve, cuda backend, host-resident experts). The looped
// expert rows reproduce the documented breakdown exactly: gate+up (mulGroup
// loops mat.mul -> one parFor per projection on non-fused kernels) + down =
// 3 dispatches per active expert, i.e. 3*K per MoE layer (~24 at K=8).
var launchCensusInventory = []LaunchCallSite{
	// GLM-DSA learned-indexer step: 4 projection GEMMs + device top-k select.
	{ID: "glm-dsa-index-qa", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "q_a_proj.weight", Class: LaunchPerIndexLayer, Factor: 1,
		Note: "indexer q_a GEMM device launch"},
	{ID: "glm-dsa-index-wqb", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.wq_b.weight", Class: LaunchPerIndexLayer, Factor: 1,
		Note: "indexer wq_b GEMM device launch"},
	{ID: "glm-dsa-index-wk", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.wk.weight", Class: LaunchPerIndexLayer, Factor: 1,
		Note: "indexer wk GEMM device launch"},
	{ID: "glm-dsa-index-weights-proj", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.weights_proj.weight", Class: LaunchPerIndexLayer, Factor: 1,
		Note: "indexer weights_proj GEMM device launch"},
	{ID: "glm-dsa-index-select", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "ik.indexSelect(", Class: LaunchPerIndexLayer, Factor: 1,
		Note: "device indexer score + top-k selection op launch"},
	// GLM-DSA MLA KV append: 4 dense projections.
	{ID: "glm-dsa-kv-qa", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "q_a_proj.weight", Class: LaunchPerKVLayer, Factor: 1,
		Note: "kv-step q_a GEMM device launch (second computation of q_a per layer)"},
	{ID: "glm-dsa-kv-qb", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "q_b_proj.weight", Class: LaunchPerKVLayer, Factor: 1,
		Note: "kv-step q_b GEMM device launch"},
	{ID: "glm-dsa-kv-kva", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "kv_a_proj_with_mqa.weight", Class: LaunchPerKVLayer, Factor: 1,
		Note: "compressed-KV GEMM device launch"},
	{ID: "glm-dsa-kv-kvb", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "kv_b_proj.weight", Class: LaunchPerKVLayer, Factor: 1,
		Note: "decompressed K/V GEMM device launch"},
	// GLM-DSA sparse attention compute + output projection.
	{ID: "glm-dsa-attend-sparse", File: "internal/model/glm_dsa_session.go", Func: "glmDsaSparseAttend",
		Pattern: "sk.sparseAttend(", Class: LaunchPerAttendLayer, Factor: 1,
		Note: "device sparse-attention op launch"},
	{ID: "glm-dsa-attend-oproj", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAttendCached",
		Pattern: "o_proj.weight", Class: LaunchPerAttendLayer, Factor: 1,
		Note: "attention output projection GEMM device launch"},
	// MoE router GEMV: exactly ONE launch per MoE layer. A model routes through
	// exactly one flavor (route OR glmRoute), so only the GLM-DSA-regime flavor
	// is COUNTED; the other stays a drift guard (Factor 0) so the census never
	// double-counts the router while still failing if it vanishes.
	{ID: "moe-router-route", File: "internal/model/moe.go", Func: "route",
		Pattern: "mat.mul(routerName(layer)", Class: LaunchPrimitive, Factor: 0,
		Note: "router GEMV (Mixtral/Qwen3-MoE flavor): drift guard — counted once via moe-router-glmroute"},
	{ID: "moe-router-glmroute", File: "internal/model/moe.go", Func: "glmRoute",
		Pattern: "mat.mul(routerName(layer)", Class: LaunchPerMoELayer, Factor: 1,
		Note: "router GEMV (GLM grouped-routing flavor): the one router launch per MoE layer"},
	// Looped expert dispatch (the documented regime lever 2 targets): one
	// dispatch per active expert x projection.
	{ID: "moe-expert-gateup-looped", File: "internal/model/moe.go", Func: "expertSwiGLU",
		Pattern: "mulGroup(mat, []string{gn, un}", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 2,
		Note: "gate+up GEMVs: mulGroup loops mat.mul -> one parFor per projection on non-fused kernels"},
	{ID: "moe-expert-down-looped", File: "internal/model/moe.go", Func: "expertSwiGLU",
		Pattern: "mat.mul(dn", Class: LaunchPerActiveExpert, Variant: variantPerExpert, Factor: 1,
		Note: "down GEMV: exactly one dispatch per expert (both branch arms)"},
	// Batched expert dispatch (lever 2, shipped for host-resident experts):
	// ONE parFor per projection across all K active experts' rows.
	{ID: "moe-expert-batched-rows", File: "internal/model/moe_host_batch.go", Func: "batchExpertRows",
		Pattern: "parForRange(total, total*inDim, body)", Class: LaunchPerMoELayer, Variant: variantBatched, Factor: 3,
		Note: "lever 2: one parFor per projection (gate/up/down) over the flattened K-expert row space"},
	// Primitives: the dispatch plumbing itself. Drift guards only (Factor 0) —
	// if these vanish the census is describing dead code.
	{ID: "parallel-parfor", File: "internal/model/parallel.go", Func: "parFor",
		Pattern: "func parFor(n, workers int, body func(lo, hi int))", Class: LaunchPrimitive, Factor: 0,
		Note: "host thread-pool dispatch primitive (every host GEMV funnels through it)"},
	{ID: "parallel-parforrange", File: "internal/model/parallel.go", Func: "parForRange",
		Pattern: "func parForRange(n, work int, body func(lo, hi int))", Class: LaunchPrimitive, Factor: 0,
		Note: "work-weighted host dispatch primitive (the quantized GEMV kernels)"},
	{ID: "backend-matmul-submit", File: "internal/model/kernel.go", Func: "mul",
		Pattern: "be.Read(be.MatMul(wt, xt))", Class: LaunchPrimitive, Factor: 0,
		Note: "backendKernel.mul: the per-GEMV backend submit seam"},
	{ID: "moe-expert-batched-q4k-scaffold", File: "internal/model/moe_host_batch.go", Func: "q4kBatchRows",
		Pattern: "batchExpertRows(K, rowsPer", Class: LaunchPrimitive, Variant: variantBatched, Factor: 0,
		Note: "q4k batched gate/up rows: funnels into the single batchExpertRows parFor"},
	{ID: "moe-expert-batched-kquant-scaffold", File: "internal/model/moe_host_batch.go", Func: "kQuantBatchRows",
		Pattern: "batchExpertRows(K, rowsPer", Class: LaunchPrimitive, Variant: variantBatched, Factor: 0,
		Note: "k-quant batched down rows: funnels into the single batchExpertRows parFor"},
	{ID: "cuda-copy-h2d", File: "internal/compute/cuda.go", Func: "",
		Pattern: "C.fcuda_h2d(", Class: LaunchPrimitive, Factor: 0,
		Note: "host->device copy submission (all upload paths funnel through it)"},
	{ID: "cuda-copy-d2h", File: "internal/compute/cuda.go", Func: "",
		Pattern: "C.fcuda_d2h_on_checked(", Class: LaunchPrimitive, Factor: 0,
		Note: "device->host copy submission (Read's stream-syncing drain)"},
	{ID: "cuda-matmul-submit", File: "internal/compute/cuda.go", Func: "cudaMatMulLocked",
		Pattern: "C.fcuda_matmul_f32(", Class: LaunchPrimitive, Factor: 0,
		Note: "CUDA GEMM kernel submission (the f32 arm of the quant-arm ladder in cudaBackend's locked MatMul path)"},
}

// LaunchShape parameterizes the census math. Layer-class counters default to
// Layers when zero (the documented assumption: every layer pays that step).
// ActiveExperts is the MoE top-k; it must be explicit whenever MoELayers > 0.
type LaunchShape struct {
	Name            string `json:"name,omitempty"`
	Layers          int    `json:"layers"`
	IndexStepLayers int    `json:"index_step_layers,omitempty"`
	KVAppendLayers  int    `json:"kv_append_layers,omitempty"`
	AttendLayers    int    `json:"attend_layers,omitempty"`
	MoELayers       int    `json:"moe_layers,omitempty"`
	ActiveExperts   int    `json:"active_experts,omitempty"`
}

// normalize fills defaulted layer counters and validates them. It returns the
// normalized copy so callers keep the input untouched.
func (s LaunchShape) normalize() (LaunchShape, error) {
	out := s
	if out.Layers <= 0 {
		return out, fmt.Errorf("model shape needs layers >= 1 (got %d)", out.Layers)
	}
	check := func(name string, v *int) error {
		if *v == 0 {
			*v = out.Layers
		}
		if *v < 0 || *v > out.Layers {
			return fmt.Errorf("%s=%d out of range [0, layers=%d]", name, *v, out.Layers)
		}
		return nil
	}
	for _, pair := range []struct {
		name string
		v    *int
	}{
		{"index_step_layers", &out.IndexStepLayers},
		{"kv_append_layers", &out.KVAppendLayers},
		{"attend_layers", &out.AttendLayers},
	} {
		if err := check(pair.name, pair.v); err != nil {
			return out, err
		}
	}
	// MoE layers: an explicit moe_layers=0 WITH active_experts=0 is a dense-only
	// shape (no MoE work at all); a zero moe_layers with a stated top-k means
	// "every layer is MoE" (the documented assumption). Anything else must be
	// in range.
	if out.MoELayers == 0 && out.ActiveExperts > 0 {
		out.MoELayers = out.Layers
	}
	if out.MoELayers < 0 || out.MoELayers > out.Layers {
		return out, fmt.Errorf("moe_layers=%d out of range [0, layers=%d]", out.MoELayers, out.Layers)
	}
	if out.MoELayers > 0 && out.ActiveExperts < 1 {
		return out, fmt.Errorf("active_experts must be >= 1 when moe_layers=%d (the MoE top-k is a census input, not a guess)", out.MoELayers)
	}
	return out, nil
}

// launchCensusPresets are the named shapes. glm-dsa mirrors the documented
// GLM-5.2-shaped scenario (92 layers, ~89 MoE layers, top-k 8 — the exact
// assumptions behind the ~24 parFors/MoE-layer figure); tiny-4 exists for
// shape-math regression tests.
var launchCensusPresets = map[string]LaunchShape{
	"glm-dsa": {Name: "glm-dsa", Layers: 92, MoELayers: 89, ActiveExperts: 8},
	"tiny-4":  {Name: "tiny-4", Layers: 4, MoELayers: 4, ActiveExperts: 2},
}

// documentedGLMDsaExpertParFors is the hand-counted reference from
// docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md lever 2 ("~24 tiny
// parFors/layer"). It is a REFERENCE the census checks itself against — the
// computed figure always comes from inventory x shape, never from this
// constant.
const (
	documentedGLMDsaExpertParFors = 24
	// launchCensusGLMDsaRefDoc names the same note as glue-census's
	// documentedGLMDsaRefDoc (a package-level const cannot be shared under two
	// names without a rename); the string is the reference for --check.
	launchCensusGLMDsaRefDoc = "docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md"
)

// computeLaunches turns the inventory into estimated launch events per decoded
// token for the given shape and dispatch variant. Rows whose Variant is set
// and differs from the requested variant are excluded (looped and batched are
// mutually exclusive representations of the expert dispatch); primitive rows
// contribute nothing.
func computeLaunches(inv []LaunchCallSite, shape LaunchShape, variant string) int {
	total := 0
	for _, s := range inv {
		if s.Variant != launchVariantShared && s.Variant != variant {
			continue
		}
		switch s.Class {
		case LaunchPerIndexLayer:
			total += s.Factor * shape.IndexStepLayers
		case LaunchPerKVLayer:
			total += s.Factor * shape.KVAppendLayers
		case LaunchPerAttendLayer:
			total += s.Factor * shape.AttendLayers
		case LaunchPerMoELayer:
			total += s.Factor * shape.MoELayers
		case LaunchPerActiveExpert:
			total += s.Factor * shape.ActiveExperts * shape.MoELayers
		case LaunchPerToken, LaunchPerRequest:
			total += s.Factor
		default: // LaunchPrimitive and unknown classes: drift-guard rows only.
		}
	}
	return total
}

// expertDispatchPerMoELayer returns the computed expert-dispatch launch count
// for ONE MoE layer under the given variant: the sum of that variant's
// per-active-expert factors times ActiveExperts (looped) or the sum of its
// per-moe-layer factors (batched).
func expertDispatchPerMoELayer(inv []LaunchCallSite, shape LaunchShape, variant string) int {
	sum := 0
	for _, s := range inv {
		if s.Variant != variant {
			continue
		}
		switch s.Class {
		case LaunchPerActiveExpert:
			sum += s.Factor * shape.ActiveExperts
		case LaunchPerMoELayer:
			sum += s.Factor
		default:
			// Other and unknown cadence classes do not contribute to this expert-only subtotal.
			continue
		}
	}
	return sum
}

// LaunchDrift is one inventory row that no longer matches the live source.
type LaunchDrift struct {
	Site        LaunchCallSite `json:"site"`
	Reason      string         `json:"reason"` // file-missing | parse-error | func-missing | pattern-absent
	Occurrences int            `json:"occurrences"`
}

// verifyLaunchInventory re-checks every inventory row against the live tree
// (reusing the glue-census goFileScan) and returns both the drift findings and
// the per-row occurrence counts.
func verifyLaunchInventory(root string, inv []LaunchCallSite) ([]LaunchDrift, map[string]int) {
	fileScanMap := map[string]*goFileScan{}
	get := func(rel string) *goFileScan {
		if sc, ok := fileScanMap[rel]; ok {
			return sc
		}
		sc, _ := scanGoFile(root, rel)
		fileScanMap[rel] = sc
		return sc
	}
	var drift []LaunchDrift
	occ := map[string]int{}
	for _, site := range inv {
		sc := get(site.File)
		if sc.src == nil {
			drift = append(drift, LaunchDrift{Site: site, Reason: "file-missing"})
			continue
		}
		if sc.parseErr != nil {
			drift = append(drift, LaunchDrift{Site: site, Reason: "parse-error"})
			continue
		}
		var text string
		if site.Func == "" {
			text = string(sc.src)
		} else if ft, ok := sc.funcText(site.Func); ok {
			text = ft
		} else {
			drift = append(drift, LaunchDrift{Site: site, Reason: "func-missing"})
			continue
		}
		n := strings.Count(text, site.Pattern)
		occ[site.ID] = n
		if n == 0 {
			drift = append(drift, LaunchDrift{Site: site, Reason: "pattern-absent"})
		}
	}
	return drift, occ
}

// launchCensusDiscoverFiles is the version-controlled decode-path file set the
// uncovered-site scan covers. Extend it deliberately when the decode path
// gains new launch-emitting files.
var launchCensusDiscoverFiles = []string{
	"internal/model/moe.go",
	"internal/model/moe_host_batch.go",
	"internal/model/glm_dsa_session.go",
	"internal/model/kernel.go",
	"internal/model/hal.go",
	"internal/model/parallel.go",
	"internal/compute/cuda.go",
}

// launchCensusMarkers are the high-signal Go-visible launch/dispatch
// primitives. A hit inside a file region with NO inventory coverage is an
// UNCOVERED candidate — the new-regression signal. (CUDA <<<grid,block>>>
// sites live in the .cu sources behind these cgo wrappers, so the Go-side
// wrapper calls are the census boundary.)
var launchCensusMarkers = []string{
	"parFor(",
	"parForRange(",
	"be.MatMul(",
	"C.fcuda_h2d",
	"C.fcuda_d2h",
}

// LaunchUncovered is a launch-marker hit the inventory does not pin.
type LaunchUncovered struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Marker  string `json:"marker"`
	Snippet string `json:"snippet"`
}

// discoverLaunchUncovered scans the decode-path files for launch markers whose
// lines no inventory row covers (a line is covered when it falls inside an
// inventoried function span, or the file carries a file-level inventory row).
func discoverLaunchUncovered(root string, inv []LaunchCallSite) []LaunchUncovered {
	type coverage struct {
		fileLevel bool
		spans     [][2]int // line-based [startLine, endLine]
	}
	coveredBy := map[string]*coverage{}
	fileScanMap := map[string]*goFileScan{}
	for _, rel := range launchCensusDiscoverFiles {
		sc, err := scanGoFile(root, rel)
		if err != nil || sc.src == nil || sc.parseErr != nil {
			continue
		}
		fileScanMap[rel] = sc
		starts := sc.lineStarts()
		lineOf := func(off int) int {
			lo, hi := 0, len(starts)-1
			for lo <= hi {
				mid := (lo + hi) / 2
				if starts[mid] <= off {
					lo = mid + 1
				} else {
					hi = mid - 1
				}
			}
			return lo // 1-based line number
		}
		cov := coveredBy[rel]
		if cov == nil {
			cov = &coverage{}
			coveredBy[rel] = cov
		}
		for _, site := range inv {
			if site.File != rel {
				continue
			}
			if site.Func == "" {
				cov.fileLevel = true
				continue
			}
			for _, sp := range sc.funcSpans[site.Func] {
				cov.spans = append(cov.spans, [2]int{lineOf(sp[0]), lineOf(sp[1])})
			}
		}
	}
	inSpan := func(spans [][2]int, line int) bool {
		for _, sp := range spans {
			if line >= sp[0] && line <= sp[1] {
				return true
			}
		}
		return false
	}
	var out []LaunchUncovered
	for _, rel := range launchCensusDiscoverFiles {
		sc, ok := fileScanMap[rel]
		if !ok {
			continue
		}
		cov := coveredBy[rel]
		lines := strings.Split(string(sc.src), "\n")
		for i, line := range lines {
			lineNo := i + 1
			if (cov != nil && cov.fileLevel) || (cov != nil && inSpan(cov.spans, lineNo)) {
				continue
			}
			for _, m := range launchCensusMarkers {
				if strings.Contains(line, m) {
					snippet := strings.TrimSpace(line)
					if len(snippet) > 100 {
						snippet = snippet[:100] + "…"
					}
					out = append(out, LaunchUncovered{File: rel, Line: lineNo, Marker: m, Snippet: snippet})
					break // one report per line
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// LaunchCallSiteRow is an inventory row joined with its live occurrence count.
type LaunchCallSiteRow struct {
	LaunchCallSite
	Occurrences int `json:"occurrences"`
}

// LaunchCensusResult is the full census payload (JSON form of the report).
type LaunchCensusResult struct {
	Schema                   string              `json:"schema"`
	Root                     string              `json:"root"`
	HeadSHA                  string              `json:"head_sha,omitempty"`
	Shape                    LaunchShape         `json:"shape"`
	TotalLooped              int                 `json:"total_launches_per_token_loop"`
	TotalBatched             int                 `json:"total_launches_per_token_batched"`
	DeltaBatchedVsLooped     int                 `json:"delta_batched_vs_loop"`
	ExpertPerMoELayerLooped  int                 `json:"expert_dispatch_per_moe_layer_loop"`
	ExpertPerMoELayerBatched int                 `json:"expert_dispatch_per_moe_layer_batched"`
	Sites                    []LaunchCallSiteRow `json:"sites"`
	Drift                    []LaunchDrift       `json:"drift"`
	Uncovered                []LaunchUncovered   `json:"uncovered"`
	CrossCheck               *GlueCrossCheckRes  `json:"cross_check,omitempty"`
}

const launchCensusUsage = `usage: fak-dev launch-census [--root DIR] [--preset NAME | --model-shape JSON] [--check [REF]] [--no-discover] [--snapshot PATH] [--json]

Census the kernel-launch / parallel-for dispatch events a decoded token issues
on the public engine's decode paths: verifies the version-controlled call-site
inventory against the live source, computes launches/token for a model shape in
BOTH the looped and the batched expert-dispatch variant, lists uncovered launch
candidates, and optionally cross-checks the computed looped expert-dispatch
count per MoE layer against the documented ~24 parFors/layer reference (±20%).

Exit codes: 0 PASS, 1 drift or cross-check FAIL, 2 usage/IO error.`

// RunLaunchCensus implements the fak-dev launch-census verb.
func RunLaunchCensus(stdout, stderr io.Writer, argv []string) int {
	root := repoRoot()
	preset := "glm-dsa"
	shapeJSON := ""
	check := false
	checkRef := 0
	noDiscover := false
	snapshot := ""
	asJSON := false
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, launchCensusUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "launch-census: --root needs a value")
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		case arg == "--preset":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "launch-census: --preset needs a value")
				return 2
			}
			i++
			preset = argv[i]
		case strings.HasPrefix(arg, "--preset="):
			preset = strings.TrimPrefix(arg, "--preset=")
		case arg == "--model-shape":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "launch-census: --model-shape needs a JSON value")
				return 2
			}
			i++
			shapeJSON = argv[i]
		case strings.HasPrefix(arg, "--model-shape="):
			shapeJSON = strings.TrimPrefix(arg, "--model-shape=")
		case arg == "--check":
			check = true
			// Optional space-separated reference (--check REF); a following flag
			// or non-numeric token leaves the documented-reference default in place.
			if i+1 < len(argv) {
				if v, err := strconv.Atoi(argv[i+1]); err == nil && v > 0 {
					checkRef = v
					i++
				}
			}
		case strings.HasPrefix(arg, "--check="):
			check = true
			v, err := strconv.Atoi(strings.TrimPrefix(arg, "--check="))
			if err != nil || v <= 0 {
				fmt.Fprintln(stderr, "launch-census: --check=REF needs a positive integer reference")
				return 2
			}
			checkRef = v
		case arg == "--no-discover":
			noDiscover = true
		case arg == "--snapshot":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "launch-census: --snapshot needs a value")
				return 2
			}
			i++
			snapshot = argv[i]
		case strings.HasPrefix(arg, "--snapshot="):
			snapshot = strings.TrimPrefix(arg, "--snapshot=")
		case arg == "--json":
			asJSON = true
		default:
			fmt.Fprintln(stderr, launchCensusUsage)
			return 2
		}
	}

	var shape LaunchShape
	switch {
	case shapeJSON != "":
		if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
			fmt.Fprintf(stderr, "launch-census: parse --model-shape: %v\n", err)
			return 2
		}
		if shape.Name == "" {
			shape.Name = "custom"
		}
	default:
		ps, ok := launchCensusPresets[preset]
		if !ok {
			names := make([]string, 0, len(launchCensusPresets))
			for n := range launchCensusPresets {
				names = append(names, n)
			}
			sort.Strings(names)
			fmt.Fprintf(stderr, "launch-census: unknown preset %q (known: %s)\n", preset, strings.Join(names, ", "))
			return 2
		}
		shape = ps
	}
	normed, err := shape.normalize()
	if err != nil {
		fmt.Fprintf(stderr, "launch-census: invalid shape: %v\n", err)
		return 2
	}
	// Normalize BEFORE the math: presets like glm-dsa leave the per-class layer
	// counters zero and rely on the layers default, so computing on the raw
	// shape would silently yield 0.
	shape = normed

	drift, occ := verifyLaunchInventory(root, launchCensusInventory)
	looped := computeLaunches(launchCensusInventory, shape, variantPerExpert)
	batched := computeLaunches(launchCensusInventory, shape, variantBatched)
	var uncovered []LaunchUncovered
	if !noDiscover {
		uncovered = discoverLaunchUncovered(root, launchCensusInventory)
	}

	var xc *GlueCrossCheckRes
	if check {
		ref := checkRef
		doc := ""
		if ref == 0 {
			if shape.Name != "glm-dsa" {
				fmt.Fprintln(stderr, "launch-census: non-glm-dsa shapes need an explicit --check REF")
				return 2
			}
			ref = documentedGLMDsaExpertParFors
			doc = launchCensusGLMDsaRefDoc
		}
		computed := expertDispatchPerMoELayer(launchCensusInventory, shape, variantPerExpert)
		res := crossCheck(computed, ref)
		res.ReferenceDoc = doc
		xc = &res
	}

	result := LaunchCensusResult{
		Schema:                   "fak.dev.launch-census.v1",
		Root:                     root,
		HeadSHA:                  gitHeadSHA(root),
		Shape:                    shape,
		TotalLooped:              looped,
		TotalBatched:             batched,
		DeltaBatchedVsLooped:     batched - looped,
		ExpertPerMoELayerLooped:  expertDispatchPerMoELayer(launchCensusInventory, shape, variantPerExpert),
		ExpertPerMoELayerBatched: expertDispatchPerMoELayer(launchCensusInventory, shape, variantBatched),
		Drift:                    drift,
		Uncovered:                uncovered,
		CrossCheck:               xc,
	}
	for _, s := range launchCensusInventory {
		result.Sites = append(result.Sites, LaunchCallSiteRow{LaunchCallSite: s, Occurrences: occ[s.ID]})
	}

	if snapshot != "" {
		// The committed snapshot is machine-independent: the root path is where
		// THIS run happened, not part of the censused fact — head_sha pins the
		// tree.
		snap := result
		snap.Root = ""
		data, err := json.MarshalIndent(snap, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "launch-census: marshal snapshot: %v\n", err)
			return 2
		}
		if err := os.MkdirAll(filepath.Dir(snapshot), 0o755); err != nil {
			fmt.Fprintf(stderr, "launch-census: mkdir snapshot dir: %v\n", err)
			return 2
		}
		if err := os.WriteFile(snapshot, append(data, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "launch-census: write snapshot: %v\n", err)
			return 2
		}
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "launch-census: encode: %v\n", err)
			return 2
		}
	} else {
		writeLaunchCensusText(stdout, &result)
	}

	if len(drift) > 0 {
		return 1
	}
	if xc != nil && !xc.Pass {
		return 1
	}
	return 0
}

func writeLaunchCensusText(w io.Writer, r *LaunchCensusResult) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "LAUNCH-CENSUS\troot=%s shape=%s(layers=%d index=%d kv=%d attend=%d moe=%d experts=%d)\n",
		r.Root, r.Shape.Name, r.Shape.Layers, r.Shape.IndexStepLayers, r.Shape.KVAppendLayers,
		r.Shape.AttendLayers, r.Shape.MoELayers, r.Shape.ActiveExperts)
	fmt.Fprintf(tw, "CALL SITE\tCLASS\tVARIANT\tFACTOR\tOCC\n")
	for _, s := range r.Sites {
		label := s.File + ":" + s.Func + "[" + s.Pattern + "]"
		if s.Func == "" {
			label = s.File + "[" + s.Pattern + "]"
		}
		if len(label) > 60 {
			label = label[:57] + "..."
		}
		variant := s.Variant
		if variant == "" {
			variant = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%d\n", label, s.Class, variant, s.Factor, s.Occurrences)
	}
	fmt.Fprintf(tw, "TOTAL estimated kernel launches / decoded token\tLOOPED\t%d\n", r.TotalLooped)
	deltaPct := 0.0
	if r.TotalLooped > 0 {
		deltaPct = float64(r.DeltaBatchedVsLooped) / float64(r.TotalLooped) * 100
	}
	fmt.Fprintf(tw, "\tBATCHED\t%d (%+.1f%%, lever-2 delta %d)\n", r.TotalBatched, deltaPct, r.DeltaBatchedVsLooped)
	fmt.Fprintf(tw, "expert-dispatch per MoE layer\tLOOPED\t%d\n", r.ExpertPerMoELayerLooped)
	fmt.Fprintf(tw, "\tBATCHED\t%d\n", r.ExpertPerMoELayerBatched)
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(w, "(table flush failed: %v)\n", err)
	}
	if r.CrossCheck != nil {
		xc := r.CrossCheck
		status := "PASS"
		if !xc.Pass {
			status = "FAIL"
		}
		doc := ""
		if xc.ReferenceDoc != "" {
			doc = " (" + xc.ReferenceDoc + ")"
		}
		fmt.Fprintf(w, "cross-check vs documented ref=%d parFors/MoE-layer%s: %s delta=%+.2f%% tol=±%d%%\n",
			xc.Reference, doc, status, xc.DeltaPct, xc.TolerancePct)
	}
	if len(r.Drift) > 0 {
		fmt.Fprintf(w, "DRIFT (%d inventory row(s) no longer match the live source):\n", len(r.Drift))
		for _, d := range r.Drift {
			fmt.Fprintf(w, "  %-58s %-14s %s\n", d.Site.File+":"+d.Site.Func+"["+d.Site.Pattern+"]", d.Reason, d.Site.Note)
		}
	}
	if len(r.Uncovered) > 0 {
		fmt.Fprintf(w, "UNCOVERED launch candidates (%d, informational — review and extend the inventory if they are decode-path dispatches):\n", len(r.Uncovered))
		for _, u := range r.Uncovered {
			fmt.Fprintf(w, "  %s:%d [%s] %s\n", u.File, u.Line, u.Marker, u.Snippet)
		}
	}
}
