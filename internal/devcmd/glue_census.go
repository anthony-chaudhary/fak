// glue-census — static census of the synchronous host<->device round-trips a
// decoded token issues through the GLM-DSA device seams.
//
// The ad-hoc method this replaces is the hand-counted prose figure in
// docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md ("~12 synchronous
// device<->host round-trips per layer ... ~1100 per decoded token"). That number
// was a one-off estimate: it cannot be recomputed after a refactor, cannot flag a
// NEW sync site added to the decode path, and cannot be checked against another
// model shape. This census converts it into a version-controlled, recomputable,
// trendable metric:
//
//   - a closed inventory of the synchronous call sites (one row per projection/op
//     seam + the underlying H2D/D2H primitives), each pinned to file + function +
//     source pattern;
//   - verification against the LIVE source (go/ast): a missing function or a
//     vanished pattern is reported as DRIFT and fails the run — the metric can no
//     longer silently rot the way a prose number does;
//   - discovery of UNCOVERED sync primitives (be.Read / fcuda_h2d / fcuda_d2h /
//     cudaMemcpy / ...) in the decode-path files that no inventory row pins — the
//     new-regression signal;
//   - shape-parameterized math: per-class layer counts turn the inventory into
//     round-trips/token for ANY model shape (--model-shape JSON or named presets);
//   - a cross-check mode that asserts the computed figure lands within 20% of the
//     documented ~1100 reference (the delta is reported, never hardcoded).
//
// Measured mode lives in internal/model (FAK_GLUE_CENSUS=1): the same seams count
// themselves when a real device forward runs, so future hardware receipts carry
// measured values beside this static estimate.

package devcmd

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// GlueClass is the cadence class of an inventoried synchronous call site.
// Per-layer classes scale with the matching model-shape counter; per-token and
// per-request are constant per decoded token / request; primitive rows are
// drift guards only (they name the H2D/D2H plumbing and are NOT summed).
type GlueClass string

const (
	GluePerIndexLayer  GlueClass = "per-index-layer"  // glmDsaIndexStep projections + indexSelect
	GluePerKVLayer     GlueClass = "per-kv-layer"     // glmDsaAppendAttentionKV projections
	GluePerAttendLayer GlueClass = "per-attend-layer" // sparse attend + o_proj
	GluePerRouterLayer GlueClass = "per-router-layer" // MoE router GEMM
	GluePerToken       GlueClass = "per-token"
	GluePerRequest     GlueClass = "per-request"
	GluePrimitive      GlueClass = "primitive"
)

// GlueCallSite is one version-controlled inventory row. Pattern must occur in
// the enclosing function's source text (or anywhere in the file when Func is
// empty) — verification re-checks this against the live tree on every run.
type GlueCallSite struct {
	ID      string    `json:"id"`
	File    string    `json:"file"`
	Func    string    `json:"func"` // "" = file-level
	Pattern string    `json:"pattern"`
	Class   GlueClass `json:"class"`
	Factor  int       `json:"factor"` // round-trips contributed per unit of Class
	Note    string    `json:"note"`
}

// glueCensusInventory is the closed set of synchronous host<->device call sites
// on the GLM-DSA small-batch DECODE path (hybrid cpu-offload serve, cuda
// backend). The twelve counted rows reproduce the documented breakdown exactly:
// glmDsaIndexStep 4 projections + indexSelect, glmDsaAppendAttentionKV 4
// projections, glmDsaAttendCached sparseAttend + o_proj, and the MoE router.
// q_a_proj is intentionally listed twice — it is computed once in the index
// step AND once in the KV step (the doc calls this out explicitly).
var glueCensusInventory = []GlueCallSite{
	// glmDsaIndexStep: learned-indexer projections + device top-k selection.
	{ID: "glm-dsa-index-q-a", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "q_a_proj.weight", Class: GluePerIndexLayer, Factor: 1,
		Note: "indexer q_a GEMM (H2D activation + MatMul + D2H read)"},
	{ID: "glm-dsa-index-wq-b", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.wq_b.weight", Class: GluePerIndexLayer, Factor: 1,
		Note: "indexer wq_b GEMM"},
	{ID: "glm-dsa-index-wk", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.wk.weight", Class: GluePerIndexLayer, Factor: 1,
		Note: "indexer wk GEMM"},
	{ID: "glm-dsa-index-weights-proj", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "indexer.weights_proj.weight", Class: GluePerIndexLayer, Factor: 1,
		Note: "indexer weights_proj GEMM"},
	{ID: "glm-dsa-index-select", File: "internal/model/glm_dsa_session.go", Func: "glmDsaIndexStep",
		Pattern: "ik.indexSelect(", Class: GluePerIndexLayer, Factor: 1,
		Note: "device indexer score + top-k selection (fenced readback of the selection)"},
	// glmDsaAppendAttentionKV: MLA dense projections.
	{ID: "glm-dsa-kv-q-a", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "q_a_proj.weight", Class: GluePerKVLayer, Factor: 1,
		Note: "kv-step q_a GEMM (second computation of q_a per layer)"},
	{ID: "glm-dsa-kv-q-b", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "q_b_proj.weight", Class: GluePerKVLayer, Factor: 1,
		Note: "kv-step q_b GEMM"},
	{ID: "glm-dsa-kv-kv-a", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "kv_a_proj_with_mqa.weight", Class: GluePerKVLayer, Factor: 1,
		Note: "compressed-KV GEMM"},
	{ID: "glm-dsa-kv-kv-b", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAppendAttentionKV",
		Pattern: "kv_b_proj.weight", Class: GluePerKVLayer, Factor: 1,
		Note: "decompressed K/V GEMM"},
	// glmDsaAttendCached: sparse attention compute + output projection.
	{ID: "glm-dsa-attend-sparse", File: "internal/model/glm_dsa_session.go", Func: "glmDsaSparseAttend",
		Pattern: "sk.sparseAttend(", Class: GluePerAttendLayer, Factor: 1,
		Note: "device sparse-attention op (3 operand uploads + fenced result read)"},
	{ID: "glm-dsa-attend-o-proj", File: "internal/model/glm_dsa_session.go", Func: "glmDsaAttendCached",
		Pattern: "o_proj.weight", Class: GluePerAttendLayer, Factor: 1,
		Note: "attention output projection GEMM"},
	// MoE router (experts stay host-resident under --cpu-offload-experts).
	{ID: "glm-moe-router", File: "internal/model/moe.go", Func: "route",
		Pattern: "mat.mul(routerName(", Class: GluePerRouterLayer, Factor: 1,
		Note: "router GEMM (the only device work of the MoE FFN in the hybrid split)"},
	// Primitives: the H2D/D2H plumbing itself. Drift guards only (Factor 0) — if
	// these vanish the whole census is describing dead code.
	{ID: "cuda-primitive-h2d", File: "internal/compute/cuda.go", Func: "",
		Pattern: "C.fcuda_h2d(", Class: GluePrimitive, Factor: 0,
		Note: "host->device copy primitive (all upload paths funnel through it)"},
	{ID: "cuda-primitive-d2h", File: "internal/compute/cuda.go", Func: "",
		Pattern: "C.fcuda_d2h_on_checked(", Class: GluePrimitive, Factor: 0,
		Note: "device->host copy primitive (Read's stream-syncing drain)"},
	{ID: "seam-mul", File: "internal/model/kernel.go", Func: "mul",
		Pattern: "be.Read(be.MatMul(wt, xt))", Class: GluePrimitive, Factor: 0,
		Note: "backendKernel.mul: the per-projection H2D+GEMM+D2H round-trip seam"},
	{ID: "seam-sparse-attend", File: "internal/model/kernel.go", Func: "sparseAttend",
		Pattern: "be.Read(sb.DSASparseAttend(", Class: GluePrimitive, Factor: 0,
		Note: "backendKernel.sparseAttend round-trip seam"},
	{ID: "seam-index-select", File: "internal/model/kernel.go", Func: "indexSelect",
		Pattern: "ib.DSAIndexSelect(", Class: GluePrimitive, Factor: 0,
		Note: "backendKernel.indexSelect round-trip seam (fenced by the host readback)"},
	{ID: "seam-upload-helper", File: "internal/model/hal.go", Func: "uploadHostF32Class",
		Pattern: "UploadClass(src", Class: GluePrimitive, Factor: 0,
		Note: "the shared activation-upload helper every seam entry above funnels through"},
}

// ModelShape parameterizes the census math. Layer-class counters default to
// Layers when zero (the documented assumption: every layer pays that step).
type ModelShape struct {
	Name            string `json:"name,omitempty"`
	Layers          int    `json:"layers"`
	IndexStepLayers int    `json:"index_step_layers,omitempty"`
	KVAppendLayers  int    `json:"kv_append_layers,omitempty"`
	AttendLayers    int    `json:"attend_layers,omitempty"`
	RouterLayers    int    `json:"router_layers,omitempty"`
}

// normalize fills defaulted layer counters and validates them. It returns the
// normalized copy so callers keep the input untouched.
func (s ModelShape) normalize() (ModelShape, error) {
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
	if err := check("index_step_layers", &out.IndexStepLayers); err != nil {
		return out, err
	}
	if err := check("kv_append_layers", &out.KVAppendLayers); err != nil {
		return out, err
	}
	if err := check("attend_layers", &out.AttendLayers); err != nil {
		return out, err
	}
	if err := check("router_layers", &out.RouterLayers); err != nil {
		return out, err
	}
	return out, nil
}

// glueCensusPresets are the named shapes. glm-dsa mirrors the documented
// GLM-5.2-shaped hand-count scenario (92 layers, every layer pays every step —
// the exact assumption behind the ~1100 figure); tiny-4 exists for shape-math
// regression tests.
var glueCensusPresets = map[string]ModelShape{
	"glm-dsa": {Name: "glm-dsa", Layers: 92, IndexStepLayers: 92, KVAppendLayers: 92, AttendLayers: 92, RouterLayers: 92},
	"tiny-4":  {Name: "tiny-4", Layers: 4},
}

// documentedGLMDsaRoundTrips is the hand-counted reference from
// docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md. It is a REFERENCE the
// census checks itself against — the computed figure always comes from
// inventory x shape, never from this constant.
const (
	documentedGLMDsaRoundTrips = 1100
	documentedGLMDsaRefDoc     = "docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md"
	glueCensusTolerancePct     = 20
)

// computeRoundTrips turns the inventory into estimated synchronous round-trips
// per decoded token for the given shape. Primitive rows contribute nothing.
func computeRoundTrips(inv []GlueCallSite, shape ModelShape) int {
	total := 0
	for _, s := range inv {
		switch s.Class {
		case GluePerIndexLayer:
			total += s.Factor * shape.IndexStepLayers
		case GluePerKVLayer:
			total += s.Factor * shape.KVAppendLayers
		case GluePerAttendLayer:
			total += s.Factor * shape.AttendLayers
		case GluePerRouterLayer:
			total += s.Factor * shape.RouterLayers
		case GluePerToken, GluePerRequest:
			total += s.Factor
		default: // GluePrimitive and unknown classes: drift-guard rows only.
		}
	}
	return total
}

// goFileScan is the parsed view of one Go file: function-name -> source spans
// (byte offsets; methods of different receivers sharing a name merge into one
// span list, which is safe because inventory patterns are specific enough not
// to collide across receivers).
type goFileScan struct {
	path        string
	src         []byte
	funcSpans   map[string][][2]int
	parseErr    error
	fileLevelOK bool // true once the file parsed (used by discovery coverage)
}

func scanGoFile(root, rel string) (*goFileScan, error) {
	full := filepath.Join(root, filepath.FromSlash(rel))
	sc := &goFileScan{path: full, funcSpans: map[string][][2]int{}}
	src, err := os.ReadFile(full)
	if err != nil {
		return sc, err
	}
	sc.src = src
	fset := token.NewFileSet()
	fn, perr := parser.ParseFile(fset, full, src, 0)
	if perr != nil {
		sc.parseErr = perr
		return sc, nil
	}
	sc.fileLevelOK = true
	for _, decl := range fn.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Name == nil {
			continue
		}
		start := fset.Position(fd.Pos()).Offset
		end := fset.Position(fd.End()).Offset
		sc.funcSpans[fd.Name.Name] = append(sc.funcSpans[fd.Name.Name], [2]int{start, end})
	}
	return sc, nil
}

// funcText returns the concatenated source text of every function named name in
// the scan ("" when the name is absent).
func (sc *goFileScan) funcText(name string) (string, bool) {
	spans, ok := sc.funcSpans[name]
	if !ok {
		return "", false
	}
	var b strings.Builder
	for _, sp := range spans {
		b.Write(sc.src[sp[0]:sp[1]])
	}
	return b.String(), true
}

// lineStarts precomputes byte offsets of line beginnings for line-based scans.
func (sc *goFileScan) lineStarts() []int {
	starts := []int{0}
	for i, b := range sc.src {
		if b == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// GlueDrift is one inventory row that no longer matches the live source.
type GlueDrift struct {
	Site        GlueCallSite `json:"site"`
	Reason      string       `json:"reason"` // file-missing | parse-error | func-missing | pattern-absent
	Occurrences int          `json:"occurrences"`
}

// verifyInventory re-checks every inventory row against the live tree and
// returns both the drift findings and the per-row occurrence counts.
func verifyInventory(root string, inv []GlueCallSite) ([]GlueDrift, map[string]int) {
	scanCache := map[string]*goFileScan{}
	get := func(rel string) *goFileScan {
		if sc, ok := scanCache[rel]; ok {
			return sc
		}
		sc, _ := scanGoFile(root, rel)
		scanCache[rel] = sc
		return sc
	}
	var drift []GlueDrift
	occ := map[string]int{}
	for _, site := range inv {
		sc := get(site.File)
		if sc.src == nil {
			drift = append(drift, GlueDrift{Site: site, Reason: "file-missing"})
			continue
		}
		if sc.parseErr != nil {
			drift = append(drift, GlueDrift{Site: site, Reason: "parse-error"})
			continue
		}
		var text string
		if site.Func == "" {
			text = string(sc.src)
		} else if ft, ok := sc.funcText(site.Func); ok {
			text = ft
		} else {
			drift = append(drift, GlueDrift{Site: site, Reason: "func-missing"})
			continue
		}
		n := strings.Count(text, site.Pattern)
		occ[site.ID] = n
		if n == 0 {
			drift = append(drift, GlueDrift{Site: site, Reason: "pattern-absent"})
		}
	}
	return drift, occ
}

// glueCensusDiscoverFiles is the version-controlled decode-path file set the
// uncovered-site scan covers. Extend it deliberately when the decode path gains
// new host-glue files.
var glueCensusDiscoverFiles = []string{
	"internal/model/glm_dsa_session.go",
	"internal/model/kernel.go",
	"internal/model/moe.go",
	"internal/model/moe_offload.go",
	"internal/model/hal.go",
	"internal/compute/cuda.go",
}

// glueCensusSyncMarkers are the high-signal synchronous host<->device
// primitives. A hit inside a file that has NO inventory coverage for its line
// is an UNCOVERED candidate — the new-regression signal.
var glueCensusSyncMarkers = []string{
	"be.Read(",
	".Read(be.",
	"fcuda_h2d",
	"fcuda_d2h",
	"uploadHostF32Class(",
	"cudaMemcpy",
	"StreamSynchronize",
	"cudaDeviceSynchronize",
}

// GlueUncovered is a sync-primitive hit the inventory does not pin.
type GlueUncovered struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Marker  string `json:"marker"`
	Snippet string `json:"snippet"`
}

// discoverUncovered scans the decode-path files for sync markers whose lines no
// inventory row covers (a line is covered when it falls inside an inventoried
// function span, or the file carries a file-level inventory row).
func discoverUncovered(root string, inv []GlueCallSite) []GlueUncovered {
	type coverage struct {
		fileLevel bool
		spans     [][2]int // line-based [startLine, endLine]
	}
	coveredBy := map[string]*coverage{}
	scanCache := map[string]*goFileScan{}
	for _, rel := range glueCensusDiscoverFiles {
		sc, err := scanGoFile(root, rel)
		if err != nil || sc.src == nil || sc.parseErr != nil {
			continue
		}
		scanCache[rel] = sc
		starts := sc.lineStarts()
		lineOf := func(off int) int {
			// binary search: last start <= off
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
	var out []GlueUncovered
	for _, rel := range glueCensusDiscoverFiles {
		sc, ok := scanCache[rel]
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
			for _, m := range glueCensusSyncMarkers {
				if strings.Contains(line, m) {
					snippet := strings.TrimSpace(line)
					if len(snippet) > 100 {
						snippet = snippet[:100] + "…"
					}
					out = append(out, GlueUncovered{File: rel, Line: lineNo, Marker: m, Snippet: snippet})
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

// GlueCensusResult is the full census payload (JSON form of the report).
type GlueCensusResult struct {
	Root            string             `json:"root"`
	Shape           ModelShape         `json:"shape"`
	TotalRoundTrips int                `json:"total_round_trips_per_token"`
	Sites           []GlueCallSiteRow  `json:"sites"`
	Drift           []GlueDrift        `json:"drift"`
	Uncovered       []GlueUncovered    `json:"uncovered"`
	CrossCheck      *GlueCrossCheckRes `json:"cross_check,omitempty"`
}

// GlueCallSiteRow is an inventory row joined with its live occurrence count.
type GlueCallSiteRow struct {
	GlueCallSite
	Occurrences int `json:"occurrences"`
}

// GlueCrossCheckRes records the assertion against a documented reference.
type GlueCrossCheckRes struct {
	Reference    int     `json:"reference"`
	ReferenceDoc string  `json:"reference_doc,omitempty"`
	Computed     int     `json:"computed"`
	DeltaPct     float64 `json:"delta_pct"`
	TolerancePct int     `json:"tolerance_pct"`
	Pass         bool    `json:"pass"`
}

// crossCheck asserts computed lands within ±tolerancePct of ref.
func crossCheck(computed, ref int) GlueCrossCheckRes {
	delta := float64(computed-ref) / float64(ref) * 100
	res := GlueCrossCheckRes{
		Reference:    ref,
		Computed:     computed,
		DeltaPct:     delta,
		TolerancePct: glueCensusTolerancePct,
	}
	res.Pass = delta >= -float64(glueCensusTolerancePct) && delta <= float64(glueCensusTolerancePct)
	return res
}

const glueCensusUsage = `usage: fak-dev glue-census [--root DIR] [--preset NAME | --model-shape JSON] [--check [REF]] [--no-discover] [--json]

Census the synchronous host<->device round-trips a decoded token issues on the
GLM-DSA device path: verifies the version-controlled call-site inventory
against the live source, computes round-trips/token for a model shape, lists
uncovered sync candidates, and optionally cross-checks the result against the
documented ~1100 reference (±20%).

Exit codes: 0 PASS, 1 drift or cross-check FAIL, 2 usage/IO error.`

// RunGlueCensus implements the fak-dev glue-census verb.
func RunGlueCensus(stdout, stderr io.Writer, argv []string) int {
	root := repoRoot()
	preset := "glm-dsa"
	shapeJSON := ""
	check := false
	checkRef := 0
	noDiscover := false
	asJSON := false
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, glueCensusUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "glue-census: --root needs a value")
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		case arg == "--preset":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "glue-census: --preset needs a value")
				return 2
			}
			i++
			preset = argv[i]
		case strings.HasPrefix(arg, "--preset="):
			preset = strings.TrimPrefix(arg, "--preset=")
		case arg == "--model-shape":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "glue-census: --model-shape needs a JSON value")
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
				fmt.Fprintln(stderr, "glue-census: --check=REF needs a positive integer reference")
				return 2
			}
			checkRef = v
		case arg == "--no-discover":
			noDiscover = true
		case arg == "--json":
			asJSON = true
		default:
			fmt.Fprintln(stderr, glueCensusUsage)
			return 2
		}
	}

	var shape ModelShape
	switch {
	case shapeJSON != "":
		if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
			fmt.Fprintf(stderr, "glue-census: parse --model-shape: %v\n", err)
			return 2
		}
		if shape.Name == "" {
			shape.Name = "custom"
		}
	default:
		ps, ok := glueCensusPresets[preset]
		if !ok {
			names := make([]string, 0, len(glueCensusPresets))
			for n := range glueCensusPresets {
				names = append(names, n)
			}
			sort.Strings(names)
			fmt.Fprintf(stderr, "glue-census: unknown preset %q (known: %s)\n", preset, strings.Join(names, ", "))
			return 2
		}
		shape = ps
	}
	normed, err := shape.normalize()
	if err != nil {
		fmt.Fprintf(stderr, "glue-census: invalid shape: %v\n", err)
		return 2
	}
	// Normalize BEFORE the math: presets like tiny-4 leave the per-class layer
	// counters zero and rely on the layers default, so computing on the raw
	// shape would silently yield 0.
	shape = normed

	drift, occ := verifyInventory(root, glueCensusInventory)
	total := computeRoundTrips(glueCensusInventory, shape)
	var uncovered []GlueUncovered
	if !noDiscover {
		uncovered = discoverUncovered(root, glueCensusInventory)
	}

	var xc *GlueCrossCheckRes
	if check {
		ref := checkRef
		doc := ""
		if ref == 0 {
			if shape.Name != "glm-dsa" {
				fmt.Fprintln(stderr, "glue-census: non-glm-dsa shapes need an explicit --check REF")
				return 2
			}
			ref = documentedGLMDsaRoundTrips
			doc = documentedGLMDsaRefDoc
		}
		res := crossCheck(total, ref)
		res.ReferenceDoc = doc
		xc = &res
	}

	result := GlueCensusResult{
		Root:            root,
		Shape:           shape,
		TotalRoundTrips: total,
		Drift:           drift,
		Uncovered:       uncovered,
		CrossCheck:      xc,
	}
	for _, s := range glueCensusInventory {
		result.Sites = append(result.Sites, GlueCallSiteRow{GlueCallSite: s, Occurrences: occ[s.ID]})
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "glue-census: encode: %v\n", err)
			return 2
		}
	} else {
		writeGlueCensusText(stdout, &result)
	}

	if len(drift) > 0 {
		return 1
	}
	if xc != nil && !xc.Pass {
		return 1
	}
	return 0
}

func writeGlueCensusText(w io.Writer, r *GlueCensusResult) {
	fmt.Fprintf(w, "glue-census root=%s shape=%s(layers=%d index=%d kv=%d attend=%d router=%d)\n",
		r.Root, r.Shape.Name, r.Shape.Layers, r.Shape.IndexStepLayers, r.Shape.KVAppendLayers, r.Shape.AttendLayers, r.Shape.RouterLayers)
	fmt.Fprintf(w, "%-72s %-16s %6s %5s\n", "call site", "class", "factor", "occ")
	for _, s := range r.Sites {
		label := s.File + ":" + s.Func + "[" + s.Pattern + "]"
		if s.Func == "" {
			label = s.File + "[" + s.Pattern + "]"
		}
		if len(label) > 72 {
			label = label[:69] + "..."
		}
		fmt.Fprintf(w, "%-72s %-16s %6d %5d\n", label, s.Class, s.Factor, s.Occurrences)
	}
	fmt.Fprintf(w, "TOTAL estimated synchronous host<->device round-trips / decoded token: %d\n", r.TotalRoundTrips)
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
		fmt.Fprintf(w, "cross-check vs documented ref=%d%s: %s delta=%+.2f%% tol=±%d%%\n",
			xc.Reference, doc, status, xc.DeltaPct, xc.TolerancePct)
	}
	if len(r.Drift) > 0 {
		fmt.Fprintf(w, "DRIFT (%d inventory row(s) no longer match the live source):\n", len(r.Drift))
		for _, d := range r.Drift {
			fmt.Fprintf(w, "  %-58s %-14s %s\n", d.Site.File+":"+d.Site.Func+"["+d.Site.Pattern+"]", d.Reason, d.Site.Note)
		}
	}
	if len(r.Uncovered) > 0 {
		fmt.Fprintf(w, "UNCOVERED sync candidates (%d, informational — review and extend the inventory if they are decode-path glue):\n", len(r.Uncovered))
		for _, u := range r.Uncovered {
			fmt.Fprintf(w, "  %s:%d [%s] %s\n", u.File, u.Line, u.Marker, u.Snippet)
		}
	}
}
