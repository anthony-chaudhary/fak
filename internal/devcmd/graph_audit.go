// graph-audit — presence-vs-invokability audit of device-graph capture/replay on
// the small-batch decode paths of the public engine.
//
// The ad-hoc method this replaces is the study-note claim ("the CUDA backend
// captures the decode step into a CUDA graph; Vulkan/Metal do not"), which is a
// one-off prose assertion: it cannot be recomputed after a refactor, cannot flag
// a NEW decode path that silently bypasses the captured fast path, and cannot
// distinguish "the replay call site exists" (presence) from "the backend
// primitive it reaches into still exists and is gated on" (invokability). This
// audit converts it into a version-controlled, recomputable, trendable table:
//
//   - a closed inventory of decode paths (backend x model-path), each pinned to
//     file + function + source pattern and classified CAPTURED /
//     PER_LAYER_HOST_SETUP / NO_GRAPH_SUPPORT / UNKNOWN;
//   - verification against the LIVE source (go/ast, the glue-census scanner): a
//     vanished replay call site, a removed cudaGraph* primitive, or a newly
//     appearing capture marker where none was allowed is reported as DRIFT and
//     fails the run — the table can no longer silently rot the way a prose
//     claim does;
//   - a coverage rollup counting UNCOVERED decode paths (anything still
//     PER_LAYER_HOST_SETUP or NO_GRAPH_SUPPORT), the regression signal the
//     host-overhead goal tracks;
//   - --fail-on uncovered turns that count into a gate; --snapshot writes the
//     machine-readable table (with the HEAD SHA) so it trends over time.
//
// Classification vocabulary:
//   - CAPTURED: the path replays its op stream as a captured device graph
//     (one launch per decoded token) — the persistent-kernel-style fast path.
//   - PER_LAYER_HOST_SETUP: the path issues per-layer/per-op host-side setup
//     (per-op kernel launches, parameter marshaling, H2D/D2H round-trips).
//   - NO_GRAPH_SUPPORT: the backend implements no graph capture/replay at all;
//     every model path on it is host-setup bound.
//   - UNKNOWN: no live evidence pins the verdict (drift fallback).

package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// GraphClass is the audited classification of one decode path or primitive pin.
type GraphClass string

const (
	GraphCaptured          GraphClass = "CAPTURED"
	GraphPerLayerHostSetup GraphClass = "PER_LAYER_HOST_SETUP"
	GraphNoGraphSupport    GraphClass = "NO_GRAPH_SUPPORT"
	GraphUnknown           GraphClass = "UNKNOWN"
)

// GraphRole distinguishes counted decode paths from drift-guard primitive pins.
type GraphRole string

const (
	GraphPath      GraphRole = "path"      // a decode path, counted in coverage
	GraphPrimitive GraphRole = "primitive" // pins the underlying API; NOT summed
)

// GraphSite is one version-controlled inventory row. Pattern must occur in the
// enclosing function's source text (or anywhere in the file when Func is
// empty); NegPatterns must NOT occur there. Verification re-checks both
// against the live tree on every run — a missing Pattern drops the row to
// UNKNOWN, a present NegPattern signals a verdict flip.
type GraphSite struct {
	ID          string     `json:"id"`
	Backend     string     `json:"backend"`    // cuda | vulkan | metal | cpu-ref | host | any
	ModelPath   string     `json:"model_path"` // hal-dense-decode | glm-moe-dsa-decode | ...
	File        string     `json:"file"`
	Func        string     `json:"func"` // "" = file-level
	Pattern     string     `json:"pattern"`
	NegPatterns []string   `json:"neg_patterns,omitempty"`
	Class       GraphClass `json:"classification"`
	Role        GraphRole  `json:"role"`
	Note        string     `json:"note"`
}

// graphAuditInventory is the closed set of decode paths (and the CUDA graph
// primitives they invoke) in the public engine. Model-level rows pin WHERE a
// path gets its cadence; primitive rows pin that the device APIs behind the
// CAPTURED verdicts still exist (presence AND invokability); backend rows pin
// which backends have no graph capability at all.
var graphAuditInventory = []GraphSite{
	// --- Decode paths: model level -------------------------------------------
	{ID: "hal-capture-once-replay", Backend: "cuda", ModelPath: "hal-dense-decode",
		File: "internal/model/hal.go", Func: "tokenHALOutput",
		Pattern: "once.GraphOnceReplay(", Class: GraphCaptured, Role: GraphPath,
		Note: "steady-state capture-once replay: the dense F32 HAL decode step replays as ONE graph launch per token (the primary small-batch fast path)"},
	{ID: "hal-classic-graph", Backend: "cuda", ModelPath: "hal-dense-decode",
		File: "internal/model/hal.go", Func: "tokenHALOutput",
		Pattern: "gr.GraphBegin()", Class: GraphCaptured, Role: GraphPath,
		Note: "classic per-token capture+replay fallback when capture-once is ineligible (still one launch per token under capture)"},
	{ID: "glm-dsa-band-step", Backend: "any", ModelPath: "glm-moe-dsa-decode",
		File: "internal/model/glm_dsa_session.go", Func: "decodeBandGLMDsa",
		Pattern: "s.glmDsaMatKernel()", NegPatterns: []string{"GraphBegin(", "GraphOnceReplay("},
		Class: GraphPerLayerHostSetup, Role: GraphPath,
		Note: "per-layer GEMMs through the matKernel round-trip; DSA index/sparse glue stays host-resident; no graph capture on this band"},
	{ID: "legacy-f32-blockstep", Backend: "host", ModelPath: "legacy-f32-session",
		File: "internal/model/session_token_hidden.go", Func: "tokenHiddenRow",
		Pattern: "s.blockStep(", NegPatterns: []string{"GraphBegin("},
		Class: GraphPerLayerHostSetup, Role: GraphPath,
		Note: "direct []float32 session loop (pre-HAL legacy path): per-layer blockStep host setup, no device graph concept"},
	// --- Decode paths: backend level (all model paths on that backend) --------
	{ID: "vulkan-no-graph", Backend: "vulkan", ModelPath: "all-model-paths",
		File:    "internal/compute/vulkan.go",
		Pattern: "func (v *vulkanBackend) MatMul(", NegPatterns: []string{"GraphBegin(", "GraphOnceReplay("},
		Class: GraphNoGraphSupport, Role: GraphPath,
		Note: "Vulkan backend dispatches per-op through the C ABI (fvk_*); no graph capture/replay surface in the public tree"},
	{ID: "metal-no-graph", Backend: "metal", ModelPath: "all-model-paths",
		File:    "internal/compute/metal.go",
		Pattern: "func (c *metalBackend) MatMul(", NegPatterns: []string{"GraphBegin(", "GraphOnceReplay("},
		Class: GraphNoGraphSupport, Role: GraphPath,
		Note: "Metal backend delegates every op to the flat C ABI shim; no graph capture/replay surface"},
	{ID: "cpuref-no-graph", Backend: "cpu-ref", ModelPath: "all-model-paths",
		File:    "internal/compute/cpuref.go",
		Pattern: "func (c *cpuBackend) MatMul(", NegPatterns: []string{"GraphBegin(", "GraphOnceReplay("},
		Class: GraphNoGraphSupport, Role: GraphPath,
		Note: "host reference backend: Upload/Read are identity, there is no device to graph"},
	// --- CUDA graph primitives: drift guards behind the CAPTURED verdicts ----
	{ID: "cuda-graph-flag", Backend: "cuda", ModelPath: "-",
		File:    "internal/compute/cuda.go",
		Pattern: "FAK_CUDA_GRAPH", Class: GraphCaptured, Role: GraphPrimitive,
		Note: "backend graph consent flag gating the reusable CUDA-graph decode path"},
	{ID: "cuda-graph-begin", Backend: "cuda", ModelPath: "-",
		File: "internal/compute/cuda.go", Func: "GraphBegin",
		Pattern: "C.fcuda_graph_begin(", Class: GraphCaptured, Role: GraphPrimitive,
		Note: "capture-open primitive (records the op stream on the capture stream)"},
	{ID: "cuda-graph-end-launch", Backend: "cuda", ModelPath: "-",
		File: "internal/compute/cuda.go", Func: "GraphEndLaunch",
		Pattern: "C.fcuda_graph_end_launch(", Class: GraphCaptured, Role: GraphPrimitive,
		Note: "instantiate+launch+fence primitive (the replay half of GraphBegin)"},
	{ID: "cuda-graph-once-create", Backend: "cuda", ModelPath: "-",
		File: "internal/compute/cuda_graph_once_qualification.go", Func: "GraphOnceBegin",
		Pattern: "C.fak_graph_once_create(", Class: GraphCaptured, Role: GraphPrimitive,
		Note: "capture-once graph creation (fixed-KV, reusable exec graph)"},
	{ID: "cuda-graph-once-replay", Backend: "cuda", ModelPath: "-",
		File: "internal/compute/cuda_graph_once_qualification.go", Func: "GraphOnceReplay",
		Pattern: "C.fak_graph_once_replay(", Class: GraphCaptured, Role: GraphPrimitive,
		Note: "capture-once replay primitive (one launch per token in steady state)"},
}

// GraphDrift is one inventory row that no longer matches the live source.
type GraphDrift struct {
	Site   GraphSite `json:"site"`
	Reason string    `json:"reason"` // file-missing | parse-error | func-missing | pattern-absent | negative-pattern-present
}

// verifyGraphInventory re-checks every inventory row against the live tree
// (reusing the glue-census go/ast scanner) and returns the drift findings plus
// per-row occurrence counts of the pinned pattern.
func verifyGraphInventory(root string, inv []GraphSite) ([]GraphDrift, map[string]int) {
	scanCache := map[string]*goFileScan{}
	get := func(rel string) *goFileScan {
		if sc, ok := scanCache[rel]; ok {
			return sc
		}
		sc, _ := scanGoFile(root, rel)
		scanCache[rel] = sc
		return sc
	}
	var drift []GraphDrift
	occ := map[string]int{}
	for _, site := range inv {
		sc := get(site.File)
		if sc.src == nil {
			drift = append(drift, GraphDrift{Site: site, Reason: "file-missing"})
			continue
		}
		if sc.parseErr != nil {
			drift = append(drift, GraphDrift{Site: site, Reason: "parse-error"})
			continue
		}
		var text string
		if site.Func == "" {
			text = string(sc.src)
		} else if ft, ok := sc.funcText(site.Func); ok {
			text = ft
		} else {
			drift = append(drift, GraphDrift{Site: site, Reason: "func-missing"})
			continue
		}
		n := strings.Count(text, site.Pattern)
		occ[site.ID] = n
		if n == 0 {
			drift = append(drift, GraphDrift{Site: site, Reason: "pattern-absent"})
			continue
		}
		for _, np := range site.NegPatterns {
			if strings.Contains(text, np) {
				drift = append(drift, GraphDrift{Site: site, Reason: "negative-pattern-present"})
				break
			}
		}
	}
	return drift, occ
}

// GraphSiteRow is an inventory row joined with its live occurrence count and
// verdict. Verdict is the inventoried Class unless the row drifted, in which
// case it degrades to UNKNOWN (the table never asserts what it cannot see).
type GraphSiteRow struct {
	GraphSite
	Occurrences int        `json:"occurrences"`
	Verdict     GraphClass `json:"verdict"`
}

// GraphCoverage rolls the path-role rows up into the coverage numbers the
// host-overhead goal tracks. Primitive rows are drift guards only and are not
// summed.
type GraphCoverage struct {
	Captured          int `json:"captured"`
	PerLayerHostSetup int `json:"per_layer_host_setup"`
	NoGraphSupport    int `json:"no_graph_support"`
	Unknown           int `json:"unknown"`
	TotalPaths        int `json:"total_decode_paths"`
	UncoveredPaths    int `json:"uncovered_decode_paths"` // PER_LAYER_HOST_SETUP + NO_GRAPH_SUPPORT
}

func computeGraphCoverage(rows []GraphSiteRow) GraphCoverage {
	var c GraphCoverage
	for _, r := range rows {
		if r.Role != GraphPath {
			continue
		}
		c.TotalPaths++
		switch r.Verdict {
		case GraphCaptured:
			c.Captured++
		case GraphPerLayerHostSetup:
			c.PerLayerHostSetup++
			c.UncoveredPaths++
		case GraphNoGraphSupport:
			c.NoGraphSupport++
			c.UncoveredPaths++
		default:
			c.Unknown++
		}
	}
	return c
}

// GraphAuditResult is the full audit payload (JSON form of the report).
type GraphAuditResult struct {
	Root     string         `json:"root"`
	HeadSHA  string         `json:"head_sha,omitempty"`
	Sites    []GraphSiteRow `json:"sites"`
	Drift    []GraphDrift   `json:"drift"`
	Coverage GraphCoverage  `json:"coverage"`
}

// gitHeadSHA best-effort resolves the HEAD short SHA of root ("" outside a git
// tree or when git is unavailable — fixtures audit cleanly without one).
func gitHeadSHA(root string) string {
	out, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

const graphAuditUsage = `usage: fak-dev graph-audit [--root DIR] [--json] [--fail-on uncovered] [--snapshot PATH]

Presence-vs-invokability audit of CUDA/Vulkan graph capture/replay on the
small-batch decode paths: verifies the version-controlled decode-path
inventory against the live source, classifies each path CAPTURED /
PER_LAYER_HOST_SETUP / NO_GRAPH_SUPPORT / UNKNOWN, and reports the count of
uncovered decode paths.

Flags:
  --root DIR           repository root to audit (default: auto-discovered)
  --json               emit the machine-readable result
  --fail-on uncovered  exit non-zero when any decode path is still
                       PER_LAYER_HOST_SETUP or NO_GRAPH_SUPPORT
  --snapshot PATH      write the machine-readable snapshot (with HEAD SHA) to
                       PATH so the table trends over time

Exit codes: 0 PASS, 1 drift or --fail-on hit, 2 usage/IO error.`

// RunGraphAudit implements the fak-dev graph-audit verb.
func RunGraphAudit(stdout, stderr io.Writer, argv []string) int {
	root := repoRoot()
	asJSON := false
	failOn := ""
	snapshot := ""
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, graphAuditUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "graph-audit: --root needs a value")
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		case arg == "--json":
			asJSON = true
		case arg == "--fail-on":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "graph-audit: --fail-on needs a value")
				return 2
			}
			i++
			failOn = argv[i]
		case strings.HasPrefix(arg, "--fail-on="):
			failOn = strings.TrimPrefix(arg, "--fail-on=")
		case arg == "--snapshot":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "graph-audit: --snapshot needs a value")
				return 2
			}
			i++
			snapshot = argv[i]
		case strings.HasPrefix(arg, "--snapshot="):
			snapshot = strings.TrimPrefix(arg, "--snapshot=")
		default:
			fmt.Fprintln(stderr, graphAuditUsage)
			return 2
		}
	}
	if failOn != "" && failOn != "uncovered" {
		fmt.Fprintf(stderr, "graph-audit: --fail-on supports only \"uncovered\" (got %q)\n", failOn)
		return 2
	}

	drift, occ := verifyGraphInventory(root, graphAuditInventory)
	drifted := map[string]bool{}
	for _, d := range drift {
		drifted[d.Site.ID] = true
	}
	result := GraphAuditResult{
		Root:     root,
		HeadSHA:  gitHeadSHA(root),
		Drift:    drift,
		Coverage: GraphCoverage{},
	}
	for _, s := range graphAuditInventory {
		row := GraphSiteRow{GraphSite: s, Occurrences: occ[s.ID], Verdict: s.Class}
		if drifted[s.ID] {
			row.Verdict = GraphUnknown
		}
		result.Sites = append(result.Sites, row)
	}
	result.Coverage = computeGraphCoverage(result.Sites)

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "graph-audit: encode: %v\n", err)
			return 2
		}
	} else {
		writeGraphAuditText(stdout, &result)
	}

	if snapshot != "" {
		// The committed snapshot is machine-independent: the root path is where
		// THIS run happened, not part of the audited fact — head_sha pins the tree.
		snap := result
		snap.Root = ""
		body, err := json.MarshalIndent(snap, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "graph-audit: marshal snapshot: %v\n", err)
			return 2
		}
		if err := os.MkdirAll(filepath.Dir(snapshot), 0o755); err != nil {
			fmt.Fprintf(stderr, "graph-audit: mkdir snapshot dir: %v\n", err)
			return 2
		}
		if err := os.WriteFile(snapshot, append(body, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "graph-audit: write snapshot: %v\n", err)
			return 2
		}
		if !asJSON {
			fmt.Fprintf(stdout, "snapshot written: %s\n", snapshot)
		}
	}

	if len(drift) > 0 {
		return 1
	}
	if failOn == "uncovered" && result.Coverage.UncoveredPaths > 0 {
		return 1
	}
	return 0
}

func writeGraphAuditText(w io.Writer, r *GraphAuditResult) {
	head := ""
	if r.HeadSHA != "" {
		head = " head=" + r.HeadSHA[:12]
	}
	fmt.Fprintf(w, "graph-audit root=%s%s\n", r.Root, head)
	rows := append([]GraphSiteRow(nil), r.Sites...)
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Role != b.Role {
			return a.Role < b.Role
		}
		if a.Backend != b.Backend {
			return a.Backend < b.Backend
		}
		return a.ID < b.ID
	})
	fmt.Fprintf(w, "%-26s %-8s %-18s %-40s %-20s %5s\n", "decode path", "backend", "model-path", "evidence", "class", "occ")
	for _, s := range rows {
		label := s.File + "[" + s.Pattern + "]"
		if s.Func != "" {
			label = s.File + ":" + s.Func + "[" + s.Pattern + "]"
		}
		if len(label) > 40 {
			label = label[:37] + "..."
		}
		cls := string(s.Verdict)
		if s.Role == GraphPrimitive {
			cls += "*"
		}
		fmt.Fprintf(w, "%-26s %-8s %-18s %-40s %-20s %5d\n", s.ID, s.Backend, s.ModelPath, label, cls, s.Occurrences)
	}
	c := r.Coverage
	fmt.Fprintf(w, "decode paths: total=%d captured=%d per_layer_host_setup=%d no_graph_support=%d unknown=%d\n",
		c.TotalPaths, c.Captured, c.PerLayerHostSetup, c.NoGraphSupport, c.Unknown)
	fmt.Fprintf(w, "UNCOVERED decode paths (still host-setup bound or graph-less): %d\n", c.UncoveredPaths)
	if len(r.Drift) > 0 {
		fmt.Fprintf(w, "DRIFT (%d inventory row(s) no longer match the live source):\n", len(r.Drift))
		for _, d := range r.Drift {
			fmt.Fprintf(w, "  %-26s %-14s %s\n", d.Site.ID, d.Reason, d.Site.Note)
		}
	}
}
