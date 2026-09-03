// serve_sizing_json.go implements `fak serve --plan-json` (#4361): emit the versioned,
// header-derived memory sizing artifact — the classed demands the serve arm's fit check
// would size, the disk/ram/vram tier rollup, the per-pool usable bytes after headroom,
// and any would-be refusals as warnings — as JSON on stdout, then exit BEFORE any load.
// Nothing is allocated and no listener binds (the colibri-inspired pre-load inspection
// dry-run: read the numbers FIRST, before committing hundreds of GB and the RAM budget).
// It mirrors the --policy-check early-exit and reuses the exact serveGGUF…MemoryPlan
// helpers the live arms run, so the emitted numbers are the admission numbers.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveSizingVersion stamps the artifact schema so a consumer can bind to a shape,
// not a build. Bump on any breaking field change.
const serveSizingVersion = "fak.serve.memory-sizing.v1"

// serveSizingDemandRow is one classed demand of the fit check, json-shaped.
// Scope is always concrete ("device"/"host") — the empty compat scope is resolved.
type serveSizingDemandRow struct {
	Class  string `json:"class"`
	Bytes  int64  `json:"bytes"`
	Detail string `json:"detail,omitempty"`
	Scope  string `json:"scope"`
	DType  string `json:"dtype,omitempty"`
}

// serveSizingTierRollup folds the demands into the three placement tiers an operator
// budgets against: bytes read from disk, bytes resident in host RAM, bytes in VRAM.
type serveSizingTierRollup struct {
	DiskBytes int64 `json:"disk_bytes"`
	RAMBytes  int64 `json:"ram_bytes"`
	VRAMBytes int64 `json:"vram_bytes"`
}

// serveSizingPool reports one finite capacity pool and the usable budget the fit
// check would size against (the same BudgetAfterHeadroom number, so an emitted
// usable_bytes provably matches a later admission). CapacityKnown=false means the
// pool is unprobeable and every fit check on it fails open.
type serveSizingPool struct {
	Pool          string  `json:"pool"` // "device" | "host"
	Backend       string  `json:"backend,omitempty"`
	TotalBytes    int64   `json:"total_bytes"`
	FreeBytes     int64   `json:"free_bytes"`
	CapacityKnown bool    `json:"capacity_known"`
	Headroom      float64 `json:"headroom_fraction"`
	UsableBytes   int64   `json:"usable_bytes"`
}

// serveSizingArtifact is the emitted dry-run artifact: every derived sizing number
// the selected serve arm would admit against, plus the warnings a live boot would
// print or refuse on. Warnings never empty-marshal to null (always an array).
type serveSizingArtifact struct {
	Version             string                 `json:"version"`
	Model               string                 `json:"model"`
	Arm                 string                 `json:"arm"`
	ContextBudgetTokens int                    `json:"context_budget_tokens"` // 0 = auto-sized to the box
	Demands             []serveSizingDemandRow `json:"demands"`
	Tiers               serveSizingTierRollup  `json:"tiers"`
	Pools               []serveSizingPool      `json:"pools"`
	Warnings            []string               `json:"warnings"`
}

// serveSizingArm names the load arm loadServeInKernelModel's switch would select for
// these inputs — the same case order, so the artifact labels the path a real boot takes.
func serveSizingArm(be compute.Backend, cpuOffloadExperts bool) string {
	switch {
	case be != nil && cpuOffloadExperts:
		return "device-resident-q4k-cpu-offload-experts"
	case serveDeviceResidentQ4K(be):
		return "device-resident-q4k"
	case be != nil && be.Caps().UploadDtype:
		return "device-lean-q8"
	case be != nil:
		return "device-f32"
	case os.Getenv("FAK_Q4K") != "":
		return "cpu-resident-q4k"
	default:
		return "cpu-lean-q8"
	}
}

// buildServeSizingArtifact sizes the SAME classed demands the selected serve arm's
// fit check builds (serveGGUFMemoryPlan / serveGGUFCPUOffloadMemoryPlan, header-only,
// alloc-free) and folds them into the versioned artifact. A demand set that a live
// boot would REFUSE is reported in warnings[] instead of failing: the dry-run's job
// is inspection, so it always emits.
func buildServeSizingArtifact(ws *ggufload.WeightSource, be compute.Backend, cpuOffloadExperts bool, contextBudgetTokens int, model string, diskBytes int64) (serveSizingArtifact, error) {
	warnings := []string{}
	var demands compute.MemoryPlan
	var err error
	switch {
	case be != nil && cpuOffloadExperts:
		if !be.Caps().UploadDtype {
			warnings = append(warnings, fmt.Sprintf("--cpu-offload-experts requires backend %q to advertise quantized UploadDtype (Q8_0 upload); a live serve refuses this combination", be.Name()))
		}
		// The sizing dry-run inspects a single unsharded process, so it plans the whole
		// routed-expert set (ranks=1) — the same demands a non-EP serve boot would build.
		demands, err = serveGGUFCPUOffloadMemoryPlan(ws, 1, contextBudgetTokens, serveDeviceFitBudget(be))
	case be != nil:
		demands, err = serveGGUFMemoryPlan(ws, !be.Caps().UploadDtype, contextBudgetTokens, serveDeviceFitBudget(be))
	default:
		demands, err = serveGGUFMemoryPlan(ws, false, contextBudgetTokens, serveHostFitBudget())
	}
	if err != nil {
		return serveSizingArtifact{}, err
	}

	// Re-run the arm's admission checks and downgrade any refusal to a warning: the
	// operator reading this artifact wants to SEE the shortfall, not lose the numbers.
	if be != nil {
		if ferr := compute.RefuseMemoryPlanIfTooBig(be, demands, serveGGUFDeviceHeadroom); ferr != nil {
			warnings = append(warnings, "device fit would refuse: "+ferr.Error())
		}
		if cpuOffloadExperts {
			if ferr := compute.RefuseHostScopedPlanIfTooBigForHost(demands, serveGGUFHostHeadroom); ferr != nil {
				warnings = append(warnings, "host expert pool would refuse: "+ferr.Error())
			}
		}
		if _, _, known := compute.DeviceMemoryInfo(be); !known {
			warnings = append(warnings, fmt.Sprintf("backend %q reports no device capacity — device fit checks fail open", be.Name()))
		}
	} else {
		if ferr := compute.RefuseMemoryPlanIfTooBigForHost(demands, serveGGUFHostHeadroom); ferr != nil {
			warnings = append(warnings, "host fit would refuse: "+ferr.Error())
		}
	}
	if _, _, known := compute.HostSystemMemoryInfo(); !known {
		warnings = append(warnings, "host memory capacity unknown on this platform — host fit checks fail open")
	}

	rows := make([]serveSizingDemandRow, 0, len(demands))
	for _, d := range demands {
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		rows = append(rows, serveSizingDemandRow{
			Class:  string(class),
			Bytes:  d.Bytes,
			Detail: d.Detail,
			Scope:  string(d.ScopeOrDefault()),
			DType:  d.DType,
		})
	}

	tiers := serveSizingTierRollup{DiskBytes: max(diskBytes, 0)}
	if be != nil {
		tiers.VRAMBytes = demands.DeviceTotal()
		tiers.RAMBytes = demands.HostTotal()
	} else {
		// A pure-CPU serve copies every demand into anonymous host RAM — the grand
		// total, matching RefuseMemoryPlanIfTooBigForHost's ceiling.
		tiers.RAMBytes = demands.Total()
	}

	pools := []serveSizingPool{}
	if be != nil {
		total, free, known := compute.DeviceMemoryInfo(be)
		pools = append(pools, serveSizingPool{
			Pool:          "device",
			Backend:       be.Name(),
			TotalBytes:    max(total, 0),
			FreeBytes:     max(free, 0),
			CapacityKnown: known,
			Headroom:      serveGGUFDeviceHeadroom,
			UsableBytes:   max(serveDeviceFitBudget(be).avail(), 0),
		})
	}
	hostTotal, hostFree, hostKnown := compute.HostSystemMemoryInfo()
	pools = append(pools, serveSizingPool{
		Pool:          "host",
		TotalBytes:    max(hostTotal, 0),
		FreeBytes:     max(hostFree, 0),
		CapacityKnown: hostKnown,
		Headroom:      serveGGUFHostHeadroom,
		UsableBytes:   max(serveHostFitBudget().avail(), 0),
	})

	return serveSizingArtifact{
		Version:             serveSizingVersion,
		Model:               model,
		Arm:                 serveSizingArm(be, cpuOffloadExperts),
		ContextBudgetTokens: contextBudgetTokens,
		Demands:             rows,
		Tiers:               tiers,
		Pools:               pools,
		Warnings:            warnings,
	}, nil
}

// runServeSizingJSON is the --plan-json early-exit body: resolve the backend the way
// resolveCompute would (Lookup, fail-loud on a typo), open the GGUF header, build the
// artifact, print it, and return without loading a tensor byte or binding a listener.
func runServeSizingJSON(sf *serveFlags) {
	if *sf.ggufPath == "" {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailWeightsRequired,
			Summary: "--plan-json requires --gguf WEIGHTS (the artifact is header-derived)",
			Knobs: []bailKnob{
				bailFlag("plan-json", "true"),
				bailFlag("gguf", "").want("a GGUF path, an hf:// URI, or a registry alias"),
			},
			Check: "fak ls   # the locally cached models --gguf can name",
		})
		os.Exit(2)
	}
	be, err := resolveServeChatBackend(*sf.backendName)
	if err != nil {
		writeBackendUnavailableBail(os.Stderr, "fak serve", *sf.backendName)
		os.Exit(2)
	}
	ws, err := ggufload.OpenWeights(*sf.ggufPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: --plan-json:", err)
		os.Exit(1)
	}
	defer ws.Close()
	var diskBytes int64
	if st, statErr := os.Stat(*sf.ggufPath); statErr == nil {
		diskBytes = st.Size()
	}
	art, err := buildServeSizingArtifact(ws, be, *sf.cpuOffloadExperts, sf.effectiveAdmissionTokenBudget(), *sf.ggufPath, diskBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: --plan-json:", err)
		os.Exit(1)
	}
	// The #1062 slow-load-path advisory rides in warnings[] too: on NFS/CIFS weights the
	// multi-minute load tax is exactly what a pre-load reader wants to know about.
	if w := compute.WarnSlowLoadPath(compute.ProbeLoadPath(*sf.ggufPath)); w != "" {
		art.Warnings = append(art.Warnings, w)
	}
	out, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: --plan-json:", err)
		os.Exit(1)
	}
	fmt.Println(string(out))
}
