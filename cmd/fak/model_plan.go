package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	v41 "github.com/anthony-chaudhary/fak/internal/model/v41"
	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
	"github.com/anthony-chaudhary/fak/pkg/hwspecs"
	"github.com/anthony-chaudhary/fak/pkg/hwutil"
)

// modelPlanProbeGPUName resolves the host's primary telemetry GPU name for
// `fak model plan --device-spec auto` (first device from the hwutil probe);
// stubbed in tests.
var modelPlanProbeGPUName = func() (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	snap, err := hwutil.Collect(ctx, hwutil.DefaultOptions())
	if err != nil || len(snap.GPUs) == 0 {
		return "", false
	}
	return snap.GPUs[0].Name, true
}

// resolveModelPlanDeviceSpec turns a --device-spec value (a catalog key, a
// human spelling, or "auto") into a cited spec row, printing the typed refusal
// and returning false when unresolvable.
func resolveModelPlanDeviceSpec(name string, stderr io.Writer) (hwspecs.DeviceSpec, bool) {
	query := name
	if name == "auto" {
		probed, ok := modelPlanProbeGPUName()
		if !ok {
			fmt.Fprintln(stderr, "fak model plan: --device-spec auto found no GPU surface on this host (nvidia-smi/rocm probe empty); name the card instead")
			return hwspecs.DeviceSpec{}, false
		}
		query = probed
	}
	sp, ok := hwspecs.Lookup(query)
	if !ok {
		fmt.Fprintf(stderr, "fak model plan: no cited hardware spec for %q (catalog: %s)\n", query, strings.Join(hwspecs.Keys(), ", "))
		return hwspecs.DeviceSpec{}, false
	}
	return sp, true
}

// ErrModelPlanMultiGPUUnsupported marks a non-V4.1 placement leaf asked for
// more than one GPU. The qwen38 leaf places the full model on exactly one
// device; two-GPU tensor-parallel serving is a separate capability with its
// own provenance (fak tpcheck), not a planner flag.
var ErrModelPlanMultiGPUUnsupported = errors.New("the qwen38 placement leaf supports exactly one GPU; two-GPU full-model placement is not implemented (TP=2 collectives are proven separately via fak tpcheck)")

// validateModelPlanGPUs is the closed-contract check over --gpus before any
// leaf runs. Below one GPU is a CLI misuse (exit 2 at the call site); above
// one on a non-V4.1 leaf is the typed multi-GPU refusal (exit 3, matching the
// V4.1 planner's own single-GPU refusal). V4.1 passes through untouched: its
// planner owns the GPU-count verdict.
func validateModelPlanGPUs(gpus int, isV41 bool) error {
	if gpus < 1 {
		return fmt.Errorf("fak model plan: --gpus must be >= 1 (got %d)", gpus)
	}
	if !isV41 && gpus > 1 {
		return ErrModelPlanMultiGPUUnsupported
	}
	return nil
}

func runModelPlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	setup := fs.String("setup", "personal", "harness setup: personal, shared, or batch")
	goal := fs.String("goal", "balanced", "selection goal: balanced, quality, latency, or cost")
	local := fs.String("local", "auto", "local loading policy: auto, require, or disable")
	memory := fs.String("memory", "unified", "memory topology: unified or split")
	deviceGiB := fs.Float64("device-gib", 0, "usable device or unified memory in GiB (0 means unknown)")
	deviceSpec := fs.String("device-spec", "", "resolve device memory from a cited hardware spec: card key/spelling (e.g. h200) or auto (probe the host GPU); mutually exclusive with --device-gib")
	hostGiB := fs.Float64("host-gib", 0, "usable host GiB; V4.1: cache budget AFTER OS/CPU decoder/pinned-transfer reserves (0 unknown)")
	diskGiB := fs.Float64("disk-gib", 0, "usable model-cache disk in GiB (0 means unknown)")
	jsonOut := fs.Bool("json", false, "emit the versioned plan as JSON")
	gpus := fs.Int("gpus", 1, "placement GPU count; qwen38 places exactly one GPU (refused above one), V4.1 passes through to its planner")
	hfConfig := fs.String("hf-config", "", "V4.1 HF config JSON; metadata only")
	hfIndex := fs.String("hf-index", "", "pinned1017 model.safetensors.index.json; hashed metadata only")
	revision := fs.String("revision", v41.V41Placement1017Revision, "V4.1 packed checkpoint revision (1017 only)")
	contextTokens := fs.Int("context-tokens", 256, "V4.1 context charge in tokens")
	workspaceGiB := fs.Float64("workspace-gib", 0, "V4.1 DEVICE activation/staging/runtime-cast reserve; host reserves must already be subtracted from --host-gib")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak model plan [qwen38:27b] [--setup personal|shared|batch] [--goal balanced|quality|latency|cost] [--local auto|require|disable] [--memory unified|split] [--device-gib N | --device-spec KEY|auto] [--host-gib N] [--disk-gib N] [--json]")
		fmt.Fprintln(stderr, "       fak model plan deepseek-v41-flash --memory split --hf-config config.json --hf-index model.safetensors.index.json --device-gib N --host-gib N --disk-gib N --workspace-gib N [--gpus 1] [--context-tokens 256] [--revision SHA] [--json]")
	}
	modelArg := ""
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		modelArg, args = strings.ToLower(args[0]), args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	isV41 := modelArg == "deepseek-v41-flash"
	if fs.NArg() > 0 || (modelArg != "" && !isV41 && !oneOfString(modelArg, "qwen38", "qwen38:27b", "qwen/qwen3.8-27b")) {
		fmt.Fprintln(stderr, "fak model plan: supported models are qwen38:27b (default) and deepseek-v41-flash (offline placement)")
		return 2
	}
	if err := validateModelPlanGPUs(*gpus, isV41); err != nil {
		if errors.Is(err, ErrModelPlanMultiGPUUnsupported) {
			fmt.Fprintln(stderr, err)
			return 3
		}
		fmt.Fprintln(stderr, err)
		return 2
	}
	deviceSpecNote := ""
	var deviceSpecRow hwspecs.DeviceSpec
	if *deviceSpec != "" {
		if isV41 {
			fmt.Fprintln(stderr, "fak model plan: --device-spec is not applicable to deepseek-v41-flash (its planner declares its own device bytes)")
			return 2
		}
		if *deviceGiB != 0 {
			fmt.Fprintln(stderr, "fak model plan: --device-spec and --device-gib are mutually exclusive")
			return 2
		}
		sp, ok := resolveModelPlanDeviceSpec(*deviceSpec, stderr)
		if !ok {
			return 2
		}
		if sp.VRAMGiB <= 0 || sp.VRAMGiB >= float64(math.MaxInt64)/(1<<30) {
			fmt.Fprintf(stderr, "fak model plan: cited spec %s carries an unusable VRAM figure (%g)\n", sp.Key, sp.VRAMGiB)
			return 2
		}
		deviceSpecRow = sp
		deviceSpecNote = fmt.Sprintf("cited hardware spec %s (%s, %s, %g GiB %s @ %g GB/s): %s", sp.Key, sp.Name, sp.Architecture, sp.VRAMGiB, sp.MemoryType, sp.BandwidthGBps, sp.Source)
	}
	toBytes := func(name string, n float64) (int64, bool) {
		if n < 0 || math.IsNaN(n) || math.IsInf(n, 0) || n >= float64(math.MaxInt64)/(1<<30) {
			fmt.Fprintf(stderr, "fak model plan: --%s must be finite, nonnegative and representable in bytes\n", name)
			return 0, false
		}
		return int64(n * (1 << 30)), true
	}
	device, ok := toBytes("device-gib", *deviceGiB)
	if !ok {
		return 2
	}
	host, ok := toBytes("host-gib", *hostGiB)
	if !ok {
		return 2
	}
	disk, ok := toBytes("disk-gib", *diskGiB)
	if !ok {
		return 2
	}
	if deviceSpecRow.Key != "" {
		device = int64(deviceSpecRow.VRAMGiB * (1 << 30))
	}
	if isV41 {
		// Do not infer this split-device capacity from the host running the CLI.
		// No artifact download, payload load, hardware probe or serve admission.
		if *memory != "split" || *local == "disable" || *hfConfig == "" || *hfIndex == "" {
			fmt.Fprintln(stderr, "fak model plan: V4.1 requires --memory split, --hf-config, --hf-index and local loading enabled")
			return 2
		}
		workspace, ok := toBytes("workspace-gib", *workspaceGiB)
		if !ok {
			return 2
		}
		f, err := os.Open(*hfConfig)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		// Config is metadata. Refuse trailing data and oversized inputs.
		data, err := io.ReadAll(io.LimitReader(f, (1<<20)+1))
		closeErr := f.Close()
		if err != nil || closeErr != nil || len(data) > 1<<20 {
			fmt.Fprintln(stderr, "fak model plan: config read failed or exceeds 1MiB")
			return 1
		}
		index, err := os.Open(*hfIndex)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		hash := sha256.New()
		n, readErr := io.Copy(hash, io.LimitReader(index, (8<<20)+1))
		indexCloseErr := index.Close()
		if readErr != nil || indexCloseErr != nil || n > 8<<20 {
			fmt.Fprintln(stderr, "fak model plan: index read failed or exceeds8MiB")
			return 1
		}
		plan, planErr := v41.PlanDeepSeekV41Placement(data, v41.V41PlacementTarget{CheckpointRevision: *revision, IndexSHA256: hex.EncodeToString(hash.Sum(nil)), GPUs: *gpus, DeviceBytes: device, HostBytes: host, NVMeBytes: disk, ContextTokens: *contextTokens, WorkspaceBytes: workspace})
		if *jsonOut {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(plan); err != nil {
				fmt.Fprintln(stderr, err)
				return 1
			}
		} else {
			fmt.Fprintf(stdout, "V4.1 offline placement only: device=%d host=%d streamed=%d spill_layers=%d effective_kv=%s\n", plan.Device.Total(), plan.Host.Total(), plan.StreamedBytes, plan.SpillLayers, plan.EffectiveKVPrecision)
			for _, limit := range plan.Limitations {
				fmt.Fprintln(stdout, limit)
			}
		}
		if planErr != nil {
			fmt.Fprintln(stderr, planErr)
			return 3
		}
		return 0
	}
	if device == 0 {
		if total, _, known := compute.HostSystemMemoryInfo(); known && total > 0 {
			if *memory == "unified" || (runtime.GOOS == "darwin" && runtime.GOARCH == "arm64") {
				device = total
			} else {
				host = total
			}
		}
	}
	if disk == 0 {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			cacheDir = os.TempDir()
		}
		if _, freeDisk, known := compute.DiskInfo(cacheDir); known && freeDisk > 0 {
			disk = freeDisk
		}
	}
	plan, err := modelloadplan.Build(modelloadplan.Request{Setup: *setup, Goal: *goal, LocalPolicy: *local, Memory: *memory, DeviceBytes: device, HostBytes: host, DiskBytes: disk})
	if err != nil {
		fmt.Fprintf(stderr, "fak model plan: %v\n", err)
		return 2
	}
	if deviceSpecNote != "" {
		plan.Provenance = append(plan.Provenance, deviceSpecNote)
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(plan); err != nil {
			fmt.Fprintf(stderr, "fak model plan: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "model: %s\nsetup: %s  goal: %s  local: %s  memory: %s\n", plan.Model, plan.Request.Setup, plan.Request.Goal, plan.Request.LocalPolicy, plan.Request.Memory)
	if deviceSpecNote != "" {
		fmt.Fprintf(stdout, "device basis: %s\n", deviceSpecNote)
	}
	if plan.Selected == nil {
		fmt.Fprintln(stdout, "selected: NONE (no candidate fits the declared policy and capacities)")
	} else {
		fmt.Fprintf(stdout, "selected: %s", plan.Selected.ID)
		if plan.Selected.Quantization != "" {
			fmt.Fprintf(stdout, " (%s)", plan.Selected.Quantization)
		}
		fmt.Fprintf(stdout, "\nuri: %s\nnext: %s\n", plan.Selected.URI, plan.Selected.NextCommand)
	}
	fmt.Fprintln(stdout, "candidates:")
	for _, c := range plan.Candidates {
		verdict := "REJECT"
		if c.Fits {
			verdict = "FIT"
		}
		fmt.Fprintf(stdout, "  %-6s %-22s %s\n", verdict, c.ID, strings.Join(c.Reasons, "; "))
	}
	return 0
}

func oneOfString(v string, values ...string) bool {
	for _, x := range values {
		if v == x {
			return true
		}
	}
	return false
}
