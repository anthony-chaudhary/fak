package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	v41 "github.com/anthony-chaudhary/fak/internal/model/v41"
	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
)

func runModelPlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	setup := fs.String("setup", "personal", "harness setup: personal, shared, or batch")
	goal := fs.String("goal", "balanced", "selection goal: balanced, quality, latency, or cost")
	local := fs.String("local", "auto", "local loading policy: auto, require, or disable")
	memory := fs.String("memory", "unified", "memory topology: unified or split")
	deviceGiB := fs.Float64("device-gib", 0, "usable device or unified memory in GiB (0 means unknown)")
	hostGiB := fs.Float64("host-gib", 0, "usable host GiB; V4.1: cache budget AFTER OS/CPU decoder/pinned-transfer reserves (0 unknown)")
	diskGiB := fs.Float64("disk-gib", 0, "usable model-cache disk in GiB (0 means unknown)")
	jsonOut := fs.Bool("json", false, "emit the versioned plan as JSON")
	gpus := fs.Int("gpus", 1, "V4.1 placement GPU count (first leaf supports one)")
	hfConfig := fs.String("hf-config", "", "V4.1 HF config JSON; metadata only")
	hfIndex := fs.String("hf-index", "", "pinned1017 model.safetensors.index.json; hashed metadata only")
	revision := fs.String("revision", v41.V41Placement1017Revision, "V4.1 packed checkpoint revision (1017 only)")
	contextTokens := fs.Int("context-tokens", 256, "V4.1 context charge in tokens")
	workspaceGiB := fs.Float64("workspace-gib", 0, "V4.1 DEVICE activation/staging/runtime-cast reserve; host reserves must already be subtracted from --host-gib")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak model plan [qwen38:27b] [--setup personal|shared|batch] [--goal balanced|quality|latency|cost] [--local auto|require|disable] [--memory unified|split] [--device-gib N] [--host-gib N] [--disk-gib N] [--json]")
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
