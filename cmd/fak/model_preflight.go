package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

const modelPreflightUsage = "usage: fak model preflight (--gguf <path|shard1> | --safetensors <dir>) [--hf-config <config.json>] [--device cuda --arch sm_XY] [--json]"

// modelPreflightReport wraps ggufload.ManifestPreflight with two additive omitempty fields:
// format (set only for the safetensors leg) and device (set only with --device). Without either
// flag the JSON is byte-identical to the bare manifest report.
type modelPreflightReport struct {
	ggufload.ManifestPreflight
	Format string                `json:"format,omitempty"`
	Device *modelPreflightDevice `json:"device,omitempty"`
}

// runModelPreflight checks a checkpoint header (every shard, no tensor bytes) for unread keys,
// HF-config mismatches, missing/orphan tensors, dtypes without a host kernel, and — with
// --device cuda --arch — weight ops without a CUDA device kernel at that arch, all in one pass.
// Exit 0 ready, 3 refused, 2 usage, 1 I/O.
func runModelPreflight(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model preflight", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gguf := fs.String("gguf", "", "GGUF checkpoint path (any shard of a split checkpoint)")
	safetensors := fs.String("safetensors", "", "safetensors checkpoint directory (index + shards, or a single model.safetensors)")
	hfConfig := fs.String("hf-config", "", "optional HF config.json the checkpoint keys must agree with")
	device := fs.String("device", "", "device to classify weight ops for (cuda)")
	arch := fs.String("arch", "", "CUDA architecture target sm_XY (with --device cuda)")
	asJSON := fs.Bool("json", false, "emit the fak.model-manifest-preflight/1 report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	*gguf = pathutil.ExpandTilde(*gguf)
	*safetensors = pathutil.ExpandTilde(*safetensors)
	*hfConfig = pathutil.ExpandTilde(*hfConfig)
	if (*gguf == "") == (*safetensors == "") {
		fmt.Fprintln(stderr, modelPreflightUsage)
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, modelPreflightUsage)
		return 2
	}
	if (*device != "") != (*arch != "") {
		fmt.Fprintln(stderr, "--device and --arch are a pair")
		fmt.Fprintln(stderr, modelPreflightUsage)
		return 2
	}
	if *device != "" && *device != "cuda" {
		fmt.Fprintf(stderr, "fak model preflight: --device supports only cuda, got %q\n", *device)
		return 2
	}
	if *device != "" {
		if _, _, ok := compute.ParseCUDADeviceOpArch(*arch); !ok {
			fmt.Fprintf(stderr, "fak model preflight: --arch must be a canonical sm_XY target such as sm_103, got %q\n", *arch)
			return 2
		}
	}

	var hfBytes []byte
	if *hfConfig != "" {
		var err error
		if hfBytes, err = os.ReadFile(*hfConfig); err != nil {
			fmt.Fprintf(stderr, "fak model preflight: read hf config: %v\n", err)
			return 1
		}
	}

	report := modelPreflightReport{}
	var (
		ggufTensors []ggufload.TensorInfo
		stTensors   []fakmodel.SafetensorsPreflightTensor
		cfg         fakmodel.Config
	)
	switch {
	case *gguf != "":
		f, err := ggufload.OpenHeaders(*gguf)
		if err != nil {
			fmt.Fprintf(stderr, "fak model preflight: %v\n", err)
			return 1
		}
		in := ggufload.ManifestPreflightInput{Path: *gguf, File: f}
		if len(hfBytes) > 0 {
			in.HFConfig = hfBytes
		}
		report.ManifestPreflight = ggufload.BuildManifestPreflight(in)
		ggufTensors = f.Tensors
		cfg, _ = f.Config()
	default:
		plan, err := fakmodel.BuildSafetensorsPreflight(fakmodel.SafetensorsPreflightInput{Dir: *safetensors, HFConfig: hfBytes})
		if err != nil {
			fmt.Fprintf(stderr, "fak model preflight: %v\n", err)
			return 1
		}
		report.ManifestPreflight = manifestFromSafetensorsPlan(*safetensors, plan)
		report.Format = "safetensors"
		stTensors = plan.Tensors
		cfg = plan.Config
	}

	if *device != "" {
		table := compute.CUDADeviceOpTable()
		decision := cudaStaticKQuantDecision(table, *arch)
		block, refs := buildDeviceReport(table, *arch, decision.retained, ggufTensors, stTensors, cfg)
		report.Device = block
		report.Refusals = append(report.Refusals, refs...)
	}

	finalizePreflightReport(&report.ManifestPreflight)
	if code := emitJSONOrRender(stdout, stderr, "fak model preflight", *asJSON, &report, func(w io.Writer) {
		fmt.Fprint(w, report.Render())
		if report.Device != nil {
			fmt.Fprint(w, renderDeviceBlock(report.Device))
		}
	}); code != 0 {
		return code
	}
	if report.Refused() {
		return 3
	}
	return 0
}

// finalizePreflightReport recomputes counts and verdict after any appended refusals, applying
// the same stable order ggufload.ManifestPreflight.finish uses.
func finalizePreflightReport(p *ggufload.ManifestPreflight) {
	sort.SliceStable(p.Refusals, func(i, j int) bool {
		a, b := p.Refusals[i], p.Refusals[j]
		if a.Reason != b.Reason {
			return a.Reason < b.Reason
		}
		return a.Axis < b.Axis
	})
	p.Counts = map[string]int{}
	for _, r := range p.Refusals {
		p.Counts[string(r.Reason)]++
	}
	p.Verdict = ggufload.ManifestPreflightReady
	if p.Refused() {
		p.Verdict = ggufload.ManifestPreflightRefused
	}
}

// manifestFromSafetensorsPlan maps the header-only safetensors plan onto the shared manifest
// report: admission becomes one config refusal, index/header consistency becomes per-tensor
// missing/orphan refusals, scale-pair gaps become named refusals, host-decode gaps take the
// host census's own host_dequant(<dtype>) axis shape, and role metadata failures name the
// tensor. The census always continues past an admission refusal, so a refused config still
// reports every other seam.
func manifestFromSafetensorsPlan(dir string, plan *fakmodel.SafetensorsPreflight) ggufload.ManifestPreflight {
	out := ggufload.ManifestPreflight{
		Schema:    ggufload.ManifestPreflightSchema,
		Path:      dir,
		ModelType: plan.Config.ModelType,
		DTypes:    plan.DTypes,
		Refusals:  []newmodel.Refusal{},
		Tensors:   ggufload.ManifestTensorCoverage{Mode: "safetensors-index", Present: len(plan.Tensors), Missing: len(plan.Missing), Orphan: len(plan.Orphans)},
	}
	if plan.AdmissionErr != nil {
		out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalHeaderConfigInvalid, "config", plan.AdmissionErr.Error()))
	}
	for _, name := range plan.Missing {
		out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalTensorMissing, "index:"+name, "weight_map entry absent from its declared shard header"))
	}
	for _, name := range plan.Orphans {
		out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalTensorOrphan, "header:"+name, "header tensor claimed by no weight_map entry"))
	}
	for _, g := range plan.ScaleGaps {
		if g.Dtype == "" {
			out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalTensorMissing, "scale:"+g.Weight, fmt.Sprintf("expected F8_E8M0 sibling %s is absent", g.Sibling)))
		} else {
			out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalUnsupportedDtype, "scale:"+g.Weight, fmt.Sprintf("sibling %s has dtype %s, want F8_E8M0", g.Sibling, g.Dtype)))
		}
	}
	keys := make([]string, 0, len(plan.HostGaps))
	for k := range plan.HostGaps {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, key := range keys {
		names := plan.HostGaps[key]
		sort.Strings(names)
		shown := names
		if len(shown) > 6 {
			shown = shown[:6]
		}
		out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(newmodel.RefusalUnsupportedDtype, key, fmt.Sprintf("%d tensors, e.g. %s", len(names), strings.Join(shown, ", "))))
	}
	roleNames := make([]string, 0, len(plan.RoleErrors))
	for n := range plan.RoleErrors {
		roleNames = append(roleNames, n)
	}
	sort.Strings(roleNames)
	for _, name := range roleNames {
		msg := plan.RoleErrors[name]
		reason := newmodel.RefusalHeaderConfigInvalid
		if strings.Contains(msg, "shape") {
			reason = newmodel.RefusalTensorShape
		}
		out.Refusals = append(out.Refusals, newmodel.HeaderRefusal(reason, "role:"+name, msg))
	}
	return out
}

// renderDeviceBlock appends a short human summary of the device classification.
func renderDeviceBlock(d *modelPreflightDevice) string {
	var b strings.Builder
	fmt.Fprintf(&b, "device %s @ %s (%s): scope %s; ", d.Device, d.Arch, d.Coverage, d.Scope)
	counts := map[string]int{}
	for _, c := range d.Classes {
		counts[c]++
	}
	classes := make([]string, 0, len(counts))
	for c := range counts {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	for i, c := range classes {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s=%d", c, counts[c])
	}
	for _, g := range d.Gaps {
		fmt.Fprintf(&b, "\n  gap %s: %d tensors", g.Group, len(g.Tensors))
	}
	b.WriteString("\n")
	return b.String()
}
