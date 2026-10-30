package devcmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// cudaV41CoverageResult is the coverage subcommand's receipt: the tag set the
// report was compiled with plus the in-binary op table.
type cudaV41CoverageResult struct {
	Schema   string                    `json:"schema"`
	Tags     string                    `json:"tags"`
	Defaults []string                  `json:"v41_defaults,omitempty"`
	Missing  []string                  `json:"require_device_missing,omitempty"`
	Report   compute.V41CoverageReport `json:"report"`
}

// runCUDAV41Coverage compiles and runs TestV41OpCoverageReport with the same
// Go tags and cgo environment as a CUDA build of cfg, against the
// libfakcuda.a a prior `build` left in the compute package. The report is the
// view from inside that tagged binary, so it reflects the artifact's selection.
func runCUDAV41Coverage(stdout, stderr io.Writer, cfg *computebuild.CUDAConfig, jsonOut bool, requireDevice string, command func(context.Context, string, []string, string, []string, io.Writer, io.Writer) error) int {
	root, err := computebuild.ResolveRepoRoot(cfg.RepoRoot)
	if err != nil {
		fmt.Fprintln(stderr, "build-cuda coverage:", err)
		return 2
	}
	pkgDir := cfg.PkgDir
	if pkgDir == "" {
		pkgDir = filepath.Join(root, "internal", "compute")
	}
	tc := cfg.Toolchain
	if tc == nil {
		if tc, err = computebuild.DiscoverToolchain(); err != nil || tc == nil {
			// The tagged go test reports the real link failure if cgo cannot resolve.
			fmt.Fprintln(stderr, "build-cuda coverage: toolchain not discovered; using ambient cgo env:", err)
			tc = &computebuild.Toolchain{}
		}
	}
	tags := computebuild.CUDAGoTags(cfg)
	out, err := os.CreateTemp("", "fak-v41-coverage-*.json")
	if err != nil {
		fmt.Fprintln(stderr, "build-cuda coverage:", err)
		return 2
	}
	outPath := out.Name()
	_ = out.Close()
	defer os.Remove(outPath)
	env := computebuild.MergeEnviron(os.Environ(), computebuild.SynthesizeCUDACgoEnv(tc, pkgDir, cfg.EnableNCCL, nil))
	if len(tc.VsDevEnv) > 0 {
		env = computebuild.MergeEnviron(env, tc.VsDevEnv)
	}
	env = computebuild.MergeEnviron(env, map[string]string{"FAK_V41_COVERAGE_OUT": outPath})
	args := []string{"test", "-tags", tags, "-count=1", "-run", "^TestV41OpCoverageReport$", "./internal/compute/"}
	if err := command(context.Background(), "go", args, root, env, stderr, stderr); err != nil {
		fmt.Fprintln(stderr, "build-cuda coverage:", err)
		return 1
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		fmt.Fprintln(stderr, "build-cuda coverage: report not written:", err)
		return 1
	}
	res := cudaV41CoverageResult{Schema: "fak.cuda-v41-coverage/1", Tags: tags, Defaults: cfg.V41Defaults}
	if err := json.Unmarshal(data, &res.Report); err != nil {
		fmt.Fprintln(stderr, "build-cuda coverage: decoding report:", err)
		return 1
	}
	res.Missing = cudaV41CoverageMissing(res.Report, requireDevice)
	if jsonOut {
		_ = json.NewEncoder(stdout).Encode(res)
	} else {
		fmt.Fprintf(stdout, "tags: %s\nbackend: %s %s\n", res.Tags, res.Report.Backend, res.Report.Tier)
		for _, row := range res.Report.Ops {
			fmt.Fprintf(stdout, "  %-34s %-13s %s\n", row.Op, row.Status, row.Probe)
		}
		for _, dt := range res.Report.WeightTypes {
			fmt.Fprintf(stdout, "  weight %-8s device=%v\n", dt.Dtype, dt.Device)
		}
		fmt.Fprintf(stdout, "device ops: %d/%d\n", res.Report.DeviceOps, res.Report.TotalOps)
	}
	if len(res.Missing) > 0 {
		fmt.Fprintf(stderr, "build-cuda coverage: not on device: %s\n", strings.Join(res.Missing, ","))
		return 1
	}
	return 0
}

// cudaV41CoverageMissing returns the required ops (comma list) that are not
// device rows; an unknown op name counts as missing.
func cudaV41CoverageMissing(r compute.V41CoverageReport, require string) []string {
	status := make(map[string]string, len(r.Ops))
	for _, row := range r.Ops {
		status[row.Op] = row.Status
	}
	var missing []string
	for _, op := range strings.Split(require, ",") {
		op = strings.TrimSpace(op)
		if op != "" && status[op] != compute.V41OpDevice {
			missing = append(missing, op)
		}
	}
	return missing
}
