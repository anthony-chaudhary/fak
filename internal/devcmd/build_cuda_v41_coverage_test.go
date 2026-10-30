package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms lane=default
// A bare sm_120 build selects the default V4.1 set; --v41-defaults=false and
// FAK_CUDA_V41=0 opt out; an explicit pool-only flag keeps the ds4 compressor off.
func TestBuildCUDAV41Defaults(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		env                               string
		args                              []string
		compressor, pool, router, sharedA bool
	}{
		{"default", "", nil, true, false, true, true},
		{"flag opt out", "", []string{"--v41-defaults=false"}, false, false, false, false},
		{"env opt out", "0", nil, false, false, false, false},
		{"pool explicit", "", []string{"--v41-compressor-pool"}, false, true, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(computebuild.CUDAV41DefaultsEnv, tc.env)
			var out, errout bytes.Buffer
			called := false
			args := append([]string{"build", "--arch", "sm_120"}, tc.args...)
			code := runBuildCUDA(&out, &errout, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				if cfg.EnableV41Compressor != tc.compressor || cfg.EnableV41CompressorPool != tc.pool || cfg.EnableV41Router != tc.router || cfg.EnableV41SharedAttention != tc.sharedA {
					t.Fatalf("selection = compressor %v pool %v router %v shared %v", cfg.EnableV41Compressor, cfg.EnableV41CompressorPool, cfg.EnableV41Router, cfg.EnableV41SharedAttention)
				}
				if cfg.EnableV41TailRoPE || cfg.EnableV41BF16Publish || cfg.EnableV41MHC || cfg.EnableV41EngramGate || cfg.EnableV41RoutedSwiGLU {
					t.Fatal("default selected an unwitnessed candidate")
				}
				return nil
			})
			if code != 0 || !called {
				t.Fatalf("code=%d called=%v stderr=%s", code, called, errout.String())
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
// The coverage subcommand runs the in-package report with the build's tags
// and fails closed on --require-device rows that are not device.
func TestBuildCUDAV41CoverageCommand(t *testing.T) {
	t.Setenv(computebuild.CUDAV41DefaultsEnv, "")
	report := compute.V41CoverageReport{Schema: compute.V41CoverageReportSchema, Backend: "cuda", Ops: []compute.V41OpStatus{
		{Op: "V41RouterTopKBackend", Status: compute.V41OpDevice, Probe: compute.V41ProbeAvailable},
		{Op: "V41MHCBackend", Status: compute.V41OpAbsent},
	}, DeviceOps: 1, TotalOps: 2}
	run := func(require string) (int, string, []string) {
		var out, errout bytes.Buffer
		var gotArgs []string
		args := []string{"coverage", "--arch", "sm_120", "--json", "--require-device", require}
		code := runBuildCUDAWith(&out, &errout, args, func(context.Context, *computebuild.CUDAConfig) error {
			t.Fatal("coverage must not rebuild")
			return nil
		}, func(_ context.Context, name string, args []string, _ string, env []string, _, _ io.Writer) error {
			gotArgs = args
			for _, kv := range env {
				if path, ok := strings.CutPrefix(kv, "FAK_V41_COVERAGE_OUT="); ok {
					data, _ := json.Marshal(report)
					return os.WriteFile(path, data, 0o644)
				}
			}
			t.Fatal("coverage output path not passed")
			return nil
		})
		return code, out.String(), gotArgs
	}
	code, out, args := run("V41RouterTopKBackend")
	if code != 0 {
		t.Fatalf("code=%d out=%s", code, out)
	}
	var res cudaV41CoverageResult
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatal(err)
	}
	if res.Tags != "cuda,fak_cuda_v41_compressor,fak_cuda_v41_router,fak_cuda_v41_shared_attention" || res.Report.DeviceOps != 1 || len(res.Missing) != 0 {
		t.Fatalf("result = %+v", res)
	}
	if !containsCUDAArg(args, res.Tags) || !containsCUDAArg(args, "^TestV41OpCoverageReport$") {
		t.Fatalf("go test args = %v", args)
	}
	if code, _, _ := run("V41RouterTopKBackend,V41MHCBackend"); code != 1 {
		t.Fatalf("missing device row exit = %d, want 1", code)
	}
}
