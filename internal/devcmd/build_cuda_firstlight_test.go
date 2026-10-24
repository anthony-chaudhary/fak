package devcmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

const cudaFirstLightFixtureExecutable = "fixture executable"

func cudaFirstLightFixture(t *testing.T) (string, []string) {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                          "module example.com/firstlight\n",
		"internal/compute/cuda_arch.txt":  "sm_89\nsm_100\nsm_103\nsm_120\n",
		"internal/compute/cuda_backend.h": "void hello(void);\n",
		"weights/config.json":             `{"eos_token_id":99}`,
		"weights/model.safetensors":       "fixture",
	}
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root, []string{"--repo-root", root, "--arch", "sm_120", "--hf", "weights", "--tokens", "2", "--json"}
}

func cudaFirstLightAcceptanceFixture() map[string]any {
	return map[string]any{
		"schema": "fak.gpucheck/1", "engine": "fak-native", "pass": true,
		"requested_backend": "cuda", "observed_backend": "cuda", "observed_tier": "sm_120",
		"device_memory": true, "backend_observed": true, "fallback_active": false,
		"requested_tokens": 2, "compared_tokens": 2, "reference_tokens": []int{10, 20}, "device_tokens": []int{10, 20},
	}
}

func cudaFirstLightFakeDeps(t *testing.T, calls *[]string, acceptance map[string]any) cudaFirstLightDeps {
	t.Helper()
	return cudaFirstLightDeps{
		discover: func() (*computebuild.Toolchain, error) {
			*calls = append(*calls, "discover")
			return &computebuild.Toolchain{CC: "cc", NVCC: "nvcc"}, nil
		},
		build: func(ctx context.Context, cfg *computebuild.CUDAConfig) error {
			*calls = append(*calls, "build")
			if cfg.Command != "binary" || cfg.OutPkg != "./cmd/gpucheck" || cfg.Arch != "sm_120" || !cfg.SkipSmoke || !cfg.SkipSign {
				t.Fatalf("unexpected native build configuration: %+v", cfg)
			}
			if _, ok := ctx.Deadline(); !ok {
				t.Fatal("build is missing its deadline")
			}
			return os.WriteFile(cfg.OutBin, []byte(cudaFirstLightFixtureExecutable), 0o600)
		},
		run: func(ctx context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error {
			if _, ok := ctx.Deadline(); !ok {
				t.Fatal("child is missing its deadline")
			}
			if name == "cc" {
				*calls = append(*calls, "header")
				expectedPrefix := []string{"-x", "c", "-std=c11", "-fsyntax-only", "-Wall", "-Werror"}
				if len(args) < len(expectedPrefix) || !reflect.DeepEqual(args[:len(expectedPrefix)], expectedPrefix) {
					t.Fatalf("header args: %v; want prefix: %v", args, expectedPrefix)
				}
				return nil
			}
			*calls = append(*calls, "accept")
			want := []string{"-hf", filepath.Join(dir, "weights"), "-backend", "cuda", "-n", "2", "-json", "-require-native"}
			if !reflect.DeepEqual(args, want) {
				t.Fatalf("gpucheck arguments: %v", args)
			}
			return json.NewEncoder(stdout).Encode(acceptance)
		},
	}
}

func readCUDAFirstLightReceipt(t *testing.T, root string) cudaFirstLightReceipt {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, ".fak", "cuda-first-light.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r cudaFirstLightReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// fak-test:runtime fast est=1s
// fak-test:justify why=contract when=changed:internal/devcmd/**
func TestCUDAFirstLightOrdinaryPlan(t *testing.T) {
	root, args := cudaFirstLightFixture(t)
	var stdout, stderr bytes.Buffer
	if code := RunBuildCUDA(&stdout, &stderr, append([]string{"first-light"}, append(args, "--plan")...)); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, &stderr)
	}
	r := readCUDAFirstLightReceipt(t, root)
	if r.Outcome != "planned" || !r.Plan || r.NativeModelParity || r.HardwareQualified || len(r.Pending) != 4 || len(r.Acceptance) != 0 {
		t.Fatalf("plan misstates its evidence: %+v", r)
	}
	for _, s := range r.Stages {
		if s.Outcome != "pending" {
			t.Fatalf("plan executed %s", s.Name)
		}
	}
	if _, err := os.Stat(r.Binary); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("plan created a binary: %v", err)
	}
	var printed cudaFirstLightReceipt
	if err := json.Unmarshal(stdout.Bytes(), &printed); err != nil {
		t.Fatal(err)
	}
	if printed.Scope != "native-hf-model-parity" {
		t.Fatalf("scope=%s", printed.Scope)
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=contract when=changed:internal/devcmd/**
func TestCUDAFirstLightExecutionAndFailFirst(t *testing.T) {
	wantBinarySHA256 := fmt.Sprintf("%x", sha256.Sum256([]byte(cudaFirstLightFixtureExecutable)))
	for _, failure := range []string{"", "header", "build", "accept", "timeout"} {
		t.Run(failure, func(t *testing.T) {
			root, args := cudaFirstLightFixture(t)
			var calls []string
			deps := cudaFirstLightFakeDeps(t, &calls, cudaFirstLightAcceptanceFixture())
			realRun, realBuild := deps.run, deps.build
			sentinel := errors.New("fixture stage failure")
			deps.run = func(ctx context.Context, name string, argv []string, dir string, env []string, out, errout io.Writer) error {
				stage := "accept"
				if name == "cc" {
					stage = "header"
				}
				if failure == stage {
					calls = append(calls, stage)
					return sentinel
				}
				if failure == "timeout" {
					calls = append(calls, stage)
					<-ctx.Done()
					return ctx.Err()
				}
				return realRun(ctx, name, argv, dir, env, out, errout)
			}
			deps.build = func(ctx context.Context, cfg *computebuild.CUDAConfig) error {
				if failure == "build" {
					calls = append(calls, "build")
					return sentinel
				}
				return realBuild(ctx, cfg)
			}
			if failure == "timeout" {
				args = append(args, "--timeout", "1s")
			}
			var stdout, stderr bytes.Buffer
			code := runCUDAFirstLightWith(&stdout, &stderr, args, deps)
			r := readCUDAFirstLightReceipt(t, root)
			want := []string{"discover", "header", "build", "accept"}
			switch failure {
			case "header", "timeout":
				want = want[:2]
			case "build":
				want = want[:3]
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls=%v want=%v", calls, want)
			}
			if failure == "" {
				if code != 0 || r.Outcome != "passed" || !r.NativeModelParity || r.BinarySHA256 != wantBinarySHA256 {
					t.Fatalf("exit=%d receipt=%+v", code, r)
				}
			} else if code == 0 || r.Outcome != "failed" || r.NativeModelParity {
				t.Fatalf("failure accepted: exit=%d receipt=%+v", code, r)
			}
			if r.HardwareQualified || len(r.Pending) != 4 {
				t.Fatalf("preparation became qualification: %+v", r)
			}
		})
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=guard when=changed:internal/devcmd/** twin=TestCUDAFirstLightExecutionAndFailFirst
func TestCUDAFirstLightRejectsFalseNativeReceipt(t *testing.T) {
	for _, bad := range []string{"empty", "fallback", "wrong architecture", "host memory", "token mismatch", "missing identity", "invalid json", "oversized"} {
		t.Run(bad, func(t *testing.T) {
			root, args := cudaFirstLightFixture(t)
			m := cudaFirstLightAcceptanceFixture()
			switch bad {
			case "empty":
				m["compared_tokens"] = 0
				m["reference_tokens"] = []int{}
				m["device_tokens"] = []int{}
			case "fallback":
				m["fallback_active"] = true
			case "wrong architecture":
				m["observed_tier"] = "sm_89"
			case "host memory":
				m["device_memory"] = false
			case "token mismatch":
				m["device_tokens"] = []int{10, 21}
			case "oversized":
				m["extra"] = string(make([]byte, 1<<20))
			case "missing identity":
				delete(m, "backend_observed")
			}
			var calls []string
			deps := cudaFirstLightFakeDeps(t, &calls, m)
			if bad == "invalid json" || bad == "oversized" {
				run := deps.run
				deps.run = func(ctx context.Context, name string, argv []string, dir string, env []string, out, errout io.Writer) error {
					if name == "cc" {
						return run(ctx, name, argv, dir, env, out, errout)
					}
					if bad == "invalid json" {
						_, err := fmt.Fprint(out, "{}\n{}\n")
						return err
					}
					encoded, err := json.Marshal(m)
					if err != nil {
						return err
					}
					_, err = io.Copy(out, io.LimitReader(bytes.NewReader(encoded), int64(len(encoded))))
					return err
				}
			}
			var stdout, stderr bytes.Buffer
			if code := runCUDAFirstLightWith(&stdout, &stderr, args, deps); code == 0 {
				t.Fatal("false native acceptance passed")
			}
			r := readCUDAFirstLightReceipt(t, root)
			if r.Outcome != "failed" || r.NativeModelParity || r.Stages[3].Outcome != "failed" {
				t.Fatalf("false receipt: %+v", r)
			}
		})
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=guard when=changed:internal/devcmd/** twin=TestCUDAFirstLightExecutionAndFailFirst
func TestCUDAFirstLightPreflightStopsBeforeToolchain(t *testing.T) {
	root, args := cudaFirstLightFixture(t)
	if err := os.Remove(filepath.Join(root, "weights", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	deps := cudaFirstLightDeps{discover: func() (*computebuild.Toolchain, error) {
		t.Fatal("missing checkpoint reached toolchain")
		return nil, nil
	}}
	var stdout, stderr bytes.Buffer
	if code := runCUDAFirstLightWith(&stdout, &stderr, args, deps); code == 0 {
		t.Fatal("missing checkpoint accepted")
	}
	r := readCUDAFirstLightReceipt(t, root)
	if r.Stages[0].Outcome != "failed" || r.Stages[1].Outcome != "pending" {
		t.Fatalf("preflight continued: %+v", r.Stages)
	}
}
