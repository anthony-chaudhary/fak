package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=20ms
func TestCUDABuildEnvironmentCompatibility(t *testing.T) {
	t.Setenv("FAK_CUDA_ARCH", "89")
	t.Setenv("FAK_CUDA_NCCL", "1")
	t.Setenv("NCCL_HOME", t.TempDir())
	t.Setenv("FAK_NVCC_CCBIN", "legacy-cxx")
	for _, override := range []bool{false, true} {
		args := []string{"binary", "--json", "--skip-sign", "--smoke=false", "./cmd/fak", "result"}
		if override {
			args = append([]string{"binary", "--arch=sm_120", "--nccl=false", "--cxx=explicit-cxx"}, args[1:]...)
		}
		var out, errOut bytes.Buffer
		code := runBuildCUDA(&out, &errOut, args, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
			arch, cxx, nccl := "89", "legacy-cxx", true
			if override {
				arch, cxx, nccl = "sm_120", "explicit-cxx", false
			}
			if cfg.Arch != arch || cfg.EnableNCCL != nccl || cfg.NCCLHome != os.Getenv("NCCL_HOME") || cfg.Toolchain == nil || cfg.Toolchain.CXX != cxx {
				t.Fatalf("environment/flag mapping: arch=%q nccl=%v home=%q toolchain=%v", cfg.Arch, cfg.EnableNCCL, cfg.NCCLHome, cfg.Toolchain)
			}
			if cfg.OutPkg != "./cmd/fak" || cfg.OutBin != "result" || !cfg.SkipSign || !cfg.SkipSmoke {
				t.Fatal("binary options lost")
			}
			return nil
		})
		if code != 0 || !json.Valid(out.Bytes()) {
			t.Fatalf("exit=%d stdout=%s stderr=%s", code, &out, &errOut)
		}
	}
}

func init() {
	if os.Getenv("FAK_CUDA_ENTRYPOINT_HELPER") != "1" {
		return
	}
	var out, errOut bytes.Buffer
	code := runBuildCUDA(&out, &errOut, os.Args[1:], func(_ context.Context, cfg *computebuild.CUDAConfig) error {
		return json.NewEncoder(&out).Encode([]string{cfg.Command, cfg.Arch, cfg.OutPkg, cfg.OutBin})
	})
	_, _ = io.Copy(os.Stdout, &out)
	_, _ = io.Copy(os.Stderr, &errOut)
	os.Exit(code)
}

// fak-test:runtime fast est=100ms
func TestCUDABuildEntrypointSubprocess(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "binary", "--arch=sm_120", "--skip-sign", "--smoke=false", "./cmd/fak", "out/fak")
	cmd.Env = append(os.Environ(), "FAK_CUDA_ENTRYPOINT_HELPER=1")
	output, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	if err := json.Unmarshal(output, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"binary", "sm_120", "./cmd/fak", "out/fak"}) {
		t.Fatalf("dispatch=%v", got)
	}
}

// fak-test:runtime fast est=10ms
func TestCUDABuildMakeEntrypoints(t *testing.T) {
	root, err := computebuild.ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(root, "Makefile"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, target := range []string{"cuda-check", "cuda-build", "cuda-test", "cuda-build-sm100", "cuda-build-sm103", "cuda-build-sm120", "cuda-accept-model", "cuda-accept"} {
		start := strings.Index(text, "\n"+target+":")
		if start < 0 {
			t.Fatalf("target %s missing", target)
		}
		body := strings.SplitN(text[start+1:], "\n\n", 2)[0]
		if !strings.Contains(body, "$(CUDA_BUILD)") || strings.Contains(body, ".sh") {
			t.Fatalf("target %s does not use native driver: %s", target, body)
		}
	}
}

// fak-test:runtime fast est=30ms
func TestCUDABuildAcceptanceFamiliesFailClosed(t *testing.T) {
	for _, mode := range []string{"pass", "skip", "missing", "crash"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			for name, data := range map[string]string{
				"go.mod": "module acceptance-test\n", "internal/compute/cuda_arch.txt": "sm_120\n",
				"model/config.json": "{}", "model/model.safetensors": "fixture",
			} {
				path := filepath.Join(root, filepath.FromSlash(name))
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(data), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var out, errOut bytes.Buffer
			families := cudaAcceptanceFamilies()
			familyIndex, builds := 0, 0
			code := runBuildCUDAWith(&out, &errOut, []string{"accept", "--repo-root", root, "--arch=sm_120", "--hf=model", "--tokens=1", "--skip-sign", "--smoke=false"}, func(_ context.Context, cfg *computebuild.CUDAConfig) error {
				builds++
				if cfg.Command != "binary" || cfg.OutPkg != "./cmd/gpucheck" || !cfg.SkipSmoke || !cfg.SkipSign {
					t.Fatal("acceptance dropped build options")
				}
				cfg.Toolchain = &computebuild.Toolchain{}
				return nil
			}, func(_ context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error {
				if filepath.Clean(dir) != filepath.Clean(root) {
					t.Fatalf("dir=%q", dir)
				}
				if name != "go" {
					_, err := io.WriteString(stdout, `{"schema":"fak.gpucheck/1","engine":"fak-native","pass":true,"requested_backend":"cuda","observed_backend":"cuda","observed_tier":"sm_120","device_memory":true,"backend_observed":true,"fallback_active":false,"requested_tokens":1,"compared_tokens":1,"reference_tokens":[4],"device_tokens":[4]}`)
					return err
				}
				if !containsCUDAArg(args, "-json") {
					return nil
				}
				family := families[familyIndex]
				familyIndex++
				if !containsCUDAArg(args, "^("+strings.Join(family.Tests, "|")+")$") || args[len(args)-1] != family.Package {
					t.Fatalf("family invocation=%v", args)
				}
				if mode == "crash" {
					return errors.New("fixture exit")
				}
				for i, test := range family.Tests {
					if mode == "missing" && i == 0 {
						continue
					}
					action := "pass"
					if mode == "skip" && i == 0 {
						action = "skip"
					}
					fmt.Fprintf(stdout, "{\"Action\":%q,\"Test\":%q}\n", action, test)
				}
				return nil
			})
			want := 0
			if mode != "pass" {
				want = 1
			}
			if code != want || builds != 1 || familyIndex != len(families) {
				t.Fatalf("exit=%d builds=%d families=%d stderr=%s", code, builds, familyIndex, &errOut)
			}
			var result struct {
				Success  bool
				Families []cudaAcceptanceFamily
			}
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Success != (mode == "pass") || len(result.Families) != len(families) {
				t.Fatal("acceptance verdict mismatch")
			}
		})
	}
}

func containsCUDAArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}
