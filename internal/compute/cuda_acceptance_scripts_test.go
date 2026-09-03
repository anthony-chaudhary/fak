package compute_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestBuildCUDAScriptNormalizesCRLFArchitectureManifest(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash is required to exercise build_cuda.sh")
	}
	if out, err := exec.Command(bash, "-c", "exit 0").CombinedOutput(); err != nil {
		t.Skipf("bash is not operational: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	script, err := os.ReadFile("build_cuda.sh")
	if err != nil {
		t.Fatal(err)
	}
	computeDir := filepath.Join(t.TempDir(), "internal", "compute")
	if err := os.MkdirAll(computeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(computeDir, "build_cuda.sh")
	if err := os.WriteFile(scriptPath, script, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "sm_80\r\nsm_89\r\nsm_90\r\nsm_100\r\n"
	if err := os.WriteFile(filepath.Join(computeDir, "cuda_arch.txt"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	bashScriptPath := filepath.ToSlash(scriptPath)
	if runtime.GOOS == "windows" && strings.Contains(strings.ToLower(bash), `\windows\system32\bash.exe`) {
		volume := filepath.VolumeName(scriptPath)
		bashScriptPath = "/mnt/" + strings.ToLower(strings.TrimSuffix(volume, ":")) + filepath.ToSlash(strings.TrimPrefix(scriptPath, volume))
	}
	goBin := filepath.Join(t.TempDir(), "bin")
	if err := os.MkdirAll(goBin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(goBin, "go"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	bashGoBin := filepath.ToSlash(goBin)
	if runtime.GOOS == "windows" && strings.Contains(strings.ToLower(bash), `\windows\system32\bash.exe`) {
		volume := filepath.VolumeName(goBin)
		bashGoBin = "/mnt/" + strings.ToLower(strings.TrimSuffix(volume, ":")) + filepath.ToSlash(strings.TrimPrefix(goBin, volume))
	}
	quotedGoBin := "'" + strings.ReplaceAll(bashGoBin, "'", `'"'"'`) + "'"
	quotedScriptPath := "'" + strings.ReplaceAll(bashScriptPath, "'", `'"'"'`) + "'"
	quotedStubPath := "'" + strings.ReplaceAll(strings.TrimSuffix(bashGoBin, "/")+"/go", "'", `'"'"'`) + "'"
	run := func(arch string) (string, error) {
		command := "PATH=" + quotedGoBin + ":/usr/bin:/bin FAK_CUDA_ARCH=" + arch + " PYTHON=" + quotedStubPath + " CC=" + quotedStubPath + " bash " + quotedScriptPath + " check"
		cmd := exec.Command(bash, "-c", command)
		out, err := cmd.CombinedOutput()
		return string(out), err
	}
	if out, err := run("sm_89"); err != nil {
		t.Fatalf("supported CRLF architecture refused: %v\n%s", err, out)
	} else if !strings.Contains(out, "[cuda] OK check") {
		t.Fatalf("supported CRLF architecture did not complete validation:\n%s", out)
	}
	out, err := run("sm_999")
	if err == nil {
		t.Fatalf("unknown architecture accepted:\n%s", out)
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() != 2 {
		t.Fatalf("unknown architecture exit = %v, want status 2:\n%s", err, out)
	}
	if !strings.Contains(out, "unsupported CUDA arch 'sm_999'") {
		t.Fatalf("unknown architecture did not fail at the exact membership check:\n%s", out)
	}
}

func TestBuildCUDAScriptRejectsIncompleteGoToolchainBeforeNVCC(t *testing.T) {
	data, err := os.ReadFile("build_cuda.sh")
	if err != nil {
		t.Fatal(err)
	}
	script := string(data)
	preflight := strings.Index(script, "src/encoding/base32/base32.go")
	nvcc := strings.Index(script, `echo "[cuda] nvcc compile kernels`)
	if preflight < 0 || nvcc < 0 {
		t.Fatalf("build_cuda.sh is missing the complete-toolchain preflight or nvcc boundary")
	}
	if preflight > nvcc {
		t.Fatalf("complete-toolchain preflight occurs after the expensive nvcc build")
	}
	for _, required := range []string{"src/encoding/base32/base32.go", "src/go/types/api.go"} {
		if !strings.Contains(script, required) {
			t.Errorf("build_cuda.sh does not verify %s", required)
		}
	}
	if !strings.Contains(script, "go env GOROOT") || !strings.Contains(script, "incomplete Go toolchain") {
		t.Error("build_cuda.sh does not resolve and clearly refuse an incomplete GOROOT")
	}
}

func TestGPUAcceptanceScriptsResolveCleanEnvironmentToolkit(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("..", "..", "tools", "run_*_acceptance_on_gpu.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no GPU acceptance scripts found")
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		script := string(data)
		if strings.Contains(script, "${CUDA_HOME:-$HOME/") {
			t.Errorf("%s expands unset HOME under `set -u`", filepath.Base(path))
		}
		for _, cache := range []string{"GOCACHE", "GOMODCACHE", "GOPATH"} {
			if !strings.Contains(script, "export "+cache+"=") {
				t.Errorf("%s does not provision %s for a HOME-less clean session", filepath.Base(path), cache)
			}
		}
		if !strings.Contains(script, "[ -x /usr/local/cuda/bin/nvcc ]") {
			t.Errorf("%s does not probe the standard CUDA toolkit outside PATH", filepath.Base(path))
		}
	}
}
